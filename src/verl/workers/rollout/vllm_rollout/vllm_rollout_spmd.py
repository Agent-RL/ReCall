# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""
import os
import numpy as np
from typing import List
from contextlib import contextmanager
from omegaconf import DictConfig
import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn
from typing import Any, Union
from verl import DataProto
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.workers.rollout.base import BaseRollout
from vllm.distributed import parallel_state as vllm_ps
from vllm import LLM, SamplingParams
from verl.third_party.vllm import vllm_version

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


class vLLMRollout(BaseRollout):

    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        assert not (not config.enforce_eager and config.free_cache_engine), \
            "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get('tensor_model_parallel_size', 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), \
            "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get('max_num_batched_tokens', 8192)

        if kwargs.get('train_tp', None) is not None:
            # deployed with megatron
            os.environ['CUDA_TIMER_STREAM_KAFKA_ENABLE'] = '0'
            os.environ['MEGATRON_IMPORT_TIMERS'] = '0'
            if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3'):
                train_tp = kwargs.get('train_tp', None)
                num_tp_per_train_tp = train_tp // tensor_parallel_size
                vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size,
                                                  num_tp_per_train_tp=num_tp_per_train_tp)
            else:
                vllm_ps.initialize_model_parallel(tensor_model_parallel_size=tensor_parallel_size)

        assert model_hf_config.max_position_embeddings >= config.prompt_length + config.response_length, \
            "model context length should be greater than total sequence length"

        max_model_len = self.config.max_model_len if self.config.max_model_len \
                        else config.prompt_length + config.response_length
        max_model_len = int(max_model_len)

        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError('Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill')

        trust_remote_code = kwargs.get('trust_remote_code', False)
        load_format = 'dummy' if config.load_format.startswith('dummy') else config.load_format

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=True,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            disable_mm_preprocessor_cache=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=int(os.getenv("RANK", "0")) // tensor_parallel_size,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # # we may detokenize the result all together later
        if vllm_version != '0.3.1':
            kwargs['detokenize'] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        idx = prompts.batch['input_ids']  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']

        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if 'raw_prompt_ids' not in non_tensor_batch:
            non_tensor_batch['raw_prompt_ids'] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object)

        if batch_size != len(non_tensor_batch['raw_prompt_ids']):
            raise RuntimeError('vllm sharding manager is not work properly.')

        if 'multi_modal_data' in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(non_tensor_batch.pop('raw_prompt_ids'),
                                                        non_tensor_batch.pop('multi_modal_data')):
                vllm_inputs.append({'prompt_token_ids': raw_prompt_ids, 'multi_modal_data': multi_modal_data})
        else:
            vllm_inputs = [{
                'prompt_token_ids': raw_prompt_ids
            } for raw_prompt_ids in non_tensor_batch.pop('raw_prompt_ids')]

        # ensure the type of `prompt_token_ids` passed to vllm is list[int]
        # https://github.com/volcengine/verl/pull/772
        for input_data in vllm_inputs:
            if isinstance(input_data['prompt_token_ids'], np.ndarray):
                input_data['prompt_token_ids'] = input_data['prompt_token_ids'].tolist()
            elif not isinstance(input_data['prompt_token_ids'], list):
                raise TypeError(
                    f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}")

        do_sample = prompts.meta_info.get('do_sample', True)
        is_validate = prompts.meta_info.get('validate', False)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                'top_k': self.config.val_kwargs.top_k,
                'top_p': self.config.val_kwargs.top_p,
                'temperature': self.config.val_kwargs.temperature,
                'n': 1,  # if validate, already repeat in ray_trainer
            }

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                use_tqdm=False)

            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

            response = []
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    response.append(output.outputs[sample_id].token_ids)

            response = pad_2d_list_to_length(response, self.pad_token_id,
                                             max_length=self.config.response_length).to(idx.device)

            if self.sampling_params.n > 1 and do_sample:
                idx = _repeat_interleave(idx, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                batch_size = batch_size * self.sampling_params.n
                if 'multi_modal_inputs' in non_tensor_batch.keys():
                    non_tensor_batch['multi_modal_inputs'] = _repeat_interleave(non_tensor_batch['multi_modal_inputs'],
                                                                                self.sampling_params.n)

            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(response_id=response,
                                                    eos_token=eos_token_id,
                                                    dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                'prompts': idx,
                'responses': response,
                'input_ids': seq,  # here input_ids become the whole sentences
                # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                'attention_mask': attention_mask,
                'position_ids': position_ids
            },
            batch_size=batch_size)

        # free vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

import re
import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from verl.utils.torch_functional import pad_sequence_to_length

class vLLMRolloutWithTool(vLLMRollout):
    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        super().__init__(model_path, config, tokenizer, model_hf_config, **kwargs)
        self.tokenizer = tokenizer
        self.tp_rank = vllm_ps.get_tensor_model_parallel_rank()

        self.gen_str = "\n<|im_start|>assistant\n<think>"
        self.gen_ids = self.tokenizer.encode(self.gen_str)
    
    def format_tool_call(self, tool_call_str: str):
        """Convert JSON function call description to Python executable code string."""
        try:
            call_json = json.loads(tool_call_str)
            func_name = call_json['name']
            arguments = call_json.get('arguments', {})
            
            args_str = ', '.join(f"{k}={repr(v)}" for k, v in arguments.items())
            return f"{func_name}({args_str})"
        except Exception as e:
            return f"Parse tool call failed: {e}"

    def validate_tool_calls(self, output_str):
        start_tags = re.findall(r'<tool_call>', output_str)
        end_tags = re.findall(r'</tool_call>', output_str)
        
        if len(start_tags) != len(end_tags):
            return False
            
        start_positions = [m.start() for m in re.finditer(r'<tool_call>', output_str)]
        end_positions = [m.start() for m in re.finditer(r'</tool_call>', output_str)]
        
        for start, end in zip(start_positions, end_positions):
            if start >= end:
                return False
                
        return True

    def extract_tool_calls(self, output_str):
        if not self.validate_tool_calls(output_str):
            return []

        try:
            pattern = r'<tool_call>((?:(?!</tool_call>).)*)</tool_call>'
            matches = re.finditer(pattern, output_str, re.DOTALL)
            
            return [match.group(1).strip() for match in matches]
        except Exception as e:
            return []
    
    def batch_execute(self, env_list: List[str], tool_calls_list: List[List[str]]):
        def exe_tool_call(env, call):
            url = f'{self.config.sandbox_url}/execute'

            call_str = self.format_tool_call(call)
            if call_str.startswith("Parse tool call failed"):
                return call_str
            
            try:
                data = {
                    'env': env,
                    'call': call_str
                }                
                response = requests.post(url, json=data, timeout=10)
                if response.status_code != 200:
                    return f"error: {response.status_code}"
                response = response.json()
                ret_str = ''
                if response['result']:
                    ret_str += f'result: \n{response["result"]}\n'
                if response['output']:
                    ret_str += f'output: \n{response["output"]}\n'
                if response['error']:
                    ret_str += f'error: \n{response["error"]}\n'
                return ret_str.strip()
            except requests.exceptions.Timeout:
                return "error: execution timed out"
            except Exception as e:
                return str(e)

        # flatten all tasks
        all_tasks = []
        task_indices = []
        for env_idx, (env, tool_calls) in enumerate(zip(env_list, tool_calls_list)):
            for call_idx, tool_call in enumerate(tool_calls):
                all_tasks.append((env, tool_call))
                task_indices.append((env_idx, call_idx))

        # parallel execute all tasks
        all_results = [None] * len(all_tasks)
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_index = {executor.submit(exe_tool_call, env, call): i 
                            for i, (env, call) in enumerate(all_tasks)}
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                all_results[index] = future.result()

        # reorganize results to original structure
        results_list = [[None for _ in range(len(tool_calls_list[i]))] for i, _ in enumerate(env_list)]
        for (env_idx, call_idx), result in zip(task_indices, all_results):
            results_list[env_idx][call_idx] = result

        return results_list

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:        
        # rebuild vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        ori_input_ids = prompts.batch['input_ids']  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']

        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']

        batch_size = ori_input_ids.size(0)

        idx_list = []
        # parse idx from torch.Tensor to List[List[str]]
        for i in range(batch_size):
            idx_list.append(_pre_process_inputs(self.pad_token_id, ori_input_ids[i]))

        do_sample = prompts.meta_info.get('do_sample', True)
        is_validate = prompts.meta_info.get('validate', False)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                'top_k': self.config.val_kwargs.top_k,
                'top_p': self.config.val_kwargs.top_p,
                'temperature': self.config.val_kwargs.temperature,
                'n': 1,  # if validate, already repeat in ray_trainer
            }

        with self.update_sampling_params(**kwargs):
            # prepare n copies for each input
            rollout_n = self.sampling_params.n
            curr_inputs = []
            for input_ids in idx_list:
                for _ in range(rollout_n):
                    curr_inputs.append(input_ids.copy())
            init_inputs = [ids.copy() for ids in curr_inputs]

            # if there are envs, prepare n copies for each env
            env_list = None
            if 'env' in prompts.non_tensor_batch:
                env_list = []
                for env in prompts.non_tensor_batch['env']:
                    for _ in range(rollout_n):
                        env_list.append(env)

            # track the status of each input
            curr_max_tokens = [self.sampling_params.max_tokens] * len(curr_inputs)
            active_indices = list(range(len(curr_inputs)))

            # collect the result mask of each rollout, 1 for non-result, 0 for tool call result or pad
            result_mask_list = [[] for _ in range(len(curr_inputs))]

            # generate until all inputs are completed
            for step in range(self.config.max_turns):
                if len(active_indices) == 0:
                    break

                # only process the active inputs
                active_inputs = [curr_inputs[i] for i in active_indices]
                active_max_tokens = [curr_max_tokens[i] for i in active_indices]
                
                with self.update_sampling_params(
                    n=1, 
                    max_tokens=min(512, max(active_max_tokens)),
                    stop_token_ids=[151644],
                    top_p=0.99,
                ):  # 512 at most, and add <|im_start|> as stop for corner case
                    vllm_inputs = [{
                        'prompt_token_ids': raw_prompt_ids
                    } for raw_prompt_ids in active_inputs]
                    outputs = self.inference_engine.generate(
                        prompts=vllm_inputs,
                        sampling_params=self.sampling_params,
                        use_tqdm=False
                    )

                # collect all tool calls
                tool_calls_list: List[List[str]] = []
                call_indices: List[int] = []

                # process each output
                new_active_indices = []
                for i, idx in enumerate(active_indices):
                    output_ids = outputs[i].outputs[0].token_ids
                    finish_reason = outputs[i].outputs[0].finish_reason
                    stop_reason = outputs[i].outputs[0].stop_reason
                    
                    if finish_reason == 'stop' and (stop_reason == None or stop_reason == self.tokenizer.pad_token_id):
                        curr_inputs[idx] += output_ids
                        result_mask_list[idx] += [1] * len(output_ids)

                        output_str = self.tokenizer.decode(output_ids)
                        tool_calls: List[str] = self.extract_tool_calls(output_str)
                        if tool_calls:
                            tool_calls_list.append(tool_calls)
                            call_indices.append(idx)
                            new_active_indices.append(idx)
                        else:
                            pass # no tool calls
                    elif finish_reason == 'length':
                        # output over max tokens
                        curr_inputs[idx] += output_ids
                        result_mask_list[idx] += [1] * len(output_ids)
                    elif finish_reason == 'stop' and stop_reason == 151644: # 151644 is the id of <|im_start|>, is a illigal stop, we stop here
                        curr_inputs[idx] += output_ids
                        result_mask_list[idx] += [1] * len(output_ids)
                    else:
                        raise ValueError(f"unknown stop reason. finish_reason: {finish_reason}, stop_reason: {stop_reason}")

                # batch process tool calls
                if tool_calls_list:
                    # Only tp_rank 0 executes the tools
                    if self.tp_rank == 0:
                        active_env_list = [env_list[i] for i in call_indices]
                        tool_responses_list = self.batch_execute(active_env_list, tool_calls_list)
                        
                        # Prepare data for broadcasting
                        broadcast_data = {
                            'tool_calls_list': tool_calls_list,
                            'call_indices': call_indices,
                            'tool_responses_list': tool_responses_list
                        }
                    else:
                        broadcast_data = None
                    
                    broadcast_data = vllm_ps._TP.broadcast_object(broadcast_data, src=0)
                    
                    # All ranks process the broadcasted data
                    if broadcast_data is not None:
                        tool_calls_list = broadcast_data['tool_calls_list']
                        call_indices = broadcast_data['call_indices']
                        tool_responses_list = broadcast_data['tool_responses_list']

                        for idx, tool_calls, tool_responses in zip(call_indices, tool_calls_list, tool_responses_list):
                            tool_response_str = ''
                            for call, response in zip(tool_calls, tool_responses):
                                tool_response_str += f"<tool_response>{call}\n{response}\n</tool_response>\n"
                            tool_response_str = "\n<|im_start|>user\n" + tool_response_str + "<|im_end|>"
                            output_ids = self.tokenizer.encode(tool_response_str)
                            curr_inputs[idx] += output_ids
                            result_mask_list[idx] += [0] * len(output_ids)

                            curr_inputs[idx] += self.gen_ids
                            result_mask_list[idx] += [0] * len(self.gen_ids)

                # check if need to truncate, if yes, truncate, and remove from active; if no, update curr_max_tokens
                length_checked_active_indices = []
                for idx in active_indices:
                    assert len(curr_inputs[idx]) - len(init_inputs[idx]) == len(result_mask_list[idx]), f"curr_inputs: {len(curr_inputs[idx])}, init_inputs: {len(init_inputs[idx])}, result_mask_list: {len(result_mask_list[idx])}"
                    if len(curr_inputs[idx]) - len(init_inputs[idx]) >= self.config.response_length:
                        curr_inputs[idx] = init_inputs[idx] \
                            + curr_inputs[idx][len(init_inputs[idx]):len(init_inputs[idx])+self.config.response_length]
                        result_mask_list[idx] = result_mask_list[idx][:self.config.response_length]
                    else:
                        curr_max_tokens[idx] = self.config.response_length - len(curr_inputs[idx]) + len(init_inputs[idx])
                        if idx in new_active_indices:
                            length_checked_active_indices.append(idx)
                active_indices = length_checked_active_indices

            output_ids_list = []
            # collect the all rollouts
            for i, input_ids in enumerate(idx_list):
                for j in range(rollout_n):
                    idx = i * rollout_n + j
                    input_len = len(input_ids)
                    output_ids_list.append(curr_inputs[idx][input_len:])

        response_attention_mask_list = []
        response_list = []
        result_mask_list_padded = []
        for output_ids, result_mask in zip(output_ids_list, result_mask_list):
            assert len(output_ids) == len(result_mask), f"output_ids: {len(output_ids)}, result_mask: {len(result_mask)}"
            # to tensor 
            response = torch.tensor(output_ids, device=ori_input_ids.device)
            result_mask = torch.tensor(result_mask, device=ori_input_ids.device)
            # response attention mask, 1 for valid, 0 for invalid
            response_attention_mask = torch.ones_like(response, dtype=torch.int64)
            response_attention_mask = pad_sequence_to_length(response_attention_mask, self.config.response_length, 0)
            response_attention_mask_list.append(response_attention_mask)
            # response, pad to response_length
            response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)
            response_list.append(response)
            # result mask, 1 for non-result, 0 for result or pad
            result_mask = pad_sequence_to_length(result_mask, self.config.response_length, 0)
            result_mask_list_padded.append(result_mask)
        response_attention_mask = torch.stack(response_attention_mask_list, dim=0)
        response = torch.stack(response_list, dim=0)
        result_mask = torch.stack(result_mask_list_padded, dim=0)

        if rollout_n > 1:
            ori_input_ids = ori_input_ids.repeat_interleave(rollout_n, dim=0)
            attention_mask = attention_mask.repeat_interleave(rollout_n, dim=0)
            position_ids = position_ids.repeat_interleave(rollout_n, dim=0)
            batch_size = batch_size * rollout_n
        seq = torch.cat([ori_input_ids, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
                
        # concat attenion_mask for input and response
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # result mask: result part is 0, other part is 1
        loss_mask = result_mask * response_attention_mask
        
        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict({
            'prompts': ori_input_ids,
            'responses': response,
            'input_ids': seq,  # here input_ids become the whole sentences
            'attention_mask': attention_mask,
            'loss_mask': loss_mask,
            'position_ids': position_ids
        }, batch_size=batch_size)

        # free vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch)
