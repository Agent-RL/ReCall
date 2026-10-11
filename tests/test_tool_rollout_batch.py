import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto


@pytest.fixture
def rollout_module(monkeypatch):
    # Load the real rollout without initializing a GPU or importing vLLM kernels.
    vllm = ModuleType("vllm")
    vllm.LLM = vllm.SamplingParams = object
    distributed = ModuleType("vllm.distributed")
    distributed.parallel_state = SimpleNamespace(
        _TP=SimpleNamespace(broadcast_object=lambda data, src: data)
    )
    third_party = ModuleType("verl.third_party.vllm")
    third_party.vllm_version = None
    for name, module in {
        "vllm": vllm,
        "vllm.distributed": distributed,
        "verl.third_party.vllm": third_party,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    path = (
        Path(__file__).parents[1]
        / "src/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py"
    )
    spec = importlib.util.spec_from_file_location("tool_rollout_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Tokenizer:
    pad_token_id = 0

    def encode(self, text):
        return [ord(c) + 100 for c in text]

    def decode(self, tokens):
        return "".join(chr(int(token) - 100) for token in tokens if token >= 100)


@pytest.mark.parametrize(
    "metadata,overrides,repeats",
    [
        ({"do_sample": True}, {}, 3),
        ({"do_sample": True, "validate": True}, {}, 1),
        ({"do_sample": False, "validate": True}, {}, 1),
        ({"do_sample": True}, {"n": 2}, 2),
    ],
    ids=["training", "sampled-validation", "greedy-validation", "override"],
)
def test_tool_rollout_packs_the_generated_batch(
    rollout_module, monkeypatch, metadata, overrides, repeats
):
    tokenizer = Tokenizer()
    rollout = object.__new__(rollout_module.vLLMRolloutWithTool)
    rollout.config = OmegaConf.create(
        {
            "n": 3,
            "response_length": 512,
            "max_turns": 3,
            "sandbox_url": "http://sandbox.invalid",
            "val_kwargs": {"top_k": -1, "top_p": 1.0, "temperature": 0.7},
        }
    )
    rollout.sampling_params = SimpleNamespace(
        n=3,
        max_tokens=512,
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        min_p=0.0,
        best_of=1,
        stop_token_ids=[],
    )
    rollout.tokenizer = tokenizer
    rollout.pad_token_id = 0
    rollout.tp_rank = 0
    rollout.gen_ids = tokenizer.encode("\n<|im_start|>assistant\n<think>")
    dispatched = []

    def generate(prompts, sampling_params, use_tqdm):
        assert sampling_params.n == 1
        dispatched.append(prompts)
        outputs = []
        for prompt in prompts:
            source = prompt["prompt_token_ids"][0]
            text = f"answer-{source}"
            if len(dispatched) == 1 and source == 10:
                text = '<tool_call>{"name":"lookup","arguments":{}}</tool_call>'
            outputs.append(
                SimpleNamespace(
                    outputs=[
                        SimpleNamespace(
                            token_ids=tokenizer.encode(text) + [1],
                            finish_reason="stop",
                            stop_reason=None,
                        )
                    ]
                )
            )
        return outputs

    rollout.inference_engine = SimpleNamespace(generate=generate)
    monkeypatch.setattr(
        rollout_module.requests,
        "post",
        lambda url, json, timeout: SimpleNamespace(
            status_code=200,
            json=lambda: {"result": json["env"], "output": "", "error": None},
        ),
    )
    # Validation already repeats examples in the trainer before calling the worker.
    ids = torch.tensor([[0, 10], [0, 10], [0, 20], [0, 20]])
    attention = ids.ne(0).long()
    positions = torch.zeros_like(ids)
    prompts = DataProto.from_dict(
        {"input_ids": ids, "attention_mask": attention, "position_ids": positions},
        {
            "env": np.array(
                ["evidence-10", "evidence-10", "evidence-20", "evidence-20"],
                dtype=object,
            )
        },
        {"eos_token_id": 1, **metadata},
    )

    result = rollout.generate_sequences(prompts, **overrides)

    assert len(result) == len(prompts) * repeats
    torch.testing.assert_close(
        result.batch["prompts"], ids.repeat_interleave(repeats, dim=0)
    )
    torch.testing.assert_close(
        result.batch["attention_mask"][:, :2],
        attention.repeat_interleave(repeats, dim=0),
    )
    torch.testing.assert_close(
        result.batch["position_ids"][:, :2], positions.repeat_interleave(repeats, dim=0)
    )
    assert [len(batch) for batch in dispatched] == [4 * repeats, 2 * repeats]
    for row, source in enumerate(ids[:, -1].repeat_interleave(repeats)):
        response = result.batch["responses"][row]
        assert tokenizer.decode(response).endswith(f"answer-{source}")
        if source == 10:
            assert "evidence-10" in tokenizer.decode(response)
            tool_tokens = (
                result.batch["attention_mask"][row, 2:] - result.batch["loss_mask"][row]
            )
            assert "evidence-10" in tokenizer.decode(response[tool_tokens.bool()])
    assert rollout.sampling_params.n == 3
