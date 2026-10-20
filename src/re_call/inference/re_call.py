import re
import json
import requests
import time
import uuid
from typing import List
from functools import wraps

def retry(max: int=10, sleep: int=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(max):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"[retry] try {i} times")
                    if i == max - 1:
                        raise Exception("Retry {} failed after {} times".format(func.__name__, max))
                    elif sleep:
                        time.sleep(sleep)
        return wrapper
    return decorator

class ReCall():
    system_prompt = """In this environment you have access to a set of tools you can use to assist with the user query. \
You may perform multiple rounds of function calls. \
In each round, you can call one or more functions. \

Here are available functions in JSONSchema format: \n```json\n{func_schemas}\n```

In your response, you need to first think about the reasoning process in the mind and then conduct function calling to get the information or perform the actions if needed. \
The reasoning process and function calling are enclosed within <think> </think> and <tool_call> </tool_call> tags. \
The results of the function calls will be given back to you after execution, \
and you can continue to call functions until you get the final answer for the user's question. \
Finally, if you have got the answer, enclose it within \\boxed{{}} with latex format and do not continue to call functions, \
i.e., <think> Based on the response from the function call, I get the weather information. </think> The weather in Beijing on 2025-04-01 is \\[ \\boxed{{20C}} \\].

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""

    def __init__(self, model_url, executor_url=None, *, search_enabled=True, search_timeout=30):
        self.model_url = model_url
        self.executor_url = executor_url
        self.search_enabled = search_enabled
        self.search_timeout = search_timeout
        
    def init_prompt(self, func_schemas, question):
        system_prompt = f"<|im_start|>system\n{self.system_prompt.format(func_schemas=func_schemas)}<|im_end|>"
        user_prompt = f"<|im_start|>user\n{question}<|im_end|>"
        assistant_prefix = f"<|im_start|>assistant\n<think>"
        return system_prompt + "\n" + user_prompt + "\n" + assistant_prefix

    def cat_assistant_response(self, curr_prompt, assistant_response):
        return curr_prompt + assistant_response + "<|im_end|>"
    
    def cat_tool_results(self, curr_prompt, tool_calls, results):
        tool_response_str = ""
        for tool_call, result in zip(tool_calls, results):
            tool_response_str += f"<tool_response>{tool_call}\n{result}\n</tool_response>\n"
        tool_response_str = f"<|im_start|>user\n{tool_response_str}<|im_end|>"
        assistant_prefix = f"<|im_start|>assistant\n<think>"
        return curr_prompt + "\n" + tool_response_str + "\n" + assistant_prefix

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
    
    def execute_tool_calls(self, env: str, tool_calls: List[str]) -> List[str]:
        def exe_tool_call(env, call):
            url = self.executor_url + '/execute'

            call_str = self.format_tool_call(call)
            if call_str.startswith("error: parse tool call failed"):
                return call_str

            try:
                data = {
                    'env': env,
                    'call': call_str
                }
                response = requests.post(url, json=data, timeout=3)
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
        
        results = []
        for tool_call in tool_calls:
            result = exe_tool_call(env, tool_call)
            results.append(result)
        return results
    
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
        
    def run(self, env=None, func_schemas=None, question=None):
        """Run inference with explicit sandbox tools or unconfigured web tools.

        Each call starts an independent conversation. This is a synchronous API;
        async applications should call it in a worker thread.
        """
        if question is None:
            raise ValueError("question is required")
        if env is None and func_schemas is None and self.executor_url is None:
            if self.search_enabled:
                from .parallel_search import ParallelSearch

                search = ParallelSearch(timeout=self.search_timeout)
                session_id = str(uuid.uuid4())
                func_schemas = search.schemas()
                execute = lambda calls: search.execute(calls, session_id)
            else:
                func_schemas = "[]"
                execute = lambda calls: ["error: web tools are disabled"] * len(calls)
        elif env is None and func_schemas in ([], "[]") and self.executor_url is None:
            execute = lambda calls: ["error: no tools are configured"] * len(calls)
        else:
            # A partially configured sandbox is never an invitation to use the web.
            if env is None or func_schemas is None or self.executor_url is None:
                raise ValueError("sandbox tools require env, func_schemas and executor_url")
            execute = lambda calls: self.execute_tool_calls(env, calls)
        # Keep the ID and selected executor across the existing generation retry.
        return self._run(func_schemas, question, execute)

    @retry(max=5, sleep=1)
    def _run(self, func_schemas, question, execute):
        curr_prompt = self.init_prompt(func_schemas, question)
        for _ in range(5):
            response = requests.post(
                f'{self.model_url}/generate', 
                json={
                    "text": curr_prompt,
                    "sampling_params": {
                        "temperature": 0.0,
                        "max_new_tokens": 512
                    }
                }
            ).json()
            curr_prompt = self.cat_assistant_response(curr_prompt, response['text'])

            tool_calls: List[str] = self.extract_tool_calls(response['text'])
            if len(tool_calls) == 0:
                break

            results: List[str] = execute(tool_calls)
            curr_prompt = self.cat_tool_results(curr_prompt, tool_calls, results)

        return curr_prompt
