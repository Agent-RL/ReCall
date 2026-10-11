import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import requests

from re_call import ReCall


@pytest.fixture
def training_rollout(monkeypatch):
    # Exercise the real batch executor without loading GPU inference kernels.
    vllm = ModuleType("vllm")
    vllm.LLM = vllm.SamplingParams = object
    distributed = ModuleType("vllm.distributed")
    distributed.parallel_state = SimpleNamespace()
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
    spec = importlib.util.spec_from_file_location("tool_result_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rollout = object.__new__(module.vLLMRolloutWithTool)
    rollout.config = SimpleNamespace(sandbox_url="http://sandbox.invalid")
    return rollout


@pytest.mark.parametrize(
    "value,output,error,expected",
    [
        (0, "", None, "result: \n0"),
        (False, "", None, "result: \nFalse"),
        ([], "", None, "result: \n[]"),
        ({}, "", None, "result: \n{}"),
        ("", "", None, "result:"),
        (None, "", None, ""),
        (42, "logged", None, "result: \n42\noutput: \nlogged"),
        (None, "logged", "failed", "output: \nlogged\nerror: \nfailed"),
    ],
)
def test_tool_results_reach_training_and_inference(
    training_rollout, monkeypatch, value, output, error, expected
):
    payload = json.dumps({"result": value, "output": output, "error": error})
    monkeypatch.setattr(
        requests,
        "post",
        lambda *args, **kwargs: SimpleNamespace(
            status_code=200, json=lambda: json.loads(payload)
        ),
    )
    call = '{"name": "lookup", "arguments": {}}'
    agent = ReCall("http://model.invalid", "http://sandbox.invalid")

    inference_results = agent.execute_tool_calls("environment", [call])
    training_results = training_rollout.batch_execute(["environment"], [[call]])
    assert (inference_results, training_results) == ([expected], [[expected]])


def test_next_reasoning_turn_receives_a_false_tool_result(monkeypatch):
    prompts = []

    def post(url, json, **kwargs):
        if url.endswith("/execute"):
            payload = {"result": False, "output": "", "error": None}
        else:
            prompts.append(json["text"])
            text = (
                '</think><tool_call>{"name":"is_available","arguments":{}}</tool_call>'
            )
            if len(prompts) > 1:
                text = r"</think>\boxed{No}"
            payload = {"text": text}
        return SimpleNamespace(status_code=200, json=lambda: payload)

    monkeypatch.setattr(requests, "post", post)
    agent = ReCall("http://model.invalid", "http://sandbox.invalid")
    transcript = agent.run("def is_available(): return False", "[]", "Is it available?")

    assert len(prompts) == 2
    assert "result: \nFalse\n</tool_response>" in prompts[1]
    assert r"\boxed{No}" in transcript
