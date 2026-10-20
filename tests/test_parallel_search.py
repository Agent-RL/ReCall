"""Exercise the inference loop with a scripted model and real MCP SDK transport."""

import asyncio
import json
import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import httpx
import requests

from re_call import ReCall
from re_call.inference import parallel_search


ASYNC_CLIENT = httpx.AsyncClient
SOURCE_URL = "https://www.python.org/downloads/"


def tool_call(name, **arguments):
    return '<tool_call>' + json.dumps({"name": name, "arguments": arguments}) + '</tool_call>'


class ModelResponse:
    def __init__(self, text):
        self.text = text

    def json(self):
        return {"text": self.text}


class MCPFixture:
    """Only HTTP is replaced: SDK initialize/list/call and parsing all execute."""

    def __init__(self):
        self.requests = []
        self.failure = None
        self.clients = []

    def client(self, **kwargs):
        client = ASYNC_CLIENT(transport=httpx.MockTransport(self.handle), **kwargs)
        self.clients.append(client)
        return client

    async def handle(self, request):
        message = json.loads(request.content)
        self.requests.append((request, message))
        method = message["method"]
        if method == "notifications/initialized":
            return httpx.Response(202)
        if method == "initialize":
            result = {
                "protocolVersion": message["params"]["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fixture", "version": "1.0"},
            }
        elif method == "tools/list":
            result = {"tools": [
                {"name": name, "description": name, "inputSchema": {
                    "type": "object", "properties": {
                        "session_id": {"type": "string"},
                        "model_name": {"type": "string"},
                        field: {"type": "array", "items": {"type": "string"}},
                    }, "required": [field],
                }}
                for name, field in [("web_search", "search_queries"), ("web_fetch", "urls")]
            ]}
        elif method == "tools/call":
            if self.failure == "timeout":
                raise httpx.ReadTimeout("fixture read deadline", request=request)
            if self.failure == "hang":
                await asyncio.sleep(10)
            result = {"content": [{"type": "text", "text": json.dumps({
                "url": SOURCE_URL, "excerpts": ["Python release notes"],
            })}], "isError": False}
            if self.failure == "quota":
                result = {"content": [{"type": "text", "text": "anonymous quota exceeded"}], "isError": True}
            elif self.failure == "malformed":
                return httpx.Response(200, text="not JSON", headers={"content-type": "application/json"})
        else:
            raise AssertionError(method)
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": message["id"], "result": result})

    def calls(self):
        return [msg["params"] for _, msg in self.requests if msg["method"] == "tools/call"]


class InferenceSearchTests(unittest.TestCase):
    def setUp(self):
        self.fixture = MCPFixture()
        self.addCleanup(patch.stopall)
        patch.object(parallel_search.httpx, "AsyncClient", side_effect=self.fixture.client).start()
        patch.dict(os.environ, {"HOME": "/nonexistent-recall-test-home"}, clear=True).start()

    def scripted_model(self, responses):
        self.prompts = []
        iterator = iter(responses)

        def generate(url, json, **kwargs):
            self.assertEqual(url, "http://model/generate")
            self.prompts.append(json["text"])
            response = next(iterator)
            if isinstance(response, BaseException):
                raise response
            return ModelResponse(response)

        return patch("re_call.inference.re_call.requests.post", side_effect=generate)

    def test_default_search_fetch_followup_are_consumed_with_one_run_id(self):
        with self.scripted_model([
            tool_call("web_search", search_queries=["Python recent release notes"], session_id="model-chosen", model_name="invented"),
            tool_call("web_fetch", urls=[SOURCE_URL]),
            tool_call("web_search", search_queries=["Python release security changes"]),
            "The source is " + SOURCE_URL,
        ]):
            result = ReCall("http://model").run(question="Find Python releases")
        self.assertIn("Python release notes", self.prompts[1])
        self.assertIn(SOURCE_URL, self.prompts[2])
        self.assertIn("The source is " + SOURCE_URL, result)
        schemas = self.prompts[0].split('```json\n', 1)[1].split('\n```', 1)[0]
        for schema in json.loads(schemas):
            self.assertNotIn("session_id", schema["parameters"]["properties"])
            self.assertNotIn("model_name", schema["parameters"]["properties"])
        calls = self.fixture.calls()
        ids = {call["arguments"]["session_id"] for call in calls}
        self.assertEqual(len(ids), 1)
        self.assertNotIn("model-chosen", ids)
        self.assertEqual(len(next(iter(ids))), 36)
        self.assertEqual(calls[1]["arguments"]["urls"], [SOURCE_URL])
        for request, _ in self.fixture.requests:
            self.assertEqual(str(request.url), parallel_search.ENDPOINT)
            self.assertEqual(request.headers["user-agent"], parallel_search.USER_AGENT)
            self.assertNotIn("authorization", request.headers)
        self.assertTrue(all("model_name" not in call["arguments"] for call in calls))
        self.assertTrue(all(client.is_closed for client in self.fixture.clients))

    def test_new_runs_on_same_agent_start_new_conversations(self):
        agent = ReCall("http://model")
        for _ in range(2):
            with self.scripted_model([tool_call("web_search", search_queries=["Python release notes"]), "done"]):
                agent.run(question="Find releases")
        self.assertEqual(len({c["arguments"]["session_id"] for c in self.fixture.calls()}), 2)

    def test_interleaved_runs_do_not_share_conversation_identity(self):
        agent = ReCall("http://model")

        def generate(url, json, **kwargs):
            prompt = json["text"]
            question = "task-A" if "task-A" in prompt else "task-B"
            count = prompt.count("<tool_response>")
            if count == 0:
                return ModelResponse(tool_call("web_search", search_queries=[question]))
            if count == 1:
                return ModelResponse(tool_call("web_fetch", urls=[SOURCE_URL], objective=question))
            return ModelResponse("done")

        with patch("re_call.inference.re_call.requests.post", side_effect=generate):
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(lambda q: agent.run(question=q), ["task-A", "task-B"]))
        ids = {}
        for call in self.fixture.calls():
            args = call["arguments"]
            task = args.get("objective") or args["search_queries"][0]
            ids.setdefault(task, set()).add(args["session_id"])
        self.assertEqual([len(ids[k]) for k in sorted(ids)], [1, 1])
        self.assertNotEqual(ids["task-A"], ids["task-B"])
        self.assertTrue(all("done" in r for r in results))

    def test_generation_retry_keeps_identity(self):
        search = tool_call("web_search", search_queries=["Python release notes"])
        with self.scripted_model([search, requests.ConnectionError("model reset"), search, "done"]):
            with patch("re_call.inference.re_call.time.sleep"):
                ReCall("http://model").run(question="Find releases")
        self.assertEqual(len(self.fixture.calls()), 2)
        self.assertEqual(len({c["arguments"]["session_id"] for c in self.fixture.calls()}), 1)

    def test_disabled_and_explicit_empty_schemas_cannot_execute_web_tools(self):
        for kwargs, run_kwargs in [({"search_enabled": False}, {}), ({}, {"func_schemas": []}), ({}, {"func_schemas": "[]"})]:
            with self.subTest(kwargs=kwargs, run_kwargs=run_kwargs):
                with self.scripted_model([tool_call("web_search", search_queries=["should not run"]), "done"]):
                    result = ReCall("http://model", **kwargs).run(question="Offline", **run_kwargs)
                self.assertIn("error:", result)
        self.assertEqual(self.fixture.requests, [])

    def test_explicit_sandbox_preserved_even_on_failure(self):
        for status in [200, 503]:
            seen = []

            def post(url, json, **kwargs):
                seen.append((url, json))
                if url == "http://sandbox/execute":
                    response = requests.Response()
                    response.status_code = status
                    response._content = b'{"result":"custom result","output":"","error":""}'
                    return response
                return ModelResponse(tool_call("custom", query="Python") if len(seen) == 1 else "done")

            with patch("re_call.inference.re_call.requests.post", side_effect=post):
                result = ReCall("http://model", "http://sandbox").run("custom env", "custom schemas", "Question")
            self.assertEqual(seen[1], ("http://sandbox/execute", {"env": "custom env", "call": "custom(query='Python')"}))
            self.assertIn("custom result" if status == 200 else "error: 503", result)
        self.assertEqual(self.fixture.requests, [])

    def test_partial_sandbox_never_falls_back(self):
        for agent, kwargs in [
            (ReCall("http://model", "http://sandbox"), {}),
            (ReCall("http://model"), {"env": "explicit"}),
            (ReCall("http://model"), {"func_schemas": "custom"}),
        ]:
            with self.assertRaises(ValueError):
                agent.run(question="Question", **kwargs)
        self.assertEqual(self.fixture.requests, [])

    def test_unknown_and_malformed_calls_denied_before_http(self):
        with self.scripted_model([
            tool_call("delete_files", path="/"),
            '<tool_call>{"name":"web_search","arguments":[]}</tool_call>',
            '<tool_call>bad json</tool_call>', "done",
        ]):
            result = ReCall("http://model").run(question="Question")
        self.assertIn("error: Parallel Search MCP", result)
        self.assertEqual(self.fixture.calls(), [])

    def test_errors_and_timeouts_are_model_observations_without_fallback(self):
        for failure in ["quota", "timeout", "malformed", "hang"]:
            with self.subTest(failure=failure):
                self.fixture.failure = failure
                with self.scripted_model([
                    tool_call("web_search", search_queries=["Python release notes"]),
                    tool_call("web_fetch", urls=[SOURCE_URL]), "done",
                ]):
                    result = ReCall("http://model", search_timeout=0.15).run(question="Question")
                self.assertIn("error:", self.prompts[1])
                self.assertIn("done", result)
                self.assertEqual(len({c["arguments"]["session_id"] for c in self.fixture.calls()[-2:]}), 1)
                self.assertTrue(all(client.is_closed for client in self.fixture.clients))

    def test_discovery_failure_stops_before_model_execution(self):
        with patch.object(self.fixture, "handle", side_effect=httpx.ConnectError("unavailable")):
            with patch("re_call.inference.re_call.requests.post") as model:
                with self.assertRaises(Exception):
                    ReCall("http://model", search_timeout=0.15).run(question="Question")
                model.assert_not_called()

    def test_cancellation_propagates_and_closes_sdk_transport(self):
        self.fixture.failure = "hang"
        run_async = asyncio.run

        async def cancel_request(request):
            task = asyncio.create_task(request)
            while not self.fixture.calls():
                await asyncio.sleep(0.001)
            task.cancel()
            return await task

        def run_with_cancellation(request):
            # Discovery completes normally. Cancel the active tool request at
            # the sync/async boundary used by the actual ReCall.run dispatcher.
            if any(msg["method"] == "tools/list" for _, msg in self.fixture.requests):
                return run_async(cancel_request(request))
            return run_async(request)

        with self.scripted_model([tool_call("web_search", search_queries=["Python release notes"])]):
            with patch.object(parallel_search.asyncio, "run", side_effect=run_with_cancellation):
                with self.assertRaises(asyncio.CancelledError):
                    ReCall("http://model", search_timeout=2).run(question="Question")
        self.assertEqual(len(self.prompts), 1)
        self.assertTrue(all(client.is_closed for client in self.fixture.clients))


if __name__ == "__main__":
    unittest.main()
