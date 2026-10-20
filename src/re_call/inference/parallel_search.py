"""Anonymous web tools for a single ReCall inference run."""

import asyncio
import copy
import json
import math
from datetime import timedelta

import anyio
import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


ENDPOINT = "https://search.parallel.ai/mcp"
USER_AGENT = "ReCall (+https://github.com/Agent-RL/ReCall)"
TOOL_NAMES = frozenset({"web_search", "web_fetch"})


class ParallelSearch:
    """Stateless sync adapter; conversation identity is supplied by ReCall.run."""

    def __init__(self, timeout=30):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("search_timeout must be finite and positive")
        self.timeout = timeout

    async def _request(self, name=None, arguments=None):
        # Bound discovery and tool execution, including transport setup/cleanup.
        with anyio.fail_after(self.timeout):
            async with httpx.AsyncClient(
                headers={"User-Agent": USER_AGENT}, timeout=self.timeout,
            ) as client:
                async with streamable_http_client(ENDPOINT, http_client=client) as streams:
                    async with ClientSession(
                        streams[0], streams[1],
                        read_timeout_seconds=timedelta(seconds=self.timeout),
                    ) as session:
                        await session.initialize()
                        if name is None:
                            return await session.list_tools()
                        return await session.call_tool(name, arguments)

    def schemas(self):
        """Discover model-visible schemas without exposing host-owned metadata."""
        tools = asyncio.run(self._request()).tools
        schemas = []
        for tool in tools:
            if tool.name not in TOOL_NAMES:
                continue
            parameters = copy.deepcopy(tool.inputSchema)
            for key in ("session_id", "model_name"):
                parameters.get("properties", {}).pop(key, None)
                if key in parameters.get("required", []):
                    parameters["required"].remove(key)
            schemas.append({
                "name": tool.name, "description": tool.description or tool.name,
                "parameters": parameters,
            })
        if {tool["name"] for tool in schemas} != TOOL_NAMES:
            raise RuntimeError("Parallel Search MCP did not provide web_search and web_fetch")
        return json.dumps(schemas)

    def execute(self, tool_calls, session_id):
        results = []
        for raw_call in tool_calls:
            try:
                call = json.loads(raw_call)
                name = call["name"]
                arguments = call.get("arguments", {})
                if name not in TOOL_NAMES or not isinstance(arguments, dict):
                    raise ValueError("only web_search and web_fetch with object arguments are allowed")
                arguments = dict(arguments)
                # The model cannot replace the run's identity or invent its model ID.
                arguments.pop("model_name", None)
                arguments["session_id"] = session_id
                result = asyncio.run(self._request(name, arguments))
                text = "\n".join(
                    block.text for block in result.content if block.type == "text"
                )
                if not text and result.structuredContent is not None:
                    text = json.dumps(result.structuredContent)
                results.append(("error: " if result.isError else "result: \n") + text)
            except Exception as error:
                # Cancellation/KeyboardInterrupt propagate; ordinary failures become
                # observations, never a silent switch to a different provider.
                results.append("error: Parallel Search MCP: " + self._error_text(error))
        return results

    @staticmethod
    def _error_text(error):
        children = getattr(error, "exceptions", None)
        if children:
            return "; ".join(ParallelSearch._error_text(child) for child in children)
        return str(error) or type(error).__name__
