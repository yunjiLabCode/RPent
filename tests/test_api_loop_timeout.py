# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fake-planner checks for timeout transcripts.

A normal stop keeps the pre-change transcript shape. A wall-clock timeout or
``ModelTimeoutError`` keeps the messages already appended and sets
``terminated_by``.
"""

from __future__ import annotations

import asyncio
import queue
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _mod(name: str, **attrs: object) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


class TextPart:
    def __init__(self, content: str) -> None:
        self.content = content


class ThinkingPart:
    def __init__(self, content: str = "") -> None:
        self.content = content


class ToolCallPart:
    def __init__(self) -> None:
        self.tool_name = "move_to"
        self.tool_call_id = "c1"

    def args_as_dict(self) -> dict:
        return {}


class ModelResponse:
    def __init__(self, parts: list) -> None:
        self.parts = parts


class ToolPart:
    def __init__(self) -> None:
        self.content = "moved"
        self.tool_name = "move_to"
        self.tool_call_id = "c1"
        self.is_error = False


class FunctionToolCallEvent:
    pass


class FunctionToolResultEvent:
    def __init__(self) -> None:
        self.part = ToolPart()


class RunUsage:
    def __init__(self) -> None:
        self.input_tokens = 3
        self.output_tokens = 4
        self.cache_read_tokens = 0
        self.cache_write_tokens = 0
        self.requests = 1


class UsageLimits:
    def __init__(self, request_limit: int | None = None) -> None:
        self.request_limit = request_limit


class UsageLimitExceeded(Exception):
    pass


class ModelHTTPError(Exception):
    pass


class ModelTimeoutError(Exception):
    pass


class Agent:
    @staticmethod
    def is_call_tools_node(node: object) -> bool:
        return isinstance(node, CallNode)

    @staticmethod
    def is_end_node(node: object) -> bool:
        return isinstance(node, EndNode)


_mod(
    "pydantic_ai",
    Agent=Agent,
    BinaryContent=object,
    ModelSettings=object,
    Tool=object,
    ToolReturn=object,
)
_mod("pydantic_ai.capabilities", ProcessHistory=object, Thinking=object)
_mod(
    "pydantic_ai.exceptions",
    ModelHTTPError=ModelHTTPError,
    UsageLimitExceeded=UsageLimitExceeded,
)
_mod(
    "pydantic_ai.messages",
    FunctionToolCallEvent=FunctionToolCallEvent,
    FunctionToolResultEvent=FunctionToolResultEvent,
    ModelMessage=object,
    ModelResponse=ModelResponse,
    TextPart=TextPart,
    ThinkingPart=ThinkingPart,
    ToolCallPart=ToolCallPart,
    UserPromptPart=object,
)
_mod("pydantic_ai.models", Model=object)
_mod("pydantic_ai.usage", RunUsage=RunUsage, UsageLimits=UsageLimits)
_mod("prompt_toolkit", PromptSession=object)
_mod("prompt_toolkit.history", InMemoryHistory=object)
_mod("prompt_toolkit.key_binding", KeyBindings=object)
_mod("prompt_toolkit.key_binding.bindings", named_commands=types.ModuleType("nc"))
_mod("prompt_toolkit.key_binding.bindings.named_commands", get_by_name=lambda _n: None)
_mod("prompt_toolkit.patch_stdout", patch_stdout=lambda: None)
_mod("prompt_toolkit.styles", Style=object)
_mod("imageio")
_mod("imageio.v2")

from rpent.planner.api_loop import ApiAgentLoop  # noqa: E402

USER = "put the moka pot on the stove"
MODE = {"name": "end"}
_PRE_CHANGE_KEYS = ("agent_error", "finish", "stats", "messages")


class _Stream:
    def __init__(self) -> None:
        self._sent = False

    async def __aenter__(self) -> _Stream:
        return self

    async def __aexit__(self, *_args: object) -> bool:
        return False

    def __aiter__(self) -> _Stream:
        return self

    async def __anext__(self) -> FunctionToolResultEvent:
        if self._sent:
            raise StopAsyncIteration
        self._sent = True
        return FunctionToolResultEvent()


class CallNode:
    model_response = ModelResponse([TextPart("looking at the stove")])

    def stream(self, _ctx: object) -> _Stream:
        return _Stream()


class EndNode:
    pass


class FakeRun:
    def __init__(self) -> None:
        self.usage = RunUsage()
        self.ctx = None
        self._yielded = False

    async def __aenter__(self) -> FakeRun:
        return self

    async def __aexit__(self, *_args: object) -> bool:
        return False

    def __aiter__(self) -> FakeRun:
        return self

    async def __anext__(self) -> object:
        if not self._yielded:
            self._yielded = True
            if MODE["name"] == "end":
                return EndNode()
            return CallNode()
        if MODE["name"] == "hang":
            await asyncio.Event().wait()
        if MODE["name"] == "model":
            raise ModelTimeoutError("the next request timed out")
        if MODE["name"] == "usage":
            raise UsageLimitExceeded("over budget")
        raise StopAsyncIteration

    def all_messages(self) -> list:
        return []


class FakeAgent:
    def iter(self, *_args: object, **_kwargs: object) -> FakeRun:
        return FakeRun()


class Toolkit:
    def __init__(self) -> None:
        self.cancelled = 0

    def cancel_active_and_wait(self) -> None:
        self.cancelled += 1


class Sink:
    def emit(self, _event: object) -> None:
        return None


def _loop(timeout_s: float | None = 30) -> ApiAgentLoop:
    loop = ApiAgentLoop.__new__(ApiAgentLoop)
    loop._model = None
    loop._sampling = {}
    loop._max_tokens = 8
    loop._dashboard_events = Sink()
    loop._no_images = True
    loop._timeout_s = timeout_s
    loop._reasoning_effort = "none"
    loop._build_agent = lambda *_a, **_k: FakeAgent()  # type: ignore[method-assign]
    return loop


def _saved(result: object) -> dict:
    """Same keys ``main`` writes, including ``terminated_by`` only when set."""
    record = {
        "agent_error": result.error,  # type: ignore[attr-defined]
        "finish": result.finish_result,  # type: ignore[attr-defined]
        "stats": result.stats,  # type: ignore[attr-defined]
        "messages": result.messages,  # type: ignore[attr-defined]
    }
    if result.terminated_by:  # type: ignore[attr-defined]
        record["terminated_by"] = result.terminated_by  # type: ignore[attr-defined]
    return record


class TimeoutTranscriptTests(unittest.TestCase):
    def test_normal_stop_matches_pre_change_shape(self) -> None:
        MODE["name"] = "turn"
        result = _loop().solve(
            system_prompt="sys",
            user_message=USER,
            toolkit=Toolkit(),
            max_turns=4,
        )
        record = _saved(result)
        self.assertEqual(tuple(record), _PRE_CHANGE_KEYS)
        self.assertIsNone(result.terminated_by)
        self.assertIsNone(result.error)
        self.assertIsNone(result.finish_result)
        self.assertEqual(
            result.messages,
            [
                {"role": "user", "content": USER},
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "looking at the stove"}],
                },
                {
                    "role": "tool",
                    "name": "move_to",
                    "tool_call_id": "c1",
                    "content": "moved",
                },
            ],
        )
        self.assertEqual(
            set(result.stats),
            {
                "turns_used",
                "tool_calls",
                "total_input_tokens",
                "total_output_tokens",
                "cache_read_tokens",
                "cache_write_tokens",
                "requests",
            },
        )
        self.assertNotIn("terminated_by", result.stats)

    def test_wall_clock_timeout_keeps_messages_and_marks_why(self) -> None:
        MODE["name"] = "hang"
        toolkit = Toolkit()
        result = _loop(0.2).solve(
            system_prompt="sys",
            user_message=USER,
            toolkit=toolkit,
            max_turns=4,
        )
        record = _saved(result)
        self.assertGreater(len(result.messages), 1)
        self.assertEqual(result.messages[2]["role"], "tool")
        self.assertEqual(result.terminated_by, "planner_timeout")
        self.assertEqual(record["terminated_by"], "planner_timeout")
        self.assertIsNone(result.finish_result)
        self.assertEqual(result.stats, {})
        self.assertEqual(result.error, "API planner timed out after 0.2s")
        self.assertEqual(toolkit.cancelled, 1)

    def test_model_timeout_keeps_messages_and_marks_why(self) -> None:
        MODE["name"] = "model"
        result = _loop().solve(
            system_prompt="sys",
            user_message=USER,
            toolkit=Toolkit(),
            max_turns=4,
        )
        self.assertGreater(len(result.messages), 1)
        self.assertEqual(result.messages[1]["role"], "assistant")
        self.assertEqual(result.terminated_by, "model_timeout")
        self.assertEqual(_saved(result)["terminated_by"], "model_timeout")
        self.assertIsNone(result.finish_result)
        self.assertEqual(result.error, "ModelTimeoutError: the next request timed out")
        self.assertEqual(result.stats["turns_used"], 1)

    def test_usage_limit_and_interactive_omit_the_marker(self) -> None:
        MODE["name"] = "usage"
        limited = _loop().solve(
            system_prompt="sys",
            user_message=USER,
            toolkit=Toolkit(),
            max_turns=4,
        )
        self.assertIsNone(limited.terminated_by)
        self.assertIsNone(limited.error)
        self.assertNotIn("terminated_by", _saved(limited))

        MODE["name"] = "end"
        interactive = queue.Queue()
        interactive.put(None)
        result = _loop().solve(
            system_prompt="sys",
            user_message=USER,
            toolkit=Toolkit(),
            max_turns=4,
            input_queue=interactive,
        )
        self.assertIsNone(result.terminated_by)
        self.assertEqual(result.messages, [{"role": "user", "content": USER}])
        self.assertNotIn("terminated_by", _saved(result))


if __name__ == "__main__":
    unittest.main()
