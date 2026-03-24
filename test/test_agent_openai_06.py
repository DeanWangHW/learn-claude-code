#!/usr/bin/env python3
"""Smoke tests for agents_openai/s06_context_compact.py."""

import copy
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_PATH = REPO_ROOT / "agents_openai" / "s06_context_compact.py"


class _FakeOpenAI:
    def __init__(self, *args, **kwargs):
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=lambda **_: None)
        )


def _load_module():
    os.environ.setdefault("OPENAI_API_KEY", "test-key")
    os.environ.setdefault("MODEL_ID", "gpt-4.1-mini")

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = _FakeOpenAI
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda **_: None

    spec = importlib.util.spec_from_file_location("agents_openai_s06_context_compact", TARGET_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None

    with patch.dict(sys.modules, {"openai": fake_openai, "dotenv": fake_dotenv}):
        spec.loader.exec_module(module)
    return module


def _tool_call(call_id: str, name: str, args: dict):
    return types.SimpleNamespace(
        id=call_id,
        function=types.SimpleNamespace(name=name, arguments=json.dumps(args)),
    )


class TestAgentOpenAI06(unittest.TestCase):
    def test_run_bash_blocks_dangerous_command(self):
        module = _load_module()
        result = module.run_bash("sudo ls")
        self.assertEqual(result, "Error: Dangerous command blocked")

    def test_tools_expose_compact(self):
        module = _load_module()
        tool_names = {item["function"]["name"] for item in module.TOOLS}
        self.assertIn("compact", tool_names)

    def test_micro_compact_replaces_only_old_tool_outputs(self):
        module = _load_module()

        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "bash", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "A" * 200},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c2",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c2", "content": "B" * 200},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c3",
                        "type": "function",
                        "function": {"name": "write_file", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c3", "content": "C" * 200},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c4",
                        "type": "function",
                        "function": {"name": "edit_file", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c4", "content": "D" * 200},
        ]

        module.micro_compact(messages)

        self.assertEqual(messages[1]["content"], "[Previous: used bash]")
        self.assertEqual(messages[3]["content"], "B" * 200)
        self.assertEqual(messages[5]["content"], "C" * 200)
        self.assertEqual(messages[7]["content"], "D" * 200)

    def test_auto_compact_writes_transcript_and_returns_summary_messages(self):
        module = _load_module()

        response = types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(content="compact summary")
                )
            ]
        )

        with tempfile.TemporaryDirectory() as td:
            module.TRANSCRIPT_DIR = Path(td) / ".transcripts"
            module.client = types.SimpleNamespace(
                chat=types.SimpleNamespace(
                    completions=types.SimpleNamespace(create=lambda **_: response)
                )
            )

            compacted = module.auto_compact([{"role": "user", "content": "hello"}])

            files = list(module.TRANSCRIPT_DIR.glob("transcript_*.jsonl"))
            self.assertEqual(len(files), 1)
            self.assertEqual(len(compacted), 2)
            self.assertEqual(compacted[0]["role"], "user")
            self.assertIn("compact summary", compacted[0]["content"])
            self.assertEqual(compacted[1]["role"], "assistant")

    def test_agent_loop_manual_compact_roundtrip(self):
        module = _load_module()

        first = types.SimpleNamespace(
            content="",
            tool_calls=[_tool_call("call_1", "compact", {"focus": "state"})],
        )
        second = types.SimpleNamespace(content="done", tool_calls=[])

        responses = iter(
            [
                types.SimpleNamespace(choices=[types.SimpleNamespace(message=first)]),
                types.SimpleNamespace(choices=[types.SimpleNamespace(message=second)]),
            ]
        )

        call_kwargs = []

        def fake_create(**kwargs):
            call_kwargs.append(copy.deepcopy(kwargs))
            return next(responses)

        module.client = types.SimpleNamespace(
            chat=types.SimpleNamespace(
                completions=types.SimpleNamespace(create=fake_create)
            )
        )

        summary_messages = [
            {"role": "user", "content": "[summary]"},
            {"role": "assistant", "content": "ack"},
        ]

        messages = [{"role": "user", "content": "compact now"}]
        module.THRESHOLD = 10**9  # avoid auto compaction by threshold in this test
        with patch.object(module, "auto_compact", return_value=summary_messages) as mock_auto:
            module.agent_loop(messages)

        mock_auto.assert_called_once()
        self.assertEqual(messages[-1]["role"], "assistant")
        self.assertEqual(messages[-1]["content"], "done")
        self.assertEqual(len(call_kwargs), 2)
        self.assertEqual(call_kwargs[0]["messages"][0]["role"], "system")
        self.assertEqual(call_kwargs[1]["messages"][-1]["content"], "ack")


if __name__ == "__main__":
    unittest.main()
