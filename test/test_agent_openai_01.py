#!/usr/bin/env python3
"""Smoke tests for agents_openai/s01_agent_loop.py."""

import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_PATH = REPO_ROOT / "agents_openai" / "s01_agent_loop.py"


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

    spec = importlib.util.spec_from_file_location("agents_openai_s01_agent_loop", TARGET_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None

    with patch.dict(sys.modules, {"openai": fake_openai, "dotenv": fake_dotenv}):
        spec.loader.exec_module(module)
    return module


class TestAgentOpenAI01(unittest.TestCase):
    def test_run_bash_blocks_dangerous_command(self):
        module = _load_module()
        result = module.run_bash("sudo ls")
        self.assertEqual(result, "Error: Dangerous command blocked")

    def test_agent_loop_tool_call_roundtrip(self):
        module = _load_module()

        tool_call = types.SimpleNamespace(
            id="call_1",
            function=types.SimpleNamespace(
                name="bash",
                arguments='{"command":"echo hello"}',
            ),
        )

        first_message = types.SimpleNamespace(content="", tool_calls=[tool_call])
        second_message = types.SimpleNamespace(content="All done", tool_calls=[])
        responses = iter([
            types.SimpleNamespace(choices=[types.SimpleNamespace(message=first_message)]),
            types.SimpleNamespace(choices=[types.SimpleNamespace(message=second_message)]),
        ])

        call_kwargs = []

        def fake_create(**kwargs):
            call_kwargs.append(kwargs)
            return next(responses)

        module.client = types.SimpleNamespace(
            chat=types.SimpleNamespace(
                completions=types.SimpleNamespace(create=fake_create)
            )
        )

        messages = [{"role": "user", "content": "run echo hello"}]
        with patch.object(module, "run_bash", return_value="hello\n") as mock_run:
            module.agent_loop(messages)

        mock_run.assert_called_once_with("echo hello")
        self.assertEqual(messages[-1]["role"], "assistant")
        self.assertEqual(messages[-1]["content"], "All done")
        self.assertTrue(any(m.get("role") == "tool" for m in messages))

        self.assertEqual(call_kwargs[0]["messages"][0]["role"], "system")
        self.assertEqual(call_kwargs[1]["messages"][-1]["role"], "tool")


if __name__ == "__main__":
    unittest.main()
