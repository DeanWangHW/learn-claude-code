#!/usr/bin/env python3
"""Smoke tests for agents_openai/s04_subagent.py."""

import copy
import importlib.util
import json
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_PATH = REPO_ROOT / "agents_openai" / "s04_subagent.py"


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

    spec = importlib.util.spec_from_file_location("agents_openai_s04_subagent", TARGET_PATH)
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


class TestAgentOpenAI04(unittest.TestCase):
    def test_run_bash_blocks_dangerous_command(self):
        module = _load_module()
        result = module.run_bash("sudo ls")
        self.assertEqual(result, "Error: Dangerous command blocked")

    def test_task_tool_only_exists_in_parent_tools(self):
        module = _load_module()
        child_tool_names = {item["function"]["name"] for item in module.CHILD_TOOLS}
        parent_tool_names = {item["function"]["name"] for item in module.PARENT_TOOLS}

        self.assertNotIn("task", child_tool_names)
        self.assertIn("task", parent_tool_names)

    def test_run_subagent_tool_roundtrip(self):
        module = _load_module()

        first = types.SimpleNamespace(
            content="",
            tool_calls=[_tool_call("sub_1", "bash", {"command": "echo sub"})],
        )
        second = types.SimpleNamespace(content="sub summary", tool_calls=[])
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

        with patch.object(module, "run_bash", return_value="sub-out") as mock_run:
            result = module.run_subagent("inspect")

        mock_run.assert_called_once_with("echo sub")
        self.assertEqual(result, "sub summary")
        self.assertEqual(call_kwargs[0]["messages"][0]["role"], "system")
        self.assertEqual(call_kwargs[1]["messages"][-1]["role"], "tool")
        self.assertEqual(call_kwargs[1]["messages"][-1]["content"], "sub-out")

    def test_agent_loop_dispatches_task_tool(self):
        module = _load_module()

        first = types.SimpleNamespace(
            content="",
            tool_calls=[
                _tool_call(
                    "parent_1",
                    "task",
                    {"prompt": "inspect repo", "description": "explore"},
                )
            ],
        )
        second = types.SimpleNamespace(content="parent done", tool_calls=[])
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

        messages = [{"role": "user", "content": "do a task"}]
        with patch.object(module, "run_subagent", return_value="child summary") as mock_sub:
            module.agent_loop(messages)

        mock_sub.assert_called_once_with("inspect repo")
        self.assertEqual(messages[-1]["role"], "assistant")
        self.assertEqual(messages[-1]["content"], "parent done")
        self.assertTrue(any(m.get("role") == "tool" for m in messages))
        self.assertTrue(any(m.get("content") == "child summary" for m in messages))
        self.assertEqual(call_kwargs[0]["messages"][0]["role"], "system")
        self.assertEqual(call_kwargs[1]["messages"][-1]["role"], "tool")


if __name__ == "__main__":
    unittest.main()
