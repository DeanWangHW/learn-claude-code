#!/usr/bin/env python3
"""Smoke tests for agents_openai/s03_todo_write.py."""

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
TARGET_PATH = REPO_ROOT / "agents_openai" / "s03_todo_write.py"


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

    spec = importlib.util.spec_from_file_location("agents_openai_s03_todo_write", TARGET_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None

    with patch.dict(sys.modules, {"openai": fake_openai, "dotenv": fake_dotenv}):
        spec.loader.exec_module(module)
    return module


class TestAgentOpenAI03(unittest.TestCase):
    def test_run_bash_blocks_dangerous_command(self):
        module = _load_module()
        result = module.run_bash("sudo ls")
        self.assertEqual(result, "Error: Dangerous command blocked")

    def test_todo_manager_rejects_multiple_in_progress(self):
        module = _load_module()
        todo = module.TodoManager()

        with self.assertRaises(ValueError):
            todo.update(
                [
                    {"id": "1", "text": "task-a", "status": "in_progress"},
                    {"id": "2", "text": "task-b", "status": "in_progress"},
                ]
            )

    def test_agent_loop_injects_todo_reminder(self):
        module = _load_module()

        def _tool_call(call_id: str, name: str, args: dict):
            return types.SimpleNamespace(
                id=call_id,
                function=types.SimpleNamespace(name=name, arguments=json.dumps(args)),
            )

        first = types.SimpleNamespace(content="", tool_calls=[_tool_call("c1", "bash", {"command": "echo 1"})])
        second = types.SimpleNamespace(content="", tool_calls=[_tool_call("c2", "bash", {"command": "echo 2"})])
        third = types.SimpleNamespace(content="", tool_calls=[_tool_call("c3", "bash", {"command": "echo 3"})])
        fourth = types.SimpleNamespace(content="Done", tool_calls=[])

        responses = iter(
            [
                types.SimpleNamespace(choices=[types.SimpleNamespace(message=first)]),
                types.SimpleNamespace(choices=[types.SimpleNamespace(message=second)]),
                types.SimpleNamespace(choices=[types.SimpleNamespace(message=third)]),
                types.SimpleNamespace(choices=[types.SimpleNamespace(message=fourth)]),
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

        messages = [{"role": "user", "content": "do task"}]
        with patch.object(module, "run_bash", return_value="ok"):
            module.agent_loop(messages)

        self.assertEqual(len(call_kwargs), 4)
        reminder = "<reminder>Update your todos.</reminder>"
        self.assertTrue(any(m.get("content") == reminder for m in messages if isinstance(m, dict)))
        self.assertTrue(any(m.get("role") == "tool" for m in messages))
        self.assertTrue(any(m.get("content") == reminder for m in call_kwargs[3]["messages"]))


if __name__ == "__main__":
    unittest.main()
