#!/usr/bin/env python3
"""Smoke tests for agents_openai/s07_task_system.py."""

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
TARGET_PATH = REPO_ROOT / "agents_openai" / "s07_task_system.py"


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

    spec = importlib.util.spec_from_file_location("agents_openai_s07_task_system", TARGET_PATH)
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


class TestAgentOpenAI07(unittest.TestCase):
    def test_run_bash_blocks_dangerous_command(self):
        module = _load_module()
        result = module.run_bash("sudo ls")
        self.assertEqual(result, "Error: Dangerous command blocked")

    def test_tools_expose_task_tools(self):
        module = _load_module()
        tool_names = {item["function"]["name"] for item in module.TOOLS}
        self.assertIn("task_create", tool_names)
        self.assertIn("task_update", tool_names)
        self.assertIn("task_list", tool_names)
        self.assertIn("task_get", tool_names)

    def test_task_manager_dependency_resolution(self):
        module = _load_module()

        with tempfile.TemporaryDirectory() as td:
            manager = module.TaskManager(Path(td))

            task_1 = json.loads(manager.create("task-1"))
            task_2 = json.loads(manager.create("task-2"))

            manager.update(task_1["id"], add_blocks=[task_2["id"]])
            task_2_after_link = json.loads(manager.get(task_2["id"]))
            self.assertIn(task_1["id"], task_2_after_link["blockedBy"])

            manager.update(task_1["id"], status="completed")
            task_2_after_complete = json.loads(manager.get(task_2["id"]))
            self.assertNotIn(task_1["id"], task_2_after_complete["blockedBy"])

    def test_agent_loop_task_list_roundtrip(self):
        module = _load_module()

        first = types.SimpleNamespace(
            content="",
            tool_calls=[_tool_call("call_1", "task_list", {})],
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

        messages = [{"role": "user", "content": "list tasks"}]
        with patch.object(module.TASKS, "list_all", return_value="No tasks.") as mock_list:
            module.agent_loop(messages)

        mock_list.assert_called_once()
        self.assertEqual(messages[-1]["role"], "assistant")
        self.assertEqual(messages[-1]["content"], "done")
        self.assertTrue(any(m.get("role") == "tool" for m in messages))
        self.assertTrue(any(m.get("content") == "No tasks." for m in messages))
        self.assertEqual(call_kwargs[0]["messages"][0]["role"], "system")
        self.assertEqual(call_kwargs[1]["messages"][-1]["role"], "tool")
        self.assertEqual(call_kwargs[1]["messages"][-1]["content"], "No tasks.")


if __name__ == "__main__":
    unittest.main()
