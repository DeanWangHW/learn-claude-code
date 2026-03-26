#!/usr/bin/env python3
"""Smoke tests for agents_openai/s08_background_tasks.py."""

import copy
import importlib.util
import json
import os
import subprocess
import sys
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_PATH = REPO_ROOT / "agents_openai" / "s08_background_tasks.py"


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

    spec = importlib.util.spec_from_file_location(
        "agents_openai_s08_background_tasks",
        TARGET_PATH,
    )
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


class TestAgentOpenAI08(unittest.TestCase):
    def test_run_bash_blocks_dangerous_command(self):
        module = _load_module()
        result = module.run_bash("sudo ls")
        self.assertEqual(result, "Error: Dangerous command blocked")

    def test_tools_expose_background_tools(self):
        module = _load_module()
        tool_names = {item["function"]["name"] for item in module.TOOLS}
        self.assertIn("background_run", tool_names)
        self.assertIn("check_background", tool_names)

    def test_background_manager_run_and_notification(self):
        module = _load_module()
        manager = module.BackgroundManager()
        completed = subprocess.CompletedProcess(
            args="echo done",
            returncode=0,
            stdout="done\n",
            stderr="",
        )

        with patch.object(module.subprocess, "run", return_value=completed):
            start_text = manager.run("echo done")

            task_id = start_text.split()[2]
            for _ in range(100):
                if manager.tasks[task_id]["status"] != "running":
                    break
                time.sleep(0.01)

            self.assertEqual(manager.tasks[task_id]["status"], "completed")
            self.assertIn("[completed]", manager.check(task_id))

            first = manager.drain_notifications()
            second = manager.drain_notifications()
            self.assertEqual(len(first), 1)
            self.assertEqual(second, [])

    def test_agent_loop_injects_background_notifications_and_handles_tools(self):
        module = _load_module()

        first = types.SimpleNamespace(
            content="",
            tool_calls=[_tool_call("call_1", "check_background", {})],
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

        notifications = [
            {
                "task_id": "abc12345",
                "status": "completed",
                "command": "echo done",
                "result": "done",
            }
        ]

        messages = [{"role": "user", "content": "check bg"}]
        with patch.object(module.BG, "drain_notifications", side_effect=[notifications, []]):
            with patch.object(module.BG, "check", return_value="No background tasks.") as mock_check:
                module.agent_loop(messages)

        mock_check.assert_called_once_with(None)
        self.assertEqual(messages[-1]["role"], "assistant")
        self.assertEqual(messages[-1]["content"], "done")
        self.assertTrue(any(m.get("role") == "tool" for m in messages))
        self.assertTrue(any(m.get("content") == "No background tasks." for m in messages))

        first_call_messages = call_kwargs[0]["messages"]
        self.assertEqual(first_call_messages[0]["role"], "system")
        self.assertTrue(
            any(
                isinstance(m.get("content"), str) and "<background-results>" in m["content"]
                for m in first_call_messages
            )
        )


if __name__ == "__main__":
    unittest.main()
