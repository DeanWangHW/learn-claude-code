#!/usr/bin/env python3
"""Smoke tests for agents_openai/s11_autonomous_agents.py."""

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
TARGET_PATH = REPO_ROOT / "agents_openai" / "s11_autonomous_agents.py"


class _FakeOpenAI:
    def __init__(self, *args, **kwargs):
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=lambda **_: None)
        )


def _load_module():
    os.environ.setdefault("OPENAI_API_KEY", "test-key")
    os.environ.setdefault("MODEL_ID", "gpt-4.1-mini")
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = _FakeOpenAI
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda **_: None

    spec = importlib.util.spec_from_file_location(
        "agents_openai_s11_autonomous_agents",
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


class TestAgentOpenAI11(unittest.TestCase):
    def test_tools_expose_autonomous_operations(self):
        module = _load_module()
        tool_names = {item["function"]["name"] for item in module.TOOLS}
        self.assertIn("idle", tool_names)
        self.assertIn("claim_task", tool_names)
        self.assertIn("shutdown_request", tool_names)
        self.assertIn("plan_approval", tool_names)

    def test_scan_unclaimed_tasks_filters_owned_and_blocked(self):
        module = _load_module()

        with tempfile.TemporaryDirectory() as td:
            tasks_dir = Path(td)
            with patch.object(module, "TASKS_DIR", tasks_dir):
                (tasks_dir / "task_1.json").write_text(
                    json.dumps({"id": 1, "subject": "one", "status": "pending"})
                )
                (tasks_dir / "task_2.json").write_text(
                    json.dumps(
                        {
                            "id": 2,
                            "subject": "two",
                            "status": "pending",
                            "owner": "alice",
                        }
                    )
                )
                (tasks_dir / "task_3.json").write_text(
                    json.dumps({"id": 3, "subject": "three", "status": "in_progress"})
                )
                (tasks_dir / "task_4.json").write_text(
                    json.dumps(
                        {
                            "id": 4,
                            "subject": "four",
                            "status": "pending",
                            "blockedBy": [1],
                        }
                    )
                )

                unclaimed = module.scan_unclaimed_tasks()
                self.assertEqual([task["id"] for task in unclaimed], [1])

    def test_claim_task_sets_owner_and_status(self):
        module = _load_module()

        with tempfile.TemporaryDirectory() as td:
            tasks_dir = Path(td)
            with patch.object(module, "TASKS_DIR", tasks_dir):
                (tasks_dir / "task_7.json").write_text(
                    json.dumps(
                        {
                            "id": 7,
                            "subject": "demo",
                            "status": "pending",
                            "owner": "",
                        }
                    )
                )
                result = module.claim_task(7, "coder")
                self.assertEqual(result, "Claimed task #7 for coder")

                task = json.loads((tasks_dir / "task_7.json").read_text())
                self.assertEqual(task["owner"], "coder")
                self.assertEqual(task["status"], "in_progress")
                self.assertIn("Task 99 not found", module.claim_task(99, "coder"))

    def test_agent_loop_injects_inbox_and_handles_idle_tool(self):
        module = _load_module()

        first = types.SimpleNamespace(
            content="",
            tool_calls=[_tool_call("call_1", "idle", {})],
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

        inbox_payload = [{"type": "message", "from": "alice", "content": "ping"}]
        messages = [{"role": "user", "content": "status"}]

        with patch.object(module.BUS, "read_inbox", side_effect=[inbox_payload, []]):
            module.agent_loop(messages)

        self.assertEqual(messages[-1]["role"], "assistant")
        self.assertEqual(messages[-1]["content"], "done")
        self.assertTrue(any(m.get("role") == "tool" for m in messages))
        self.assertTrue(any(m.get("content") == "Lead does not idle." for m in messages))

        first_call_messages = call_kwargs[0]["messages"]
        self.assertEqual(first_call_messages[0]["role"], "system")
        self.assertTrue(
            any(
                isinstance(m.get("content"), str) and "<inbox>" in m["content"]
                for m in first_call_messages
            )
        )


if __name__ == "__main__":
    unittest.main()
