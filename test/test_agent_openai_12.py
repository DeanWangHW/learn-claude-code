#!/usr/bin/env python3
"""Smoke tests for agents_openai/s12_worktree_task_isolation.py."""

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
TARGET_PATH = REPO_ROOT / "agents_openai" / "s12_worktree_task_isolation.py"


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
        "agents_openai_s12_worktree_task_isolation",
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


class TestAgentOpenAI12(unittest.TestCase):
    def test_tools_expose_task_and_worktree_operations(self):
        module = _load_module()
        tool_names = {item["function"]["name"] for item in module.TOOLS}
        self.assertIn("task_create", tool_names)
        self.assertIn("worktree_create", tool_names)
        self.assertIn("worktree_remove", tool_names)
        self.assertIn("worktree_events", tool_names)

    def test_task_manager_create_update_and_bind(self):
        module = _load_module()
        with tempfile.TemporaryDirectory() as td:
            tm = module.TaskManager(Path(td))
            task = json.loads(tm.create("Build auth", "details"))
            self.assertEqual(task["id"], 1)
            self.assertEqual(task["status"], "pending")
            self.assertTrue(tm.exists(1))

            updated = json.loads(tm.update(1, status="in_progress", owner="alice"))
            self.assertEqual(updated["status"], "in_progress")
            self.assertEqual(updated["owner"], "alice")

            bound = json.loads(tm.bind_worktree(1, "auth-lane", owner="alice"))
            self.assertEqual(bound["worktree"], "auth-lane")
            self.assertEqual(bound["owner"], "alice")
            self.assertIn("#1: Build auth", tm.list_all())

    def test_event_bus_emits_and_lists_recent(self):
        module = _load_module()
        with tempfile.TemporaryDirectory() as td:
            event_path = Path(td) / "events.jsonl"
            bus = module.EventBus(event_path)
            bus.emit("worktree.create.before", task={"id": 1}, worktree={"name": "lane-1"})
            bus.emit("worktree.keep", task={"id": 1}, worktree={"name": "lane-1"})
            recent = json.loads(bus.list_recent(limit=5))
            self.assertEqual(len(recent), 2)
            self.assertEqual(recent[0]["event"], "worktree.create.before")
            self.assertEqual(recent[1]["event"], "worktree.keep")

    def test_worktree_manager_keep_updates_index(self):
        module = _load_module()
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            tasks = module.TaskManager(repo / ".tasks")
            events = module.EventBus(repo / ".worktrees" / "events.jsonl")
            wm = module.WorktreeManager(repo, tasks, events)

            idx = wm._load_index()
            idx["worktrees"].append(
                {
                    "name": "lane-a",
                    "path": str(repo / ".worktrees" / "lane-a"),
                    "branch": "wt/lane-a",
                    "task_id": None,
                    "status": "active",
                    "created_at": 1.0,
                }
            )
            wm._save_index(idx)

            kept = json.loads(wm.keep("lane-a"))
            self.assertEqual(kept["status"], "kept")
            listing = wm.list_all()
            self.assertIn("[kept] lane-a", listing)

    def test_agent_loop_handles_tool_call_and_records_tool_result(self):
        module = _load_module()

        first = types.SimpleNamespace(
            content="",
            tool_calls=[_tool_call("call_1", "worktree_events", {"limit": 2})],
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

        with patch.object(module.EVENTS, "list_recent", return_value='[{"event":"x"}]'):
            messages = [{"role": "user", "content": "show events"}]
            module.agent_loop(messages)

        self.assertEqual(messages[-1]["role"], "assistant")
        self.assertEqual(messages[-1]["content"], "done")
        self.assertTrue(any(m.get("role") == "tool" for m in messages))
        self.assertTrue(any(m.get("content") == '[{"event":"x"}]' for m in messages))

        first_call_messages = call_kwargs[0]["messages"]
        self.assertEqual(first_call_messages[0]["role"], "system")


if __name__ == "__main__":
    unittest.main()
