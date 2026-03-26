#!/usr/bin/env python3
"""Smoke tests for agents_openai/s09_agent_teams.py."""

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
TARGET_PATH = REPO_ROOT / "agents_openai" / "s09_agent_teams.py"


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
        "agents_openai_s09_agent_teams",
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


class TestAgentOpenAI09(unittest.TestCase):
    def test_tools_expose_team_operations(self):
        module = _load_module()
        tool_names = {item["function"]["name"] for item in module.TOOLS}
        self.assertIn("spawn_teammate", tool_names)
        self.assertIn("list_teammates", tool_names)
        self.assertIn("send_message", tool_names)
        self.assertIn("read_inbox", tool_names)
        self.assertIn("broadcast", tool_names)

    def test_message_bus_send_read_and_broadcast(self):
        module = _load_module()

        with tempfile.TemporaryDirectory() as td:
            bus = module.MessageBus(Path(td))
            send_text = bus.send("lead", "alice", "hello")
            self.assertIn("Sent message to alice", send_text)

            inbox = bus.read_inbox("alice")
            self.assertEqual(len(inbox), 1)
            self.assertEqual(inbox[0]["from"], "lead")
            self.assertEqual(inbox[0]["content"], "hello")

            broadcast_text = bus.broadcast("lead", "sync", ["lead", "alice", "bob"])
            self.assertEqual("Broadcast to 2 teammates", broadcast_text)
            self.assertEqual(len(bus.read_inbox("alice")), 1)
            self.assertEqual(len(bus.read_inbox("bob")), 1)

    def test_agent_loop_injects_inbox_and_handles_tool_call(self):
        module = _load_module()

        first = types.SimpleNamespace(
            content="",
            tool_calls=[_tool_call("call_1", "list_teammates", {})],
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

        inbox_payload = [{"type": "message", "from": "alice", "content": "update"}]
        messages = [{"role": "user", "content": "status"}]

        with patch.object(module.BUS, "read_inbox", side_effect=[inbox_payload, []]):
            with patch.object(module.TEAM, "list_all", return_value="Team: default") as mock_list:
                module.agent_loop(messages)

        mock_list.assert_called_once()
        self.assertEqual(messages[-1]["role"], "assistant")
        self.assertEqual(messages[-1]["content"], "done")
        self.assertTrue(any(m.get("role") == "tool" for m in messages))
        self.assertTrue(any(m.get("content") == "Team: default" for m in messages))

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
