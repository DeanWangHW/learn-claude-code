#!/usr/bin/env python3
"""Smoke tests for agents_openai/s10_team_protocols.py."""

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
TARGET_PATH = REPO_ROOT / "agents_openai" / "s10_team_protocols.py"


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
        "agents_openai_s10_team_protocols",
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


class TestAgentOpenAI10(unittest.TestCase):
    def test_tools_expose_protocol_operations(self):
        module = _load_module()
        tool_names = {item["function"]["name"] for item in module.TOOLS}
        self.assertIn("shutdown_request", tool_names)
        self.assertIn("shutdown_response", tool_names)
        self.assertIn("plan_approval", tool_names)

    def test_shutdown_request_records_status_and_sends_message(self):
        module = _load_module()
        module.shutdown_requests.clear()

        with tempfile.TemporaryDirectory() as td:
            bus = module.MessageBus(Path(td))
            with patch.object(module, "BUS", bus):
                result = module.handle_shutdown_request("alice")

                request_id = result.split()[2]
                status = json.loads(module._check_shutdown_status(request_id))
                self.assertEqual(status["target"], "alice")
                self.assertEqual(status["status"], "pending")

                inbox = bus.read_inbox("alice")
                self.assertEqual(len(inbox), 1)
                self.assertEqual(inbox[0]["type"], "shutdown_request")
                self.assertEqual(inbox[0]["request_id"], request_id)

    def test_plan_review_updates_tracker_and_notifies_teammate(self):
        module = _load_module()
        module.plan_requests.clear()
        module.plan_requests["req12345"] = {
            "from": "alice",
            "plan": "Plan text",
            "status": "pending",
        }

        with tempfile.TemporaryDirectory() as td:
            bus = module.MessageBus(Path(td))
            with patch.object(module, "BUS", bus):
                result = module.handle_plan_review("req12345", True, "Looks good")
                self.assertEqual(result, "Plan approved for 'alice'")
                self.assertEqual(module.plan_requests["req12345"]["status"], "approved")

                inbox = bus.read_inbox("alice")
                self.assertEqual(len(inbox), 1)
                self.assertEqual(inbox[0]["type"], "plan_approval_response")
                self.assertTrue(inbox[0]["approve"])
                self.assertEqual(inbox[0]["feedback"], "Looks good")

                not_found = module.handle_plan_review("missing", False)
                self.assertIn("Unknown plan request_id", not_found)

    def test_agent_loop_injects_inbox_and_checks_shutdown_status(self):
        module = _load_module()
        module.shutdown_requests.clear()
        module.shutdown_requests["req11111"] = {"target": "alice", "status": "pending"}

        first = types.SimpleNamespace(
            content="",
            tool_calls=[_tool_call("call_1", "shutdown_response", {"request_id": "req11111"})],
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

        inbox_payload = [{"type": "plan_approval_response", "request_id": "req11111"}]
        messages = [{"role": "user", "content": "status"}]

        with patch.object(module.BUS, "read_inbox", side_effect=[inbox_payload, []]):
            module.agent_loop(messages)

        self.assertEqual(messages[-1]["role"], "assistant")
        self.assertEqual(messages[-1]["content"], "done")
        self.assertTrue(any(m.get("role") == "tool" for m in messages))
        self.assertTrue(
            any(
                isinstance(m.get("content"), str) and '"status": "pending"' in m["content"]
                for m in messages
            )
        )

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
