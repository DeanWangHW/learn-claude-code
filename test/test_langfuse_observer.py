#!/usr/bin/env python3
"""Tests for agents_openai/observability/langfuse_observer.py."""

import types
import unittest
from unittest.mock import patch

from agents_openai.observability.langfuse_observer import (
    LangfuseObserver,
    NoopObserver,
    create_observer,
)


class _FakeChildObservation:
    def __init__(self):
        self.end_calls = 0

    def end(self):
        self.end_calls += 1
        return self


class _FakeRootObservation:
    def __init__(self):
        self.trace_id = "trace_test_123"
        self.id = "obs_root_123"
        self.started_children = []
        self.events = []
        self.updated = []
        self.end_calls = 0

    def start_observation(self, **kwargs):
        self.started_children.append(kwargs)
        return _FakeChildObservation()

    def create_event(self, **kwargs):
        self.events.append(kwargs)
        return self

    def update(self, **kwargs):
        self.updated.append(kwargs)
        return self

    def end(self, **kwargs):
        self.end_calls += 1
        return self


class _FakeLangfuseClient:
    last_instance = None

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.root = _FakeRootObservation()
        self.flush_calls = 0
        _FakeLangfuseClient.last_instance = self

    def start_observation(self, **kwargs):
        self.start_kwargs = kwargs
        return self.root

    def flush(self):
        self.flush_calls += 1


class _FakeLangfuseModule(types.SimpleNamespace):
    pass


class TestLangfuseObserver(unittest.TestCase):
    def test_create_observer_returns_noop_without_keys(self):
        with patch.dict(
            "os.environ",
            {
                "LANGFUSE_ENABLED": "1",
                "LANGFUSE_PUBLIC_KEY": "",
                "LANGFUSE_SECRET_KEY": "",
            },
            clear=False,
        ):
            observer = create_observer(agent_name="s01_agent_loop", model="gpt-test")
        self.assertIsInstance(observer, NoopObserver)

    def test_create_observer_returns_langfuse_observer_with_keys(self):
        fake_module = _FakeLangfuseModule(Langfuse=_FakeLangfuseClient)
        with patch.dict("sys.modules", {"langfuse": fake_module}):
            with patch.dict(
                "os.environ",
                {
                    "LANGFUSE_ENABLED": "1",
                    "LANGFUSE_PUBLIC_KEY": "pk-lf-test",
                    "LANGFUSE_SECRET_KEY": "sk-lf-test",
                },
                clear=False,
            ):
                observer = create_observer(agent_name="s01_agent_loop", model="gpt-test")

        self.assertIsInstance(observer, LangfuseObserver)
        self.assertTrue(observer.enabled)
        self.assertIsNotNone(_FakeLangfuseClient.last_instance)

    def test_langfuse_observer_lifecycle_uses_v4_observation_api(self):
        fake_module = _FakeLangfuseModule(Langfuse=_FakeLangfuseClient)

        with patch.dict("sys.modules", {"langfuse": fake_module}):
            with patch.dict(
                "os.environ",
                {
                    "LANGFUSE_ENABLED": "1",
                    "LANGFUSE_PUBLIC_KEY": "pk-lf-test",
                    "LANGFUSE_SECRET_KEY": "sk-lf-test",
                    "LANGFUSE_DEBUG": "0",
                },
                clear=False,
            ):
                observer = create_observer(agent_name="s01_agent_loop", model="gpt-test")

        ctx = observer.start_trace(user_input="list files", history_len=3)
        self.assertIsNotNone(ctx)
        self.assertIsNotNone(ctx.root_observation)

        observer.on_model_response(
            ctx,
            assistant_text="I'll run bash",
            tool_calls=[{"id": "c1", "function": {"name": "bash"}}],
        )
        observer.on_tool_result(
            ctx,
            tool_name="bash",
            tool_input={"command": "ls -la"},
            output="file1\nfile2\n",
        )
        observer.finish_trace(ctx, final_output="done")
        observer.flush()

        client = _FakeLangfuseClient.last_instance
        self.assertIsNotNone(client)
        self.assertEqual(client.flush_calls, 1)

        root = client.root
        self.assertEqual(root.end_calls, 1)
        self.assertTrue(root.started_children)
        self.assertEqual(root.started_children[0]["name"], "model_turn")
        self.assertEqual(root.started_children[0]["as_type"], "generation")
        self.assertEqual(root.started_children[0]["model"], "gpt-test")

        self.assertTrue(root.events)
        self.assertEqual(root.events[0]["name"], "tool:bash")
        self.assertEqual(root.events[0]["input"], {"command": "ls -la"})

        self.assertTrue(root.updated)
        self.assertEqual(root.updated[-1]["output"], "done")


if __name__ == "__main__":
    unittest.main()
