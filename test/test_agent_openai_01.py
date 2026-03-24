#!/usr/bin/env python3
"""Smoke tests for agents_openai/s01_agent_loop.py."""

import importlib.util
import os
import sys
import types
import tempfile
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


def _load_module_real():
    spec = importlib.util.spec_from_file_location(
        "agents_openai_s01_agent_loop_real",
        TARGET_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
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

    def test_real_openai_list_python_files_in_directory(self):
        if os.getenv("RUN_OPENAI_REAL_TEST") != "1":
            self.skipTest("Set RUN_OPENAI_REAL_TEST=1 to enable real OpenAI integration test")
        if importlib.util.find_spec("openai") is None:
            self.skipTest("openai package is not installed")

        if importlib.util.find_spec("dotenv") is not None:
            from dotenv import load_dotenv

            load_dotenv(override=True)

        if not os.getenv("OPENAI_API_KEY"):
            self.skipTest("OPENAI_API_KEY is required for real OpenAI integration test")
        os.environ.setdefault("MODEL_ID", os.getenv("OPENAI_REAL_MODEL", "gpt-4.1-mini"))

        module = _load_module_real()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            (tmp / "alpha.py").write_text("print('alpha')\n", encoding="utf-8")
            (tmp / "beta.py").write_text("print('beta')\n", encoding="utf-8")
            (tmp / "notes.txt").write_text("not python\n", encoding="utf-8")

            old_cwd = os.getcwd()
            try:
                os.chdir(tmp)
                module.SYSTEM = (
                    f"You are a coding agent at {os.getcwd()}. "
                    "Use bash to solve tasks. Act, don't explain."
                )
                history = [{"role": "user", "content": "List all Python files in this directory"}]
                module.agent_loop(history)
            finally:
                os.chdir(old_cwd)

            response = history[-1].get("content", "")
            self.assertTrue(response.strip(), "Empty assistant response from real API")

            expected_py_files = sorted(p.name for p in tmp.glob("*.py"))
            missing = [name for name in expected_py_files if name not in response]
            self.assertFalse(
                missing,
                f"Missing files in model response: {missing}. Response: {response!r}",
            )
            self.assertNotIn("notes.txt", response)


if __name__ == "__main__":
    unittest.main()
