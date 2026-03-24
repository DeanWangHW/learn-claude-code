#!/usr/bin/env python3
"""Smoke tests for agents_openai/s05_skill_loading.py."""

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
TARGET_PATH = REPO_ROOT / "agents_openai" / "s05_skill_loading.py"


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

    spec = importlib.util.spec_from_file_location("agents_openai_s05_skill_loading", TARGET_PATH)
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


class TestAgentOpenAI05(unittest.TestCase):
    def test_run_bash_blocks_dangerous_command(self):
        module = _load_module()
        result = module.run_bash("sudo ls")
        self.assertEqual(result, "Error: Dangerous command blocked")

    def test_skill_loader_reads_frontmatter_and_body(self):
        module = _load_module()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "pdf").mkdir(parents=True)
            (root / "pdf" / "SKILL.md").write_text(
                "---\n"
                "name: pdf\n"
                "description: Process PDF files\n"
                "tags: docs,pdf\n"
                "---\n"
                "Use this skill for PDFs.\n",
                encoding="utf-8",
            )
            (root / "code-review").mkdir(parents=True)
            (root / "code-review" / "SKILL.md").write_text(
                "Review code for correctness.\n",
                encoding="utf-8",
            )

            loader = module.SkillLoader(root)
            descriptions = loader.get_descriptions()
            self.assertIn("pdf: Process PDF files [docs,pdf]", descriptions)
            self.assertIn("code-review: No description", descriptions)

            content = loader.get_content("pdf")
            self.assertTrue(content.startswith("<skill name=\"pdf\">"))
            self.assertIn("Use this skill for PDFs.", content)

            unknown = loader.get_content("missing")
            self.assertIn("Unknown skill 'missing'", unknown)
            self.assertIn("pdf", unknown)

    def test_tools_expose_load_skill(self):
        module = _load_module()
        tool_names = {item["function"]["name"] for item in module.TOOLS}
        self.assertIn("load_skill", tool_names)

    def test_agent_loop_tool_call_roundtrip(self):
        module = _load_module()

        first = types.SimpleNamespace(
            content="",
            tool_calls=[_tool_call("call_1", "load_skill", {"name": "pdf"})],
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

        messages = [{"role": "user", "content": "load skill pdf"}]
        with patch.object(module.SKILL_LOADER, "get_content", return_value="<skill name=\"pdf\">body</skill>") as mock_get:
            module.agent_loop(messages)

        mock_get.assert_called_once_with("pdf")
        self.assertEqual(messages[-1]["role"], "assistant")
        self.assertEqual(messages[-1]["content"], "done")
        self.assertTrue(any(m.get("role") == "tool" for m in messages))
        self.assertTrue(any(m.get("content") == "<skill name=\"pdf\">body</skill>" for m in messages))
        self.assertEqual(call_kwargs[0]["messages"][0]["role"], "system")
        self.assertEqual(call_kwargs[1]["messages"][-1]["role"], "tool")
        self.assertEqual(call_kwargs[1]["messages"][-1]["content"], "<skill name=\"pdf\">body</skill>")


if __name__ == "__main__":
    unittest.main()
