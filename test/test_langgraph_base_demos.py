#!/usr/bin/env python3
"""Smoke tests for the LangGraph base demos."""

from __future__ import annotations

import importlib.util
import py_compile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LANGGRAPH_BASE = REPO_ROOT / "langgraph_base"

DEMO_FILES = [
    "01_react_demo.py",
    "02_human_review_demo.py",
    "03_subagent_demo.py",
    "04_todo_demo.py",
    "05_context_compression_demo.py",
    "06_streaming_demo.py",
    "07_checkpointer_demo.py",
    "08_stream_events_demo.py",
]


class TestLanggraphBaseDemos(unittest.TestCase):
    def test_demo_files_exist(self):
        for filename in DEMO_FILES:
            with self.subTest(filename=filename):
                self.assertTrue((LANGGRAPH_BASE / filename).exists())

    def test_demo_files_compile(self):
        for filename in DEMO_FILES:
            with self.subTest(filename=filename):
                py_compile.compile(str(LANGGRAPH_BASE / filename), doraise=True)

    def test_demo_modules_expose_build_graph_and_main(self):
        for filename in DEMO_FILES:
            path = LANGGRAPH_BASE / filename
            module_name = f"test_{path.stem}"
            with self.subTest(filename=filename):
                spec = importlib.util.spec_from_file_location(module_name, path)
                self.assertIsNotNone(spec)
                self.assertIsNotNone(spec.loader)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self.assertTrue(callable(getattr(module, "build_graph", None)))
                self.assertTrue(callable(getattr(module, "main", None)))


if __name__ == "__main__":
    unittest.main()
