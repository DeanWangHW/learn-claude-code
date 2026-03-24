#!/usr/bin/env python3
"""Shared tool implementations and tool schemas for OpenAI agents."""

from copy import deepcopy
from pathlib import Path
import subprocess
from typing import Callable


MAX_TOOL_OUTPUT_CHARS = 50000
DANGEROUS_COMMAND_SNIPPETS = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]


def function_tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


_BASH_TOOL = function_tool(
    "bash",
    "Run a shell command.",
    {"command": {"type": "string"}},
    ["command"],
)

_READ_FILE_TOOL = function_tool(
    "read_file",
    "Read file contents.",
    {"path": {"type": "string"}, "limit": {"type": "integer"}},
    ["path"],
)

_WRITE_FILE_TOOL = function_tool(
    "write_file",
    "Write content to file.",
    {"path": {"type": "string"}, "content": {"type": "string"}},
    ["path", "content"],
)

_EDIT_FILE_TOOL = function_tool(
    "edit_file",
    "Replace exact text in file.",
    {
        "path": {"type": "string"},
        "old_text": {"type": "string"},
        "new_text": {"type": "string"},
    },
    ["path", "old_text", "new_text"],
)


def bash_tool() -> dict:
    return deepcopy(_BASH_TOOL)


def file_tools() -> list[dict]:
    return [
        deepcopy(_BASH_TOOL),
        deepcopy(_READ_FILE_TOOL),
        deepcopy(_WRITE_FILE_TOOL),
        deepcopy(_EDIT_FILE_TOOL),
    ]


def safe_path(workdir: Path, relative_path: str) -> Path:
    path = (workdir / relative_path).resolve()
    if not path.is_relative_to(workdir):
        raise ValueError(f"Path escapes workspace: {relative_path}")
    return path


def run_bash(command: str, workdir: Path) -> str:
    if any(s in command for s in DANGEROUS_COMMAND_SNIPPETS):
        return "Error: Dangerous command blocked"
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=120,
        )
        output = (result.stdout + result.stderr).strip()
        return output[:MAX_TOOL_OUTPUT_CHARS] if output else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def run_read(path: str, workdir: Path, limit: int | None = None) -> str:
    try:
        text = safe_path(workdir, path).read_text()
        lines = text.splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)[:MAX_TOOL_OUTPUT_CHARS]
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str, workdir: Path) -> str:
    try:
        fp = safe_path(workdir, path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str, workdir: Path) -> str:
    try:
        fp = safe_path(workdir, path)
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


def make_file_tool_functions(
    workdir: Path,
) -> tuple[
    Callable[[str], Path],
    Callable[[str], str],
    Callable[[str, int | None], str],
    Callable[[str, str], str],
    Callable[[str, str, str], str],
]:
    def _safe_path(p: str) -> Path:
        return safe_path(workdir, p)

    def _run_bash(command: str) -> str:
        return run_bash(command, workdir)

    def _run_read(path: str, limit: int | None = None) -> str:
        return run_read(path, workdir, limit)

    def _run_write(path: str, content: str) -> str:
        return run_write(path, content, workdir)

    def _run_edit(path: str, old_text: str, new_text: str) -> str:
        return run_edit(path, old_text, new_text, workdir)

    return _safe_path, _run_bash, _run_read, _run_write, _run_edit
