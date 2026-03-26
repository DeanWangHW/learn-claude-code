#!/usr/bin/env python3
"""Shared helpers for the LangGraph learning demos."""

from __future__ import annotations

import ast
import operator
import os
import uuid
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


WORKDIR = Path.cwd()
DEFAULT_MODEL = "gpt-4o-mini"
MAX_GRAPH_STEPS = 8

_ALLOWED_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def load_demo_env() -> None:
    load_dotenv(override=True)


def build_chat_model(*, tools: list[Any] | None = None, temperature: float = 0) -> ChatOpenAI:
    load_demo_env()
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("OPENAI_BASE_URL"):
        raise RuntimeError(
            "Missing auth config. Set OPENAI_API_KEY in .env, "
            "or provide OPENAI_BASE_URL for a compatible endpoint."
        )

    model = ChatOpenAI(
        model=os.getenv("MODEL_ID", DEFAULT_MODEL),
        temperature=temperature,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
    if tools:
        return model.bind_tools(tools)
    return model


def next_thread_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BIN_OPS:
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return _ALLOWED_BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY_OPS:
        return _ALLOWED_UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Unsupported expression")


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression safely."""
    try:
        parsed = ast.parse(expression, mode="eval")
        result = _safe_eval(parsed.body)
        if result.is_integer():
            return str(int(result))
        return str(result)
    except Exception as exc:  # pragma: no cover - guard path
        return f"calculator error: {exc}"


@tool
def current_workdir() -> str:
    """Return the current working directory."""
    return str(WORKDIR)


@tool
def list_workspace_files(limit: int = 10) -> str:
    """List a few files from the current workspace."""
    files = sorted(
        path.relative_to(WORKDIR).as_posix()
        for path in WORKDIR.rglob("*")
        if path.is_file() and ".git/" not in path.as_posix()
    )
    if not files:
        return "(no files found)"
    clipped = files[: max(1, min(limit, 50))]
    return "\n".join(clipped)


def extract_text(message: BaseMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    if isinstance(message.content, list):
        parts: list[str] = []
        for block in message.content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = str(block.get("text", "")).strip()
                if text:
                    parts.append(text)
        if parts:
            return "\n".join(parts)
    return str(message.content)


def last_ai_text(messages: Iterable[BaseMessage]) -> str:
    for message in reversed(list(messages)):
        if isinstance(message, AIMessage):
            return extract_text(message)
    return ""


def summarize_messages_locally(messages: list[BaseMessage]) -> str:
    lines: list[str] = []
    for message in messages:
        role = message.type
        text = extract_text(message).strip()
        if not text:
            continue
        compact = " ".join(text.split())
        lines.append(f"- {role}: {compact[:140]}")
    return "\n".join(lines[:12]) if lines else "(empty summary)"
