#!/usr/bin/env python3
"""A minimal LangGraph agent scaffold for this repository.

Run:
    python langgraph_base/basic_agent.py

Required packages (install as needed):
    langgraph, langchain-OPENAI, langchain-core
"""

from __future__ import annotations

import ast
import operator
import os
from typing import Annotated, Any, TypedDict

from dotenv import load_dotenv
# from langchain_OPENAI import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

SYSTEM_PROMPT = (
    "You are a helpful LangGraph demo agent. Keep responses concise and accurate. "
    "Use tools when math or local context lookup is needed."
)
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


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    steps: int


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
    except Exception as exc:  # pragma: no cover - simple guard path
        return f"calculator error: {exc}"


@tool
def current_workdir() -> str:
    """Return current working directory."""
    return os.getcwd()


def _extract_text(message: AIMessage) -> str:
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


def _build_model(tools: list[Any]) -> ChatOpenAI:
    model_id = os.getenv("MODEL_ID", "claude-3-5-sonnet-latest")
    return ChatOpenAI(
        model=model_id,
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    ).bind_tools(tools)


def build_graph():
    tools = [calculator, current_workdir]
    llm = _build_model(tools)
    tool_node = ToolNode(tools)

    def agent_node(state: AgentState) -> dict[str, Any]:
        messages: list[AnyMessage] = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT), *messages]
        response = llm.invoke(messages)
        return {
            "messages": [response],
            "steps": state.get("steps", 0) + 1,
        }

    def route_after_agent(state: AgentState) -> str:
        if state.get("steps", 0) >= MAX_GRAPH_STEPS:
            return "end"

        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"

        return "end"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "tools": "tools",
            "end": END,
        },
    )
    graph.add_edge("tools", "agent")

    return graph.compile()


def main() -> None:
    load_dotenv(override=True)

    if not os.getenv("OPENAI_API_KEY") and not os.getenv("OPENAI_BASE_URL"):
        raise RuntimeError(
            "Missing auth config. Set OPENAI_API_KEY in .env, "
            "or provide OPENAI_BASE_URL for a compatible endpoint."
        )

    app = build_graph()
    history: list[AnyMessage] = []

    print("LangGraph base agent ready. Type 'exit' to quit.")

    while True:
        try:
            user_input = input("langgraph_base >> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input or user_input.lower() in {"q", "quit", "exit"}:
            break

        history.append(HumanMessage(content=user_input))
        result = app.invoke({"messages": history, "steps": 0})
        history = result["messages"]

        for message in reversed(history):
            if isinstance(message, AIMessage):
                print(_extract_text(message))
                break


if __name__ == "__main__":
    main()
