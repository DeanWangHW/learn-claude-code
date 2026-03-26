#!/usr/bin/env python3
"""Chapter 3 demo: subagent delegation through a task tool."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from langgraph_base._shared import (
    MAX_GRAPH_STEPS,
    build_chat_model,
    calculator,
    current_workdir,
    last_ai_text,
    list_workspace_files,
)


SYSTEM_PROMPT = (
    "You are the chapter-3 orchestration demo agent. Use the delegate_task tool "
    "for complex analysis that can be pushed to a specialized subagent."
)


class SubagentDemoState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    steps: int


def _build_subagent(system_prompt: str, tools: list[Any]):
    llm = build_chat_model(tools=tools)
    tool_node = ToolNode(tools)

    def agent_node(state: SubagentDemoState) -> dict[str, Any]:
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_prompt), *messages]
        response = llm.invoke(messages)
        return {"messages": [response], "steps": state.get("steps", 0) + 1}

    def route_after_agent(state: SubagentDemoState) -> str:
        if state.get("steps", 0) >= MAX_GRAPH_STEPS:
            return "end"
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
        return "end"

    graph = StateGraph(SubagentDemoState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", "end": END},
    )
    graph.add_edge("tools", "agent")
    return graph.compile()


def build_graph():
    shared_tools = [calculator, current_workdir, list_workspace_files]
    subagents = {
        "general-purpose": _build_subagent(
            "You are a general-purpose subagent. Investigate the request carefully, "
            "use tools if needed, and return a concise but complete answer.",
            shared_tools,
        ),
        "code-analyzer": _build_subagent(
            "You are a code analysis subagent. Focus on repository structure, code intent, "
            "and implementation detail. Prefer file and directory inspection tools.",
            [current_workdir, list_workspace_files],
        ),
        "math-helper": _build_subagent(
            "You are a math-focused subagent. Use the calculator tool when arithmetic helps.",
            [calculator],
        ),
    }

    @tool
    def delegate_task(description: str, subagent_type: str = "general-purpose") -> str:
        """Delegate work to a specialized subagent.

        Allowed subagent types: general-purpose, code-analyzer, math-helper.
        """
        subagent = subagents.get(subagent_type)
        if subagent is None:
            return f"Unknown subagent_type: {subagent_type}"
        result = subagent.invoke({"messages": [HumanMessage(content=description)], "steps": 0})
        return last_ai_text(result["messages"])

    tools = [delegate_task, calculator, current_workdir, list_workspace_files]
    llm = build_chat_model(tools=tools)
    tool_node = ToolNode(tools)

    def agent_node(state: SubagentDemoState) -> dict[str, Any]:
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT), *messages]
        response = llm.invoke(messages)
        return {"messages": [response], "steps": state.get("steps", 0) + 1}

    def route_after_agent(state: SubagentDemoState) -> str:
        if state.get("steps", 0) >= MAX_GRAPH_STEPS:
            return "end"
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
        return "end"

    graph = StateGraph(SubagentDemoState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", "end": END},
    )
    graph.add_edge("tools", "agent")
    return graph.compile()


def main() -> None:
    app = build_graph()
    history: list[BaseMessage] = []

    print("Chapter 3 / Subagent demo ready. Type 'exit' to quit.")
    while True:
        try:
            user_input = input("subagent_demo >> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input or user_input.lower() in {"q", "quit", "exit"}:
            break

        history.append(HumanMessage(content=user_input))
        result = app.invoke({"messages": history, "steps": 0})
        history = result["messages"]
        print(last_ai_text(history))


if __name__ == "__main__":
    main()

