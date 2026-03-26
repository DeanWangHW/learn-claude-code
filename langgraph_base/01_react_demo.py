#!/usr/bin/env python3
"""Chapter 1 demo: the minimal ReAct-style LangGraph agent."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
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
    "You are the chapter-1 LangGraph demo agent. Keep replies short, "
    "and use tools when arithmetic or workspace inspection is useful."
)


class ReactDemoState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    steps: int


def build_graph():
    tools = [calculator, current_workdir, list_workspace_files]
    llm = build_chat_model(tools=tools)
    tool_node = ToolNode(tools)

    def agent_node(state: ReactDemoState) -> dict[str, Any]:
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT), *messages]
        response = llm.invoke(messages)
        return {"messages": [response], "steps": state.get("steps", 0) + 1}

    def route_after_agent(state: ReactDemoState) -> str:
        if state.get("steps", 0) >= MAX_GRAPH_STEPS:
            return "end"

        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
        return "end"

    graph = StateGraph(ReactDemoState)
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

    print("Chapter 1 / ReAct demo ready. Type 'exit' to quit.")
    while True:
        try:
            user_input = input("react_demo >> ").strip()
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

