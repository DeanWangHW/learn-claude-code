#!/usr/bin/env python3
"""Focused demo: low-level event streaming with astream_events()."""

from __future__ import annotations

import asyncio
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
    extract_text,
    list_workspace_files,
)


SYSTEM_PROMPT = (
    "You are the event-stream demo. Use tools when useful, and keep the final answer short."
)


class StreamEventsDemoState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    steps: int


def build_graph():
    tools = [calculator, current_workdir, list_workspace_files]
    llm = build_chat_model(tools=tools)
    tool_node = ToolNode(tools)

    def agent_node(state: StreamEventsDemoState) -> dict[str, Any]:
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT), *messages]
        response = llm.invoke(messages)
        return {"messages": [response], "steps": state.get("steps", 0) + 1}

    def route_after_agent(state: StreamEventsDemoState) -> str:
        if state.get("steps", 0) >= MAX_GRAPH_STEPS:
            return "end"
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
        return "end"

    graph = StateGraph(StreamEventsDemoState)
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


async def _print_event_stream(app, user_input: str) -> None:
    final_output: dict[str, Any] | None = None
    async for event in app.astream_events(
        {"messages": [HumanMessage(content=user_input)], "steps": 0},
        version="v2",
    ):
        event_name = event["event"]
        name = event.get("name", "")

        if event_name == "on_chat_model_stream":
            chunk = event.get("data", {}).get("chunk")
            if chunk is not None:
                text = extract_text(chunk).strip()
                if text:
                    print(f"[model_chunk] {text}")
        elif event_name in {"on_tool_start", "on_tool_end"}:
            print(f"[{event_name}] {name}")
        elif event_name == "on_chain_stream" and name in {"agent", "tools"}:
            print(f"[node_stream] {name}: {event.get('data', {}).get('chunk')}")
        elif event_name == "on_chain_end" and name == "LangGraph":
            final_output = event.get("data", {}).get("output")

    if final_output:
        messages = final_output.get("messages", [])
        if messages:
            print("[final_output]")
            print(extract_text(messages[-1]))


def main() -> None:
    app = build_graph()

    print("Focused stream-events demo ready. Type 'exit' to quit.")
    while True:
        try:
            user_input = input("stream_events_demo >> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input or user_input.lower() in {"q", "quit", "exit"}:
            break

        asyncio.run(_print_event_stream(app, user_input))


if __name__ == "__main__":
    main()
