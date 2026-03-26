#!/usr/bin/env python3
"""Chapter 2 demo: human review with interrupts and checkpointing."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

from langgraph_base._shared import (
    MAX_GRAPH_STEPS,
    build_chat_model,
    calculator,
    current_workdir,
    last_ai_text,
    list_workspace_files,
    next_thread_id,
)


SYSTEM_PROMPT = (
    "You are the chapter-2 LangGraph demo agent. Ask to use tools when they help, "
    "and wait for the review gate before any tool actually runs."
)


class HumanReviewState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    steps: int


def _parse_review_reply(raw: str) -> dict[str, str]:
    normalized = raw.strip()
    lowered = normalized.lower()
    if lowered in {"y", "yes", "approve", "approved"}:
        return {"action": "approve", "feedback": ""}
    return {
        "action": "reject",
        "feedback": normalized or "Do not run the tool. Answer directly instead.",
    }


def build_graph():
    tools = [calculator, current_workdir, list_workspace_files]
    llm = build_chat_model(tools=tools)
    tool_node = ToolNode(tools)

    def agent_node(state: HumanReviewState) -> dict[str, Any]:
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT), *messages]
        response = llm.invoke(messages)
        return {"messages": [response], "steps": state.get("steps", 0) + 1}

    def review_node(state: HumanReviewState) -> Command:
        last_message = state["messages"][-1]
        decision = interrupt(
            {
                "question": "Approve the pending tool calls?",
                "tool_calls": getattr(last_message, "tool_calls", []),
            }
        )
        payload = decision if isinstance(decision, dict) else {"action": str(decision)}
        action = str(payload.get("action", "approve")).lower()
        feedback = str(payload.get("feedback", "")).strip()

        if action in {"approve", "approved", "y", "yes"}:
            return Command(
                goto="tools",
                update={"messages": [HumanMessage(content="User approved the tool call.")]},
            )

        rejection_text = feedback or "User rejected the tool call. Continue without tools."
        return Command(
            goto="agent",
            update={"messages": [HumanMessage(content=rejection_text)]},
        )

    def route_after_agent(state: HumanReviewState) -> str:
        if state.get("steps", 0) >= MAX_GRAPH_STEPS:
            return "end"

        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "review"
        return "end"

    graph = StateGraph(HumanReviewState)
    graph.add_node("agent", agent_node)
    graph.add_node("review", review_node)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {"review": "review", "end": END},
    )
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=MemorySaver())


def main() -> None:
    app = build_graph()

    print("Chapter 2 / Human review demo ready. Type 'exit' to quit.")
    while True:
        try:
            user_input = input("human_review_demo >> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input or user_input.lower() in {"q", "quit", "exit"}:
            break

        config = {"configurable": {"thread_id": next_thread_id("human-review")}}
        result = app.invoke({"messages": [HumanMessage(content=user_input)], "steps": 0}, config)

        while "__interrupt__" in result:
            interrupt_payload = result["__interrupt__"][0]
            print(f"[interrupt] {interrupt_payload.value}")
            reviewer_input = input("approve? [y / feedback] >> ").strip()
            result = app.invoke(Command(resume=_parse_review_reply(reviewer_input)), config)

        print(last_ai_text(result["messages"]))


if __name__ == "__main__":
    main()
