#!/usr/bin/env python3
"""Chapter 5 demo: context compression before the model call."""

from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from langgraph_base._shared import build_chat_model, last_ai_text, summarize_messages_locally


SYSTEM_PROMPT = (
    "You are the chapter-5 context engineering demo. Use the supplied summary when "
    "it exists, and answer based on the most recent turns."
)
KEEP_LAST_MESSAGES = 4
RAW_MESSAGE_LIMIT = 6


class CompressionDemoState(TypedDict):
    messages: list[BaseMessage]
    summary: str
    compression_count: int


def build_graph():
    llm = build_chat_model()

    def compress_context(state: CompressionDemoState) -> dict[str, Any]:
        messages = list(state.get("messages", []))
        if len(messages) <= RAW_MESSAGE_LIMIT:
            return {}

        older_messages = messages[:-KEEP_LAST_MESSAGES]
        recent_messages = messages[-KEEP_LAST_MESSAGES:]
        new_summary = summarize_messages_locally(older_messages)
        merged_summary = "\n".join(
            part for part in [state.get("summary", "").strip(), new_summary] if part
        ).strip()

        return {
            "summary": merged_summary,
            "messages": recent_messages,
            "compression_count": state.get("compression_count", 0) + 1,
        }

    def agent_node(state: CompressionDemoState) -> dict[str, Any]:
        system_sections = [SYSTEM_PROMPT]
        if state.get("summary"):
            system_sections.append(f"Compressed conversation summary:\n{state['summary']}")

        response = llm.invoke(
            [SystemMessage(content="\n\n".join(system_sections)), *state["messages"]]
        )
        return {"messages": [*state["messages"], response]}

    graph = StateGraph(CompressionDemoState)
    graph.add_node("compress_context", compress_context)
    graph.add_node("agent", agent_node)
    graph.add_edge(START, "compress_context")
    graph.add_edge("compress_context", "agent")
    graph.add_edge("agent", END)
    return graph.compile()


def main() -> None:
    app = build_graph()
    state: CompressionDemoState = {
        "messages": [],
        "summary": "",
        "compression_count": 0,
    }

    print("Chapter 5 / Context compression demo ready. Type 'exit' to quit.")
    while True:
        try:
            user_input = input("compression_demo >> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input or user_input.lower() in {"q", "quit", "exit"}:
            break

        state["messages"] = [*state["messages"], HumanMessage(content=user_input)]
        state = app.invoke(state)
        print(last_ai_text(state["messages"]))
        print(
            f"[compression_count={state['compression_count']}] "
            f"retained_messages={len(state['messages'])}"
        )


if __name__ == "__main__":
    main()

