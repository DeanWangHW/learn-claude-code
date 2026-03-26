#!/usr/bin/env python3
"""Focused demo: explicit checkpoint persistence and resume flow."""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from langgraph_base._shared import next_thread_id


class CheckpointerDemoState(TypedDict):
    task: str
    plan: str
    approved: bool
    review_note: str
    result: str


def _build_plan(task: str) -> str:
    return "\n".join(
        [
            f"Task: {task}",
            "1. Inspect the requirement.",
            "2. Draft the implementation steps.",
            "3. Review and approve the draft before execution.",
        ]
    )


def _parse_review_reply(raw: str) -> dict[str, str]:
    normalized = raw.strip()
    lowered = normalized.lower()
    if lowered in {"y", "yes", "approve", "approved"}:
        return {"action": "approve", "feedback": "Approved by reviewer."}
    return {
        "action": "reject",
        "feedback": normalized or "Rejected by reviewer.",
    }


def build_graph():
    def draft_plan(state: CheckpointerDemoState) -> dict[str, Any]:
        return {"plan": _build_plan(state["task"])}

    def review_plan(state: CheckpointerDemoState) -> Command:
        decision = interrupt(
            {
                "question": "Review the generated plan before continuing.",
                "plan": state["plan"],
            }
        )
        payload = decision if isinstance(decision, dict) else {"action": str(decision)}
        action = str(payload.get("action", "approve")).lower()
        feedback = str(payload.get("feedback", "")).strip()
        approved = action in {"approve", "approved", "y", "yes"}
        return Command(
            goto="finalize",
            update={
                "approved": approved,
                "review_note": feedback or ("Approved." if approved else "Rejected."),
            },
        )

    def finalize(state: CheckpointerDemoState) -> dict[str, Any]:
        if state["approved"]:
            result = (
                "Plan approved.\n\n"
                f"{state['plan']}\n\n"
                f"Reviewer note: {state['review_note']}"
            )
        else:
            result = (
                "Plan rejected.\n\n"
                f"{state['plan']}\n\n"
                f"Reviewer note: {state['review_note']}"
            )
        return {"result": result}

    graph = StateGraph(CheckpointerDemoState)
    graph.add_node("draft_plan", draft_plan)
    graph.add_node("review_plan", review_plan)
    graph.add_node("finalize", finalize)
    graph.add_edge(START, "draft_plan")
    graph.add_edge("draft_plan", "review_plan")
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=MemorySaver())


def main() -> None:
    app = build_graph()

    print("Focused checkpointer demo ready. Type 'exit' to quit.")
    while True:
        try:
            task = input("checkpointer_demo >> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not task or task.lower() in {"q", "quit", "exit"}:
            break

        config = {"configurable": {"thread_id": next_thread_id("checkpoint")}}
        result = app.invoke(
            {
                "task": task,
                "plan": "",
                "approved": False,
                "review_note": "",
                "result": "",
            },
            config,
        )

        if "__interrupt__" in result:
            snapshot = app.get_state(config)
            print("[checkpoint_saved]")
            print(snapshot.values["plan"])
            review_reply = input("approve? [y / feedback] >> ").strip()
            result = app.invoke(Command(resume=_parse_review_reply(review_reply)), config)

        print(result["result"])


if __name__ == "__main__":
    main()

