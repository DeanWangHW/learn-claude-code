#!/usr/bin/env python3
"""Chapter 4 demo: todo management embedded in graph state."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from langgraph_base._shared import MAX_GRAPH_STEPS, build_chat_model, last_ai_text


SYSTEM_PROMPT = (
    "You are the chapter-4 task manager demo. Use todo tools whenever the user asks "
    "to plan, track, start, finish, or review work. Keep the todo list current."
)


class TodoItem(TypedDict):
    id: int
    title: str
    status: str


class TodoDemoState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    todos: list[TodoItem]
    next_todo_id: int
    steps: int


@tool
def add_todo(title: str) -> str:
    """Create a new todo item."""
    return ""


@tool
def start_todo(todo_id: int) -> str:
    """Mark a todo item as in progress."""
    return ""


@tool
def complete_todo(todo_id: int) -> str:
    """Mark a todo item as completed."""
    return ""


@tool
def list_todos() -> str:
    """List the current todo items."""
    return ""


def _format_todos(todos: list[TodoItem]) -> str:
    if not todos:
        return "No todos yet."
    return "\n".join(f"- #{item['id']} [{item['status']}] {item['title']}" for item in todos)


def _find_todo(todos: list[TodoItem], todo_id: int) -> TodoItem | None:
    for item in todos:
        if item["id"] == todo_id:
            return item
    return None


def build_graph():
    tools = [add_todo, start_todo, complete_todo, list_todos]
    llm = build_chat_model(tools=tools)

    def agent_node(state: TodoDemoState) -> dict[str, Any]:
        todo_context = _format_todos(state.get("todos", []))
        system_prompt = (
            f"{SYSTEM_PROMPT}\n\nCurrent todo state:\n{todo_context}\n"
            "Call todo tools when the user is managing tasks."
        )
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_prompt), *messages]
        else:
            messages[0] = SystemMessage(content=system_prompt)
        response = llm.invoke(messages)
        return {"messages": [response], "steps": state.get("steps", 0) + 1}

    def todo_tools_node(state: TodoDemoState) -> dict[str, Any]:
        todos = [dict(item) for item in state.get("todos", [])]
        next_todo_id = state.get("next_todo_id", 1)
        outputs: list[ToolMessage] = []

        for tool_call in state["messages"][-1].tool_calls:
            name = tool_call["name"]
            args = tool_call["args"]

            if name == "add_todo":
                todo = {"id": next_todo_id, "title": args["title"], "status": "pending"}
                todos.append(todo)
                content = f"Created todo #{next_todo_id}: {args['title']}"
                next_todo_id += 1
            elif name == "start_todo":
                todo = _find_todo(todos, int(args["todo_id"]))
                if todo is None:
                    content = f"Todo #{args['todo_id']} not found."
                else:
                    todo["status"] = "in_progress"
                    content = f"Todo #{todo['id']} is now in progress."
            elif name == "complete_todo":
                todo = _find_todo(todos, int(args["todo_id"]))
                if todo is None:
                    content = f"Todo #{args['todo_id']} not found."
                else:
                    todo["status"] = "completed"
                    content = f"Todo #{todo['id']} is completed."
            elif name == "list_todos":
                content = _format_todos(todos)
            else:  # pragma: no cover - guard path
                content = f"Unknown tool: {name}"

            outputs.append(
                ToolMessage(
                    content=content,
                    name=name,
                    tool_call_id=tool_call["id"],
                )
            )

        return {"messages": outputs, "todos": todos, "next_todo_id": next_todo_id}

    def route_after_agent(state: TodoDemoState) -> str:
        if state.get("steps", 0) >= MAX_GRAPH_STEPS:
            return "end"
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
        return "end"

    graph = StateGraph(TodoDemoState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", todo_tools_node)
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
    todos: list[TodoItem] = []
    next_todo_id = 1

    print("Chapter 4 / Todo demo ready. Type 'exit' to quit.")
    while True:
        try:
            user_input = input("todo_demo >> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input or user_input.lower() in {"q", "quit", "exit"}:
            break

        history.append(HumanMessage(content=user_input))
        result = app.invoke(
            {
                "messages": history,
                "todos": todos,
                "next_todo_id": next_todo_id,
                "steps": 0,
            }
        )
        history = result["messages"]
        todos = result["todos"]
        next_todo_id = result["next_todo_id"]
        print(last_ai_text(history))
        print(_format_todos(todos))


if __name__ == "__main__":
    main()
