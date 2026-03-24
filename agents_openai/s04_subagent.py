#!/usr/bin/env python3
# Harness: context isolation -- protecting the model's clarity of thought.
"""
s04_subagent.py - Subagents (OpenAI format)

Spawn a child agent with fresh messages=[]. The child works in its own
context, sharing the filesystem, then returns only a summary to the parent.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
try:
    from agents_openai.tools import (
        file_tools,
        function_tool,
        make_file_tool_functions,
    )
except ImportError:
    from tools import (
        file_tools,
        function_tool,
        make_file_tool_functions,
    )

load_dotenv(override=True)

WORKDIR = Path.cwd()
client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.getenv("OPENAI_BASE_URL"),
)
MODEL = os.environ["MODEL_ID"]

SYSTEM = (
    f"You are a coding agent at {WORKDIR}. "
    "Use the task tool to delegate exploration or subtasks."
)
SUBAGENT_SYSTEM = (
    f"You are a coding subagent at {WORKDIR}. "
    "Complete the given task, then summarize your findings."
)

safe_path, run_bash, run_read, run_write, run_edit = make_file_tool_functions(WORKDIR)


TOOL_HANDLERS = {
    "bash": lambda **kw: run_bash(kw["command"]),
    "read_file": lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file": lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
}


CHILD_TOOLS = file_tools()

PARENT_TOOLS = CHILD_TOOLS + [
    function_tool(
        "task",
        "Spawn a subagent with fresh context. It shares the filesystem but not conversation history.",
        {
            "prompt": {"type": "string"},
            "description": {
                "type": "string",
                "description": "Short description of the task",
            },
        },
        ["prompt"],
    )
]


def parse_tool_input(tool_call) -> dict:
    try:
        return json.loads(tool_call.function.arguments or "{}")
    except json.JSONDecodeError:
        return {}


def assistant_to_history(assistant) -> dict:
    turn = {"role": "assistant", "content": assistant.content or ""}
    if assistant.tool_calls:
        turn["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in assistant.tool_calls
        ]
    return turn


def call_model(system_prompt: str, messages: list, tools: list):
    return client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system_prompt}, *messages],
        tools=tools,
    )


def run_subagent(prompt: str) -> str:
    sub_messages = [{"role": "user", "content": prompt}]  # fresh context
    final_text = "(no summary)"
    for _ in range(30):  # safety limit
        response = call_model(SUBAGENT_SYSTEM, sub_messages, CHILD_TOOLS)
        assistant = response.choices[0].message
        sub_messages.append(assistant_to_history(assistant))

        if assistant.content:
            final_text = assistant.content
        if not assistant.tool_calls:
            break

        for tool_call in assistant.tool_calls:
            tool_input = parse_tool_input(tool_call)
            handler = TOOL_HANDLERS.get(tool_call.function.name)
            try:
                output = handler(**tool_input) if handler else f"Unknown tool: {tool_call.function.name}"
            except Exception as e:
                output = f"Error: {e}"
            sub_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(output)[:50000],
                }
            )

    return final_text


def agent_loop(messages: list):
    while True:
        response = call_model(SYSTEM, messages, PARENT_TOOLS)
        assistant = response.choices[0].message
        messages.append(assistant_to_history(assistant))
        if not assistant.tool_calls:
            return

        for tool_call in assistant.tool_calls:
            tool_input = parse_tool_input(tool_call)
            if tool_call.function.name == "task":
                desc = tool_input.get("description", "subtask")
                prompt = tool_input.get("prompt", "")
                print(f"> task ({desc}): {prompt[:80]}")
                output = run_subagent(prompt) if prompt else "Error: prompt is required"
            else:
                handler = TOOL_HANDLERS.get(tool_call.function.name)
                try:
                    output = handler(**tool_input) if handler else f"Unknown tool: {tool_call.function.name}"
                except Exception as e:
                    output = f"Error: {e}"

            print(f"  {str(output)[:200]}")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(output),
                }
            )


if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms04 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        response_content = history[-1].get("content", "")
        if response_content:
            print(response_content)
        print()
