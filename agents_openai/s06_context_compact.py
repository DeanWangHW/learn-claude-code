#!/usr/bin/env python3
# Harness: compression -- clean memory for infinite sessions.
"""
s06_context_compact.py - Compact (OpenAI format)

Three-layer compression pipeline so the agent can work for long sessions:
1) micro_compact: shrink old tool outputs in-place.
2) auto_compact: persist transcript + summarize when context is large.
3) compact tool: allow model-triggered immediate compaction.
"""

import json
import os
import time
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
TRANSCRIPT_DIR = WORKDIR / ".transcripts"

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.getenv("OPENAI_BASE_URL"),
)
MODEL = os.environ["MODEL_ID"]

SYSTEM = f"You are a coding agent at {WORKDIR}. Use tools to solve tasks."

THRESHOLD = 50000
KEEP_RECENT = 3

safe_path, run_bash, run_read, run_write, run_edit = make_file_tool_functions(WORKDIR)


def estimate_tokens(messages: list) -> int:
    """Rough token count: ~4 chars per token."""
    return len(str(messages)) // 4


def micro_compact(messages: list) -> list:
    """Layer 1: replace old long tool outputs with compact placeholders."""
    tool_messages = []
    for msg_idx, msg in enumerate(messages):
        if msg.get("role") == "tool":
            tool_messages.append((msg_idx, msg))

    if len(tool_messages) <= KEEP_RECENT:
        return messages

    tool_name_map = {}
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for tool_call in msg.get("tool_calls") or []:
            call_id = tool_call.get("id", "")
            call_name = tool_call.get("function", {}).get("name", "unknown")
            if call_id:
                tool_name_map[call_id] = call_name

    for _, tool_msg in tool_messages[:-KEEP_RECENT]:
        content = tool_msg.get("content", "")
        if isinstance(content, str) and len(content) > 100:
            tool_call_id = tool_msg.get("tool_call_id", "")
            tool_name = tool_name_map.get(tool_call_id, "unknown")
            tool_msg["content"] = f"[Previous: used {tool_name}]"

    return messages


def auto_compact(messages: list) -> list:
    """Layer 2: persist full transcript and replace history with summary."""
    TRANSCRIPT_DIR.mkdir(exist_ok=True)
    transcript_path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with open(transcript_path, "w") as f:
        for msg in messages:
            f.write(json.dumps(msg, default=str, ensure_ascii=False) + "\n")
    print(f"[transcript saved: {transcript_path}]")

    conversation_text = json.dumps(messages, default=str, ensure_ascii=False)[:80000]
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You summarize coding-agent conversations for continuity. "
                    "Preserve only critical actionable context."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Summarize this conversation for continuity. Include: "
                    "1) What was accomplished, 2) Current state, 3) Key decisions made. "
                    "Be concise but preserve critical details.\n\n"
                    + conversation_text
                ),
            },
        ],
    )
    summary = (response.choices[0].message.content or "(no summary)").strip()

    return [
        {
            "role": "user",
            "content": f"[Conversation compressed. Transcript: {transcript_path}]\n\n{summary}",
        },
        {
            "role": "assistant",
            "content": "Understood. I have the context from the summary. Continuing.",
        },
    ]


TOOL_HANDLERS = {
    "bash": lambda **kw: run_bash(kw["command"]),
    "read_file": lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file": lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "compact": lambda **kw: "Manual compression requested.",
}

TOOLS = file_tools() + [
    function_tool(
        "compact",
        "Trigger manual conversation compression.",
        {
            "focus": {
                "type": "string",
                "description": "What to preserve in the summary",
            }
        },
        [],
    )
]


def parse_tool_input(tool_call) -> dict:
    try:
        return json.loads(tool_call.function.arguments or "{}")
    except json.JSONDecodeError:
        return {}


def agent_loop(messages: list):
    while True:
        micro_compact(messages)

        if estimate_tokens(messages) > THRESHOLD:
            print("[auto_compact triggered]")
            messages[:] = auto_compact(messages)

        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM}, *messages],
            tools=TOOLS,
        )

        assistant = response.choices[0].message
        assistant_turn = {"role": "assistant", "content": assistant.content or ""}
        if assistant.tool_calls:
            assistant_turn["tool_calls"] = [
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
        messages.append(assistant_turn)

        if not assistant.tool_calls:
            return

        manual_compact = False
        for tool_call in assistant.tool_calls:
            tool_input = parse_tool_input(tool_call)

            if tool_call.function.name == "compact":
                manual_compact = True
                output = "Compressing..."
            else:
                handler = TOOL_HANDLERS.get(tool_call.function.name)
                try:
                    output = handler(**tool_input) if handler else f"Unknown tool: {tool_call.function.name}"
                except Exception as e:
                    output = f"Error: {e}"

            print(f"\033[33m> {tool_call.function.name}\033[0m")
            print(str(output)[:200])
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(output),
                }
            )

        if manual_compact:
            print("[manual compact]")
            messages[:] = auto_compact(messages)


if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms06 >> \033[0m")
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
