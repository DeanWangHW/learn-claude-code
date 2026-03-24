#!/usr/bin/env python3
"""
s03_todo_write.py - TodoWrite

The loop from s02 stays the same. We add:
1) a TodoManager state object
2) a todo tool
3) a reminder after several rounds without todo updates
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
try:
    from agents_openai.observability import create_observer
    from agents_openai.tools import (
        file_tools,
        function_tool,
        make_file_tool_functions,
    )
except ImportError:
    from observability import create_observer
    from tools import (
        file_tools,
        function_tool,
        make_file_tool_functions,
    )

# 加载 .env，便于本地通过环境变量注入模型与密钥配置。
load_dotenv(override=True)

# 初始化 OpenAI 客户端；支持通过 OPENAI_BASE_URL 接入代理或兼容网关。
client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.getenv("OPENAI_BASE_URL"),
)
MODEL = os.environ["MODEL_ID"]
OBSERVER = create_observer(agent_name="s03_todo_write", model=MODEL)

WORKDIR = Path.cwd()
SYSTEM = f"""You are a coding agent at {WORKDIR}.
Use the todo tool to plan multi-step tasks. Mark in_progress before starting, completed when done.
Prefer tools over prose."""


class TodoManager:
    """Structured todo state that the model updates via the todo tool."""

    def __init__(self):
        self.items = []

    def update(self, items: list) -> str:
        if len(items) > 20:
            raise ValueError("Max 20 todos allowed")

        validated = []
        in_progress_count = 0
        for i, item in enumerate(items):
            text = str(item.get("text", "")).strip()
            status = str(item.get("status", "pending")).lower()
            item_id = str(item.get("id", str(i + 1)))

            if not text:
                raise ValueError(f"Item {item_id}: text required")
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Item {item_id}: invalid status '{status}'")
            if status == "in_progress":
                in_progress_count += 1

            validated.append({"id": item_id, "text": text, "status": status})

        if in_progress_count > 1:
            raise ValueError("Only one task can be in_progress at a time")

        self.items = validated
        return self.render()

    def render(self) -> str:
        if not self.items:
            return "No todos."

        lines = []
        for item in self.items:
            marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}[item["status"]]
            lines.append(f"{marker} #{item['id']}: {item['text']}")

        done = sum(1 for t in self.items if t["status"] == "completed")
        lines.append(f"\n({done}/{len(self.items)} completed)")
        return "\n".join(lines)


TODO = TodoManager()

safe_path, run_bash, run_read, run_write, run_edit = make_file_tool_functions(WORKDIR)


# 工具分发表：将模型请求的工具名映射到本地处理函数。
TOOL_HANDLERS = {
    "bash": lambda **kw: run_bash(kw["command"]),
    "read_file": lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file": lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "todo": lambda **kw: TODO.update(kw["items"]),
}

TOOLS = file_tools() + [
    function_tool(
        "todo",
        "Update task list. Track progress on multi-step tasks.",
        {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "text": {"type": "string"},
                        "status": {
                            "type": "string",
                            "enum": ["pending", "in_progress", "completed"],
                        },
                    },
                    "required": ["id", "text", "status"],
                    "additionalProperties": False,
                },
            },
        },
        ["items"],
    )
]


def agent_loop(messages: list):
    # -- The core pattern: a while loop that calls tools until the model stops --
    user_input = ""
    if messages and messages[-1].get("role") == "user":
        user_input = str(messages[-1].get("content", ""))
    trace_ctx = OBSERVER.start_trace(user_input=user_input, history_len=len(messages))
    final_output = ""

    rounds_since_todo = 0
    while True:
        # 1) 让模型在当前消息上下文下做一次决策（回答或发起工具调用）。
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM}, *messages],
            tools=TOOLS,
        )

        assistant = response.choices[0].message
        # 2) 先把 assistant 回合写入历史，保证后续上下文完整。
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
        tool_calls_payload = assistant_turn.get("tool_calls", [])
        OBSERVER.on_model_response(
            trace_ctx,
            assistant_text=assistant.content or "",
            tool_calls=tool_calls_payload,
        )

        # 3) 如果这轮没有工具调用，说明模型决定直接收敛，循环结束。
        if not assistant.tool_calls:
            final_output = assistant.content or ""
            OBSERVER.finish_trace(trace_ctx, final_output=final_output)
            OBSERVER.flush()
            return

        # 4) 执行模型请求的工具，并把 tool 结果回填到消息历史中。
        used_todo = False
        for tool_call in assistant.tool_calls:
            try:
                tool_input = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                tool_input = {}

            handler = TOOL_HANDLERS.get(tool_call.function.name)
            try:
                output = handler(**tool_input) if handler else f"Unknown tool: {tool_call.function.name}"
            except Exception as e:
                output = f"Error: {e}"

            print(f"\033[33m> {tool_call.function.name}\033[0m")
            print(str(output)[:200])
            OBSERVER.on_tool_result(
                trace_ctx,
                tool_name=tool_call.function.name,
                tool_input=tool_input,
                output=str(output),
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(output),
                }
            )

            if tool_call.function.name == "todo":
                used_todo = True

        # 5) 多轮未更新 todo 时，注入提醒，促使模型同步计划进度。
        rounds_since_todo = 0 if used_todo else rounds_since_todo + 1
        if rounds_since_todo >= 3:
            messages.append({"role": "user", "content": "<reminder>Update your todos.</reminder>"})


if __name__ == "__main__":
    # REPL 入口：持续读取用户输入，交给 agent_loop 处理。
    history = []
    while True:
        try:
            query = input("\033[36ms03 >> \033[0m")
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
