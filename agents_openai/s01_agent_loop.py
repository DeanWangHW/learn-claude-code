#!/usr/bin/env python3
# Harness: the loop -- the model's first connection to the real world.
"""
s01_agent_loop.py - The Agent Loop

The entire secret of an AI coding agent in one pattern:

    while stop_reason == "tool_use":
        response = LLM(messages, tools)
        execute tools
        append results

    +----------+      +-------+      +---------+
    |   User   | ---> |  LLM  | ---> |  Tool   |
    |  prompt  |      |       |      | execute |
    +----------+      +---+---+      +----+----+
                          ^               |
                          |   tool_result |
                          +---------------+
                          (loop continues)

This is the core loop: feed tool results back to the model
until the model decides to stop. Production agents layer
policy, hooks, and lifecycle controls on top.
"""

import os
import json
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv
try:
    from agents_openai.observability import create_observer
    from agents_openai.tools import bash_tool, run_bash as shared_run_bash
except ImportError:
    from observability import create_observer
    from tools import bash_tool, run_bash as shared_run_bash

# 加载 .env，便于本地通过环境变量注入模型与密钥配置。
load_dotenv(override=True)

# 初始化 OpenAI 客户端；支持通过 OPENAI_BASE_URL 接入代理或兼容网关。
client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.getenv("OPENAI_BASE_URL"),
)
MODEL = os.environ["MODEL_ID"]
OBSERVER = create_observer(agent_name="s01_agent_loop", model=MODEL)

SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."

# 声明给模型可用的工具，这里只暴露一个 bash 执行器。
TOOLS = [bash_tool()]


def run_bash(command: str) -> str:
    return shared_run_bash(command, Path.cwd())


# -- The core pattern: a while loop that calls tools until the model stops --
def agent_loop(messages: list):
    user_input = ""
    if messages and messages[-1].get("role") == "user":
        user_input = str(messages[-1].get("content", ""))
    trace_ctx = OBSERVER.start_trace(user_input=user_input, history_len=len(messages))
    final_output = ""

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
            assistant_turn["tool_calls"] = [{
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            } for tc in assistant.tool_calls]
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
        for tool_call in assistant.tool_calls:
            if tool_call.function.name == "bash":
                try:
                    tool_input = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError:
                    tool_input = {}
                command = tool_input.get("command", "")
                print(f"\033[33m$ {command}\033[0m")
                output = run_bash(command)
                print(output[:200])
                OBSERVER.on_tool_result(
                    trace_ctx,
                    tool_name="bash",
                    tool_input={"command": command},
                    output=output,
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": output,
                })


if __name__ == "__main__":
    # REPL 入口：持续读取用户输入，交给 agent_loop 处理。
    history = []
    while True:
        try:
            query = input("\033[36ms01 >> \033[0m")
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
