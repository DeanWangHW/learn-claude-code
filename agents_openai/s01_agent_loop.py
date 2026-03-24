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
import subprocess
import json

from openai import OpenAI
from dotenv import load_dotenv

# 加载 .env，便于本地通过环境变量注入模型与密钥配置。
load_dotenv(override=True)

# 初始化 OpenAI 客户端；支持通过 OPENAI_BASE_URL 接入代理或兼容网关。
client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.getenv("OPENAI_BASE_URL"),
)
MODEL = os.environ["MODEL_ID"]

SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."

# 声明给模型可用的工具，这里只暴露一个 bash 执行器。
TOOLS = [{
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a shell command.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
            "additionalProperties": False,
        },
    },
}]


def run_bash(command: str) -> str:
    # 最小安全防护：拦截高危命令，避免误删系统或关机等操作。
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        # 在当前工作目录执行命令，并限制超时，防止任务无限挂起。
        r = subprocess.run(command, shell=True, cwd=os.getcwd(),
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        # 控制返回长度，避免把超长日志直接塞回上下文。
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


# -- The core pattern: a while loop that calls tools until the model stops --
def agent_loop(messages: list):
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

        # 3) 如果这轮没有工具调用，说明模型决定直接收敛，循环结束。
        if not assistant.tool_calls:
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
