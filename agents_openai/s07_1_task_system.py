#!/usr/bin/env python3
# Harness: persistent tasks -- goals that outlive any single conversation.
"""
s07_1_task_system.py - Tasks + Delete (OpenAI format)

Tasks persist as JSON files in .tasks/ so they survive context compression.
Each task has a dependency graph (blockedBy/blocks).
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

load_dotenv(override=True)

WORKDIR = Path.cwd()
TASKS_DIR = WORKDIR / ".tasks"

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.getenv("OPENAI_BASE_URL"),
)
MODEL = os.environ["MODEL_ID"]
OBSERVER = create_observer(agent_name="s07_1_task_system", model=MODEL)

SYSTEM = f"You are a coding agent at {WORKDIR}. Use task tools to plan and track work."

safe_path, run_bash, run_read, run_write, run_edit = make_file_tool_functions(WORKDIR)


class TaskManager:
    """CRUD with dependency graph, persisted as JSON files."""

    def __init__(self, tasks_dir: Path):
        self.dir = tasks_dir
        self.dir.mkdir(exist_ok=True)
        self._next_id = self._max_id() + 1

    def _max_id(self) -> int:
        ids = [int(f.stem.split("_")[1]) for f in self.dir.glob("task_*.json")]
        return max(ids) if ids else 0

    def _load(self, task_id: int) -> dict:
        path = self.dir / f"task_{task_id}.json"
        if not path.exists():
            raise ValueError(f"Task {task_id} not found")
        return json.loads(path.read_text())

    def _save(self, task: dict):
        path = self.dir / f"task_{task['id']}.json"
        path.write_text(json.dumps(task, indent=2))

    def create(self, subject: str, description: str = "") -> str:
        task = {
            "id": self._next_id,
            "subject": subject,
            "description": description,
            "status": "pending",
            "blockedBy": [],
            "blocks": [],
            "owner": "",
        }
        self._save(task)
        self._next_id += 1
        return json.dumps(task, indent=2)

    def get(self, task_id: int) -> str:
        return json.dumps(self._load(task_id), indent=2)

    def update(
        self,
        task_id: int,
        status: str | None = None,
        add_blocked_by: list | None = None,
        add_blocks: list | None = None,
    ) -> str:
        task = self._load(task_id)

        if status:
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Invalid status: {status}")
            task["status"] = status
            if status == "completed":
                self._clear_dependency(task_id)

        if add_blocked_by:
            task["blockedBy"] = list(set(task["blockedBy"] + add_blocked_by))

        if add_blocks:
            task["blocks"] = list(set(task["blocks"] + add_blocks))
            for blocked_id in add_blocks:
                try:
                    blocked = self._load(blocked_id)
                    if task_id not in blocked["blockedBy"]:
                        blocked["blockedBy"].append(task_id)
                        self._save(blocked)
                except ValueError:
                    pass

        self._save(task)
        return json.dumps(task, indent=2)

    def _clear_dependency(self, completed_id: int):
        for f in self.dir.glob("task_*.json"):
            task = json.loads(f.read_text())
            if completed_id in task.get("blockedBy", []):
                task["blockedBy"].remove(completed_id)
                self._save(task)

    def list_all(self) -> str:
        tasks = [json.loads(f.read_text()) for f in sorted(self.dir.glob("task_*.json"))]
        if not tasks:
            return "No tasks."

        lines = []
        for task in tasks:
            marker = {
                "pending": "[ ]",
                "in_progress": "[>]",
                "completed": "[x]",
            }.get(task["status"], "[?]")
            blocked = f" (blocked by: {task['blockedBy']})" if task.get("blockedBy") else ""
            lines.append(f"{marker} #{task['id']}: {task['subject']}{blocked}")
        return "\n".join(lines)

    def delete(self, task_id: int, force: bool = False) -> str:
        task = self._load(task_id)
        if task["status"] != "completed" and not force:
            raise ValueError(f"Task {task_id} is {task['status']}; set force=true to delete unfinished tasks")

        removed_from_blocked_by = []
        removed_from_blocks = []
        for f in self.dir.glob("task_*.json"):
            other = json.loads(f.read_text())
            if other["id"] == task_id:
                continue

            changed = False
            if task_id in other.get("blockedBy", []):
                other["blockedBy"] = [i for i in other.get("blockedBy", []) if i != task_id]
                removed_from_blocked_by.append(other["id"])
                changed = True

            if task_id in other.get("blocks", []):
                other["blocks"] = [i for i in other.get("blocks", []) if i != task_id]
                removed_from_blocks.append(other["id"])
                changed = True

            if changed:
                self._save(other)

        (self.dir / f"task_{task_id}.json").unlink()
        return json.dumps(
            {
                "deleted": task,
                "removedFromBlockedBy": sorted(set(removed_from_blocked_by)),
                "removedFromBlocks": sorted(set(removed_from_blocks)),
            },
            indent=2,
        )


TASKS = TaskManager(TASKS_DIR)

TOOL_HANDLERS = {
    "bash": lambda **kw: run_bash(kw["command"]),
    "read_file": lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file": lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "task_create": lambda **kw: TASKS.create(kw["subject"], kw.get("description", "")),
    "task_update": lambda **kw: TASKS.update(
        kw["task_id"],
        kw.get("status"),
        kw.get("addBlockedBy"),
        kw.get("addBlocks"),
    ),
    "task_list": lambda **kw: TASKS.list_all(),
    "task_get": lambda **kw: TASKS.get(kw["task_id"]),
    "task_delete": lambda **kw: TASKS.delete(kw["task_id"], kw.get("force", False)),
}

TOOLS = file_tools() + [
    function_tool(
        "task_create",
        "Create a new task.",
        {
            "subject": {"type": "string"},
            "description": {"type": "string"},
        },
        ["subject"],
    ),
    function_tool(
        "task_update",
        "Update a task's status or dependencies.",
        {
            "task_id": {"type": "integer"},
            "status": {
                "type": "string",
                "enum": ["pending", "in_progress", "completed"],
            },
            "addBlockedBy": {"type": "array", "items": {"type": "integer"}},
            "addBlocks": {"type": "array", "items": {"type": "integer"}},
        },
        ["task_id"],
    ),
    function_tool(
        "task_list",
        "List all tasks with status summary.",
        {},
        [],
    ),
    function_tool(
        "task_get",
        "Get full details of a task by ID.",
        {"task_id": {"type": "integer"}},
        ["task_id"],
    ),
    function_tool(
        "task_delete",
        "Delete a task and clean dependency references. By default only completed tasks can be deleted.",
        {
            "task_id": {"type": "integer"},
            "force": {"type": "boolean"},
        },
        ["task_id"],
    ),
]


def parse_tool_input(tool_call) -> dict:
    try:
        return json.loads(tool_call.function.arguments or "{}")
    except json.JSONDecodeError:
        return {}


def agent_loop(messages: list):
    user_input = ""
    if messages and messages[-1].get("role") == "user":
        user_input = str(messages[-1].get("content", ""))
    trace_ctx = OBSERVER.start_trace(user_input=user_input, history_len=len(messages))
    final_output = ""

    while True:
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
        OBSERVER.on_model_response(
            trace_ctx,
            assistant_text=assistant.content or "",
            tool_calls=assistant_turn.get("tool_calls", []),
        )

        if not assistant.tool_calls:
            final_output = assistant.content or ""
            OBSERVER.finish_trace(trace_ctx, final_output=final_output)
            OBSERVER.flush()
            return

        for tool_call in assistant.tool_calls:
            tool_input = parse_tool_input(tool_call)
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


if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms07_1 >> \033[0m")
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
