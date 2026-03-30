#!/usr/bin/env python3
# Harness: directory isolation -- parallel execution lanes that never collide.
"""
s12_worktree_task_isolation.py - Worktree + Task Isolation (OpenAI format)

Directory-level isolation for parallel task execution.
Tasks are the control plane and worktrees are the execution plane.
"""

import json
import os
import re
import subprocess
import time
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
client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.getenv("OPENAI_BASE_URL"),
)
MODEL = os.environ["MODEL_ID"]
OBSERVER = create_observer(agent_name="s12_worktree_task_isolation", model=MODEL)


def detect_repo_root(cwd: Path) -> Path | None:
    """Return git repo root if cwd is inside a repo, else None."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None
        root = Path(result.stdout.strip())
        return root if root.exists() else None
    except Exception:
        return None


REPO_ROOT = detect_repo_root(WORKDIR) or WORKDIR

SYSTEM = (
    f"You are a coding agent at {WORKDIR}. "
    "Use task + worktree tools for multi-task work. "
    "For parallel or risky changes: create tasks, allocate worktree lanes, "
    "run commands in those lanes, then choose keep/remove for closeout. "
    "Use worktree_events when you need lifecycle visibility."
)

safe_path, run_bash, run_read, run_write, run_edit = make_file_tool_functions(WORKDIR)


class EventBus:
    """Append-only lifecycle events for observability."""

    def __init__(self, event_log_path: Path):
        self.path = event_log_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("")

    def emit(
        self,
        event: str,
        task: dict | None = None,
        worktree: dict | None = None,
        error: str | None = None,
    ):
        payload = {
            "event": event,
            "ts": time.time(),
            "task": task or {},
            "worktree": worktree or {},
        }
        if error:
            payload["error"] = error
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

    def list_recent(self, limit: int = 20) -> str:
        n = max(1, min(int(limit or 20), 200))
        lines = self.path.read_text(encoding="utf-8").splitlines()
        recent = lines[-n:]
        items = []
        for line in recent:
            try:
                items.append(json.loads(line))
            except Exception:
                items.append({"event": "parse_error", "raw": line})
        return json.dumps(items, indent=2)


class TaskManager:
    """Persistent task board with optional worktree binding."""

    def __init__(self, tasks_dir: Path):
        self.dir = tasks_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self._next_id = self._max_id() + 1

    def _max_id(self) -> int:
        ids = []
        for f in self.dir.glob("task_*.json"):
            try:
                ids.append(int(f.stem.split("_")[1]))
            except Exception:
                pass
        return max(ids) if ids else 0

    def _path(self, task_id: int) -> Path:
        return self.dir / f"task_{task_id}.json"

    def _load(self, task_id: int) -> dict:
        path = self._path(task_id)
        if not path.exists():
            raise ValueError(f"Task {task_id} not found")
        return json.loads(path.read_text())

    def _save(self, task: dict):
        self._path(task["id"]).write_text(json.dumps(task, indent=2))

    def create(self, subject: str, description: str = "") -> str:
        task = {
            "id": self._next_id,
            "subject": subject,
            "description": description,
            "status": "pending",
            "owner": "",
            "worktree": "",
            "blockedBy": [],
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        self._save(task)
        self._next_id += 1
        return json.dumps(task, indent=2)

    def get(self, task_id: int) -> str:
        return json.dumps(self._load(task_id), indent=2)

    def exists(self, task_id: int) -> bool:
        return self._path(task_id).exists()

    def update(self, task_id: int, status: str | None = None, owner: str | None = None) -> str:
        task = self._load(task_id)
        if status:
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Invalid status: {status}")
            task["status"] = status
        if owner is not None:
            task["owner"] = owner
        task["updated_at"] = time.time()
        self._save(task)
        return json.dumps(task, indent=2)

    def bind_worktree(self, task_id: int, worktree: str, owner: str = "") -> str:
        task = self._load(task_id)
        task["worktree"] = worktree
        if owner:
            task["owner"] = owner
        if task["status"] == "pending":
            task["status"] = "in_progress"
        task["updated_at"] = time.time()
        self._save(task)
        return json.dumps(task, indent=2)

    def unbind_worktree(self, task_id: int) -> str:
        task = self._load(task_id)
        task["worktree"] = ""
        task["updated_at"] = time.time()
        self._save(task)
        return json.dumps(task, indent=2)

    def list_all(self) -> str:
        tasks = []
        for f in sorted(self.dir.glob("task_*.json")):
            tasks.append(json.loads(f.read_text()))
        if not tasks:
            return "No tasks."
        lines = []
        for task in tasks:
            marker = {
                "pending": "[ ]",
                "in_progress": "[>]",
                "completed": "[x]",
            }.get(task["status"], "[?]")
            owner = f" owner={task['owner']}" if task.get("owner") else ""
            wt = f" wt={task['worktree']}" if task.get("worktree") else ""
            lines.append(f"{marker} #{task['id']}: {task['subject']}{owner}{wt}")
        return "\n".join(lines)


TASKS = TaskManager(REPO_ROOT / ".tasks")
EVENTS = EventBus(REPO_ROOT / ".worktrees" / "events.jsonl")


class WorktreeManager:
    """Create/list/run/remove git worktrees + lifecycle index."""

    def __init__(self, repo_root: Path, tasks: TaskManager, events: EventBus):
        self.repo_root = repo_root
        self.tasks = tasks
        self.events = events
        self.dir = repo_root / ".worktrees"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.dir / "index.json"
        if not self.index_path.exists():
            self.index_path.write_text(json.dumps({"worktrees": []}, indent=2))
        self.git_available = self._is_git_repo()

    def _is_git_repo(self) -> bool:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except Exception:
            return False

    def _run_git(self, args: list[str]) -> str:
        if not self.git_available:
            raise RuntimeError("Not in a git repository. worktree tools require git.")
        result = subprocess.run(
            ["git", *args],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            msg = (result.stdout + result.stderr).strip()
            raise RuntimeError(msg or f"git {' '.join(args)} failed")
        return (result.stdout + result.stderr).strip() or "(no output)"

    def _load_index(self) -> dict:
        return json.loads(self.index_path.read_text())

    def _save_index(self, data: dict):
        self.index_path.write_text(json.dumps(data, indent=2))

    def _find(self, name: str) -> dict | None:
        idx = self._load_index()
        for wt in idx.get("worktrees", []):
            if wt.get("name") == name:
                return wt
        return None

    def _validate_name(self, name: str):
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,40}", name or ""):
            raise ValueError(
                "Invalid worktree name. Use 1-40 chars: letters, numbers, ., _, -"
            )

    def create(self, name: str, task_id: int | None = None, base_ref: str = "HEAD") -> str:
        self._validate_name(name)
        if self._find(name):
            raise ValueError(f"Worktree '{name}' already exists in index")
        if task_id is not None and not self.tasks.exists(task_id):
            raise ValueError(f"Task {task_id} not found")

        path = self.dir / name
        branch = f"wt/{name}"
        self.events.emit(
            "worktree.create.before",
            task={"id": task_id} if task_id is not None else {},
            worktree={"name": name, "base_ref": base_ref},
        )
        try:
            self._run_git(["worktree", "add", "-b", branch, str(path), base_ref])

            entry = {
                "name": name,
                "path": str(path),
                "branch": branch,
                "task_id": task_id,
                "status": "active",
                "created_at": time.time(),
            }

            idx = self._load_index()
            idx["worktrees"].append(entry)
            self._save_index(idx)

            if task_id is not None:
                self.tasks.bind_worktree(task_id, name)

            self.events.emit(
                "worktree.create.after",
                task={"id": task_id} if task_id is not None else {},
                worktree={
                    "name": name,
                    "path": str(path),
                    "branch": branch,
                    "status": "active",
                },
            )
            return json.dumps(entry, indent=2)
        except Exception as e:
            self.events.emit(
                "worktree.create.failed",
                task={"id": task_id} if task_id is not None else {},
                worktree={"name": name, "base_ref": base_ref},
                error=str(e),
            )
            raise

    def list_all(self) -> str:
        idx = self._load_index()
        worktrees = idx.get("worktrees", [])
        if not worktrees:
            return "No worktrees in index."
        lines = []
        for wt in worktrees:
            suffix = f" task={wt['task_id']}" if wt.get("task_id") else ""
            lines.append(
                f"[{wt.get('status', 'unknown')}] {wt['name']} -> "
                f"{wt['path']} ({wt.get('branch', '-')}){suffix}"
            )
        return "\n".join(lines)

    def status(self, name: str) -> str:
        wt = self._find(name)
        if not wt:
            return f"Error: Unknown worktree '{name}'"
        path = Path(wt["path"])
        if not path.exists():
            return f"Error: Worktree path missing: {path}"
        result = subprocess.run(
            ["git", "status", "--short", "--branch"],
            cwd=path,
            capture_output=True,
            text=True,
            timeout=60,
        )
        text = (result.stdout + result.stderr).strip()
        return text or "Clean worktree"

    def run(self, name: str, command: str) -> str:
        dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
        if any(d in command for d in dangerous):
            return "Error: Dangerous command blocked"

        wt = self._find(name)
        if not wt:
            return f"Error: Unknown worktree '{name}'"
        path = Path(wt["path"])
        if not path.exists():
            return f"Error: Worktree path missing: {path}"

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=path,
                capture_output=True,
                text=True,
                timeout=300,
            )
            out = (result.stdout + result.stderr).strip()
            return out[:50000] if out else "(no output)"
        except subprocess.TimeoutExpired:
            return "Error: Timeout (300s)"

    def remove(self, name: str, force: bool = False, complete_task: bool = False) -> str:
        wt = self._find(name)
        if not wt:
            return f"Error: Unknown worktree '{name}'"

        self.events.emit(
            "worktree.remove.before",
            task={"id": wt.get("task_id")} if wt.get("task_id") is not None else {},
            worktree={"name": name, "path": wt.get("path")},
        )
        try:
            args = ["worktree", "remove"]
            if force:
                args.append("--force")
            args.append(wt["path"])
            self._run_git(args)

            if complete_task and wt.get("task_id") is not None:
                task_id = wt["task_id"]
                before = json.loads(self.tasks.get(task_id))
                self.tasks.update(task_id, status="completed")
                self.tasks.unbind_worktree(task_id)
                self.events.emit(
                    "task.completed",
                    task={
                        "id": task_id,
                        "subject": before.get("subject", ""),
                        "status": "completed",
                    },
                    worktree={"name": name},
                )

            idx = self._load_index()
            for item in idx.get("worktrees", []):
                if item.get("name") == name:
                    item["status"] = "removed"
                    item["removed_at"] = time.time()
            self._save_index(idx)

            self.events.emit(
                "worktree.remove.after",
                task={"id": wt.get("task_id")} if wt.get("task_id") is not None else {},
                worktree={"name": name, "path": wt.get("path"), "status": "removed"},
            )
            return f"Removed worktree '{name}'"
        except Exception as e:
            self.events.emit(
                "worktree.remove.failed",
                task={"id": wt.get("task_id")} if wt.get("task_id") is not None else {},
                worktree={"name": name, "path": wt.get("path")},
                error=str(e),
            )
            raise

    def keep(self, name: str) -> str:
        wt = self._find(name)
        if not wt:
            return f"Error: Unknown worktree '{name}'"

        idx = self._load_index()
        kept = None
        for item in idx.get("worktrees", []):
            if item.get("name") == name:
                item["status"] = "kept"
                item["kept_at"] = time.time()
                kept = item
        self._save_index(idx)

        self.events.emit(
            "worktree.keep",
            task={"id": wt.get("task_id")} if wt.get("task_id") is not None else {},
            worktree={"name": name, "path": wt.get("path"), "status": "kept"},
        )
        return json.dumps(kept, indent=2) if kept else f"Error: Unknown worktree '{name}'"


WORKTREES = WorktreeManager(REPO_ROOT, TASKS, EVENTS)

TOOL_HANDLERS = {
    "bash": lambda **kw: run_bash(kw["command"]),
    "read_file": lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file": lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "task_create": lambda **kw: TASKS.create(kw["subject"], kw.get("description", "")),
    "task_list": lambda **kw: TASKS.list_all(),
    "task_get": lambda **kw: TASKS.get(kw["task_id"]),
    "task_update": lambda **kw: TASKS.update(kw["task_id"], kw.get("status"), kw.get("owner")),
    "task_bind_worktree": lambda **kw: TASKS.bind_worktree(
        kw["task_id"], kw["worktree"], kw.get("owner", "")
    ),
    "worktree_create": lambda **kw: WORKTREES.create(
        kw["name"], kw.get("task_id"), kw.get("base_ref", "HEAD")
    ),
    "worktree_list": lambda **kw: WORKTREES.list_all(),
    "worktree_status": lambda **kw: WORKTREES.status(kw["name"]),
    "worktree_run": lambda **kw: WORKTREES.run(kw["name"], kw["command"]),
    "worktree_keep": lambda **kw: WORKTREES.keep(kw["name"]),
    "worktree_remove": lambda **kw: WORKTREES.remove(
        kw["name"], kw.get("force", False), kw.get("complete_task", False)
    ),
    "worktree_events": lambda **kw: EVENTS.list_recent(kw.get("limit", 20)),
}

TOOLS = file_tools() + [
    function_tool(
        "task_create",
        "Create a new task on the shared task board.",
        {"subject": {"type": "string"}, "description": {"type": "string"}},
        ["subject"],
    ),
    function_tool(
        "task_list",
        "List all tasks with status, owner, and worktree binding.",
        {},
        [],
    ),
    function_tool(
        "task_get",
        "Get task details by ID.",
        {"task_id": {"type": "integer"}},
        ["task_id"],
    ),
    function_tool(
        "task_update",
        "Update task status or owner.",
        {
            "task_id": {"type": "integer"},
            "status": {
                "type": "string",
                "enum": ["pending", "in_progress", "completed"],
            },
            "owner": {"type": "string"},
        },
        ["task_id"],
    ),
    function_tool(
        "task_bind_worktree",
        "Bind a task to a worktree name.",
        {
            "task_id": {"type": "integer"},
            "worktree": {"type": "string"},
            "owner": {"type": "string"},
        },
        ["task_id", "worktree"],
    ),
    function_tool(
        "worktree_create",
        "Create a git worktree and optionally bind it to a task.",
        {
            "name": {"type": "string"},
            "task_id": {"type": "integer"},
            "base_ref": {"type": "string"},
        },
        ["name"],
    ),
    function_tool(
        "worktree_list",
        "List worktrees tracked in .worktrees/index.json.",
        {},
        [],
    ),
    function_tool(
        "worktree_status",
        "Show git status for one worktree.",
        {"name": {"type": "string"}},
        ["name"],
    ),
    function_tool(
        "worktree_run",
        "Run a shell command in a named worktree directory.",
        {"name": {"type": "string"}, "command": {"type": "string"}},
        ["name", "command"],
    ),
    function_tool(
        "worktree_remove",
        "Remove a worktree and optionally mark its bound task completed.",
        {
            "name": {"type": "string"},
            "force": {"type": "boolean"},
            "complete_task": {"type": "boolean"},
        },
        ["name"],
    ),
    function_tool(
        "worktree_keep",
        "Mark a worktree as kept in lifecycle state without removing it.",
        {"name": {"type": "string"}},
        ["name"],
    ),
    function_tool(
        "worktree_events",
        "List recent worktree/task lifecycle events from .worktrees/events.jsonl.",
        {"limit": {"type": "integer"}},
        [],
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
    print(f"Repo root for s12: {REPO_ROOT}")
    if not WORKTREES.git_available:
        print("Note: Not in a git repo. worktree_* tools will return errors.")

    history = []
    while True:
        try:
            query = input("\033[36ms12 >> \033[0m")
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
