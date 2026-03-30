#!/usr/bin/env python3
# Harness: autonomy -- models that find work without being told.
"""
s11_autonomous_agents.py - Autonomous Agents (OpenAI format)

Idle cycle with task board polling, auto-claiming unclaimed tasks, and
identity re-injection after context compression. Builds on s10's protocols.
"""

import json
import os
import threading
import time
import uuid
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
TEAM_DIR = WORKDIR / ".team"
INBOX_DIR = TEAM_DIR / "inbox"
TASKS_DIR = WORKDIR / ".tasks"

POLL_INTERVAL = 5
IDLE_TIMEOUT = 60

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.getenv("OPENAI_BASE_URL"),
)
MODEL = os.environ["MODEL_ID"]
OBSERVER = create_observer(agent_name="s11_autonomous_agents", model=MODEL)

SYSTEM = f"You are a team lead at {WORKDIR}. Teammates are autonomous -- they find work themselves."

VALID_MSG_TYPES = {
    "message",
    "broadcast",
    "shutdown_request",
    "shutdown_response",
    "plan_approval_response",
}

# Request trackers.
shutdown_requests: dict[str, dict] = {}
plan_requests: dict[str, dict] = {}
_tracker_lock = threading.Lock()
_claim_lock = threading.Lock()

safe_path, run_bash, run_read, run_write, run_edit = make_file_tool_functions(WORKDIR)


class MessageBus:
    """Simple JSONL mailbox bus. Each teammate has one inbox file."""

    def __init__(self, inbox_dir: Path):
        self.dir = inbox_dir
        self.dir.mkdir(parents=True, exist_ok=True)

    def send(
        self,
        sender: str,
        to: str,
        content: str,
        msg_type: str = "message",
        extra: dict | None = None,
    ) -> str:
        if msg_type not in VALID_MSG_TYPES:
            return f"Error: Invalid type '{msg_type}'. Valid: {VALID_MSG_TYPES}"

        message = {
            "type": msg_type,
            "from": sender,
            "content": content,
            "timestamp": time.time(),
        }
        if extra:
            message.update(extra)

        inbox_path = self.dir / f"{to}.jsonl"
        with open(inbox_path, "a") as f:
            f.write(json.dumps(message) + "\n")
        return f"Sent {msg_type} to {to}"

    def read_inbox(self, name: str) -> list[dict]:
        inbox_path = self.dir / f"{name}.jsonl"
        if not inbox_path.exists():
            return []

        messages = []
        for line in inbox_path.read_text().strip().splitlines():
            if line:
                messages.append(json.loads(line))
        inbox_path.write_text("")
        return messages

    def broadcast(self, sender: str, content: str, teammates: list[str]) -> str:
        count = 0
        for name in teammates:
            if name != sender:
                self.send(sender, name, content, "broadcast")
                count += 1
        return f"Broadcast to {count} teammates"


BUS = MessageBus(INBOX_DIR)


def scan_unclaimed_tasks() -> list[dict]:
    """Return pending tasks that have no owner and are not blocked."""
    TASKS_DIR.mkdir(exist_ok=True)
    unclaimed = []
    for task_file in sorted(TASKS_DIR.glob("task_*.json")):
        task = json.loads(task_file.read_text())
        if (
            task.get("status") == "pending"
            and not task.get("owner")
            and not task.get("blockedBy")
        ):
            unclaimed.append(task)
    return unclaimed


def claim_task(task_id: int, owner: str) -> str:
    with _claim_lock:
        path = TASKS_DIR / f"task_{task_id}.json"
        if not path.exists():
            return f"Error: Task {task_id} not found"
        task = json.loads(path.read_text())
        task["owner"] = owner
        task["status"] = "in_progress"
        path.write_text(json.dumps(task, indent=2))
    return f"Claimed task #{task_id} for {owner}"


def make_identity_block(name: str, role: str, team_name: str) -> dict:
    return {
        "role": "user",
        "content": (
            f"<identity>You are '{name}', role: {role}, team: {team_name}. "
            "Continue your work.</identity>"
        ),
    }


class TeammateManager:
    """Persistent teammate metadata and autonomous worker loops."""

    def __init__(self, team_dir: Path):
        self.dir = team_dir
        self.dir.mkdir(exist_ok=True)
        self.config_path = self.dir / "config.json"
        self.config = self._load_config()
        self.threads: dict[str, threading.Thread] = {}

    def _load_config(self) -> dict:
        if self.config_path.exists():
            return json.loads(self.config_path.read_text())
        return {"team_name": "default", "members": []}

    def _save_config(self):
        self.config_path.write_text(json.dumps(self.config, indent=2))

    def _find_member(self, name: str) -> dict | None:
        for member in self.config["members"]:
            if member["name"] == name:
                return member
        return None

    def _set_status(self, name: str, status: str):
        member = self._find_member(name)
        if member:
            member["status"] = status
            self._save_config()

    def spawn(self, name: str, role: str, prompt: str) -> str:
        member = self._find_member(name)
        if member:
            if member["status"] not in ("idle", "shutdown"):
                return f"Error: '{name}' is currently {member['status']}"
            member["status"] = "working"
            member["role"] = role
        else:
            member = {"name": name, "role": role, "status": "working"}
            self.config["members"].append(member)
        self._save_config()

        thread = threading.Thread(
            target=self._loop,
            args=(name, role, prompt),
            daemon=True,
        )
        self.threads[name] = thread
        thread.start()
        return f"Spawned '{name}' (role: {role})"

    def _loop(self, name: str, role: str, prompt: str):
        team_name = self.config["team_name"]
        system_prompt = (
            f"You are '{name}', role: {role}, team: {team_name}, at {WORKDIR}. "
            "Use idle tool when you have no more work. You will auto-claim new tasks."
        )
        messages = [{"role": "user", "content": prompt}]
        tools = self._teammate_tools()

        while True:
            # WORK PHASE.
            for _ in range(50):
                inbox = BUS.read_inbox(name)
                for msg in inbox:
                    if msg.get("type") == "shutdown_request":
                        self._set_status(name, "shutdown")
                        return
                    messages.append({"role": "user", "content": json.dumps(msg)})

                try:
                    response = client.chat.completions.create(
                        model=MODEL,
                        messages=[{"role": "system", "content": system_prompt}, *messages],
                        tools=tools,
                    )
                except Exception:
                    self._set_status(name, "idle")
                    return

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
                    break

                idle_requested = False
                for tool_call in assistant.tool_calls:
                    tool_input = parse_tool_input(tool_call)
                    if tool_call.function.name == "idle":
                        idle_requested = True
                        output = "Entering idle phase. Will poll for new tasks."
                    else:
                        output = self._exec(name, tool_call.function.name, tool_input)
                    print(f"  [{name}] {tool_call.function.name}: {str(output)[:120]}")
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": str(output),
                        }
                    )

                if idle_requested:
                    break

            # IDLE PHASE.
            self._set_status(name, "idle")
            resume = False
            polls = IDLE_TIMEOUT // max(POLL_INTERVAL, 1)
            for _ in range(polls):
                time.sleep(POLL_INTERVAL)
                inbox = BUS.read_inbox(name)
                if inbox:
                    for msg in inbox:
                        if msg.get("type") == "shutdown_request":
                            self._set_status(name, "shutdown")
                            return
                        messages.append({"role": "user", "content": json.dumps(msg)})
                    resume = True
                    break

                unclaimed = scan_unclaimed_tasks()
                if unclaimed:
                    task = unclaimed[0]
                    claim_task(task["id"], name)
                    task_prompt = (
                        f"<auto-claimed>Task #{task['id']}: {task['subject']}\n"
                        f"{task.get('description', '')}</auto-claimed>"
                    )
                    # Re-inject identity if context got compacted too aggressively.
                    if len(messages) <= 3:
                        messages.insert(0, make_identity_block(name, role, team_name))
                        messages.insert(1, {"role": "assistant", "content": f"I am {name}. Continuing."})
                    messages.append({"role": "user", "content": task_prompt})
                    messages.append(
                        {
                            "role": "assistant",
                            "content": f"Claimed task #{task['id']}. Working on it.",
                        }
                    )
                    resume = True
                    break

            if not resume:
                self._set_status(name, "shutdown")
                return

            self._set_status(name, "working")

    def _exec(self, sender: str, tool_name: str, args: dict) -> str:
        if tool_name == "bash":
            return run_bash(args["command"])
        if tool_name == "read_file":
            return run_read(args["path"], args.get("limit"))
        if tool_name == "write_file":
            return run_write(args["path"], args["content"])
        if tool_name == "edit_file":
            return run_edit(args["path"], args["old_text"], args["new_text"])
        if tool_name == "send_message":
            return BUS.send(sender, args["to"], args["content"], args.get("msg_type", "message"))
        if tool_name == "read_inbox":
            return json.dumps(BUS.read_inbox(sender), indent=2)
        if tool_name == "shutdown_response":
            request_id = args["request_id"]
            approve = args["approve"]
            with _tracker_lock:
                if request_id in shutdown_requests:
                    shutdown_requests[request_id]["status"] = "approved" if approve else "rejected"
            BUS.send(
                sender,
                "lead",
                args.get("reason", ""),
                "shutdown_response",
                {"request_id": request_id, "approve": approve},
            )
            return f"Shutdown {'approved' if approve else 'rejected'}"
        if tool_name == "plan_approval":
            plan_text = args.get("plan", "")
            request_id = str(uuid.uuid4())[:8]
            with _tracker_lock:
                plan_requests[request_id] = {"from": sender, "plan": plan_text, "status": "pending"}
            BUS.send(
                sender,
                "lead",
                plan_text,
                "plan_approval_response",
                {"request_id": request_id, "plan": plan_text},
            )
            return f"Plan submitted (request_id={request_id}). Waiting for approval."
        if tool_name == "claim_task":
            return claim_task(args["task_id"], sender)
        return f"Unknown tool: {tool_name}"

    def _teammate_tools(self) -> list[dict]:
        return file_tools() + [
            function_tool(
                "send_message",
                "Send message to a teammate.",
                {
                    "to": {"type": "string"},
                    "content": {"type": "string"},
                    "msg_type": {"type": "string", "enum": list(VALID_MSG_TYPES)},
                },
                ["to", "content"],
            ),
            function_tool(
                "read_inbox",
                "Read and drain your inbox.",
                {},
                [],
            ),
            function_tool(
                "shutdown_response",
                "Respond to a shutdown request.",
                {
                    "request_id": {"type": "string"},
                    "approve": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                ["request_id", "approve"],
            ),
            function_tool(
                "plan_approval",
                "Submit a plan for lead approval.",
                {"plan": {"type": "string"}},
                ["plan"],
            ),
            function_tool(
                "idle",
                "Signal that you have no more work. Enters idle polling phase.",
                {},
                [],
            ),
            function_tool(
                "claim_task",
                "Claim a task from the task board by ID.",
                {"task_id": {"type": "integer"}},
                ["task_id"],
            ),
        ]

    def list_all(self) -> str:
        if not self.config["members"]:
            return "No teammates."

        lines = [f"Team: {self.config['team_name']}"]
        for member in self.config["members"]:
            lines.append(f"  {member['name']} ({member['role']}): {member['status']}")
        return "\n".join(lines)

    def member_names(self) -> list[str]:
        return [m["name"] for m in self.config["members"]]


TEAM = TeammateManager(TEAM_DIR)


def handle_shutdown_request(teammate: str) -> str:
    request_id = str(uuid.uuid4())[:8]
    with _tracker_lock:
        shutdown_requests[request_id] = {"target": teammate, "status": "pending"}
    BUS.send(
        "lead",
        teammate,
        "Please shut down gracefully.",
        "shutdown_request",
        {"request_id": request_id},
    )
    return f"Shutdown request {request_id} sent to '{teammate}'"


def handle_plan_review(request_id: str, approve: bool, feedback: str = "") -> str:
    with _tracker_lock:
        req = plan_requests.get(request_id)
    if not req:
        return f"Error: Unknown plan request_id '{request_id}'"

    with _tracker_lock:
        req["status"] = "approved" if approve else "rejected"
    BUS.send(
        "lead",
        req["from"],
        feedback,
        "plan_approval_response",
        {"request_id": request_id, "approve": approve, "feedback": feedback},
    )
    return f"Plan {req['status']} for '{req['from']}'"


def _check_shutdown_status(request_id: str) -> str:
    with _tracker_lock:
        return json.dumps(shutdown_requests.get(request_id, {"error": "not found"}))


TOOL_HANDLERS = {
    "bash": lambda **kw: run_bash(kw["command"]),
    "read_file": lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file": lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "spawn_teammate": lambda **kw: TEAM.spawn(kw["name"], kw["role"], kw["prompt"]),
    "list_teammates": lambda **kw: TEAM.list_all(),
    "send_message": lambda **kw: BUS.send("lead", kw["to"], kw["content"], kw.get("msg_type", "message")),
    "read_inbox": lambda **kw: json.dumps(BUS.read_inbox("lead"), indent=2),
    "broadcast": lambda **kw: BUS.broadcast("lead", kw["content"], TEAM.member_names()),
    "shutdown_request": lambda **kw: handle_shutdown_request(kw["teammate"]),
    "shutdown_response": lambda **kw: _check_shutdown_status(kw.get("request_id", "")),
    "plan_approval": lambda **kw: handle_plan_review(kw["request_id"], kw["approve"], kw.get("feedback", "")),
    "idle": lambda **kw: "Lead does not idle.",
    "claim_task": lambda **kw: claim_task(kw["task_id"], "lead"),
}

TOOLS = file_tools() + [
    function_tool(
        "spawn_teammate",
        "Spawn an autonomous teammate.",
        {
            "name": {"type": "string"},
            "role": {"type": "string"},
            "prompt": {"type": "string"},
        },
        ["name", "role", "prompt"],
    ),
    function_tool(
        "list_teammates",
        "List all teammates.",
        {},
        [],
    ),
    function_tool(
        "send_message",
        "Send a message to a teammate.",
        {
            "to": {"type": "string"},
            "content": {"type": "string"},
            "msg_type": {"type": "string", "enum": list(VALID_MSG_TYPES)},
        },
        ["to", "content"],
    ),
    function_tool(
        "read_inbox",
        "Read and drain the lead's inbox.",
        {},
        [],
    ),
    function_tool(
        "broadcast",
        "Send a message to all teammates.",
        {"content": {"type": "string"}},
        ["content"],
    ),
    function_tool(
        "shutdown_request",
        "Request a teammate to shut down.",
        {"teammate": {"type": "string"}},
        ["teammate"],
    ),
    function_tool(
        "shutdown_response",
        "Check shutdown request status.",
        {"request_id": {"type": "string"}},
        ["request_id"],
    ),
    function_tool(
        "plan_approval",
        "Approve or reject a teammate's plan.",
        {
            "request_id": {"type": "string"},
            "approve": {"type": "boolean"},
            "feedback": {"type": "string"},
        },
        ["request_id", "approve"],
    ),
    function_tool(
        "idle",
        "Enter idle state (for lead -- rarely used).",
        {},
        [],
    ),
    function_tool(
        "claim_task",
        "Claim a task from the board by ID.",
        {"task_id": {"type": "integer"}},
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
        inbox = BUS.read_inbox("lead")
        if inbox:
            messages.append(
                {
                    "role": "user",
                    "content": f"<inbox>{json.dumps(inbox, indent=2)}</inbox>",
                }
            )
            messages.append({"role": "assistant", "content": "Noted inbox messages."})

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
            query = input("\033[36ms11 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        if query.strip() == "/team":
            print(TEAM.list_all())
            continue
        if query.strip() == "/inbox":
            print(json.dumps(BUS.read_inbox("lead"), indent=2))
            continue
        if query.strip() == "/tasks":
            TASKS_DIR.mkdir(exist_ok=True)
            for task_file in sorted(TASKS_DIR.glob("task_*.json")):
                task = json.loads(task_file.read_text())
                marker = {
                    "pending": "[ ]",
                    "in_progress": "[>]",
                    "completed": "[x]",
                }.get(task.get("status"), "[?]")
                owner = f" @{task['owner']}" if task.get("owner") else ""
                print(f"  {marker} #{task['id']}: {task['subject']}{owner}")
            continue
        history.append({"role": "user", "content": query})
        agent_loop(history)
        response_content = history[-1].get("content", "")
        if response_content:
            print(response_content)
        print()
