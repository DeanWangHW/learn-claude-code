import json
import os
from dataclasses import dataclass
from typing import Any

from langchain_openai import ChatOpenAI


@dataclass
class TextBlock:
    type: str
    text: str


@dataclass
class ToolUseBlock:
    type: str
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class ModelResponse:
    content: list[Any]
    stop_reason: str


class LangChainModelIO:
    """Model I/O adapter built on LangChain ChatOpenAI + bind_tools."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        http_proxy: str | None = None,
        timeout_seconds: float = 120.0,
    ):
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.http_proxy = http_proxy
        self.timeout_seconds = timeout_seconds

    def create(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 8000,
    ) -> ModelResponse:
        lc_messages = self._to_langchain_messages(system=system, messages=messages)
        lc_tools = self._to_langchain_tools(tools or [])

        model_kwargs = {}
        if self.http_proxy:
            model_kwargs["openai_proxy"] = self.http_proxy

        llm = ChatOpenAI(
            model=model,
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout_seconds,
            max_tokens=max_tokens,
            model_kwargs=model_kwargs,
        )
        if lc_tools:
            llm = llm.bind_tools(lc_tools)

        ai_msg = llm.invoke(lc_messages)

        content: list[Any] = []
        if isinstance(ai_msg.content, str) and ai_msg.content:
            content.append(TextBlock(type="text", text=ai_msg.content))

        for call in getattr(ai_msg, "tool_calls", []) or []:
            content.append(
                ToolUseBlock(
                    type="tool_use",
                    id=call.get("id", ""),
                    name=call.get("name", ""),
                    input=call.get("args", {}),
                )
            )

        stop_reason = "tool_use" if getattr(ai_msg, "tool_calls", None) else "end_turn"
        return ModelResponse(content=content, stop_reason=stop_reason)

    def _to_langchain_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        mapped = []
        for t in tools:
            mapped.append(
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
                    },
                }
            )
        return mapped

    def _to_langchain_messages(self, *, system: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            if isinstance(content, str):
                out.append({"role": role, "content": content})
                continue

            if isinstance(content, list):
                if role == "assistant":
                    assistant_text_parts = []
                    tool_calls = []
                    for part in content:
                        p_type = getattr(part, "type", None) or (part.get("type") if isinstance(part, dict) else None)
                        if p_type == "text":
                            text = getattr(part, "text", None) or (part.get("text") if isinstance(part, dict) else "")
                            if text:
                                assistant_text_parts.append(text)
                        elif p_type == "tool_use":
                            call_id = getattr(part, "id", None) or (part.get("id") if isinstance(part, dict) else None)
                            name = getattr(part, "name", None) or (part.get("name") if isinstance(part, dict) else None)
                            tool_input = getattr(part, "input", None) or (part.get("input") if isinstance(part, dict) else {})
                            tool_calls.append(
                                {
                                    "id": call_id,
                                    "type": "function",
                                    "function": {
                                        "name": name,
                                        "arguments": json.dumps(tool_input, ensure_ascii=False),
                                    },
                                }
                            )
                    out.append(
                        {
                            "role": "assistant",
                            "content": "\n".join(assistant_text_parts) if assistant_text_parts else "",
                            "tool_calls": tool_calls or None,
                        }
                    )
                elif role == "user":
                    user_text_parts = []
                    for part in content:
                        if not isinstance(part, dict):
                            continue
                        if part.get("type") == "text":
                            user_text_parts.append(part.get("text", ""))
                        elif part.get("type") == "tool_result":
                            out.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": part.get("tool_use_id"),
                                    "content": str(part.get("content", "")),
                                }
                            )
                    if user_text_parts:
                        out.append({"role": "user", "content": "\n".join(user_text_parts)})
                continue

            out.append({"role": role, "content": str(content)})

        cleaned = []
        for msg in out:
            cleaned.append({k: v for k, v in msg.items() if v is not None})
        return cleaned
