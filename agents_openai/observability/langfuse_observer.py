#!/usr/bin/env python3
"""Langfuse observability adapter with safe no-op fallback (SDK v4)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


def _as_bool(value: str | None) -> bool:
    if value is None:
        return False
    return value.lower() in {"1", "true", "on", "yes"}


@dataclass
class TraceContext:
    root_observation: Any | None
    turn: int = 0


class BaseObserver:
    def start_trace(self, user_input: str, history_len: int) -> TraceContext | None:
        return None

    def on_model_response(
        self,
        ctx: TraceContext | None,
        assistant_text: str,
        tool_calls: list[dict],
    ):
        return

    def on_tool_result(
        self,
        ctx: TraceContext | None,
        tool_name: str,
        tool_input: dict,
        output: str,
    ):
        return

    def finish_trace(self, ctx: TraceContext | None, final_output: str):
        return

    def flush(self):
        return


class NoopObserver(BaseObserver):
    pass


class LangfuseObserver(BaseObserver):
    def __init__(self, agent_name: str, model: str):
        self.agent_name = agent_name
        self.model = model
        self.client = None
        self.debug = _as_bool(os.getenv("LANGFUSE_DEBUG"))

        enabled = os.getenv("LANGFUSE_ENABLED", "1").lower() not in {"0", "false", "off"}
        if not enabled:
            self._log("disabled by LANGFUSE_ENABLED")
            return

        public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
        if not public_key or not secret_key:
            self._log("disabled: LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY missing")
            return

        try:
            from langfuse import Langfuse
        except Exception as e:
            self._log(f"disabled: import langfuse failed: {e}")
            return

        host = os.getenv("LANGFUSE_HOST")
        kwargs = {"public_key": public_key, "secret_key": secret_key}
        if host:
            kwargs["host"] = host

        try:
            self.client = Langfuse(**kwargs)
            self._log("enabled")
        except Exception as e:
            self.client = None
            self._log(f"disabled: Langfuse client init failed: {e}")

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def _log(self, message: str):
        if self.debug:
            print(f"[langfuse] {message}")

    def start_trace(self, user_input: str, history_len: int) -> TraceContext | None:
        if not self.enabled:
            return None

        root = None
        try:
            root = self.client.start_observation(
                name=self.agent_name,
                as_type="span",
                input=user_input[:4000],
                metadata={
                    "model": self.model,
                    "history_len": history_len,
                },
            )
            trace_id = getattr(root, "trace_id", None)
            self._log(f"trace started: {trace_id}")
        except Exception as e:
            self._log(f"start_trace failed: {e}")
            root = None

        return TraceContext(root_observation=root)

    def on_model_response(
        self,
        ctx: TraceContext | None,
        assistant_text: str,
        tool_calls: list[dict],
    ):
        if not self.enabled or not ctx or not ctx.root_observation:
            return

        ctx.turn += 1
        output_payload = {
            "assistant_text": assistant_text[:4000],
            "tool_calls": [
                {
                    "id": tc.get("id", ""),
                    "name": tc.get("function", {}).get("name", ""),
                }
                for tc in tool_calls
            ],
        }

        try:
            obs = ctx.root_observation.start_observation(
                name="model_turn",
                as_type="generation",
                model=self.model,
                input={"turn": ctx.turn},
                output=output_payload,
            )
            if obs and hasattr(obs, "end"):
                obs.end()
        except Exception as e:
            self._log(f"on_model_response failed: {e}")

    def on_tool_result(
        self,
        ctx: TraceContext | None,
        tool_name: str,
        tool_input: dict,
        output: str,
    ):
        if not self.enabled or not ctx or not ctx.root_observation:
            return

        try:
            ctx.root_observation.create_event(
                name=f"tool:{tool_name}",
                input=tool_input,
                output=output[:8000],
            )
        except Exception as e:
            self._log(f"on_tool_result failed: {e}")

    def finish_trace(self, ctx: TraceContext | None, final_output: str):
        if not self.enabled or not ctx or not ctx.root_observation:
            return

        try:
            ctx.root_observation.update(output=final_output[:8000])
            if hasattr(ctx.root_observation, "end"):
                ctx.root_observation.end()
            trace_id = getattr(ctx.root_observation, "trace_id", None)
            self._log(f"trace finished: {trace_id}")
        except Exception as e:
            self._log(f"finish_trace failed: {e}")

    def flush(self):
        if not self.enabled:
            return
        try:
            if hasattr(self.client, "flush"):
                self.client.flush()
                self._log("flushed")
        except Exception as e:
            self._log(f"flush failed: {e}")


def create_observer(agent_name: str, model: str) -> BaseObserver:
    observer = LangfuseObserver(agent_name=agent_name, model=model)
    if observer.enabled:
        return observer
    return NoopObserver()
