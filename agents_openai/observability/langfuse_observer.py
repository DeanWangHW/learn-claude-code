#!/usr/bin/env python3
"""Langfuse 观测适配层（SDK v4）。

本模块的目标是把「可观测性」从业务 Agent Loop 中抽离出来，避免每个脚本里
都复制一整套 Langfuse 初始化、异常兜底、事件上报逻辑。

为什么要封装：
1. 降低重复代码：
   - `s01` 到 `s07` 都是类似的循环结构，如果每个文件都手写 Langfuse 调用，
     后续维护成本会线性上升。
2. 提高健壮性：
   - 生产环境里经常会遇到 `LANGFUSE_*` 缺失、SDK 未安装、网络异常。
     适配层统一提供 no-op 降级，保证主流程不因监控失败而中断。
3. 统一事件语义：
   - 所有脚本都按同样的 trace/model/tool 生命周期打点，方便横向比较与排障。
4. 降低改造成本：
   - 后续若更换 SDK 版本或更换观测后端，仅需修改这一处，不必逐文件改动。
"""

from __future__ import annotations

from contextlib import nullcontext
import importlib
import os
import re
from dataclasses import dataclass
from typing import Any


def _as_bool(value: str | None) -> bool:
    """将环境变量字符串解析为布尔值。

    Parameters
    ----------
    value : str | None
        环境变量原始字符串，例如 ``"1"``, ``"true"``, ``"on"``。

    Returns
    -------
    bool
        可识别真值时返回 ``True``，否则返回 ``False``。
    """
    if value is None:
        return False
    return value.lower() in {"1", "true", "on", "yes"}


@dataclass
class TraceContext:
    """单次 Agent 运行的 trace 上下文容器。

    Parameters
    ----------
    root_observation : Any | None
        Langfuse 根 observation（通常是 span），为 ``None`` 表示当前未启用上报。
    turn : int, default=0
        当前会话已记录的模型轮次计数，用于按回合切分 generation 事件。
    trace_name : str | None, default=None
        当前 trace 的名称，用于在 Langfuse UI 中做检索与分组管理。
    """

    root_observation: Any | None
    turn: int = 0
    trace_name: str | None = None


class BaseObserver:
    """观测接口基类。

    Notes
    -----
    这里定义的是「业务层依赖的最小接口」，而不是 Langfuse SDK 原生接口。
    Agent 代码只依赖这些方法，从而与具体观测实现解耦。
    """

    def start_trace(self, user_input: str, history_len: int) -> TraceContext | None:
        """开始一次 trace。"""
        return None

    def on_model_response(
        self,
        ctx: TraceContext | None,
        assistant_text: str,
        tool_calls: list[dict],
    ):
        """记录一次模型响应事件。"""
        return

    def on_tool_result(
        self,
        ctx: TraceContext | None,
        tool_name: str,
        tool_input: dict,
        output: str,
    ):
        """记录一次工具执行结果。"""
        return

    def finish_trace(self, ctx: TraceContext | None, final_output: str):
        """结束 trace，并写入最终输出。"""
        return

    def flush(self):
        """主动刷新缓冲区，尽量在进程退出前把事件发出。"""
        return


class NoopObserver(BaseObserver):
    """空实现：当 Langfuse 不可用时，所有方法静默返回。"""

    pass


class LangfuseObserver(BaseObserver):
    """Langfuse v4 适配器。

    Parameters
    ----------
    agent_name : str
        当前脚本/Agent 的名称，用于 trace 命名。
    model : str
        当前调用的模型 ID，写入 metadata 便于后续筛选分析。
    """

    def __init__(self, agent_name: str, model: str):
        self.agent_name = agent_name
        self.model = model
        self.client = None
        self._propagate_attributes = None
        self.debug = _as_bool(os.getenv("LANGFUSE_DEBUG"))

        # 支持显式关闭：本地调试或 CI 无需监控时可设置 LANGFUSE_ENABLED=0。
        enabled = os.getenv("LANGFUSE_ENABLED", "1").lower() not in {"0", "false", "off"}
        if not enabled:
            self._log("disabled by LANGFUSE_ENABLED")
            return

        # 缺少关键凭证时直接降级为 no-op，避免影响主流程。
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
        if not public_key or not secret_key:
            self._log("disabled: LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY missing")
            return

        try:
            langfuse_module = importlib.import_module("langfuse")
            Langfuse = getattr(langfuse_module, "Langfuse")
            self._propagate_attributes = getattr(langfuse_module, "propagate_attributes", None)
        except Exception as e:
            self._log(f"disabled: import langfuse failed: {e}")
            return

        host = os.getenv("LANGFUSE_HOST")
        kwargs = {"public_key": public_key, "secret_key": secret_key}
        if host:
            kwargs["host"] = host

        try:
            # SDK v4 客户端初始化，成功后才视为 enabled。
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

    def _build_trace_name(self, user_input: str) -> str:
        """构造稳定、可检索的 trace 名称。

        规则：
        1. 默认前缀是 ``agent_name``，便于按脚本聚合。
        2. 拼接用户输入首行摘要，提升检索可读性。
        3. 仅保留 ASCII 可见字符，避免平台约束导致丢失。
        4. 最大长度 200，符合 Langfuse 对 trace_name 的约束。
        """
        first_line = (user_input or "").strip().splitlines()[0] if user_input else ""
        normalized = re.sub(r"\s+", " ", first_line).strip()
        if normalized:
            candidate = f"{self.agent_name}: {normalized}"
        else:
            candidate = self.agent_name

        # 只保留 ASCII 可见字符，避免非法字符导致 trace_name 被忽略。
        ascii_only = "".join(ch for ch in candidate if 32 <= ord(ch) <= 126)
        if not ascii_only:
            ascii_only = self.agent_name
        return ascii_only[:200]

    def start_trace(self, user_input: str, history_len: int) -> TraceContext | None:
        """开始一次根 trace（root span）。

        Parameters
        ----------
        user_input : str
            用户最近一次输入内容，会截断以避免超长 payload。
        history_len : int
            当前会话消息总长度，便于排查上下文膨胀问题。

        Returns
        -------
        TraceContext | None
            返回可在后续步骤复用的上下文对象；若未启用监控则返回 ``None``。
        """
        if not self.enabled:
            return None

        root = None
        trace_name = self._build_trace_name(user_input)
        try:
            # Langfuse v4 的 trace 名称需要通过 propagate_attributes 注入。
            context_manager = nullcontext()
            if self._propagate_attributes is not None:
                context_manager = self._propagate_attributes(trace_name=trace_name)

            with context_manager:
                # 这里用 span 作为根 observation；后续 model/tool 事件挂在其下。
                root = self.client.start_observation(
                    name=self.agent_name,
                    as_type="span",
                    input=user_input[:4000],
                    metadata={
                        "model": self.model,
                        "history_len": history_len,
                        "trace_name": trace_name,
                    },
                )
            trace_id = getattr(root, "trace_id", None)
            self._log(f"trace started: {trace_id}, trace_name={trace_name}")
        except Exception as e:
            self._log(f"start_trace failed: {e}")
            root = None

        return TraceContext(root_observation=root, trace_name=trace_name)

    def on_model_response(
        self,
        ctx: TraceContext | None,
        assistant_text: str,
        tool_calls: list[dict],
    ):
        """记录模型回合事件。

        Notes
        -----
        - 每轮模型调用会生成一条 generation。
        - 仅保留 tool call 的结构化信息（id/name），避免日志过重。
        """
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
        """记录工具执行事件。"""
        if not self.enabled or not ctx or not ctx.root_observation:
            return

        try:
            # tool 输出可能很大，统一截断，防止上报体积过大。
            ctx.root_observation.create_event(
                name=f"tool:{tool_name}",
                input=tool_input,
                output=output[:8000],
            )
        except Exception as e:
            self._log(f"on_tool_result failed: {e}")

    def finish_trace(self, ctx: TraceContext | None, final_output: str):
        """结束 trace 并更新最终输出摘要。"""
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
        """刷新 Langfuse 客户端缓冲。"""
        if not self.enabled:
            return
        try:
            if hasattr(self.client, "flush"):
                self.client.flush()
                self._log("flushed")
        except Exception as e:
            self._log(f"flush failed: {e}")


def create_observer(agent_name: str, model: str) -> BaseObserver:
    """工厂函数：返回可直接使用的观察器实例。

    Parameters
    ----------
    agent_name : str
        当前 Agent 名称。
    model : str
        当前模型标识。

    Returns
    -------
    BaseObserver
        启用成功时返回 ``LangfuseObserver``，否则返回 ``NoopObserver``。
    """
    observer = LangfuseObserver(agent_name=agent_name, model=model)
    if observer.enabled:
        return observer
    return NoopObserver()
