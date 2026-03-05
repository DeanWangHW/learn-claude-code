import os
from dataclasses import dataclass
from typing import Optional

import httpx

from .model_io import OpenAIModelIO


@dataclass
class OpenAIClientConfig:
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    http_proxy: Optional[str] = None
    timeout_seconds: float = 120.0

    @classmethod
    def from_env(cls) -> "OpenAIClientConfig":
        return cls(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            http_proxy=(
                os.getenv("OPENAI_HTTP_PROXY")
                or os.getenv("HTTP_PROXY")
                or os.getenv("HTTPS_PROXY")
            ),
            timeout_seconds=float(os.getenv("OPENAI_TIMEOUT_SECONDS", "120")),
        )


class _MessagesAPI:
    def __init__(self, model_io: OpenAIModelIO):
        self.model_io = model_io

    def create(self, **kwargs):
        return self.model_io.create(**kwargs)


class OpenAIClient:
    """OpenAI client wrapper that provides messages.create for agent scripts."""

    def __init__(self, config: Optional[OpenAIClientConfig] = None):
        cfg = config or OpenAIClientConfig.from_env()
        http_client = None
        if cfg.http_proxy:
            http_client = httpx.Client(proxy=cfg.http_proxy, timeout=cfg.timeout_seconds)

        self._model_io = OpenAIModelIO(
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            http_client=http_client,
            timeout_seconds=cfg.timeout_seconds,
        )
        self.messages = _MessagesAPI(self._model_io)

    @classmethod
    def from_env(cls) -> "OpenAIClient":
        return cls(OpenAIClientConfig.from_env())
