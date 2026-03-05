import os
from dataclasses import dataclass
from typing import Optional

from .langchain_model_io import LangChainModelIO


@dataclass
class LangChainClientConfig:
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    http_proxy: Optional[str] = None
    timeout_seconds: float = 120.0

    @classmethod
    def from_env(cls) -> "LangChainClientConfig":
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
    def __init__(self, model_io: LangChainModelIO):
        self.model_io = model_io

    def create(self, **kwargs):
        return self.model_io.create(**kwargs)


class LangChainClient:
    def __init__(self, config: Optional[LangChainClientConfig] = None):
        cfg = config or LangChainClientConfig.from_env()
        self._model_io = LangChainModelIO(
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            http_proxy=cfg.http_proxy,
            timeout_seconds=cfg.timeout_seconds,
        )
        self.messages = _MessagesAPI(self._model_io)

    @classmethod
    def from_env(cls) -> "LangChainClient":
        return cls(LangChainClientConfig.from_env())
