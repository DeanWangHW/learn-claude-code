from .model_io import OpenAIModelIO


class _MessagesAPI:
    def __init__(self, model_io: OpenAIModelIO):
        self.model_io = model_io

    def create(self, **kwargs):
        return self.model_io.create(**kwargs)


class Anthropic:
    """Compat shim so existing examples can run on OpenAI with minimal code changes."""

    def __init__(self, base_url=None):
        self._model_io = OpenAIModelIO(base_url=base_url)
        self.messages = _MessagesAPI(self._model_io)
