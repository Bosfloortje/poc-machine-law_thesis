import logging
from typing import Any

from fastapi import Request

from .base_llm_service import BaseLLMService

logger = logging.getLogger(__name__)


class OllamaService(BaseLLMService):
    """Service for local Ollama models (llama3.1, mistral, deepseek, etc.)"""

    SESSION_KEY = "ollama_model"
    ENV_KEY = ""  # No API key needed

    def __init__(self, model_id: str, provider_name: str) -> None:
        self._model_id = model_id
        self._provider_name = provider_name

    @property
    def is_configured(self) -> bool:
        try:
            import ollama  # noqa: F401
            return True
        except ImportError:
            return False

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def model_id(self) -> str:
        return self._model_id

    def set_session_key(self, request: Request, api_key: str) -> bool:
        return True  # No key needed

    def get_api_key(self, request: Request | None = None) -> str | None:
        return None

    def configure_for_request(self, request: Request) -> None:
        pass

    def clear_session_key(self, request: Request) -> None:
        pass

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 2000,
        temperature: float = 0.7,
        system: str | None = None,
    ) -> Any | None:
        if not self.is_configured:
            return None
        import ollama
        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)
        return ollama.chat(
            model=self._model_id,
            messages=full_messages,
            options={"temperature": temperature, "num_predict": max_tokens},
        )

    def get_completion_text(self, response: Any) -> str:
        if response is None:
            return "Ollama niet beschikbaar. Zorg dat Ollama actief is: ollama serve"
        return response["message"]["content"].replace("\ufffd", "")
