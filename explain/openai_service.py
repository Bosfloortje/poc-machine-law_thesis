import os
from typing import Any

from fastapi import Request

from .base_llm_service import BaseLLMService


class OpenAIService(BaseLLMService):
    """Service for OpenAI models (gpt-4o, gpt-4o-mini, etc.)"""

    SESSION_KEY = "openai_api_key"
    ENV_KEY = "OPENAI_API_KEY"

    def __init__(self, model_id: str = "gpt-4o") -> None:
        self._model_id = model_id
        self.api_key = os.getenv(self.ENV_KEY)
        self.client = None
        self._initialize_client()

    def _initialize_client(self, key: str | None = None) -> None:
        api_key = key or self.api_key
        if not api_key:
            self.client = None
            return
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
        except Exception:
            self.client = None

    @property
    def is_configured(self) -> bool:
        return self.client is not None

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_id(self) -> str:
        return self._model_id

    def set_session_key(self, request: Request, api_key: str) -> bool:
        try:
            from openai import OpenAI
            OpenAI(api_key=api_key)
            request.session[self.SESSION_KEY] = api_key
            self._initialize_client(api_key)
            return True
        except Exception:
            return False

    def get_api_key(self, request: Request | None = None) -> str | None:
        if request and self.SESSION_KEY in request.session:
            return request.session[self.SESSION_KEY]
        return os.getenv(self.ENV_KEY)

    def configure_for_request(self, request: Request) -> None:
        if request and self.SESSION_KEY in request.session:
            self._initialize_client(request.session[self.SESSION_KEY])

    def clear_session_key(self, request: Request) -> None:
        if self.SESSION_KEY in request.session:
            del request.session[self.SESSION_KEY]
        self._initialize_client()

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 2000,
        temperature: float = 0.7,
        system: str | None = None,
    ) -> Any | None:
        if not self.is_configured:
            return None
        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)
        return self.client.chat.completions.create(
            model=self._model_id,
            messages=full_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def get_completion_text(self, response: Any) -> str:
        if response is None:
            return "OpenAI niet beschikbaar. Controleer OPENAI_API_KEY."
        return response.choices[0].message.content


# Singletons per model
gpt4o_service = OpenAIService("gpt-4o")
gpt4o_mini_service = OpenAIService("gpt-4o-mini")
