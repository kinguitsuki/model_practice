from typing import Optional

import anthropic

from config.settings import settings
from .base import BaseLLM, LLMResponse


class ClaudeLLM(BaseLLM):
    """Anthropic Claude API のラッパー。
    プロンプトキャッシュ (cache_control) を使うと
    長い system プロンプトの再利用コストを削減できる。
    """

    def __init__(self) -> None:
        if not settings.anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY が設定されていません。"
                " .env ファイルに ANTHROPIC_API_KEY=sk-ant-... を記述してください。"
            )
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._model = settings.claude_model

    def generate(self, prompt: str, system: Optional[str] = None) -> LLMResponse:
        return self.chat(
            messages=[{"role": "user", "content": prompt}],
            system=system,
        )

    def chat(
        self,
        messages: list[dict],
        system: Optional[str] = None,
    ) -> LLMResponse:
        kwargs: dict = {
            "model": self._model,
            "max_tokens": settings.max_tokens,
            "messages": messages,
        }
        if system:
            # system プロンプトにキャッシュを付与してコスト削減
            kwargs["system"] = [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ]

        response = self._client.messages.create(**kwargs)
        return LLMResponse(
            content=response.content[0].text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=self._model,
        )
