from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMResponse:
    content: str
    input_tokens: int
    output_tokens: int
    model: str

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class BaseLLM(ABC):
    """LLMプロバイダーの抽象インターフェース。
    Claude以外のモデル (OpenAI, Gemini等) に切り替えるときは
    このクラスを継承して run() / chat() を実装するだけでよい。
    """

    @abstractmethod
    def generate(self, prompt: str, system: Optional[str] = None) -> LLMResponse:
        """シングルターンの生成。"""

    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        system: Optional[str] = None,
    ) -> LLMResponse:
        """マルチターン対話。messages は [{"role": "user"|"assistant", "content": "..."}] 形式。"""
