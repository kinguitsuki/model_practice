from dataclasses import dataclass
from typing import List

from config.settings import settings


@dataclass
class Message:
    role: str    # "user" | "assistant"
    content: str


class ConversationMemory:
    """スライディングウィンドウ方式の会話履歴。

    max_window ターン分 (user + assistant のペア) だけ保持し、
    古い履歴は破棄する。これによりトークン数を制御しつつ
    直近の文脈は維持できる。
    """

    def __init__(self, max_window: int | None = None) -> None:
        self._max_window = max_window or settings.memory_window
        self._history: List[Message] = []

    def add(self, role: str, content: str) -> None:
        self._history.append(Message(role=role, content=content))

    def get_messages(self) -> List[dict]:
        """Anthropic API 形式 (role / content 辞書のリスト) で返す。"""
        recent = self._history[-(self._max_window * 2):]
        return [{"role": m.role, "content": m.content} for m in recent]

    def clear(self) -> None:
        self._history.clear()

    @property
    def turn_count(self) -> int:
        return sum(1 for m in self._history if m.role == "user")
