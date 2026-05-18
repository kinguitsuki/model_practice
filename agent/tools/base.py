from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class ToolResult:
    success: bool
    output: str
    error: Optional[str] = None


class BaseTool(ABC):
    """エージェントが呼び出せるツールの基底クラス。

    新しいツールを追加するときは:
        1. このクラスを継承
        2. name / description を定義 (エージェントがプロンプトで読む)
        3. run() を実装

    safe_run() が例外をキャッチして ToolResult に変換するので
    run() 内では例外を素直に raise して OK。
    """

    name: str
    description: str

    @abstractmethod
    def run(self, tool_input: str) -> ToolResult:
        ...

    def safe_run(self, tool_input: str) -> ToolResult:
        try:
            return self.run(tool_input)
        except Exception as exc:
            return ToolResult(success=False, output="", error=str(exc))
