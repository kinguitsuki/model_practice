from rag.pipeline import RAGPipeline
from .base import BaseTool, ToolResult


class RAGSearchTool(BaseTool):
    """ナレッジベースからセマンティック検索するツール。

    エージェントが「知識を調べたい」と判断したときに呼ぶ。
    入力はクエリ文字列、出力は関連チャンクをまとめたテキスト。
    """

    name = "rag_search"
    description = (
        "ナレッジベース（社内ドキュメント・マニュアル・技術資料）を検索します。"
        "入力: 調べたいキーワードや質問文"
    )

    def __init__(self, pipeline: RAGPipeline) -> None:
        self._pipeline = pipeline

    def run(self, tool_input: str) -> ToolResult:
        context = self._pipeline.retrieve(tool_input)
        return ToolResult(success=True, output=context)
