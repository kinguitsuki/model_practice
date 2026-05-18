"""ReActエージェントのユニットテスト。

LLM をモックして ReAct ループのロジックだけをテストする。
実行方法: python -m pytest tests/test_agent.py -v
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from agent.core.memory import ConversationMemory
from agent.tools.base import BaseTool, ToolResult
from llm.base import LLMResponse


# ── テスト用ダミーツール ───────────────────────────────────────────────────

class EchoTool(BaseTool):
    """入力をそのまま返すダミーツール。"""
    name = "echo"
    description = "入力をそのまま返します。"

    def run(self, tool_input: str) -> ToolResult:
        return ToolResult(success=True, output=f"ECHO: {tool_input}")


class FailingTool(BaseTool):
    """常に失敗するダミーツール。"""
    name = "failing_tool"
    description = "必ず失敗するツール。"

    def run(self, tool_input: str) -> ToolResult:
        raise RuntimeError("意図的なエラー")


def _make_llm_response(content: str) -> LLMResponse:
    return LLMResponse(content=content, input_tokens=10, output_tokens=20, model="test")


# ── ConversationMemory テスト ──────────────────────────────────────────────

class TestConversationMemory:
    def test_add_and_retrieve(self):
        mem = ConversationMemory(max_window=5)
        mem.add("user", "こんにちは")
        mem.add("assistant", "こんにちは！")
        messages = mem.get_messages()
        assert len(messages) == 2
        assert messages[0]["role"] == "user"

    def test_sliding_window(self):
        mem = ConversationMemory(max_window=2)
        for i in range(5):
            mem.add("user", f"メッセージ {i}")
            mem.add("assistant", f"回答 {i}")

        # max_window=2 → 直近 2 ターン = 4 メッセージだけ保持
        messages = mem.get_messages()
        assert len(messages) == 4
        assert "メッセージ 3" in messages[0]["content"]

    def test_turn_count(self):
        mem = ConversationMemory()
        mem.add("user", "Q1")
        mem.add("assistant", "A1")
        mem.add("user", "Q2")
        assert mem.turn_count == 2

    def test_clear(self):
        mem = ConversationMemory()
        mem.add("user", "テスト")
        mem.clear()
        assert len(mem.get_messages()) == 0


# ── ReActAgent テスト ──────────────────────────────────────────────────────

class TestReActAgent:
    def _make_agent(self, llm_responses: list[str], tools=None):
        """モック LLM を持つエージェントを生成するヘルパー。"""
        from agent.core.agent import ReActAgent

        mock_llm = MagicMock()
        mock_llm.chat.side_effect = [
            _make_llm_response(r) for r in llm_responses
        ]

        return ReActAgent(
            llm=mock_llm,
            tools=tools or [EchoTool()],
            verbose=False,
        )

    def test_direct_final_answer(self):
        """ツール不要でいきなり Final Answer を返すケース。"""
        agent = self._make_agent([
            "Thought: 計算不要です。\nFinal Answer: 答えは42です。"
        ])
        result = agent.run("42とは何ですか？")
        assert result == "答えは42です。"

    def test_tool_use_then_final_answer(self):
        """ツールを 1 回使ってから Final Answer を返すケース。"""
        agent = self._make_agent([
            "Thought: echo ツールを使います。\nAction: echo\nAction Input: テスト",
            "Thought: 結果を確認しました。\nFinal Answer: ECHO 結果を受け取りました。",
        ])
        result = agent.run("echo ツールを試して")
        assert "ECHO 結果を受け取りました" in result

    def test_unknown_tool_handled_gracefully(self):
        """存在しないツールを呼んでもループが壊れないこと。"""
        agent = self._make_agent([
            "Thought: 存在しないツール。\nAction: nonexistent_tool\nAction Input: 何か",
            "Thought: ツールが見つかりませんでした。\nFinal Answer: ツールがありません。",
        ])
        result = agent.run("存在しないツールを使って")
        assert result  # クラッシュせず文字列が返ること

    def test_tool_error_handled_gracefully(self):
        """ツールが例外を投げても ReAct ループが継続すること。"""
        agent = self._make_agent(
            llm_responses=[
                "Thought: failing_tool を使います。\nAction: failing_tool\nAction Input: 何か",
                "Thought: エラーが出ました。\nFinal Answer: エラーが発生しました。",
            ],
            tools=[FailingTool()],
        )
        result = agent.run("失敗するツールを使って")
        assert "エラーが発生しました" in result

    def test_max_iterations_respected(self):
        """max_iterations を超えたらフォールバックを返すこと。"""
        from agent.core.agent import ReActAgent

        mock_llm = MagicMock()
        # Action だけ返し続けて Final Answer を返さないケース
        mock_llm.chat.return_value = _make_llm_response(
            "Thought: ずっと考え中。\nAction: echo\nAction Input: ループ"
        )

        agent = ReActAgent(
            llm=mock_llm,
            tools=[EchoTool()],
            verbose=False,
        )
        agent._max_iter = 3
        result = agent.run("無限ループテスト")

        assert "最大試行回数" in result
        assert mock_llm.chat.call_count == 3
