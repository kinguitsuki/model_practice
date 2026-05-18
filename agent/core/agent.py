"""ReAct (Reasoning + Acting) エージェントの実装。

論文: "ReAct: Synergizing Reasoning and Acting in Language Models"
     Yao et al., 2022  https://arxiv.org/abs/2210.03629

ループ構造:
    ┌─────────────────────────────────────────┐
    │  Thought: (LLMが状況を分析)              │
    │  Action: ツール名                        │
    │  Action Input: ツールへの入力            │
    │  Observation: (ツール実行結果)            │
    │  → 上記を繰り返す                        │
    │  Final Answer: ユーザーへの最終回答       │
    └─────────────────────────────────────────┘
"""
from __future__ import annotations

import re
from typing import List, Optional

from config.settings import settings
from llm.base import BaseLLM
from .memory import ConversationMemory
from agent.tools.base import BaseTool

# ── ReAct プロンプト ──────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
あなたは高性能な AI アシスタントです。ユーザーの質問に対して、
以下の「Thought / Action / Action Input / Observation / Final Answer」形式に
厳密に従って回答してください。

## 利用可能なツール
{tools}

## 出力フォーマット（必ず守ること）

```
Thought: [現在の状況の分析と、次に何をすべきかの計画]
Action: [ツール名 — 上のリストから正確に選ぶ]
Action Input: [ツールに渡す入力テキスト]
```

ツール結果を見たあとは:

```
Thought: [観察結果の解釈と、次の行動の計画]
Action: [ツール名]
Action Input: [入力]
```

十分な情報が集まったら:

```
Thought: [最終的な判断]
Final Answer: [ユーザーへの回答（日本語で丁寧に）]
```

## ルール
- ツールが必要ないと判断した場合は即座に Final Answer を出す
- ツール名は完全一致。存在しないツール名は使わない
- Action と Action Input は必ずセットで記述する
- Observation はシステムが自動挿入する。自分で書かない
- 日本語で回答する
"""


class ReActAgent:
    """ReAct フレームワークを実装した AI エージェント。

    LLM の出力をパースして Action を検出 → ツール実行 → 結果を
    Observation として追記 → LLM に再度渡す、というループを
    max_iterations 回まで繰り返す。
    """

    def __init__(
        self,
        llm: BaseLLM,
        tools: List[BaseTool],
        memory: Optional[ConversationMemory] = None,
        verbose: bool = True,
    ) -> None:
        self._llm = llm
        self._tools: dict[str, BaseTool] = {t.name: t for t in tools}
        self._memory = memory or ConversationMemory()
        self._verbose = verbose
        self._max_iter = settings.max_iterations

    # ── パブリック API ─────────────────────────────────────────────────────

    def run(self, user_input: str) -> str:
        """ユーザー入力を受け取り、最終回答を返す。"""
        self._memory.add("user", user_input)
        system = _SYSTEM_PROMPT.format(tools=self._format_tools())

        # ReAct のスクラッチパッド: LLM に渡す「今までの思考経緯」
        scratchpad = user_input
        history = self._memory.get_messages()[:-1]  # 最新の user メッセージ以外

        for iteration in range(1, self._max_iter + 1):
            self._log(f"\n{'─'*50}")
            self._log(f"[ReAct ループ {iteration}/{self._max_iter}]")

            # LLM に思考させる
            response = self._llm.chat(
                messages=history + [{"role": "user", "content": scratchpad}],
                system=system,
            )
            llm_text = response.content.strip()
            self._log(f"[LLM 出力]\n{llm_text}")

            # ── Final Answer を検出 ──────────────────────────────────────
            if "Final Answer:" in llm_text:
                answer = llm_text.split("Final Answer:")[-1].strip()
                self._memory.add("assistant", answer)
                return answer

            # ── Action を検出 ────────────────────────────────────────────
            action, action_input = self._parse_action(llm_text)

            if action is None:
                # フォーマット違反 → そのままテキストを返す
                self._memory.add("assistant", llm_text)
                return llm_text

            # ── ツール実行 ───────────────────────────────────────────────
            observation = self._execute_tool(action, action_input)
            self._log(f"[Observation] {observation}")

            # スクラッチパッドに追記して次ループへ
            scratchpad = (
                f"{llm_text}\n"
                f"Observation: {observation}\n"
            )
            # 2 回目以降は history を空にして scratchpad だけを渡す
            # (トークン節約 + 直近の文脈に集中)
            history = []

        # 最大反復回数超過
        fallback = (
            "申し訳ありません、最大試行回数に達しました。"
            "より具体的な質問をお試しください。"
        )
        self._memory.add("assistant", fallback)
        return fallback

    def reset(self) -> None:
        self._memory.clear()

    # ── プライベート ──────────────────────────────────────────────────────

    def _format_tools(self) -> str:
        lines = [f"- {t.name}: {t.description}" for t in self._tools.values()]
        return "\n".join(lines)

    def _parse_action(self, text: str) -> tuple[Optional[str], str]:
        """LLM 出力から Action / Action Input を抽出する。"""
        action_match = re.search(r"Action:\s*(.+?)(?:\n|$)", text)
        input_match = re.search(r"Action Input:\s*([\s\S]+?)(?:\n(?:Thought|Action|Final Answer|Observation)|$)", text)

        if not action_match:
            return None, ""

        action = action_match.group(1).strip()
        action_input = input_match.group(1).strip() if input_match else ""
        return action, action_input

    def _execute_tool(self, tool_name: str, tool_input: str) -> str:
        self._log(f"[ツール実行] {tool_name}({tool_input!r})")

        tool = self._tools.get(tool_name)
        if tool is None:
            available = list(self._tools.keys())
            return f"ツール '{tool_name}' は存在しません。利用可能: {available}"

        result = tool.safe_run(tool_input)
        if result.success:
            return result.output
        return f"ツールエラー: {result.error}"

    def _log(self, message: str) -> None:
        if self._verbose:
            print(message)
