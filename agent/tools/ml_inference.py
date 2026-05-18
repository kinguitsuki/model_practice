from __future__ import annotations

import sys
from pathlib import Path

# プロジェクトルートを sys.path に追加 (src/ を参照するため)
_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from .base import BaseTool, ToolResult


class MLInferenceTool(BaseTool):
    """既存の線形回帰モデル (src/) を呼び出す推論ツール。

    広告費 (万円) を入力すると売上予測値を返す。
    初回呼び出し時にモデルを学習し、以降はキャッシュする。
    """

    name = "ml_inference"
    description = (
        "広告費（万円）から売上を予測します。"
        "入力: 広告費の数値 (例: '50')"
    )

    def __init__(self) -> None:
        self._model = None

    def run(self, tool_input: str) -> ToolResult:
        try:
            ad_budget = float(tool_input.strip())
        except ValueError:
            return ToolResult(
                success=False, output="", error="数値を入力してください (例: '50')"
            )

        model = self._get_model()
        import numpy as np

        prediction = model.predict(np.array([[ad_budget]]))[0][0]
        return ToolResult(
            success=True,
            output=(
                f"広告費 {ad_budget:.1f} 万円 → 予測売上: {prediction:.1f} 万円\n"
                f"(モデル: 単回帰, src/model.py の LinearRegression を使用)"
            ),
        )

    # ── プライベート ──────────────────────────────────────────

    def _get_model(self):
        if self._model is None:
            from src.data_loader import load_data
            from src.model import LinearRegression

            print("  [ml_inference] モデルを学習中...")
            X, y = load_data()
            self._model = LinearRegression()
            self._model.fit(X, y, lr=0.0001, epochs=5000)
        return self._model
