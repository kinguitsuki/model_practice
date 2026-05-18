import ast
import math
import operator
from typing import Any

from .base import BaseTool, ToolResult

# 許可する演算子のホワイトリスト (任意コード実行を防ぐ)
_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_SAFE_MATH = {
    name: fn
    for name, fn in vars(math).items()
    if callable(fn) and not name.startswith("_")
}


class CalculatorTool(BaseTool):
    """数式を安全に評価する計算ツール。

    eval() は使わず AST を手動でウォークするため、
    任意コード実行の脆弱性がない。

    例: "2 + 3 * 4", "sqrt(2)", "10 ** 2 / 4", "log(100, 10)"
    """

    name = "calculator"
    description = (
        "数学的な計算を行います。"
        "例: '2 + 3 * 4'、'sqrt(16)'、'log(100, 10)'、'10 ** 2'"
    )

    def run(self, tool_input: str) -> ToolResult:
        try:
            result = self._safe_eval(tool_input.strip())
            return ToolResult(success=True, output=str(result))
        except ZeroDivisionError:
            return ToolResult(success=False, output="", error="0 除算エラー")
        except Exception as exc:
            return ToolResult(success=False, output="", error=f"計算エラー: {exc}")

    # ── プライベート ──────────────────────────────────────────

    def _safe_eval(self, expr: str) -> Any:
        tree = ast.parse(expr, mode="eval")
        return self._eval_node(tree.body)

    def _eval_node(self, node: ast.expr) -> Any:
        if isinstance(node, ast.Constant):
            return node.value

        if isinstance(node, ast.BinOp):
            op_fn = _SAFE_OPS.get(type(node.op))
            if op_fn is None:
                raise ValueError(f"未対応の演算子: {type(node.op).__name__}")
            return op_fn(self._eval_node(node.left), self._eval_node(node.right))

        if isinstance(node, ast.UnaryOp):
            op_fn = _SAFE_OPS.get(type(node.op))
            if op_fn is None:
                raise ValueError(f"未対応の単項演算子: {type(node.op).__name__}")
            return op_fn(self._eval_node(node.operand))

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("メソッド呼び出しは許可されていません")
            fn = _SAFE_MATH.get(node.func.id)
            if fn is None:
                raise ValueError(f"未対応の関数: {node.func.id}")
            args = [self._eval_node(a) for a in node.args]
            return fn(*args)

        raise ValueError(f"未対応の式の種類: {type(node).__name__}")
