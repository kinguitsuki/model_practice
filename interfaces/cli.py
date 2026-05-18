"""CLIインターフェース。python main.py --mode chat で起動する。"""
from pathlib import Path

from agent.core.agent import ReActAgent
from agent.core.memory import ConversationMemory
from agent.tools.calculator import CalculatorTool
from agent.tools.ml_inference import MLInferenceTool
from agent.tools.rag_search import RAGSearchTool
from config.settings import settings
from llm.claude import ClaudeLLM
from rag.pipeline import RAGPipeline


def build_agent(verbose: bool = True) -> ReActAgent:
    """エージェントを初期化して返す。"""
    rag = _setup_rag()

    tools = [
        RAGSearchTool(rag),
        CalculatorTool(),
        MLInferenceTool(),
    ]
    llm = ClaudeLLM()
    memory = ConversationMemory()
    return ReActAgent(llm=llm, tools=tools, memory=memory, verbose=verbose)


def run_cli() -> None:
    print("=" * 60)
    print("  AI Agent (RAG + ReAct)  起動")
    print("=" * 60)
    print("コマンド: 'quit' で終了 / 'reset' で会話リセット")
    print("-" * 60)

    agent = build_agent(verbose=True)

    while True:
        try:
            user_input = input("\nあなた: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n終了します。")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "終了"):
            print("終了します。")
            break

        if user_input.lower() in ("reset", "リセット"):
            agent.reset()
            print("会話をリセットしました。")
            continue

        print("\nエージェント思考中...\n")
        response = agent.run(user_input)
        print(f"\n{'='*60}")
        print(f"最終回答:\n{response}")
        print("=" * 60)


# ── ユーティリティ ────────────────────────────────────────────────────────

def _setup_rag() -> RAGPipeline:
    rag = RAGPipeline()
    vs_path = Path(settings.vector_store_path)
    if (vs_path / "index.faiss").exists():
        print("📚 既存のインデックスを読み込み中...")
        rag.load_index()
    else:
        print("📚 ナレッジベースをインデックス化中...")
        rag.build_index("knowledge_base")
    return rag
