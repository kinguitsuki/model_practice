#!/usr/bin/env python3
"""AI Agent with RAG — エントリーポイント

使い方:
    python main.py                       # チャットモード（デフォルト）
    python main.py --mode chat           # チャットモード
    python main.py --mode api            # FastAPI サーバー起動
    python main.py --mode build-index    # インデックス構築のみ
    python main.py --mode build-index --docs path/to/docs
"""
import argparse
import sys
from pathlib import Path

# プロジェクトルートを sys.path に追加
sys.path.insert(0, str(Path(__file__).parent))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG + ReAct 搭載 AI エージェント",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["chat", "api", "build-index"],
        default="chat",
        help="実行モード (デフォルト: chat)",
    )
    parser.add_argument(
        "--docs",
        default="knowledge_base",
        help="インデックス構築に使うドキュメントパス (デフォルト: knowledge_base/)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="API サーバーのポート番号 (デフォルト: 8000)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="ReAct の思考ログを非表示にする",
    )
    args = parser.parse_args()

    if args.mode == "chat":
        from interfaces.cli import run_cli
        run_cli()

    elif args.mode == "build-index":
        from rag.pipeline import RAGPipeline
        pipeline = RAGPipeline()
        count = pipeline.build_index(args.docs)
        print(f"\n✅ 完了: {count} チャンクをインデックス化しました")

    elif args.mode == "api":
        try:
            import uvicorn
        except ImportError:
            print("FastAPI モードには uvicorn が必要です: pip install uvicorn")
            sys.exit(1)
        print(f"🚀 API サーバー起動: http://localhost:{args.port}")
        print(f"   Swagger UI: http://localhost:{args.port}/docs")
        import uvicorn
        uvicorn.run(
            "interfaces.api:app",
            host="0.0.0.0",
            port=args.port,
            reload=False,
        )


if __name__ == "__main__":
    main()
