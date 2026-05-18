from pathlib import Path
from typing import List, Optional

from config.settings import settings
from .chunker import RecursiveTextChunker
from .document_loader import DocumentLoader
from .embedder import EmbeddingModel
from .retriever import HybridRetriever
from .vector_store import FAISSVectorStore


class RAGPipeline:
    """RAG のライフサイクル全体を管理するファサード。

    使い方:
        pipeline = RAGPipeline()
        pipeline.build_index("knowledge_base/")   # インデックス構築
        context = pipeline.retrieve("RAGとは何ですか？")  # 検索
    """

    def __init__(self) -> None:
        self._loader = DocumentLoader()
        self._chunker = RecursiveTextChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        self._embedder: Optional[EmbeddingModel] = None
        self._store: Optional[FAISSVectorStore] = None
        self._retriever: Optional[HybridRetriever] = None

    # ── インデックス構築 ──────────────────────────────────────

    def build_index(self, documents_path: str | Path) -> int:
        """ドキュメントを読み込み FAISS インデックスを構築してディスクに保存。
        戻り値はインデックスに登録したチャンク数。
        """
        print("📄 ドキュメントを読み込み中...")
        docs = self._loader.load(documents_path)
        if not docs:
            raise RuntimeError(f"{documents_path} からドキュメントを読み込めませんでした。")
        print(f"   → {len(docs)} ファイル読み込み完了")

        print("✂️  テキストをチャンクに分割中...")
        chunks = self._chunker.split(docs)
        print(f"   → {len(chunks)} チャンク生成")

        print("🔢 埋め込みベクトルを生成中...")
        embedder = self._get_embedder()
        texts = [c.content for c in chunks]
        embeddings = embedder.embed_documents(texts)
        print(f"   → ベクトル次元数: {embeddings.shape[1]}")

        print("💾 FAISS インデックスを構築・保存中...")
        store = FAISSVectorStore(dimension=embeddings.shape[1])
        store.add(chunks, embeddings)
        store.save(settings.vector_store_path)

        self._store = store
        self._retriever = HybridRetriever(store, embedder)
        print(f"✅ インデックス構築完了 — {len(chunks)} チャンク")
        return len(chunks)

    def load_index(self) -> None:
        """保存済みインデックスをディスクから読み込む。"""
        path = Path(settings.vector_store_path)
        if not (path / "index.faiss").exists():
            raise FileNotFoundError(
                f"{path} にインデックスが見つかりません。"
                " build_index() を先に実行してください。"
            )
        self._store = FAISSVectorStore.load(path)
        embedder = self._get_embedder()
        self._retriever = HybridRetriever(self._store, embedder)
        print(f"📚 インデックス読み込み完了 — {self._store.size} チャンク")

    # ── 検索 ─────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int | None = None) -> str:
        """クエリに関連する文脈テキストを返す。LLM のプロンプトに埋め込む。"""
        if not self._retriever:
            raise RuntimeError(
                "インデックスが未構築です。build_index() または load_index() を実行してください。"
            )
        results = self._retriever.retrieve(query, top_k=top_k)
        return self._retriever.format_context(results)

    @property
    def is_ready(self) -> bool:
        return self._retriever is not None

    # ── プライベート ──────────────────────────────────────────

    def _get_embedder(self) -> EmbeddingModel:
        if self._embedder is None:
            self._embedder = EmbeddingModel(settings.embedding_model)
        return self._embedder
