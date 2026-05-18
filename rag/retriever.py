from typing import List, Tuple

from config.settings import settings
from .chunker import Chunk
from .embedder import EmbeddingModel
from .vector_store import FAISSVectorStore


class HybridRetriever:
    """セマンティック検索 + キーワードリランキングのハイブリッド検索。

    Step 1: FAISS で top_k*2 件の候補を取得 (セマンティック)
    Step 2: クエリとのキーワード一致数でスコアを加算 (BM25 ライク)
    Step 3: 再スコアリング後に top_k 件を返す

    これにより、意味的に近くてもキーワードが合わない結果を後退させ、
    両方に合致する結果を上位に引き上げる。
    """

    def __init__(
        self, vector_store: FAISSVectorStore, embedder: EmbeddingModel
    ) -> None:
        self._store = vector_store
        self._embedder = embedder

    def retrieve(self, query: str, top_k: int | None = None) -> List[Tuple[Chunk, float]]:
        top_k = top_k or settings.top_k
        query_vec = self._embedder.embed_query(query)

        # 候補を多めに取得してからリランク
        candidates = self._store.search(query_vec, top_k=top_k * 2)
        reranked = self._keyword_rerank(query, candidates)
        return reranked[:top_k]

    def format_context(self, results: List[Tuple[Chunk, float]]) -> str:
        """検索結果を LLM へ渡す文脈テキストに整形する。"""
        if not results:
            return "関連する情報が見つかりませんでした。"

        parts: List[str] = []
        for i, (chunk, score) in enumerate(results, start=1):
            source = Path_basename(chunk.metadata.get("source", "不明"))
            parts.append(
                f"【参考資料 {i}】出典: {source}  類似度: {score:.3f}\n{chunk.content}"
            )
        return "\n\n---\n\n".join(parts)

    # ── プライベート ──────────────────────────────────────────

    def _keyword_rerank(
        self, query: str, results: List[Tuple[Chunk, float]]
    ) -> List[Tuple[Chunk, float]]:
        query_tokens = set(query.lower().split())
        reranked: List[Tuple[Chunk, float]] = []
        for chunk, score in results:
            doc_tokens = set(chunk.content.lower().split())
            overlap = len(query_tokens & doc_tokens)
            # キーワード一致 1 語につき +0.02 のボーナス
            boosted_score = score + overlap * 0.02
            reranked.append((chunk, boosted_score))
        return sorted(reranked, key=lambda x: x[1], reverse=True)


def Path_basename(path_str: str) -> str:
    from pathlib import Path
    return Path(path_str).name
