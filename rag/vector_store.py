import pickle
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

from .chunker import Chunk


class FAISSVectorStore:
    """FAISS を使ったインメモリ・ベクトルストア。

    IndexFlatIP (内積) + 正規化ベクトル = コサイン類似度検索。
    save() / load() でディスクに永続化できる。
    """

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        # 内積インデックス。ベクトルを L2 正規化済みなら cosine sim と等価。
        self._index = faiss.IndexFlatIP(dimension)
        self._chunks: List[Chunk] = []

    # ── 書き込み ──────────────────────────────────────────────

    def add(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks と embeddings の件数が一致しません")
        self._index.add(embeddings.astype("float32"))
        self._chunks.extend(chunks)

    # ── 検索 ─────────────────────────────────────────────────

    def search(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> List[Tuple[Chunk, float]]:
        query = query_embedding.astype("float32").reshape(1, -1)
        scores, indices = self._index.search(query, top_k)

        results: List[Tuple[Chunk, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:  # -1 は「見つからず」を意味する
                results.append((self._chunks[idx], float(score)))
        return results

    @property
    def size(self) -> int:
        return self._index.ntotal

    # ── 永続化 ───────────────────────────────────────────────

    def save(self, dir_path: str | Path) -> None:
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(dir_path / "index.faiss"))
        with open(dir_path / "chunks.pkl", "wb") as f:
            pickle.dump(self._chunks, f)

    @classmethod
    def load(cls, dir_path: str | Path) -> "FAISSVectorStore":
        dir_path = Path(dir_path)
        index = faiss.read_index(str(dir_path / "index.faiss"))
        with open(dir_path / "chunks.pkl", "rb") as f:
            chunks = pickle.load(f)

        store = cls.__new__(cls)
        store._index = index
        store.dimension = index.d
        store._chunks = chunks
        return store
