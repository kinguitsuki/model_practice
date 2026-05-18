from typing import List

import numpy as np


class EmbeddingModel:
    """sentence-transformers を使ったローカル埋め込みモデル。

    intfloat/multilingual-e5-base を既定とする。E5 系モデルは
    クエリに "query: "、文書に "passage: " を付けると精度が上がる。
    """

    def __init__(self, model_name: str = "intfloat/multilingual-e5-base") -> None:
        from sentence_transformers import SentenceTransformer

        print(f"  埋め込みモデルを読み込み中: {model_name}")
        self._model = SentenceTransformer(model_name)
        self.model_name = model_name
        # モデルの出力次元数を取得
        self.dimension: int = self._model.get_sentence_embedding_dimension()

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """文書リストをベクトル化 (passage: プレフィックス付き)。"""
        prefixed = self._add_prefix(texts, "passage")
        vectors = self._model.encode(
            prefixed,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
        )
        return np.array(vectors, dtype="float32")

    def embed_query(self, query: str) -> np.ndarray:
        """クエリ 1 件をベクトル化 (query: プレフィックス付き)。"""
        prefixed = self._add_prefix([query], "query")[0]
        vector = self._model.encode([prefixed], normalize_embeddings=True)
        return np.array(vector[0], dtype="float32")

    # ── プライベート ──────────────────────────────────────────

    def _add_prefix(self, texts: List[str], kind: str) -> List[str]:
        if "e5" in self.model_name.lower():
            return [f"{kind}: {t}" for t in texts]
        return texts
