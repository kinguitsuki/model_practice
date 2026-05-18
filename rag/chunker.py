from dataclasses import dataclass, field
from typing import List

from .document_loader import Document


@dataclass
class Chunk:
    """Document を分割した小単位。埋め込みの入力になる。"""

    content: str
    metadata: dict = field(default_factory=dict)
    chunk_id: int = 0


class RecursiveTextChunker:
    """再帰的テキスト分割。

    LangChain の RecursiveCharacterTextSplitter と同等のロジックを
    スクラッチ実装。段落 → 文 → 単語の順に分割点を探し、
    意味のある単位でチャンクを作る。

    chunk_size    : 1 チャンクの最大文字数
    chunk_overlap : 前チャンクとの重複文字数（文脈の連続性を保つ）
    """

    _SEPARATORS = ["\n\n", "\n", "。", ".", "、", ",", " ", ""]

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, documents: List[Document]) -> List[Chunk]:
        chunks: List[Chunk] = []
        chunk_id = 0
        for doc in documents:
            for text in self._split_text(doc.content):
                if text.strip():
                    chunks.append(
                        Chunk(
                            content=text.strip(),
                            metadata=doc.metadata.copy(),
                            chunk_id=chunk_id,
                        )
                    )
                    chunk_id += 1
        return chunks

    # ── プライベート ──────────────────────────────────────────

    def _split_text(self, text: str) -> List[str]:
        return self._recursive_split(text, self._SEPARATORS)

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        # 有効な区切り文字を探す
        separator = ""
        remaining_seps = list(separators)
        for sep in separators:
            if sep == "" or sep in text:
                separator = sep
                remaining_seps = separators[separators.index(sep) + 1:]
                break

        parts = text.split(separator) if separator else [text]
        chunks: List[str] = []
        current = ""

        for part in parts:
            candidate = (current + separator + part) if current else part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                    # オーバーラップ: 前チャンクの末尾を次チャンクに引き継ぐ
                    overlap_text = current[-self.chunk_overlap:]
                    current = (overlap_text + separator + part).lstrip(separator)
                else:
                    # 1 つの部分がチャンクサイズを超えるなら再帰分割
                    if remaining_seps:
                        chunks.extend(self._recursive_split(part, remaining_seps))
                    else:
                        # 最終手段: 強制的に固定幅で分割
                        step = self.chunk_size - self.chunk_overlap
                        for i in range(0, len(part), step):
                            chunks.append(part[i : i + self.chunk_size])
                    current = ""

        if current:
            chunks.append(current)

        return chunks
