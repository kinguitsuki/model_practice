"""RAGパイプラインのユニットテスト。

LLM・外部APIに依存しないため、APIキーなしで実行できる。
実行方法: python -m pytest tests/test_rag.py -v
"""
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest


# ── Chunker テスト ─────────────────────────────────────────────────────────

class TestRecursiveTextChunker:
    from rag.chunker import RecursiveTextChunker

    def test_short_text_not_split(self):
        from rag.chunker import RecursiveTextChunker
        from rag.document_loader import Document

        chunker = RecursiveTextChunker(chunk_size=512, chunk_overlap=64)
        doc = Document(content="短いテキスト", metadata={"source": "test"})
        chunks = chunker.split([doc])

        assert len(chunks) == 1
        assert chunks[0].content == "短いテキスト"

    def test_long_text_split(self):
        from rag.chunker import RecursiveTextChunker
        from rag.document_loader import Document

        chunker = RecursiveTextChunker(chunk_size=100, chunk_overlap=10)
        long_text = "あ" * 300  # 300文字
        doc = Document(content=long_text, metadata={"source": "test"})
        chunks = chunker.split([doc])

        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.content) <= 100

    def test_paragraph_split_preference(self):
        """段落区切り (\n\n) を優先して分割することを確認。"""
        from rag.chunker import RecursiveTextChunker
        from rag.document_loader import Document

        text = "段落1の内容です。\n\n段落2の内容です。\n\n段落3の内容です。"
        chunker = RecursiveTextChunker(chunk_size=50, chunk_overlap=5)
        doc = Document(content=text, metadata={"source": "test"})
        chunks = chunker.split([doc])

        # 各段落が分離されているはず
        assert len(chunks) >= 2

    def test_chunk_id_increments(self):
        from rag.chunker import RecursiveTextChunker
        from rag.document_loader import Document

        chunker = RecursiveTextChunker(chunk_size=50, chunk_overlap=5)
        docs = [
            Document(content="A" * 100, metadata={"source": "a.txt"}),
            Document(content="B" * 100, metadata={"source": "b.txt"}),
        ]
        chunks = chunker.split(docs)
        ids = [c.chunk_id for c in chunks]
        assert ids == list(range(len(chunks)))


# ── DocumentLoader テスト ──────────────────────────────────────────────────

class TestDocumentLoader:
    def test_load_text_file(self, tmp_path):
        from rag.document_loader import DocumentLoader

        f = tmp_path / "test.txt"
        f.write_text("テストコンテンツ", encoding="utf-8")

        loader = DocumentLoader()
        docs = loader.load(str(f))

        assert len(docs) == 1
        assert docs[0].content == "テストコンテンツ"
        assert docs[0].metadata["type"] == "text"

    def test_load_md_file(self, tmp_path):
        from rag.document_loader import DocumentLoader

        f = tmp_path / "test.md"
        f.write_text("# タイトル\n\n内容", encoding="utf-8")

        loader = DocumentLoader()
        docs = loader.load(str(f))

        assert len(docs) == 1
        assert "タイトル" in docs[0].content

    def test_load_directory(self, tmp_path):
        from rag.document_loader import DocumentLoader

        (tmp_path / "a.txt").write_text("ファイルA", encoding="utf-8")
        (tmp_path / "b.md").write_text("ファイルB", encoding="utf-8")
        (tmp_path / "c.jpg").write_text("無視される", encoding="utf-8")

        loader = DocumentLoader()
        docs = loader.load(str(tmp_path))

        assert len(docs) == 2  # .jpg は除外される

    def test_unsupported_format_raises(self, tmp_path):
        from rag.document_loader import DocumentLoader

        f = tmp_path / "test.xyz"
        f.write_text("内容", encoding="utf-8")

        loader = DocumentLoader()
        with pytest.raises(ValueError, match="未対応の形式"):
            loader.load(str(f))


# ── FAISSVectorStore テスト ────────────────────────────────────────────────

class TestFAISSVectorStore:
    def test_add_and_search(self):
        from rag.chunker import Chunk
        from rag.vector_store import FAISSVectorStore

        dim = 64
        store = FAISSVectorStore(dimension=dim)

        chunks = [
            Chunk(content=f"チャンク {i}", metadata={}, chunk_id=i)
            for i in range(5)
        ]
        embeddings = np.random.rand(5, dim).astype("float32")
        # 正規化（コサイン類似度のため）
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

        store.add(chunks, embeddings)
        assert store.size == 5

        query = np.random.rand(dim).astype("float32")
        query /= np.linalg.norm(query)
        results = store.search(query, top_k=3)

        assert len(results) == 3
        for chunk, score in results:
            assert isinstance(score, float)
            assert -1.0 <= score <= 1.1  # コサイン類似度の範囲

    def test_save_and_load(self, tmp_path):
        from rag.chunker import Chunk
        from rag.vector_store import FAISSVectorStore

        dim = 32
        store = FAISSVectorStore(dimension=dim)
        chunks = [Chunk(content="保存テスト", metadata={"source": "x"}, chunk_id=0)]
        embeddings = np.ones((1, dim), dtype="float32")
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        store.add(chunks, embeddings)

        store.save(str(tmp_path))
        loaded = FAISSVectorStore.load(str(tmp_path))

        assert loaded.size == 1
        assert loaded._chunks[0].content == "保存テスト"


# ── CalculatorTool テスト ──────────────────────────────────────────────────

class TestCalculatorTool:
    def test_basic_arithmetic(self):
        from agent.tools.calculator import CalculatorTool

        calc = CalculatorTool()
        assert calc.run("2 + 3").output == "5"
        assert calc.run("10 - 4").output == "6"
        assert calc.run("3 * 4").output == "12"
        assert calc.run("10 / 4").output == "2.5"

    def test_power(self):
        from agent.tools.calculator import CalculatorTool

        calc = CalculatorTool()
        assert calc.run("2 ** 10").output == "1024"

    def test_math_functions(self):
        from agent.tools.calculator import CalculatorTool

        calc = CalculatorTool()
        result = calc.run("sqrt(16)")
        assert result.success
        assert float(result.output) == pytest.approx(4.0)

    def test_invalid_expression(self):
        from agent.tools.calculator import CalculatorTool

        calc = CalculatorTool()
        result = calc.run("import os")
        assert not result.success

    def test_zero_division(self):
        from agent.tools.calculator import CalculatorTool

        calc = CalculatorTool()
        result = calc.run("1 / 0")
        assert not result.success
        assert "0 除算" in result.error
