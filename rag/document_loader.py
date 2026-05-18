from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class Document:
    """RAGパイプラインの最小単位。ローカルファイルから読み込まれた生テキスト。"""

    content: str
    metadata: dict = field(default_factory=dict)


# サポートする拡張子
_SUPPORTED = {".txt", ".md", ".pdf", ".csv"}


class DocumentLoader:
    """複数形式のファイル・ディレクトリを読み込み Document リストを返す。

    対応形式:
        .txt / .md   : UTF-8 テキスト全文
        .csv         : pandas で読み込み、文字列化
        .pdf         : pypdf で各ページのテキスト抽出
    """

    def load(self, path: str | Path) -> List[Document]:
        path = Path(path)
        if path.is_dir():
            return self._load_directory(path)
        return [self._load_file(path)]

    # ── プライベート ──────────────────────────────────────────

    def _load_file(self, path: Path) -> Document:
        dispatch = {
            ".txt": self._load_text,
            ".md": self._load_text,
            ".csv": self._load_csv,
            ".pdf": self._load_pdf,
        }
        loader_fn = dispatch.get(path.suffix.lower())
        if not loader_fn:
            raise ValueError(f"未対応の形式: {path.suffix}  (対応: {_SUPPORTED})")
        return loader_fn(path)

    def _load_text(self, path: Path) -> Document:
        content = path.read_text(encoding="utf-8")
        return Document(content=content, metadata={"source": str(path), "type": "text"})

    def _load_csv(self, path: Path) -> Document:
        import pandas as pd

        df = pd.read_csv(path)
        content = df.to_string(index=False)
        return Document(
            content=content,
            metadata={"source": str(path), "type": "csv", "rows": len(df)},
        )

    def _load_pdf(self, path: Path) -> Document:
        try:
            import pypdf
        except ImportError as exc:
            raise ImportError("PDF 読み込みには pypdf が必要です: pip install pypdf") from exc

        with open(path, "rb") as f:
            reader = pypdf.PdfReader(f)
            pages_text = [page.extract_text() or "" for page in reader.pages]

        content = "\n\n".join(pages_text)
        return Document(
            content=content,
            metadata={"source": str(path), "type": "pdf", "pages": len(reader.pages)},
        )

    def _load_directory(self, dir_path: Path) -> List[Document]:
        docs: List[Document] = []
        for file_path in sorted(dir_path.rglob("*")):
            if file_path.is_file() and file_path.suffix.lower() in _SUPPORTED:
                try:
                    docs.append(self._load_file(file_path))
                except Exception as exc:
                    print(f"  [警告] {file_path.name} の読み込みをスキップ: {exc}")
        return docs
