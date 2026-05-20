# RAG 構成・データ登録ガイド

RAGパイプラインの全体構成、各コンポーネントの役割、およびナレッジベースへのデータ登録手順を説明する。

---

## 1. RAG 全体アーキテクチャ

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAGPipeline                              │
│  (rag/pipeline.py — 全コンポーネントのファサード)               │
│                                                                 │
│  ┌─────────────┐  ┌──────────────────┐  ┌───────────────────┐  │
│  │DocumentLoader│→ │RecursiveText     │→ │  EmbeddingModel   │  │
│  │             │  │Chunker           │  │(multilingual-e5)  │  │
│  │.md/.txt     │  │                  │  │                   │  │
│  │.pdf/.csv    │  │512文字/64重複    │  │768次元ベクトル    │  │
│  └─────────────┘  └──────────────────┘  └─────────┬─────────┘  │
│                                                    │           │
│                                          ┌─────────▼─────────┐  │
│                                          │  FAISSVectorStore │  │
│                                          │  (IndexFlatIP)    │  │
│                                          │  index.faiss      │  │
│                                          │  chunks.pkl       │  │
│                                          └─────────┬─────────┘  │
│                                                    │           │
│                                          ┌─────────▼─────────┐  │
│                                          │  HybridRetriever  │  │
│                                          │  セマンティック検索│  │
│                                          │ +キーワードリランク│  │
│                                          └───────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### データフローの2フェーズ

| フェーズ | 処理 | トリガー |
|---------|------|---------|
| **インデックス構築** | ファイル読込 → チャンク分割 → ベクトル化 → FAISS保存 | `python main.py --mode build-index` |
| **検索（推論時）** | クエリのベクトル化 → FAISS検索 → リランキング → 文脈テキスト生成 | エージェントが`rag_search`ツールを使用 |

---

## 2. コンポーネント詳細

### 2-1. DocumentLoader（`rag/document_loader.py`）

ファイルシステムからドキュメントを読み込み、`Document` オブジェクトのリストに変換する。

**対応形式**

| 拡張子 | 読み込み方法 | メタデータ |
|--------|------------|----------|
| `.md` / `.txt` | UTF-8 テキスト全文 | `source`, `type: text` |
| `.csv` | pandas で読み込み → `df.to_string()` で文字列化 | `source`, `type: csv`, `rows` |
| `.pdf` | pypdf で各ページのテキスト抽出 → 改行2つで結合 | `source`, `type: pdf`, `pages` |

ディレクトリを渡すと対応拡張子のファイルをすべて再帰的に読み込む。読み込み失敗したファイルは警告を出してスキップされる（他ファイルの処理は継続する）。

---

### 2-2. RecursiveTextChunker（`rag/chunker.py`）

長いテキストをFAISSに登録できる小単位（チャンク）に分割する。

**設定値（`config/settings.py` / `.env` で変更可）**

| パラメータ | デフォルト | 意味 |
|-----------|----------|------|
| `CHUNK_SIZE` | `512` | 1チャンクの最大文字数 |
| `CHUNK_OVERLAP` | `64` | 前後チャンクとの重複文字数 |

**分割優先順位**

```
\n\n（段落）→ \n（改行）→ 。（句点）→ .（ピリオド）→ 、→ , → 空白 → 強制固定幅
```

意味のある区切りを優先するため、埋め込みの精度が固定幅分割より高くなる。オーバーラップは境界をまたぐ文脈の損失を防ぐ。

---

### 2-3. EmbeddingModel（`rag/embedder.py`）

テキストを768次元のベクトルに変換する。

**モデル**: `intfloat/multilingual-e5-base`（HuggingFace、約270MB）

- 日本語・英語の混在テキストに対応
- L2正規化済みベクトルを出力（`normalize_embeddings=True`）

**E5 プレフィックス**

E5モデルの学習仕様に従い、用途でプレフィックスを変える：

```python
# インデックス構築時（文書チャンク）
"passage: RAGとはRetrieval-Augmented Generationの略で..."

# 検索時（ユーザーの質問）
"query: RAGとは何ですか？"
```

これにより質問と文書が異なる文体でも高い類似度を計算できる（非対称検索）。

---

### 2-4. FAISSVectorStore（`rag/vector_store.py`）

ベクトルの格納・検索・永続化を担当する。

**インデックス種類**: `IndexFlatIP`（全件総当たり内積検索）

L2正規化済みベクトルを使うため、内積 = コサイン類似度として機能する。

**永続化ファイル**

```
vector_store/
├── index.faiss   ← FAISSインデックス本体（ベクトル）
└── chunks.pkl    ← チャンクのテキストとメタデータ（Pickleシリアライズ）
```

2ファイルがペアで機能する。どちらかが欠けると読み込みエラーになる。

---

### 2-5. HybridRetriever（`rag/retriever.py`）

セマンティック検索とキーワードマッチを組み合わせてリランキングする。

**処理ステップ**

```
Step 1: クエリをベクトル化し FAISS で top_k × 2 件を取得（コサイン類似度）
Step 2: 各チャンクとクエリの共通単語数を計算
Step 3: 最終スコア = コサイン類似度 + 共通単語数 × 0.02
Step 4: スコア降順でソートし top_k 件を返す
```

**設定値**

| パラメータ | デフォルト | 意味 |
|-----------|----------|------|
| `TOP_K` | `5` | 返す検索結果の件数 |

---

## 3. データ登録手順

### ステップ 1 — ドキュメントを配置する

```
knowledge_base/
├── ai_agents.md             ← 既存
├── linear_regression_guide.md
├── rag_explained.md
└── your_new_document.md     ← ここに追加
```

サブディレクトリも再帰的に読み込まれる：

```
knowledge_base/
├── ai/
│   └── rag_explained.md
└── ml/
    └── linear_regression_guide.md
```

**対応形式**: `.md` `.txt` `.pdf` `.csv`

---

### ステップ 2 — インデックスを（再）構築する

```bash
# knowledge_base/ を対象にインデックス構築（デフォルト）
python main.py --mode build-index

# 別のディレクトリを対象にする場合
python main.py --mode build-index --docs path/to/your/docs
```

**実行時の出力例**

```
📄 ドキュメントを読み込み中...
   → 4 ファイル読み込み完了
✂️  テキストをチャンクに分割中...
   → 87 チャンク生成
🔢 埋め込みベクトルを生成中...
   → ベクトル次元数: 768
💾 FAISS インデックスを構築・保存中...
✅ インデックス構築完了 — 87 チャンク
```

構築が完了すると `vector_store/` が作成（または上書き）される。

> **注意**: `build-index` を実行するたびに既存の `vector_store/` が上書きされる。差分更新ではなく全件再構築。

---

### ステップ 3 — チャットで確認する

```bash
python main.py
# → あなた: 追加したドキュメントの内容を教えて
```

エージェントが必要に応じて `rag_search` ツールを使い、登録した内容を参照して回答する。

---

## 4. 設定のカスタマイズ

`.env` ファイル（または環境変数）で動作を変更できる。`.env.example` をコピーして作成する：

```bash
cp .env.example .env
```

**RAG関連のパラメータ**

| 変数名 | デフォルト | 変更する場面 |
|--------|----------|------------|
| `CHUNK_SIZE` | `512` | ドキュメントが長文かつ密度が高い場合は大きくする |
| `CHUNK_OVERLAP` | `64` | 文脈の連続性が重要な場合は大きくする（最大 CHUNK_SIZE の 20% 程度） |
| `TOP_K` | `5` | 回答に使う参考資料の件数を増減する |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-base` | 精度を上げたい場合は `-large` 版に変更（要GPU） |
| `VECTOR_STORE_PATH` | `vector_store` | インデックスの保存先を変えたい場合 |

設定変更後は必ずインデックスを再構築する。

---

## 5. ファイル構成まとめ

```
model_practice/
├── knowledge_base/          ← ドキュメントをここに置く
│   ├── ai_agents.md
│   ├── linear_regression_guide.md
│   └── rag_explained.md
├── vector_store/            ← インデックス（自動生成、git管理外）
│   ├── index.faiss
│   └── chunks.pkl
├── rag/                     ← RAGパイプラインの実装
│   ├── pipeline.py          ← ファサード（エントリーポイント）
│   ├── document_loader.py   ← ファイル読み込み
│   ├── chunker.py           ← テキスト分割
│   ├── embedder.py          ← ベクトル化
│   ├── vector_store.py      ← FAISS操作
│   └── retriever.py         ← ハイブリッド検索
├── config/
│   └── settings.py          ← 全設定値の定義
├── .env                     ← 環境変数（要作成）
└── main.py                  ← CLIエントリーポイント
```
