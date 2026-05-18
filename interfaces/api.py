"""FastAPI による REST API インターフェース。
python main.py --mode api で起動する (デフォルト: localhost:8000)。

エンドポイント一覧:
    POST /chat          : エージェントとのチャット
    POST /index/build   : RAG インデックスの再構築
    GET  /health        : ヘルスチェック
    GET  /docs          : Swagger UI (自動生成)
"""
from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agent.core.agent import ReActAgent

app = FastAPI(
    title="AI Agent API",
    description="RAG + ReAct 搭載 AI エージェント REST API",
    version="1.0.0",
)

# ── セッション管理 (簡易インメモリ) ──────────────────────────────────────
_agents: dict[str, ReActAgent] = {}


def _get_agent(session_id: str) -> ReActAgent:
    if session_id not in _agents:
        from interfaces.cli import build_agent

        _agents[session_id] = build_agent(verbose=False)
    return _agents[session_id]


# ── スキーマ ──────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    verbose: bool = False


class ChatResponse(BaseModel):
    response: str
    session_id: str


class BuildIndexRequest(BaseModel):
    docs_path: str = "knowledge_base"


# ── エンドポイント ────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        agent = _get_agent(request.session_id)
        response = agent.run(request.message)
        return ChatResponse(response=response, session_id=request.session_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/index/build")
async def build_index(request: BuildIndexRequest) -> dict:
    from rag.pipeline import RAGPipeline

    pipeline = RAGPipeline()
    count = pipeline.build_index(request.docs_path)
    # インデックス更新後は全セッションのエージェントをリセット
    _agents.clear()
    return {"status": "success", "chunks_indexed": count}


@app.delete("/session/{session_id}")
async def delete_session(session_id: str) -> dict:
    _agents.pop(session_id, None)
    return {"status": "deleted", "session_id": session_id}


@app.get("/health")
async def health() -> dict:
    return {"status": "healthy", "active_sessions": len(_agents)}
