from __future__ import annotations

from pydantic import BaseModel


class ChatCreateIn(BaseModel):
    title: str


class ChatPatchIn(BaseModel):
    title: str


class AskIn(BaseModel):
    query: str


class OverrideIn(BaseModel):
    llm_model: str | None = None
    embedding_model_path: str | None = None
    rerank_model_path: str | None = None

