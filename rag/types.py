from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

RouteType = Literal["memory", "rag", "direct"]


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    source: str
    text: str
    start_char: int
    end_char: int
    token_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RouterDecision:
    route: RouteType
    reason: str
    search_query: str | None = None


@dataclass(slots=True)
class RetrievedChunk:
    chunk: Chunk
    retrieval_score: float
    rerank_score: float | None = None

