from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(slots=True)
class LLMConfig:
    model: str
    endpoint: str
    timeout_seconds: int = 60


@dataclass(slots=True)
class EmbeddingConfig:
    model_path: str
    query_prompt: str
    query_mode: str = "auto"


@dataclass(slots=True)
class RerankConfig:
    model_path: str
    query_prompt: str
    min_score: float
    top_n: int
    kind: str = "auto"


@dataclass(slots=True)
class ChunkConfig:
    size: int
    overlap: int
    min_size: int = 300
    max_size: int = 800

    def validate(self) -> None:
        if self.size < self.min_size or self.size > self.max_size:
            raise ValueError(f"chunk.size must be in [{self.min_size}, {self.max_size}]")
        if self.overlap < 0:
            raise ValueError("chunk.overlap must be >= 0")
        if self.overlap >= self.size:
            raise ValueError("chunk.overlap must be < chunk.size")


@dataclass(slots=True)
class RetrievalConfig:
    top_k: int


@dataclass(slots=True)
class MemoryConfig:
    history_n: int
    enable_summary: bool = False


@dataclass(slots=True)
class LimitsConfig:
    max_context_tokens: int


@dataclass(slots=True)
class PathsConfig:
    history_file: str = "chat_history.jsonl"


@dataclass(slots=True)
class AppConfig:
    llm: LLMConfig
    embedding: EmbeddingConfig
    rerank: RerankConfig
    chunk: ChunkConfig
    retrieval: RetrievalConfig
    memory: MemoryConfig
    limits: LimitsConfig
    paths: PathsConfig


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    chunk_cfg = ChunkConfig(**raw["chunk"])
    chunk_cfg.validate()
    return AppConfig(
        llm=LLMConfig(**raw["llm"]),
        embedding=EmbeddingConfig(**raw["embedding"]),
        rerank=RerankConfig(**raw["rerank"]),
        chunk=chunk_cfg,
        retrieval=RetrievalConfig(**raw["retrieval"]),
        memory=MemoryConfig(**raw["memory"]),
        limits=LimitsConfig(**raw["limits"]),
        paths=PathsConfig(**raw.get("paths", {})),
    )
