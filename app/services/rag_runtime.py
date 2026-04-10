from __future__ import annotations

import threading
from dataclasses import dataclass

from rag.logging_utils import log_event
from rag.config import load_config
from rag.embeddings import EmbeddingService
from rag.generation import OllamaClient
from rag.rerank import Reranker
import logging

from app.core.settings import settings

logger = logging.getLogger("rag")


@dataclass
class RuntimeProfile:
    llm_model: str
    embedding_model_path: str
    rerank_model_path: str


class RuntimeRegistry:
    def __init__(self):
        self._lock = threading.RLock()
        self._embeddings: dict[str, EmbeddingService] = {}
        self._rerankers: dict[str, Reranker] = {}
        self._llms: dict[str, OllamaClient] = {}
        self._base_cfg = load_config(settings.default_config_path)

    @property
    def base_config(self):
        return self._base_cfg

    def profile(self, llm_model: str | None, emb_model: str | None, rerank_model: str | None) -> RuntimeProfile:
        return RuntimeProfile(
            llm_model=llm_model or self._base_cfg.llm.model,
            embedding_model_path=emb_model or self._base_cfg.embedding.model_path,
            rerank_model_path=rerank_model or self._base_cfg.rerank.model_path,
        )

    def get_embedding(self, path: str) -> EmbeddingService:
        with self._lock:
            if path not in self._embeddings:
                log_event(logger, "embedding_loading", model_path=path)
                self._embeddings[path] = EmbeddingService(
                    path,
                    query_prompt=self._base_cfg.embedding.query_prompt,
                    query_mode=self._base_cfg.embedding.query_mode,
                )
                log_event(logger, "embedding_loaded", model_path=path)
            return self._embeddings[path]

    def get_reranker(self, path: str) -> Reranker:
        with self._lock:
            if path not in self._rerankers:
                log_event(logger, "reranker_loading", model_path=path)
                self._rerankers[path] = Reranker(
                    path,
                    query_prompt=self._base_cfg.rerank.query_prompt,
                    min_score=self._base_cfg.rerank.min_score,
                    top_n=self._base_cfg.rerank.top_n,
                    kind=self._base_cfg.rerank.kind,
                )
                log_event(logger, "reranker_loaded", model_path=path)
            return self._rerankers[path]

    def get_llm(self, model: str) -> OllamaClient:
        with self._lock:
            key = f"{self._base_cfg.llm.endpoint}::{model}"
            if key not in self._llms:
                log_event(logger, "llm_client_init", endpoint=self._base_cfg.llm.endpoint, model=model)
                self._llms[key] = OllamaClient(self._base_cfg.llm.endpoint, model, self._base_cfg.llm.timeout_seconds)
            return self._llms[key]


runtime_registry = RuntimeRegistry()
