from __future__ import annotations

import threading
from typing import Sequence

import numpy as np
from rag.types import RetrievedChunk


class Reranker:
    def __init__(self, model_path: str, query_prompt: str, min_score: float, top_n: int, kind: str = "auto"):
        self.model_path = model_path
        self.query_prompt = query_prompt
        self.min_score = min_score
        self.top_n = top_n
        self.kind = kind
        self.mode = "lexical"
        self.model_error: str | None = None
        self._lock = threading.RLock()

        self.model = None
        self.tokenizer = None

        if kind not in {"cross_encoder", "auto"}:
            self.model_error = f"unsupported rerank kind: {kind}"
            return
        try:
            self._init_cross_encoder(model_path)
            self.mode = "cross_encoder"
        except Exception as exc:
            self.model_error = str(exc)

    def _init_cross_encoder(self, model_path: str) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
        self.model.eval()

    @staticmethod
    def _lexical_score(query: str, text: str) -> float:
        q = set(query.lower().split())
        t = set(text.lower().split())
        if not q or not t:
            return 0.0
        return len(q & t) / max(1, len(q))

    def _score_cross_encoder(self, query: str, candidates: Sequence[RetrievedChunk]) -> list[float]:
        import torch

        pairs = [[query, c.chunk.text] for c in candidates]
        with self._lock:
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

        if logits.ndim == 2 and logits.shape[1] > 1:
            scores = logits[:, 1]
        else:
            scores = logits.squeeze(-1)
        return scores.detach().cpu().numpy().astype(np.float32).tolist()

    def rerank(self, query: str, candidates: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
        if not candidates:
            return []

        if self.mode == "cross_encoder" and self.model is not None and self.tokenizer is not None:
            scores = self._score_cross_encoder(query, candidates)
        else:
            scores = [self._lexical_score(query, c.chunk.text) for c in candidates]

        scored: list[RetrievedChunk] = []
        for cand, score in zip(candidates, scores):
            cand.rerank_score = float(score)
            scored.append(cand)

        scored.sort(key=lambda c: c.rerank_score if c.rerank_score is not None else -1.0, reverse=True)
        filtered = [c for c in scored if (c.rerank_score is not None and c.rerank_score >= self.min_score)]
        if filtered:
            return filtered[: self.top_n]
        # If all scores are below threshold, still return top candidates to avoid empty-context direct fallback.
        return scored[: self.top_n]
