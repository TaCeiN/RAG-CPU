from __future__ import annotations

from typing import Sequence

import numpy as np


class EmbeddingService:
    def __init__(self, model_path: str, query_prompt: str, query_mode: str = "auto"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            model_path,
            trust_remote_code=True,
            model_kwargs={"trust_remote_code": True},
            config_kwargs={"trust_remote_code": True},
        )
        self.query_prompt = query_prompt
        self.query_mode = query_mode
        self.model_path = model_path.lower()

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self.model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        if self.query_mode == "prefix_text" or (self.query_mode == "auto" and "e5" in self.model_path):
            text = f"{self.query_prompt}{query}" if self.query_prompt else query
            vectors = self.model.encode(
                [text],
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        else:
            try:
                vectors = self.model.encode(
                    [query],
                    prompt=self.query_prompt,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
            except TypeError:
                text = f"{self.query_prompt}{query}" if self.query_prompt else query
                vectors = self.model.encode(
                    [text],
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
        return np.asarray(vectors[0], dtype=np.float32)
