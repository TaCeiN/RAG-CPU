from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rag.types import Chunk, RetrievedChunk

try:
    import faiss
except ImportError:  # pragma: no cover
    faiss = None


class FaissRetrievalStore:
    INDEX_FILE = "index.faiss"
    META_FILE = "chunks.json"

    def __init__(self, index_dir: str | Path):
        if faiss is None:
            raise RuntimeError("faiss is not installed. Install faiss-cpu in .venv")
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.index_dir / self.INDEX_FILE
        self.meta_path = self.index_dir / self.META_FILE
        self.index = None
        self.chunks: list[Chunk] = []

    def build(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if not len(chunks):
            raise ValueError("No chunks to index")
        dim = int(embeddings.shape[1])
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings.astype(np.float32))
        self.index = index
        self.chunks = chunks
        self._save()

    def _save(self) -> None:
        assert self.index is not None
        faiss.write_index(self.index, str(self.index_path))
        payload = [
            {
                "chunk_id": c.chunk_id,
                "source": c.source,
                "text": c.text,
                "start_char": c.start_char,
                "end_char": c.end_char,
                "token_count": c.token_count,
                "metadata": c.metadata,
            }
            for c in self.chunks
        ]
        self.meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self) -> None:
        if not self.index_path.exists() or not self.meta_path.exists():
            raise FileNotFoundError(f"Index files not found in {self.index_dir}")
        self.index = faiss.read_index(str(self.index_path))
        raw = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self.chunks = [Chunk(**item) for item in raw]

    def retrieve(self, query_vector: np.ndarray, top_k: int) -> list[RetrievedChunk]:
        if self.index is None:
            self.load()
        assert self.index is not None
        qv = query_vector.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(qv, top_k)
        results: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self.chunks[int(idx)]
            results.append(RetrievedChunk(chunk=chunk, retrieval_score=float(score)))
        return results

