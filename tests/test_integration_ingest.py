import importlib.util

import numpy as np
import pytest

from rag.retrieval import FaissRetrievalStore
from rag.types import Chunk


@pytest.mark.skipif(importlib.util.find_spec("faiss") is None, reason="faiss not installed")
def test_ingest_like_index_build_and_load(tmp_path):
    chunks = [
        Chunk(chunk_id="c1", source="a.md", text="one", start_char=0, end_char=3, token_count=1),
        Chunk(chunk_id="c2", source="b.md", text="two", start_char=0, end_char=3, token_count=1),
    ]
    vecs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    store = FaissRetrievalStore(tmp_path)
    store.build(chunks, vecs)
    store2 = FaissRetrievalStore(tmp_path)
    store2.load()
    got = store2.retrieve(np.array([1.0, 0.0], dtype=np.float32), top_k=1)
    assert got
    assert got[0].chunk.source == "a.md"

