from pathlib import Path

import pytest

from rag.config import load_config


def test_config_loads_default():
    cfg = load_config(Path("config/default.yaml"))
    assert cfg.retrieval.top_k == 20
    assert cfg.chunk.size == 400
    assert cfg.chunk.overlap == 60


def test_config_validation_fails_for_bad_chunk(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        """
llm: {model: "m", endpoint: "http://localhost:11434"}
embedding: {model_path: "models/e5-small-en-ru", query_prompt: "query: ", query_mode: "prefix_text"}
rerank: {model_path: "models/cross-encoder-russian-msmarco", query_prompt: "", min_score: 0.1, top_n: 3, kind: "cross_encoder"}
chunk: {size: 200, overlap: 20, min_size: 300, max_size: 800}
retrieval: {top_k: 20}
memory: {history_n: 10, enable_summary: false}
limits: {max_context_tokens: 1200}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_config(bad)
