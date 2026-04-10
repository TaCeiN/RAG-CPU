# Model Stack Recommendation (CPU, <=60s avg)

## Recommended default
- LLM: `hf.co/Vikhrmodels/QVikhr-3-4B-Instruction-GGUF:Q4_K_M`
- Embeddings: `e5-small-en-ru`
- Rerank: `cross-encoder-russian-msmarco`

Why:
- Good Russian quality/latency balance on CPU-only machines.
- Keeps RAG grounding quality with strong rerank.

## Fast profile
- LLM: `hf.co/Vikhrmodels/Vikhr-Qwen-2.5-1.5B-Instruct-GGUF:Q4_K_M`
- Embeddings: `e5-small-en-ru`
- Rerank: `cross-encoder-russian-msmarco`
- Use `config/fast.yaml`

Use when average latency is above target.

## Quality profile
- LLM: `hf.co/Vikhrmodels/QVikhr-3-4B-Instruction-GGUF:Q4_K_M`
- Embeddings: `e5-small-en-ru`
- Rerank: `cross-encoder-russian-msmarco`
- Use `config/quality.yaml`

Use when quality is priority and latency budget still acceptable.

## Notes
- `e5-small-en-ru` is intentionally kept as default for speed/quality balance in retrieval.
- Rerank remains mandatory for precision in long documents.
- For broad "what in all documents" questions, service uses per-file summaries before deep RAG.
