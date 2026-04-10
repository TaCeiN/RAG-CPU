# AI Agent Specification: Local Russian RAG Service

## Overview

This document defines architecture for a fully local Russian-first RAG service.

Goals:
- fully local runtime (no cloud dependencies in main flow)
- router-first execution (`memory|rag|direct`)
- modular components (`llm`, `embeddings`, `retrieval`, `rerank`, `memory`)
- predictable latency on CPU-limited hardware

## Core Components

### 1. LLM (Generation)
- Runtime: Ollama
- Default model: `QVikhr-3-4B-Instruction` (GGUF via Ollama)
- Responsibility: final answer generation from prepared context

### 2. Embedding Model
- Default model: `e5-small-en-ru`
- Responsibility: query/document vectorization for semantic search

### 3. Retrieval
- Storage: FAISS
- Responsibility: high-recall candidate chunk retrieval (`top_k`)

### 4. Reranker
- Default model: `cross-encoder-russian-msmarco`
- Responsibility: precision filtering and ordering (`top_n`)

### 5. Memory
- Short-term chat history (last `N` messages)
- Optional summaries for uploaded documents

### 6. Router
Returns JSON contract:

```json
{
  "route": "memory | rag | direct",
  "reason": "short explanation",
  "search_query": "optional refined query"
}
```

## Execution Flow

1. User query
2. Router decision
3. Route branch:
- `memory`: history-based response
- `rag`: embeddings -> retrieval -> rerank -> context assembly
- `direct`: LLM-only response
4. LLM generation
5. Persist assistant message

## RAG Constraints

- Chunk size: `300..800`
- Chunk overlap: `10..20%`
- Retrieval returns candidates only
- Reranker filters noise before LLM context assembly
- Context is hard-limited by token budget

## Reliability Rules

- Empty retrieval -> safe fallback (`direct` or summary path)
- Weak rerank -> fallback to top candidates (bounded)
- LLM unavailable -> return explicit local error
- Context overflow -> truncate deterministically

## Performance Rules

- Reuse loaded model instances
- Keep context short and source-grounded
- Avoid unnecessary RAG calls for pure dialogue queries
- Use document summaries for broad "what is in my files" requests
