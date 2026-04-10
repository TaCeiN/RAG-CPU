# Local Russian-first RAG Service

## Setup (.venv only)

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
```

## Run Web Service

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8090
```

- Web UI: [http://127.0.0.1:8090/](http://127.0.0.1:8090/)
- API docs: [http://127.0.0.1:8090/docs](http://127.0.0.1:8090/docs)
- Health: `GET /health`

## Features

- JWT auth: `signup/login/refresh/logout/me`
- User-isolated chats/files/messages
- Upload: `TXT/MD/PDF/DOCX`
- Background ingestion jobs + status API
- Chat ask with routing: `memory|rag|direct`
- Per-chat model overrides
- Structured JSON logs with route/retrieval/rerank/latency
- Startup warmup diagnostics for `ollama + embedding + rerank`

## Default Stack (CPU-oriented)

- LLM: `hf.co/Vikhrmodels/QVikhr-3-4B-Instruction-GGUF:Q4_K_M`
- Embeddings: `models/e5-small-en-ru`
- Rerank: `models/cross-encoder-russian-msmarco`

## Cleanup diagnostics

```powershell
.\.venv\Scripts\python.exe -m app.cleanup app_data .hf_cache .pytest_cache
```
