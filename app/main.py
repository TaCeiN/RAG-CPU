from __future__ import annotations

import logging
import time
import uuid

from fastapi import FastAPI
from fastapi import Request
from sqlalchemy import text, update

from app.api.routes import auth, chats, ui
from app.core.settings import settings
from app.db.base import Base
from app.db.session import SessionLocal, engine
from app.models.entities import IngestionJob
from app.services.rag_runtime import runtime_registry
from rag.logging_utils import log_event, setup_logger
from rag.runtime import ensure_local_hf_cache

logger = setup_logger(debug=settings.log_debug)


def _ensure_schema_extensions() -> None:
    with engine.begin() as conn:
        file_columns = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(files)")}
        if "summary" not in file_columns:
            conn.execute(text("ALTER TABLE files ADD COLUMN summary TEXT"))
        if "summary_status" not in file_columns:
            conn.execute(text("ALTER TABLE files ADD COLUMN summary_status VARCHAR(32) DEFAULT 'pending'"))
        if "summary_updated_at" not in file_columns:
            conn.execute(text("ALTER TABLE files ADD COLUMN summary_updated_at DATETIME"))


def create_app() -> FastAPI:
    ensure_local_hf_cache(".")
    settings.files_dir.mkdir(parents=True, exist_ok=True)
    settings.indexes_dir.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)
    _ensure_schema_extensions()

    # restart policy for in-process background jobs
    db = SessionLocal()
    try:
        db.execute(
            update(IngestionJob)
            .where(IngestionJob.status == "running")
            .values(status="failed", error_text="service restart during ingestion")
        )
        db.commit()
    finally:
        db.close()

    app = FastAPI(title=settings.app_name)

    @app.middleware("http")
    async def request_logging(request: Request, call_next):
        request_id = uuid.uuid4().hex[:12]
        started = time.perf_counter()
        response = await call_next(request)
        latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
        log_event(
            logger,
            "request_completed",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            latency_ms=latency_ms,
        )
        response.headers["X-Request-ID"] = request_id
        return response

    app.include_router(auth.router)
    app.include_router(chats.router)
    app.include_router(ui.router)

    @app.on_event("startup")
    def startup_log():
        cfg = runtime_registry.base_config
        warmup = {
            "llm": {"ok": False, "detail": None},
            "embedding": {"ok": False, "detail": None},
            "rerank": {"ok": False, "detail": None},
        }

        llm = runtime_registry.get_llm(cfg.llm.model)
        try:
            warmup["llm"]["detail"] = llm.probe()
            warmup["llm"]["ok"] = True
        except Exception as exc:
            warmup["llm"]["detail"] = str(exc)

        try:
            emb = runtime_registry.get_embedding(cfg.embedding.model_path)
            _ = emb.embed_query("warmup query")
            warmup["embedding"]["ok"] = True
            warmup["embedding"]["detail"] = "loaded"
        except Exception as exc:
            warmup["embedding"]["detail"] = str(exc)

        try:
            reranker = runtime_registry.get_reranker(cfg.rerank.model_path)
            mode = getattr(reranker, "mode", "unknown")
            model_error = getattr(reranker, "model_error", None)
            warmup["rerank"]["ok"] = mode == "cross_encoder" and model_error is None
            warmup["rerank"]["detail"] = {"mode": mode, "model_error": model_error}
        except Exception as exc:
            warmup["rerank"]["detail"] = str(exc)

        log_event(
            logger,
            "service_ready",
            db_url=settings.db_url,
            files_dir=str(settings.files_dir),
            indexes_dir=str(settings.indexes_dir),
            llm_model=cfg.llm.model,
            emb_model=cfg.embedding.model_path,
            rerank_model=cfg.rerank.model_path,
            model_warmup=warmup,
        )

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


app = create_app()
