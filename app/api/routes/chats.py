from __future__ import annotations

import shutil
import time
import logging
import re
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.settings import settings
from app.db.session import SessionLocal, get_db
from app.models.entities import Chat, ChatMessage, ChatModelOverride, FileRecord, IngestionJob, User
from app.schemas.chat import AskIn, ChatCreateIn, ChatPatchIn, OverrideIn
from app.services.rag_runtime import runtime_registry
from rag.generation import build_messages
from rag.ingestion import build_chunks, read_file, read_file_preserve_lines
from rag.retrieval import FaissRetrievalStore
from rag.router import route_query
from rag.types import RouterDecision
from rag.logging_utils import log_event

router = APIRouter(tags=["chats"])
logger = logging.getLogger("rag")

ALLOWED_EXT = {".txt", ".md", ".pdf", ".docx"}
OVERVIEW_PATTERNS = (
    "что в документах",
    "что в файлах",
    "что я загрузил",
    "что загружено",
    "что в документе",
    "о чем этот файл",
    "о чем этот документ",
    "кратко по каждому документу",
    "по каждому документу",
)
SOURCE_LIST_PATTERNS = (
    "список источников",
    "используемых источников",
    "использованных источников",
    "список использованных источников",
    "список литературы",
    "литература",
    "references",
    "bibliography",
)
BIBLIO_SECTION_HEADINGS = (
    "список использованных источников",
    "список используемых источников",
    "список источников",
    "список литературы",
    "использованные источники",
    "используемые источники",
    "литература",
    "references",
    "bibliography",
)
BIBLIO_STOP_HEADINGS = (
    "приложение",
    "appendix",
)


def _chat_or_404(db: Session, user_id: int, chat_id: int) -> Chat:
    chat = db.scalar(select(Chat).where(Chat.id == chat_id, Chat.user_id == user_id))
    if not chat:
        raise HTTPException(status_code=404, detail="chat not found")
    return chat


def _chat_paths(chat_id: int) -> tuple[Path, Path]:
    files_dir = settings.files_dir / f"chat_{chat_id}"
    index_dir = settings.indexes_dir / f"chat_{chat_id}"
    files_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)
    return files_dir, index_dir


def _clip_text(text: str, limit: int = 4000) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit]


def _build_summary_messages(file_name: str, text: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Ты делаешь краткие карточки документов на русском языке. "
                "Не выдумывай. Укажи: что это за документ, основные темы, важные имена или сущности, "
                "и является ли это шаблоном, примером или заполненным документом, если это можно понять."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Название файла: {file_name}\n\n"
                "Сделай краткое summary в 2-4 предложениях.\n\n"
                f"Текст документа:\n{_clip_text(text)}"
            ),
        },
    ]


def _is_overview_query(query: str, indexed_files: list[FileRecord]) -> bool:
    text = query.strip().lower()
    if not indexed_files:
        return False
    if len(indexed_files) == 1 and (
        "этот файл" in text
        or "этот документ" in text
        or "о чем файл" in text
        or "о чем документ" in text
    ):
        return True
    return any(pattern in text for pattern in OVERVIEW_PATTERNS)


def _build_overview_context(query: str, indexed_files: list[FileRecord]) -> tuple[list[str], list[str]]:
    summaries: list[str] = []
    sources: list[str] = []
    for row in indexed_files:
        summary_text = row.summary or "Summary еще не готово; известен только факт загрузки документа."
        summaries.append(f"Документ: {row.original_name}\nКраткое summary: {summary_text}")
        sources.append(row.original_name)
    return summaries, sources


def _ensure_file_summary(db: Session, file_row: FileRecord, llm_model: str) -> None:
    if file_row.summary:
        return
    try:
        llm = runtime_registry.get_llm(llm_model)
        raw_text = read_file(Path(file_row.storage_path))
        file_row.summary_status = "running"
        db.commit()
        file_row.summary = llm.chat(_build_summary_messages(file_row.original_name, raw_text))
        file_row.summary_status = "done"
        file_row.summary_updated_at = datetime.utcnow()
        db.commit()
        log_event(logger, "summary_backfilled", file_id=file_row.id, file_name=file_row.original_name)
    except Exception as exc:
        file_row.summary_status = "failed"
        db.commit()
        log_event(logger, "summary_backfill_failed", file_id=file_row.id, file_name=file_row.original_name, error=str(exc))


def _is_source_list_query(query: str) -> bool:
    text = query.strip().lower()
    return any(pattern in text for pattern in SOURCE_LIST_PATTERNS)


def _extract_bibliography_section(text: str) -> str | None:
    if not text.strip():
        return None
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    start_idx: int | None = None
    for idx, line in enumerate(lines):
        normalized = line.lower().strip(" :.-")
        if any(normalized.startswith(heading) for heading in BIBLIO_SECTION_HEADINGS):
            start_idx = idx
            break
    if start_idx is None:
        return None

    collected: list[str] = [lines[start_idx]]
    for line in lines[start_idx + 1 :]:
        normalized = line.lower().strip(" :.-")
        if any(normalized.startswith(heading) for heading in BIBLIO_STOP_HEADINGS):
            break
        collected.append(line)

    section = "\n".join(collected).strip()
    return section if section else None


def _build_bibliography_context(indexed_files: list[FileRecord]) -> tuple[list[str], list[str]]:
    sections: list[str] = []
    sources: list[str] = []
    for row in indexed_files:
        try:
            raw_text = read_file_preserve_lines(Path(row.storage_path))
        except Exception:
            continue
        section = _extract_bibliography_section(raw_text)
        if not section:
            continue
        sections.append(f"Документ: {row.original_name}\nРаздел со списком источников:\n{section}")
        sources.append(row.original_name)
    return sections, sources


def _build_bibliography_answer_messages(query: str, context: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Ты отвечаешь по разделу со списком источников. "
                "Если в контексте есть нумерованный или маркированный список, перечисли все найденные пункты без сокращений. "
                "Не придумывай и не обобщай список. Если источник не виден в контексте, не добавляй его."
            ),
        },
        {
            "role": "user",
            "content": f"Контекст:\n{context}\n\nВопрос:\n{query}",
        },
    ]


def _run_ingestion_job(job_id: int) -> None:
    db = SessionLocal()
    try:
        job = db.get(IngestionJob, job_id)
        if not job:
            return
        job.status = "running"
        db.commit()
        log_event(logger, "ingestion_started", job_id=job.id, chat_id=job.chat_id, file_id=job.file_id)

        file_row = db.get(FileRecord, job.file_id)
        if not file_row:
            job.status = "failed"
            job.error_text = "file not found"
            db.commit()
            return

        _, index_dir = _chat_paths(job.chat_id)
        cfg = runtime_registry.base_config
        raw_text = read_file(Path(file_row.storage_path))
        chunks = build_chunks(
            input_path=file_row.storage_path,
            embedding_model_path=cfg.embedding.model_path,
            chunk_size=cfg.chunk.size,
            chunk_overlap=cfg.chunk.overlap,
        )
        if not chunks:
            job.status = "failed"
            job.error_text = "no chunks produced"
            file_row.status = "failed"
            db.commit()
            return

        emb = runtime_registry.get_embedding(cfg.embedding.model_path)
        vectors = emb.embed_documents([c.text for c in chunks])
        store = FaissRetrievalStore(index_dir=index_dir)
        store.build(chunks=chunks, embeddings=vectors)

        file_row.summary_status = "running"
        db.commit()
        try:
            llm = runtime_registry.get_llm(cfg.llm.model)
            file_row.summary = llm.chat(_build_summary_messages(file_row.original_name, raw_text))
            file_row.summary_status = "done"
            file_row.summary_updated_at = datetime.utcnow()
        except Exception as exc:
            file_row.summary = None
            file_row.summary_status = "failed"
            log_event(logger, "summary_failed", job_id=job.id, chat_id=job.chat_id, file_id=job.file_id, error=str(exc))

        file_row.status = "indexed"
        job.status = "done"
        db.commit()
        log_event(
            logger,
            "ingestion_done",
            job_id=job.id,
            chat_id=job.chat_id,
            file_id=job.file_id,
            chunks=len(chunks),
            summary_status=file_row.summary_status,
        )
    except Exception as exc:
        job = db.get(IngestionJob, job_id)
        if job:
            job.status = "failed"
            job.error_text = str(exc)
            db.commit()
            log_event(logger, "ingestion_failed", job_id=job.id, chat_id=job.chat_id, error=str(exc))
    finally:
        db.close()


@router.get("/chats")
def list_chats(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = db.scalars(select(Chat).where(Chat.user_id == user.id).order_by(Chat.created_at.desc())).all()
    return [{"id": r.id, "title": r.title, "created_at": r.created_at.isoformat()} for r in rows]


@router.post("/chats")
def create_chat(payload: ChatCreateIn, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    chat = Chat(user_id=user.id, title=payload.title.strip() or "Новый чат")
    db.add(chat)
    db.commit()
    db.refresh(chat)
    _chat_paths(chat.id)
    return {"id": chat.id, "title": chat.title}


@router.get("/chats/{chat_id}")
def get_chat(chat_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    chat = _chat_or_404(db, user.id, chat_id)
    return {"id": chat.id, "title": chat.title, "created_at": chat.created_at.isoformat()}


@router.patch("/chats/{chat_id}")
def patch_chat(chat_id: int, payload: ChatPatchIn, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    chat = _chat_or_404(db, user.id, chat_id)
    chat.title = payload.title.strip() or chat.title
    db.commit()
    return {"id": chat.id, "title": chat.title}


@router.delete("/chats/{chat_id}")
def delete_chat(chat_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    chat = _chat_or_404(db, user.id, chat_id)
    files_dir, index_dir = _chat_paths(chat_id)
    db.delete(chat)
    db.commit()
    shutil.rmtree(files_dir, ignore_errors=True)
    shutil.rmtree(index_dir, ignore_errors=True)
    return {"status": "ok"}


@router.post("/chats/{chat_id}/override")
def set_override(chat_id: int, payload: OverrideIn, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    _chat_or_404(db, user.id, chat_id)
    row = db.scalar(select(ChatModelOverride).where(ChatModelOverride.chat_id == chat_id))
    if row is None:
        row = ChatModelOverride(chat_id=chat_id)
        db.add(row)
    row.llm_model = payload.llm_model
    row.embedding_model_path = payload.embedding_model_path
    row.rerank_model_path = payload.rerank_model_path
    db.commit()
    return {
        "chat_id": chat_id,
        "llm_model": row.llm_model,
        "embedding_model_path": row.embedding_model_path,
        "rerank_model_path": row.rerank_model_path,
    }


@router.post("/chats/{chat_id}/files")
def upload_file(
    chat_id: int,
    background_tasks: BackgroundTasks,
    uploaded: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _chat_or_404(db, user.id, chat_id)
    suffix = Path(uploaded.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXT:
        raise HTTPException(status_code=400, detail="unsupported file type")
    files_dir, _ = _chat_paths(chat_id)
    ts = int(time.time() * 1000)
    dst = files_dir / f"{ts}_{Path(uploaded.filename).name}"
    with dst.open("wb") as fh:
        shutil.copyfileobj(uploaded.file, fh)
    row = FileRecord(chat_id=chat_id, original_name=uploaded.filename or dst.name, storage_path=str(dst), status="uploaded")
    db.add(row)
    db.commit()
    db.refresh(row)

    job = IngestionJob(chat_id=chat_id, file_id=row.id, status="pending")
    db.add(job)
    db.commit()
    db.refresh(job)
    background_tasks.add_task(_run_ingestion_job, job.id)
    return {"file_id": row.id, "job_id": job.id, "status": "pending"}


@router.get("/chats/{chat_id}/files")
def list_files(chat_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    _chat_or_404(db, user.id, chat_id)
    rows = db.scalars(select(FileRecord).where(FileRecord.chat_id == chat_id)).all()
    return [
        {
            "id": r.id,
            "name": r.original_name,
            "status": r.status,
            "summary_status": r.summary_status,
            "summary": r.summary,
        }
        for r in rows
    ]


@router.delete("/chats/{chat_id}/files/{file_id}")
def delete_file(chat_id: int, file_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    _chat_or_404(db, user.id, chat_id)
    row = db.scalar(select(FileRecord).where(FileRecord.id == file_id, FileRecord.chat_id == chat_id))
    if not row:
        raise HTTPException(status_code=404, detail="file not found")
    try:
        Path(row.storage_path).unlink(missing_ok=True)
    except Exception:
        pass
    db.delete(row)
    db.commit()
    return {"status": "ok"}


@router.get("/jobs/{job_id}")
def get_job(job_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    row = db.get(IngestionJob, job_id)
    if not row:
        raise HTTPException(status_code=404, detail="job not found")
    _chat_or_404(db, user.id, row.chat_id)
    return {"id": row.id, "status": row.status, "error_text": row.error_text}


@router.get("/chats/{chat_id}/messages")
def list_messages(chat_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    _chat_or_404(db, user.id, chat_id)
    rows = db.scalars(select(ChatMessage).where(ChatMessage.chat_id == chat_id).order_by(ChatMessage.id.asc())).all()
    return [{"id": r.id, "role": r.role, "content": r.content, "created_at": r.created_at.isoformat()} for r in rows]


@router.post("/chats/{chat_id}/ask")
def ask_chat(chat_id: int, payload: AskIn, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    started = time.perf_counter()
    _chat_or_404(db, user.id, chat_id)
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is empty")
    db.add(ChatMessage(chat_id=chat_id, role="user", content=query))
    db.commit()

    decision = route_query(query)
    log_event(logger, "ask_route_selected", user_id=user.id, chat_id=chat_id, route=decision.route, reason=decision.reason)
    retrieved_sources: list[str] = []
    context_parts: list[str] = []
    indexed_files = db.scalars(
        select(FileRecord).where(FileRecord.chat_id == chat_id, FileRecord.status == "indexed").order_by(FileRecord.id.asc())
    ).all()
    override = db.scalar(select(ChatModelOverride).where(ChatModelOverride.chat_id == chat_id))
    profile = runtime_registry.profile(
        llm_model=(override.llm_model if override else None),
        emb_model=(override.embedding_model_path if override else None),
        rerank_model=(override.rerank_model_path if override else None),
    )

    if decision.route == "memory":
        history_n = runtime_registry.base_config.memory.history_n
        rows = db.scalars(
            select(ChatMessage).where(ChatMessage.chat_id == chat_id).order_by(ChatMessage.id.desc()).limit(history_n)
        ).all()
        rows = list(reversed(rows))
        context_parts = [f"{r.role}: {r.content}" for r in rows]
    elif decision.route == "rag":
        if _is_source_list_query(query):
            context_parts, retrieved_sources = _build_bibliography_context(indexed_files)
            if context_parts:
                decision.reason = "exact bibliography section extraction"
                log_event(
                    logger,
                    "ask_bibliography_extraction",
                    user_id=user.id,
                    chat_id=chat_id,
                    files=len(retrieved_sources),
                )
        if not context_parts and _is_overview_query(query, indexed_files):
            for row in indexed_files:
                if not row.summary:
                    _ensure_file_summary(db, row, profile.llm_model)
            context_parts, retrieved_sources = _build_overview_context(query, indexed_files)
            decision.reason = "document summaries overview"
            log_event(
                logger,
                "ask_summary_overview",
                user_id=user.id,
                chat_id=chat_id,
                files=len(retrieved_sources),
            )
        if not context_parts:
            _, index_dir = _chat_paths(chat_id)
            store = FaissRetrievalStore(index_dir=index_dir)
            emb = runtime_registry.get_embedding(profile.embedding_model_path)
            reranker = runtime_registry.get_reranker(profile.rerank_model_path)
            query_vec = emb.embed_query(decision.search_query or query)
            candidates = store.retrieve(query_vec, top_k=runtime_registry.base_config.retrieval.top_k)
            reranked = reranker.rerank(query, candidates)
            log_event(
                logger,
                "ask_retrieval",
                user_id=user.id,
                chat_id=chat_id,
                candidates=len(candidates),
                reranked=len(reranked),
            )
            if not reranked:
                decision = RouterDecision(route="direct", reason="rag fallback: no relevant chunks", search_query=None)
            else:
                matched_sources = {item.chunk.source for item in reranked}
                for row in indexed_files:
                    if not row.summary and str(Path(row.storage_path)) in matched_sources:
                        _ensure_file_summary(db, row, profile.llm_model)
                summary_parts = [
                    f"Краткое summary документа {row.original_name}: {row.summary}"
                    for row in indexed_files
                    if row.summary and str(Path(row.storage_path)) in matched_sources
                ]
                context_parts = summary_parts + [item.chunk.text for item in reranked]
                retrieved_sources = [item.chunk.source for item in reranked]

    max_ctx = runtime_registry.base_config.limits.max_context_tokens
    context = None
    if context_parts:
        out: list[str] = []
        total = 0
        for part in context_parts:
            toks = len(part.split())
            if total + toks > max_ctx:
                break
            out.append(part)
            total += toks
        context = "\n\n".join(out)

    llm = runtime_registry.get_llm(profile.llm_model)
    if _is_source_list_query(query) and context:
        messages = _build_bibliography_answer_messages(query, context)
    else:
        messages = build_messages(query, context)
    try:
        answer = llm.chat(messages)
    except Exception as exc:
        log_event(logger, "ask_llm_error", user_id=user.id, chat_id=chat_id, error=str(exc))
        answer = f"Локальная LLM недоступна: {exc}"

    db.add(ChatMessage(chat_id=chat_id, role="assistant", content=answer))
    db.commit()
    latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
    log_event(
        logger,
        "ask_completed",
        user_id=user.id,
        chat_id=chat_id,
        route=decision.route,
        retrieval_hits=len(retrieved_sources),
        latency_ms=latency_ms,
    )

    return {
        "answer": answer,
        "route": decision.route,
        "reason": decision.reason,
        "search_query": decision.search_query,
        "retrieved_sources": retrieved_sources,
        "latency_ms": latency_ms,
    }
