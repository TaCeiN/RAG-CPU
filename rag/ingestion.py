from __future__ import annotations

import re
import zipfile
from pathlib import Path

from rag.types import Chunk

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}


def iter_supported_files(input_path: str | Path) -> list[Path]:
    path = Path(input_path)
    if path.is_file():
        return [path] if path.suffix.lower() in SUPPORTED_EXTENSIONS else []
    return [p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf_file(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install pypdf to read PDF files") from exc
    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "") for page in reader.pages]
    return "\n".join(pages)


def read_docx_file(path: Path) -> str:
    try:
        from docx import Document

        return "\n".join(p.text for p in Document(str(path)).paragraphs if p.text)
    except Exception:
        with zipfile.ZipFile(path, "r") as archive:
            xml = archive.read("word/document.xml").decode("utf-8", errors="ignore")
        xml = re.sub(r"<[^>]+>", " ", xml)
        return xml


def read_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        text = read_text_file(path)
    elif suffix == ".pdf":
        text = read_pdf_file(path)
    elif suffix == ".docx":
        text = read_docx_file(path)
    else:
        text = ""
    return normalize_text(text)


def read_file_preserve_lines(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        text = read_text_file(path)
    elif suffix == ".pdf":
        text = read_pdf_file(path)
    elif suffix == ".docx":
        text = read_docx_file(path)
    else:
        text = ""
    return normalize_text_preserve_lines(text)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_text_preserve_lines(text: str) -> str:
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


class TokenizerAdapter:
    def __init__(self, model_path: str):
        self._tokenizer = None
        self._max_length: int | None = None
        try:
            from transformers import AutoConfig, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            raw_max_length = getattr(self._tokenizer, "model_max_length", None)
            if isinstance(raw_max_length, int) and 0 < raw_max_length < 100000:
                # Keep a small reserve for special tokens/prefixes.
                self._max_length = max(32, raw_max_length - 8)
            if self._max_length is None:
                cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                raw_max_length = getattr(cfg, "max_position_embeddings", None) or getattr(cfg, "n_positions", None)
                if isinstance(raw_max_length, int) and raw_max_length > 0:
                    self._max_length = max(32, raw_max_length - 8)
        except Exception:
            self._tokenizer = None

    def count_tokens(self, text: str) -> int:
        if not self._tokenizer:
            return max(1, len(text.split()))
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    def split_tokens(self, text: str) -> list[str]:
        return text.split()

    @property
    def max_length(self) -> int | None:
        return self._max_length


def chunk_text(
    source: str,
    text: str,
    tokenizer: TokenizerAdapter,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    words = tokenizer.split_tokens(text)
    if not words:
        return []
    chunks: list[Chunk] = []
    step = max(1, chunk_size - chunk_overlap)
    start = 0
    idx = 0
    while start < len(words):
        end = min(len(words), start + chunk_size)
        chunk_words = words[start:end]
        chunk_text_value = " ".join(chunk_words).strip()
        token_count = tokenizer.count_tokens(chunk_text_value) if chunk_text_value else 0

        # Whitespace-based chunking can exceed the model token limit, especially for BERT/e5 models.
        # Shrink the chunk until it fits the tokenizer's maximum supported length.
        if tokenizer.max_length is not None:
            while end > start + 1 and token_count > tokenizer.max_length:
                overflow = token_count - tokenizer.max_length
                shrink_by = max(1, min(len(chunk_words) // 2, overflow))
                end = max(start + 1, end - shrink_by)
                chunk_words = words[start:end]
                chunk_text_value = " ".join(chunk_words).strip()
                token_count = tokenizer.count_tokens(chunk_text_value) if chunk_text_value else 0

        if chunk_text_value:
            start_char = text.find(chunk_text_value[: min(len(chunk_text_value), 32)])
            end_char = start_char + len(chunk_text_value) if start_char >= 0 else -1
            chunks.append(
                Chunk(
                    chunk_id=f"{source}::chunk-{idx}",
                    source=source,
                    text=chunk_text_value,
                    start_char=max(0, start_char),
                    end_char=max(0, end_char),
                    token_count=token_count,
                )
            )
            idx += 1
        start += step
    return chunks


def build_chunks(
    input_path: str | Path,
    embedding_model_path: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    tokenizer = TokenizerAdapter(embedding_model_path)
    all_chunks: list[Chunk] = []
    for file_path in iter_supported_files(input_path):
        text = read_file(file_path)
        if not text:
            continue
        source = str(file_path)
        all_chunks.extend(chunk_text(source, text, tokenizer, chunk_size=chunk_size, chunk_overlap=chunk_overlap))
    return all_chunks
