from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname.lower(),
            "message": record.getMessage(),
        }
        if hasattr(record, "extra_fields"):
            payload.update(record.extra_fields)
        return json.dumps(payload, ensure_ascii=False)


def setup_logger(debug: bool = False) -> logging.Logger:
    logger = logging.getLogger("rag")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    return logger


def log_event(logger: logging.Logger, message: str, **fields: object) -> None:
    logger.info(message, extra={"extra_fields": fields})

