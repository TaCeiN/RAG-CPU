from __future__ import annotations

import hashlib
import hmac
import os
import uuid
from datetime import datetime, timedelta, timezone

import jwt

from app.core.settings import settings


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120_000)
    return f"{salt.hex()}:{digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    salt_hex, digest_hex = stored.split(":", 1)
    expected = hash_password(password, bytes.fromhex(salt_hex)).split(":", 1)[1]
    return hmac.compare_digest(expected, digest_hex)


def _encode(payload: dict) -> str:
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_alg)


def create_access_token(user_id: int) -> str:
    now = datetime.now(timezone.utc)
    payload = {"sub": str(user_id), "type": "access", "exp": now + timedelta(seconds=settings.access_ttl_sec), "jti": uuid.uuid4().hex}
    return _encode(payload)


def create_refresh_token(user_id: int) -> tuple[str, datetime]:
    now = datetime.now(timezone.utc)
    exp = now + timedelta(seconds=settings.refresh_ttl_sec)
    payload = {"sub": str(user_id), "type": "refresh", "exp": exp, "jti": uuid.uuid4().hex}
    return _encode(payload), exp.replace(tzinfo=None)


def decode_token(token: str, expected_type: str) -> int:
    payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_alg])
    if payload.get("type") != expected_type:
        raise ValueError("wrong token type")
    return int(payload["sub"])


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()
