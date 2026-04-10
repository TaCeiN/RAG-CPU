from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.session import get_db
from app.models.entities import RefreshToken, User
from app.schemas.auth import LoginIn, RefreshIn, SignupIn, TokenOut
from app.services.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    hash_token,
    verify_password,
)
from rag.logging_utils import log_event

router = APIRouter(prefix="/auth", tags=["auth"])
logger = logging.getLogger("rag")


@router.post("/signup", response_model=TokenOut)
def signup(payload: SignupIn, db: Session = Depends(get_db)):
    exists = db.scalar(select(User).where(User.email == payload.email))
    if exists:
        raise HTTPException(status_code=400, detail="email exists")
    user = User(email=payload.email, password_hash=hash_password(payload.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    access = create_access_token(user.id)
    refresh, exp = create_refresh_token(user.id)
    db.add(RefreshToken(user_id=user.id, token_hash=hash_token(refresh), expires_at=exp))
    db.commit()
    log_event(logger, "auth_signup", user_id=user.id, email=user.email)
    return TokenOut(access_token=access, refresh_token=refresh)


@router.post("/login", response_model=TokenOut)
def login(payload: LoginIn, db: Session = Depends(get_db)):
    user = db.scalar(select(User).where(User.email == payload.email))
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="bad credentials")
    access = create_access_token(user.id)
    refresh, exp = create_refresh_token(user.id)
    db.add(RefreshToken(user_id=user.id, token_hash=hash_token(refresh), expires_at=exp))
    db.commit()
    log_event(logger, "auth_login", user_id=user.id, email=user.email)
    return TokenOut(access_token=access, refresh_token=refresh)


@router.post("/refresh", response_model=TokenOut)
def refresh(payload: RefreshIn, db: Session = Depends(get_db)):
    try:
        user_id = decode_token(payload.refresh_token, expected_type="refresh")
    except Exception:
        raise HTTPException(status_code=401, detail="invalid refresh")
    token_hash = hash_token(payload.refresh_token)
    row = db.scalar(select(RefreshToken).where(RefreshToken.token_hash == token_hash, RefreshToken.revoked == False))
    if not row:
        raise HTTPException(status_code=401, detail="refresh revoked")
    row.revoked = True
    access = create_access_token(user_id)
    refresh_new, exp = create_refresh_token(user_id)
    db.add(RefreshToken(user_id=user_id, token_hash=hash_token(refresh_new), expires_at=exp))
    db.commit()
    log_event(logger, "auth_refresh", user_id=user_id)
    return TokenOut(access_token=access, refresh_token=refresh_new)


@router.post("/logout")
def logout(payload: RefreshIn, db: Session = Depends(get_db)):
    row = db.scalar(select(RefreshToken).where(RefreshToken.token_hash == hash_token(payload.refresh_token)))
    if row:
        row.revoked = True
        db.commit()
        log_event(logger, "auth_logout", user_id=row.user_id)
    return {"status": "ok"}


@router.get("/me")
def me(user: User = Depends(get_current_user)):
    return {"id": user.id, "email": user.email}
