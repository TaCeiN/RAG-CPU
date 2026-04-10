from __future__ import annotations

from pathlib import Path

ACTIVE_CONFIG_FILE = Path("config/quality.yaml")


class Settings:
    app_name = "RAG Service"
    db_url = "sqlite:///./app_data/app.db"
    jwt_secret = "change-me-in-prod-32-bytes-minimum-secret"
    jwt_alg = "HS256"
    access_ttl_sec = 3600
    refresh_ttl_sec = 60 * 60 * 24 * 14
    files_dir = Path("app_data/files")
    indexes_dir = Path("app_data/indexes")
    default_config_path = ACTIVE_CONFIG_FILE
    log_debug = True


settings = Settings()
