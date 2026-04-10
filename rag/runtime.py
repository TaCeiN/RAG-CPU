from __future__ import annotations

import os
from pathlib import Path


def ensure_local_hf_cache(base_dir: str | Path = ".") -> None:
    root = Path(base_dir).resolve() / ".hf_cache"
    modules = root / "modules"
    hub = root / "hub"
    root.mkdir(parents=True, exist_ok=True)
    modules.mkdir(parents=True, exist_ok=True)
    hub.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(root)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub)
    os.environ["HF_HUB_CACHE"] = str(hub)
    os.environ["HF_MODULES_CACHE"] = str(modules)
