from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class CleanupReport:
    path: str
    exists: bool
    removable: bool
    error: str | None


def check_path(path: Path) -> CleanupReport:
    if not path.exists():
        return CleanupReport(path=str(path), exists=False, removable=True, error=None)
    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        return CleanupReport(path=str(path), exists=True, removable=True, error=None)
    except Exception as exc:
        return CleanupReport(path=str(path), exists=True, removable=False, error=str(exc))


def main() -> None:
    parser = argparse.ArgumentParser(description="Cleanup diagnostics")
    parser.add_argument("paths", nargs="+", help="paths to remove/check")
    args = parser.parse_args()

    reports = [check_path(Path(p)) for p in args.paths]
    print(json.dumps([asdict(r) for r in reports], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
