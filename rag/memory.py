from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Role = Literal["user", "assistant"]


@dataclass(slots=True)
class Message:
    role: Role
    content: str


class MemoryStore:
    def __init__(self, history_file: str | Path, history_n: int, enable_summary: bool):
        self.history_file = Path(history_file)
        self.history_n = history_n
        self.enable_summary = enable_summary
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.history_file.exists():
            self.history_file.write_text("", encoding="utf-8")

    def append(self, role: Role, content: str) -> None:
        record = {"role": role, "content": content}
        with self.history_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def last_messages(self) -> list[Message]:
        lines = [line.strip() for line in self.history_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        raw = [json.loads(line) for line in lines][-self.history_n :]
        messages = [Message(role=item["role"], content=item["content"]) for item in raw]
        if self.enable_summary and len(messages) > 6:
            summary = " ".join(m.content for m in messages[:-4])[:500]
            return [Message(role="assistant", content=f"Summary: {summary}")] + messages[-4:]
        return messages

