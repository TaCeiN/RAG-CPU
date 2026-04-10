from __future__ import annotations

import re

from rag.types import RouterDecision

MEMORY_PATTERNS = [
    r"\bчто я (сказал|писал)\b",
    r"\bнапомни\b",
    r"\bвыше\b",
    r"\bв нашем чате\b",
    r"\bмы обсуждали\b",
]

DIRECT_PATTERNS = [
    r"^\s*(привет|здравствуй|спасибо|ок|понял)\s*[!.?]?\s*$",
    r"^\s*как дела\??\s*$",
]


def route_query(query: str) -> RouterDecision:
    text = query.strip().lower()
    for pattern in MEMORY_PATTERNS:
        if re.search(pattern, text):
            return RouterDecision(route="memory", reason="query references chat history", search_query=None)
    for pattern in DIRECT_PATTERNS:
        if re.search(pattern, text):
            return RouterDecision(route="direct", reason="small-talk or simple dialog", search_query=None)
    if len(text.split()) <= 2:
        return RouterDecision(route="direct", reason="too short for reliable retrieval", search_query=None)
    return RouterDecision(route="rag", reason="needs external knowledge", search_query=text)
