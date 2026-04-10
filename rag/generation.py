from __future__ import annotations

from typing import Any

import requests


class OllamaClient:
    def __init__(self, endpoint: str, model: str, timeout_seconds: int = 60):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    def chat(self, messages: list[dict[str, str]]) -> str:
        payload = {"model": self.model, "messages": messages, "stream": False}

        response = requests.post(f"{self.endpoint}/api/chat", json=payload, timeout=self.timeout_seconds)
        if response.status_code == 404:
            resolved_model = self._resolve_available_model(self.model)
            if resolved_model and resolved_model != self.model:
                payload["model"] = resolved_model
                response = requests.post(f"{self.endpoint}/api/chat", json=payload, timeout=self.timeout_seconds)
            if response.status_code == 404:
                # Some local servers expose only OpenAI-compatible route.
                response = requests.post(
                    f"{self.endpoint}/v1/chat/completions",
                    json={"model": payload["model"], "messages": messages},
                    timeout=self.timeout_seconds,
                )
                if response.ok:
                    data_v1: dict[str, Any] = response.json()
                    choices = data_v1.get("choices", [])
                    if choices:
                        return choices[0].get("message", {}).get("content", "").strip()

        if not response.ok:
            raise RuntimeError(f"LLM HTTP {response.status_code}: {response.text[:300]}")
        data: dict[str, Any] = response.json()
        return data.get("message", {}).get("content", "").strip()

    def _resolve_available_model(self, requested: str) -> str | None:
        try:
            response = requests.get(f"{self.endpoint}/api/tags", timeout=self.timeout_seconds)
            if not response.ok:
                return None
            data: dict[str, Any] = response.json()
            names = [item.get("name", "") for item in data.get("models", [])]
            if requested in names:
                return requested
            requested_l = requested.lower()
            candidates = [name for name in names if requested_l in name.lower() or name.lower() in requested_l]
            if len(candidates) == 1:
                return candidates[0]
            if candidates:
                return candidates[0]
            return names[0] if names else None
        except Exception:
            return None

    def probe(self) -> dict[str, Any]:
        status: dict[str, Any] = {
            "endpoint": self.endpoint,
            "configured_model": self.model,
            "reachable": False,
            "api_version": None,
            "models_count": 0,
            "model_available": False,
            "resolved_model": None,
            "error": None,
        }
        try:
            v = requests.get(f"{self.endpoint}/api/version", timeout=self.timeout_seconds)
            if v.ok:
                status["api_version"] = v.json().get("version")
            t = requests.get(f"{self.endpoint}/api/tags", timeout=self.timeout_seconds)
            if not t.ok:
                status["error"] = f"/api/tags http {t.status_code}"
                return status
            status["reachable"] = True
            data: dict[str, Any] = t.json()
            names = [item.get("name", "") for item in data.get("models", [])]
            status["models_count"] = len(names)
            resolved = self._resolve_available_model(self.model)
            status["resolved_model"] = resolved
            status["model_available"] = (self.model in names) or (resolved in names if resolved else False)
            return status
        except Exception as exc:
            status["error"] = str(exc)
            return status


def build_messages(query: str, context: str | None) -> list[dict[str, str]]:
    system_prompt = (
        "Ты локальный ассистент RAG. Отвечай по-русски. "
        "Если контекст дан, опирайся на него. Если контекста недостаточно, скажи об этом."
    )
    user_content = query if not context else f"Контекст:\n{context}\n\nВопрос:\n{query}"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
