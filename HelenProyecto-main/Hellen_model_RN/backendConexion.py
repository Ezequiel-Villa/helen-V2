"""Client helper that forwards recognised gestures to the Helen backend."""

from __future__ import annotations

import importlib
import json
import os
import time
import uuid
from dataclasses import dataclass
from http import client
from typing import Dict, Optional
from urllib.parse import urlparse

helpers = importlib.import_module(__name__.rsplit(".", 1)[0] + ".helpers" if "." in __name__ else "helpers")

DEFAULT_BACKEND_URL = "http://127.0.0.1:5000/gestures/gesture-key"
TIMEOUT_SECONDS = 5
http_client = client
_sequence = 0


def _resolve_backend_url(url: Optional[str] = None) -> str:
    candidate = (url or os.getenv("HELEN_BACKEND_URL") or DEFAULT_BACKEND_URL).strip()
    if not candidate:
        raise ValueError("No se especificó un endpoint para el backend Helen")
    return candidate


def _gesture_from_character(character: str) -> Optional[str]:
    normalized = character.strip()
    if not normalized:
        return None

    for value in helpers.labels_dict.values():
        if value.lower() == normalized.lower():
            return value
    return None


@dataclass(frozen=True)
class GesturePayload:
    character: str
    gesture: Optional[str]
    score: Optional[float]
    sequence: int
    session_id: str
    timestamp: float

    def to_json(self) -> bytes:
        data: Dict[str, object] = {
            "character": self.character,
            "gesture": self.gesture,
            "score": self.score,
            "sequence": self.sequence,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
        }
        return json.dumps(data, ensure_ascii=False).encode("utf-8")


def post_gesturekey(
    character: str,
    *,
    score: Optional[float] = None,
    session_id: Optional[str] = None,
    endpoint: Optional[str] = None,
) -> int:
    """Send the recognised gesture to the backend returning the HTTP status code."""

    if session_id is None:
        session_id = uuid.uuid4().hex

    global _sequence
    _sequence += 1

    payload = GesturePayload(
        character=character,
        gesture=_gesture_from_character(character),
        score=score,
        sequence=_sequence,
        session_id=session_id,
        timestamp=time.time(),
    )

    target = _resolve_backend_url(endpoint)
    parsed = urlparse(target)
    if not parsed.scheme.startswith("http"):
        raise ValueError(f"Endpoint inválido para el backend: {target}")

    port = parsed.port or (80 if parsed.scheme == "http" else 443)
    connection = http_client.HTTPConnection(parsed.hostname, port, timeout=TIMEOUT_SECONDS)
    try:
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"
        connection.request(
            "POST",
            path,
            body=payload.to_json(),
            headers={"Content-Type": "application/json", "User-Agent": "helen-backend-bridge"},
        )
        response = connection.getresponse()
        response.read()  # Ensure the response is consumed so the socket can be reused.
        status = int(response.status)
    finally:
        connection.close()

    return status


__all__ = ["post_gesturekey", "DEFAULT_BACKEND_URL", "http_client"]
