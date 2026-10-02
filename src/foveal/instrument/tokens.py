"""Image token costs: a fast estimate and a count_tokens ground truth.

Estimate follows the Vision docs (checked 2026-10-02): an image costs
ceil(w/28) * ceil(h/28) visual tokens after being downscaled, keeping aspect
ratio, to fit the model tier's long-edge and visual-token limits.
"""

from __future__ import annotations

import math
import sqlite3
import threading
from pathlib import Path
from typing import Any

PATCH = 28

# (max long edge px, max visual tokens)
HIGH_RES = (2576, 4784)  # Claude 4.7 and later
STANDARD = (1568, 1568)

_STANDARD_MODELS = (
    "claude-haiku-4-5",
    "claude-sonnet-4-6",
    "claude-opus-4-6",
    "claude-sonnet-4-5",
    "claude-opus-4-5",
    "claude-opus-4-1",
    "claude-opus-4-0",
    "claude-sonnet-4-0",
    "claude-3",
)


def tier_limits(model: str) -> tuple[int, int]:
    return STANDARD if model.startswith(_STANDARD_MODELS) else HIGH_RES


def _patches(w: int, h: int) -> int:
    return math.ceil(w / PATCH) * math.ceil(h / PATCH)


def resized_dims(w: int, h: int, model: str) -> tuple[int, int]:
    max_edge, max_tokens = tier_limits(model)
    scale = min(1.0, max_edge / max(w, h))
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    if _patches(nw, nh) > max_tokens:
        scale *= math.sqrt(max_tokens * PATCH * PATCH / (nw * nh))
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        while _patches(nw, nh) > max_tokens:
            scale *= 0.99
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    return nw, nh


def estimate_image_tokens(w: int, h: int, model: str) -> int:
    return _patches(*resized_dims(w, h, model))


class TokenCounter:
    """Ground-truth image tokens via `messages.count_tokens`, cached per (model, sha256).

    The image's cost is count(image + stub text) - count(stub text). The cache lives
    in SQLite so repeated runs never recount the same image.
    """

    STUB = "."

    def __init__(self, client: Any, cache_path: str | Path = ".cache/foveal_tokens.sqlite"):
        self._client = client
        path = Path(cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS counts (model TEXT, key TEXT, tokens INTEGER,"
            " PRIMARY KEY (model, key))"
        )
        self._lock = threading.Lock()

    def _get(self, model: str, key: str) -> int | None:
        with self._lock:
            row = self._db.execute(
                "SELECT tokens FROM counts WHERE model=? AND key=?", (model, key)
            ).fetchone()
        return row[0] if row else None

    def _put(self, model: str, key: str, tokens: int) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO counts VALUES (?, ?, ?)", (model, key, tokens))
            self._db.commit()

    def _count(self, model: str, content: list[dict[str, Any]]) -> int:
        resp = self._client.messages.count_tokens(
            model=model, messages=[{"role": "user", "content": content}]
        )
        return resp.input_tokens

    def image_tokens(self, model: str, sha256: str, block: dict[str, Any]) -> int:
        cached = self._get(model, sha256)
        if cached is not None:
            return cached
        base = self._get(model, "__stub__")
        if base is None:
            base = self._count(model, [{"type": "text", "text": self.STUB}])
            self._put(model, "__stub__", base)
        clean = {k: v for k, v in block.items() if k != "cache_control"}
        total = self._count(model, [clean, {"type": "text", "text": self.STUB}])
        tokens = max(0, total - base)
        self._put(model, sha256, tokens)
        return tokens
