"""Log record types and a thread-safe JSONL sink."""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ImageRef:
    """One image or document block as it appeared in one request."""

    sha256: str
    kind: str  # image | document
    source_type: str  # base64 | url | file
    media_type: str | None
    msg_index: int  # position of the message in the request's `messages`
    path: str  # e.g. "3.content.1.content.2" (nested inside a tool_result)
    width: int | None = None
    height: int | None = None
    phash: str | None = None
    est_tokens: int | None = None  # from the published resize + patch rule
    tokens: int | None = None  # ground truth from count_tokens, when available

    @property
    def cost_tokens(self) -> int:
        return self.tokens if self.tokens is not None else (self.est_tokens or 0)


@dataclass
class CallRecord:
    run_id: str
    call_index: int
    task_id: str | None
    agent_id: str | None
    agent_step: int
    model: str
    served_model: str | None
    started_at: float
    latency_s: float
    n_messages: int
    images: list[ImageRef]
    usage: dict[str, Any]
    stop_reason: str | None
    meta: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    type: str = "call"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class JsonlSink:
    """Append-only JSONL writer, safe to share across threads."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, record: CallRecord | dict[str, Any]) -> None:
        data = record.to_dict() if isinstance(record, CallRecord) else record
        line = json.dumps(data, ensure_ascii=False, default=str)
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
