"""A pass-through wrapper around the Anthropic client that logs image traffic.

The wrapper never changes a request. It walks the outgoing `messages`, records every
image/document block (hash, size, token cost, position), forwards the call unchanged,
and logs the response `usage` - one JSONL line per model call.

    tracer = Tracer(run_id="r1", sink=JsonlSink("runs/r1.jsonl"))
    client = InstrumentedAnthropic(anthropic.Anthropic(), tracer)
    with tracer.scope(task_id="q1", agent_id="orch"):
        client.messages.create(...)
"""

from __future__ import annotations

import contextvars
import itertools
import threading
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from foveal.instrument.hashing import decode_base64, inspect_image_bytes, sha256_hex
from foveal.instrument.pricing import call_cost_usd
from foveal.instrument.records import CallRecord, ImageRef, JsonlSink
from foveal.instrument.tokens import TokenCounter, estimate_image_tokens


@dataclass
class Scope:
    task_id: str | None = None
    agent_id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


_scope: contextvars.ContextVar[Scope | None] = contextvars.ContextVar("foveal_scope", default=None)


def current_scope() -> Scope:
    return _scope.get() or Scope()


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _as_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump(exclude_none=True)
    return dict(vars(obj))


class Tracer:
    """Holds the run's sink, call counters, scope and running cost."""

    def __init__(
        self,
        run_id: str,
        sink: JsonlSink,
        token_counter: TokenCounter | None = None,
    ):
        self.run_id = run_id
        self.sink = sink
        self.token_counter = token_counter
        self._call_index = itertools.count()
        self._steps: dict[tuple[str | None, str | None], int] = defaultdict(int)
        self._lock = threading.Lock()
        self._image_cache: dict[str, tuple[int | None, int | None, str | None]] = {}
        self.total_cost_usd = 0.0

    @contextmanager
    def scope(
        self, task_id: str | None = None, agent_id: str | None = None, **meta: Any
    ) -> Iterator[Scope]:
        parent = current_scope()
        s = Scope(
            task_id=task_id if task_id is not None else parent.task_id,
            agent_id=agent_id if agent_id is not None else parent.agent_id,
            meta={**parent.meta, **meta},
        )
        token = _scope.set(s)
        try:
            yield s
        finally:
            _scope.reset(token)

    def log_event(self, **data: Any) -> None:
        """Write a non-call record (e.g. a task result) to the same log."""
        s = current_scope()
        self.sink.write({"run_id": self.run_id, "task_id": s.task_id, **s.meta, **data})

    # -- image extraction -------------------------------------------------

    def _image_ref(self, block: dict[str, Any], msg_index: int, path: str, model: str) -> ImageRef:
        kind = block.get("type")
        source = _get(block, "source", {}) or {}
        source_type = _get(source, "type", "unknown")
        media_type = _get(source, "media_type")
        w = h = phash = None
        if source_type == "base64":
            raw = decode_base64(_get(source, "data", ""))
            sha = sha256_hex(raw)
            if kind == "image":
                if sha not in self._image_cache:
                    info = inspect_image_bytes(raw)
                    self._image_cache[sha] = (info.width, info.height, info.phash)
                w, h, phash = self._image_cache[sha]
        elif source_type == "url":
            sha = sha256_hex(("url:" + _get(source, "url", "")).encode())
        elif source_type == "file":
            sha = sha256_hex(("file:" + _get(source, "file_id", "")).encode())
        else:
            sha = sha256_hex(repr(source).encode())
        est = estimate_image_tokens(w, h, model) if (w and h) else None
        tokens = None
        if self.token_counter is not None:
            try:
                tokens = self.token_counter.image_tokens(model, sha, _as_dict(block))
            except Exception:
                tokens = None
        return ImageRef(
            sha256=sha,
            kind=kind,
            source_type=source_type,
            media_type=media_type,
            msg_index=msg_index,
            path=path,
            width=w,
            height=h,
            phash=phash,
            est_tokens=est,
            tokens=tokens,
        )

    def extract_images(self, messages: list[Any], model: str) -> list[ImageRef]:
        refs: list[ImageRef] = []

        def walk(content: Any, msg_index: int, path: str) -> None:
            if isinstance(content, str) or content is None:
                return
            for j, block in enumerate(content):
                btype = _get(block, "type")
                p = f"{path}.{j}"
                if btype in ("image", "document"):
                    refs.append(self._image_ref(_as_dict(block), msg_index, p, model))
                elif btype == "tool_result":
                    walk(_get(block, "content"), msg_index, p + ".content")

        for i, msg in enumerate(messages):
            walk(_get(msg, "content"), i, f"{i}.content")
        return refs

    # -- call recording ---------------------------------------------------

    def record(
        self,
        kwargs: dict[str, Any],
        response: Any,
        started: float,
        latency: float,
        error: str | None = None,
    ) -> CallRecord:
        s = current_scope()
        model = kwargs.get("model", "unknown")
        images = self.extract_images(kwargs.get("messages", []), model)
        usage = _as_dict(response.usage) if response is not None and response.usage else {}
        with self._lock:
            idx = next(self._call_index)
            key = (s.task_id, s.agent_id)
            self._steps[key] += 1
            step = self._steps[key]
            cost = call_cost_usd(model, usage)
            self.total_cost_usd += cost
        rec = CallRecord(
            run_id=self.run_id,
            call_index=idx,
            task_id=s.task_id,
            agent_id=s.agent_id,
            agent_step=step,
            model=model,
            served_model=_get(response, "model") if response is not None else None,
            started_at=started,
            latency_s=latency,
            n_messages=len(kwargs.get("messages", [])),
            images=images,
            usage=usage,
            stop_reason=_get(response, "stop_reason") if response is not None else None,
            meta={**s.meta, "cost_usd": cost},
            error=error,
        )
        self.sink.write(rec)
        return rec


class _InstrumentedMessages:
    def __init__(self, inner: Any, tracer: Tracer):
        self._inner = inner
        self._tracer = tracer

    def create(self, **kwargs: Any) -> Any:
        if kwargs.get("stream"):
            raise NotImplementedError("use .stream() for streaming calls")
        started = time.time()
        t0 = time.perf_counter()
        try:
            response = self._inner.create(**kwargs)
        except Exception as e:
            self._tracer.record(kwargs, None, started, time.perf_counter() - t0, repr(e))
            raise
        self._tracer.record(kwargs, response, started, time.perf_counter() - t0)
        return response

    @contextmanager
    def stream(self, **kwargs: Any) -> Iterator[Any]:
        started = time.time()
        t0 = time.perf_counter()
        with self._inner.stream(**kwargs) as s:
            yield s
            final = s.get_final_message()
        self._tracer.record(kwargs, final, started, time.perf_counter() - t0)

    def __getattr__(self, name: str) -> Any:  # count_tokens, batches, ...
        return getattr(self._inner, name)


class _InstrumentedBeta:
    def __init__(self, inner: Any, tracer: Tracer):
        self._inner = inner
        self.messages = _InstrumentedMessages(inner.messages, tracer)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class InstrumentedAnthropic:
    """Drop-in proxy for `anthropic.Anthropic` that logs `messages` calls."""

    def __init__(self, client: Any, tracer: Tracer):
        self._client = client
        self.tracer = tracer
        self.messages = _InstrumentedMessages(client.messages, tracer)
        if hasattr(client, "beta"):
            self.beta = _InstrumentedBeta(client.beta, tracer)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)
