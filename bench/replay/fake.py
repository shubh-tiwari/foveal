"""Scripted client for agent_eval --dry-run: always answers the first option with CLICK."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


class _Messages:
    def create(self, **kw: Any) -> SimpleNamespace:
        n_img = sum(1 for b in kw["messages"][0]["content"] if b.get("type") == "image")
        usage = {
            "input_tokens": 300 + 900 * n_img,
            "output_tokens": 20,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }
        return SimpleNamespace(
            model=kw["model"],
            stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text="A CLICK")],
            usage=SimpleNamespace(**usage, model_dump=lambda **_: dict(usage)),
        )


class FakeActionClient:
    def __init__(self) -> None:
        self.messages = _Messages()
