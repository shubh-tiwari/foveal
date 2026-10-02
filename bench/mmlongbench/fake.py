"""A scripted stand-in for the Anthropic client, used by --dry-run and tests.

It mimics the harness's call pattern (parallel readers, orchestrator self-viewing) and a
prompt cache that serves each conversation's previous prefix as cache reads.
"""

from __future__ import annotations

import base64
import io
import itertools
from types import SimpleNamespace
from typing import Any

from foveal.instrument.tokens import estimate_image_tokens


def _get(o: Any, k: str, d: Any = None) -> Any:
    return o.get(k, d) if isinstance(o, dict) else getattr(o, k, d)


def _image_tokens(content: Any, model: str) -> int:
    from PIL import Image

    total = 0
    if isinstance(content, str) or content is None:
        return 0
    for b in content:
        t = _get(b, "type")
        if t == "image":
            raw = base64.b64decode(_get(_get(b, "source"), "data"))
            with Image.open(io.BytesIO(raw)) as im:
                total += estimate_image_tokens(*im.size, model)
        elif t == "tool_result":
            total += _image_tokens(_get(b, "content"), model)
    return total


def _usage(**kw: int) -> SimpleNamespace:
    d = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        **kw,
    }
    return SimpleNamespace(**d, model_dump=lambda **_: dict(d))


class _Messages:
    def __init__(self) -> None:
        self._ids = itertools.count()
        self._prefix: dict[int, int] = {}

    def _block(self, **kw: Any) -> SimpleNamespace:
        return SimpleNamespace(**kw)

    def create(self, **kw: Any) -> SimpleNamespace:
        model, messages = kw["model"], kw["messages"]
        turn = sum(1 for m in messages if _get(m, "role") == "assistant")
        names = [t["name"] for t in kw.get("tools", [])]
        is_orch = "final_answer" in names
        if not names:  # captioner
            return SimpleNamespace(
                model=model,
                stop_reason="end_turn",
                content=[self._block(type="text", text="A synthetic page with a blue bar.")],
                usage=_usage(
                    input_tokens=400 + _image_tokens(messages[0]["content"], model),
                    output_tokens=15,
                ),
            )

        def tid() -> str:
            return f"toolu_{next(self._ids)}"

        if is_orch:
            script = [
                [
                    self._block(
                        type="tool_use",
                        id=tid(),
                        name="ask_reader",
                        input={"pages": [1, 2, 3], "instruction": "find the value"},
                    ),
                    self._block(
                        type="tool_use",
                        id=tid(),
                        name="ask_reader",
                        input={"pages": [3, 4, 5], "instruction": "find the value"},
                    ),
                ],
                [
                    self._block(
                        type="tool_use", id=tid(), name="view_pages", input={"pages": [3, 4]}
                    )
                ],
                [
                    self._block(
                        type="tool_use", id=tid(), name="final_answer", input={"answer": "101"}
                    )
                ],
            ]
        elif "look" in names:
            script = [
                [
                    self._block(
                        type="tool_use",
                        id=tid(),
                        name="look",
                        input={"page": 3, "detail": "glimpse"},
                    )
                ],
                [
                    self._block(
                        type="tool_use",
                        id=tid(),
                        name="look",
                        input={"page": 3, "detail": "full", "region": [0, 0, 1, 0.4]},
                    )
                ],
                [self._block(type="text", text="The value is 101 (page 3).")],
            ]
            if "note" in names:  # facts mode: record what was found before answering
                script.insert(
                    2,
                    [
                        self._block(
                            type="tool_use",
                            id=tid(),
                            name="note",
                            input={
                                "page": 3,
                                "claim": "The value on page 3 is 101",
                                "region": [0, 0, 1, 0.4],
                                "seen_at": "full",
                            },
                        )
                    ],
                )
        else:
            script = [
                [self._block(type="tool_use", id=tid(), name="view_pages", input={"pages": [6]})],
                [self._block(type="text", text="The value is 101 (page 3).")],
            ]
        content = script[min(turn, len(script) - 1)]
        stop = "tool_use" if any(b.type == "tool_use" for b in content) else "end_turn"
        total = 400 * len(messages) + sum(
            _image_tokens(_get(m, "content"), model) for m in messages
        )
        key = id(messages)
        cached = self._prefix.get(key, 0) if kw.get("cache_control") else 0
        self._prefix[key] = total
        return SimpleNamespace(
            model=model,
            stop_reason=stop,
            content=content,
            usage=_usage(
                input_tokens=total - cached, cache_read_input_tokens=cached, output_tokens=150
            ),
        )

    def count_tokens(self, **kw: Any) -> SimpleNamespace:
        content = kw["messages"][0]["content"]
        return SimpleNamespace(input_tokens=5 + _image_tokens(content, kw["model"]))


class FakeAnthropic:
    def __init__(self) -> None:
        self.messages = _Messages()
        self.beta = SimpleNamespace(messages=self.messages)
