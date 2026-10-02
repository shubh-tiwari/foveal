"""SDK middleware: rewrite outgoing message history so pixels are sent once.

    client = foveal.wrap(anthropic.Anthropic(), memory)          # Anthropic format
    client = foveal.wrap(openai_client, memory, fmt="openai")     # OpenAI-compatible format

On every call the wrapper walks `messages` in order and, for each base64 image:

  * an exact repeat of an earlier image becomes a one-line reference;
  * a new image of the same stream (same tool, same size) as the previous one becomes a
    diff: a short description plus crops of the changed regions, when that is cheaper;
  * everything else is sent unchanged (and stored in Memory, so `foveal_look` can page it
    back in at full detail).

Every decision depends only on the messages *before* the image, so a message rewrites the
same way on every call: the rewritten history is a stable, append-only prefix. Prompt
caching keeps working, and models that bind thinking blocks to the exact history (preserved
thinking) see the same history they reasoned over.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from foveal.diff import describe, diff_frames
from foveal.instrument.hashing import sha256_hex
from foveal.instrument.tokens import estimate_image_tokens

TEXT_OVERHEAD = 40  # tokens of description per rewritten image


@dataclass
class Stats:
    images: int = 0
    deduplicated: int = 0
    diffed: int = 0
    tokens_before: int = 0
    tokens_after: int = 0

    @property
    def saved(self) -> int:
        return self.tokens_before - self.tokens_after


@dataclass
class _Seen:
    asset_id: str
    where: str  # "message 4"
    image: Image.Image


@dataclass
class Rewriter:
    memory: Any
    model: str = "claude-sonnet-5-5"
    dedup: bool = True
    diff: bool = True
    min_saving: int = 150  # tokens; smaller savings are not worth the extra blocks
    _diff_cache: dict[tuple[str, str], Any] = field(default_factory=dict)

    # -- format adapters ----------------------------------------------------------

    @staticmethod
    def _anthropic_image(block: dict) -> bytes | None:
        src = block.get("source") or {}
        if block.get("type") == "image" and src.get("type") == "base64":
            return base64.b64decode(src["data"])
        return None

    @staticmethod
    def _openai_image(part: dict) -> bytes | None:
        if part.get("type") != "image_url":
            return None
        url = (part.get("image_url") or {}).get("url", "")
        if url.startswith("data:") and ";base64," in url:
            return base64.b64decode(url.split(";base64,", 1)[1])
        return None

    @staticmethod
    def _jpeg(im: Image.Image) -> tuple[str, bytes]:
        buf = io.BytesIO()
        im.convert("RGB").save(buf, "JPEG", quality=90)
        return "image/jpeg", buf.getvalue()

    def _make_image(self, im: Image.Image, fmt: str) -> dict:
        mt, data = self._jpeg(im)
        b64 = base64.b64encode(data).decode()
        if fmt == "openai":
            return {"type": "image_url", "image_url": {"url": f"data:{mt};base64,{b64}"}}
        return {"type": "image", "source": {"type": "base64", "media_type": mt, "data": b64}}

    @staticmethod
    def _make_text(text: str, fmt: str) -> dict:
        return {"type": "text", "text": text}

    # -- core ---------------------------------------------------------------------

    def _tokens(self, im: Image.Image) -> int:
        return estimate_image_tokens(*im.size, self.model)

    def _rewrite_image(
        self,
        data: bytes,
        key: str,
        where: str,
        seen: dict[str, _Seen],
        streams: dict[str, _Seen],
        fmt: str,
        stats: Stats,
    ) -> list[dict] | None:
        """Replacement blocks for one image, or None to keep it unchanged."""
        sha = sha256_hex(data)
        with Image.open(io.BytesIO(data)) as raw:
            im = raw.convert("RGB")
        full = self._tokens(im)
        stats.images += 1
        stats.tokens_before += full
        aid = self.memory.ingest_image(data, kind="image", use_ocr=False, caption=False).asset_id
        out: list[dict] | None = None
        if self.dedup and sha in seen:
            first = seen[sha]
            out = [
                self._make_text(
                    f"[foveal: identical to the image in {first.where} "
                    f"(asset {aid[:12]}); not resent]",
                    fmt,
                )
            ]
            stats.deduplicated += 1
            stats.tokens_after += TEXT_OVERHEAD
        elif self.diff and key in streams and streams[key].image.size == im.size:
            prev = streams[key]
            ck = (prev.asset_id, aid)
            if ck not in self._diff_cache:
                self._diff_cache[ck] = diff_frames(prev.image, im)
            d = self._diff_cache[ck]
            crops = [im.crop((b.x, b.y, b.x + b.w, b.y + b.h)) for b in d.regions]
            cost = TEXT_OVERHEAD + sum(self._tokens(c) for c in crops)
            if d.kind == "identical" or (d.kind == "partial" and cost + self.min_saving < full):
                note = describe(d, since=f"the previous image of this stream ({prev.where})")
                out = [self._make_text(f"[foveal: asset {aid[:12]}] {note}", fmt)]
                for b, c in zip(d.regions, crops, strict=True):
                    out.append(self._make_text(f"Region x={b.x} y={b.y} {b.w}x{b.h}:", fmt))
                    out.append(self._make_image(c, fmt))
                stats.diffed += 1
                stats.tokens_after += cost
        if out is None:
            stats.tokens_after += full
        entry = _Seen(aid, where, im)
        seen.setdefault(sha, entry)
        streams[key] = entry
        return out

    def rewrite(self, messages: list[Any], fmt: str = "anthropic") -> tuple[list[Any], Stats]:
        """Rewritten copy of `messages` (input is not modified) and what was saved."""
        stats = Stats()
        seen: dict[str, _Seen] = {}
        streams: dict[str, _Seen] = {}
        tool_names: dict[str, str] = {}
        out: list[Any] = []
        get_image = self._openai_image if fmt == "openai" else self._anthropic_image

        def walk(content: Any, where: str, key: str) -> Any:
            if not isinstance(content, list):
                return content
            new: list[Any] = []
            for block in content:
                b = block if isinstance(block, dict) else _as_dict(block)
                if b.get("type") == "tool_use":
                    tool_names[b.get("id", "")] = b.get("name", "tool")
                if b.get("type") == "tool_result":
                    tkey = "tool:" + tool_names.get(b.get("tool_use_id", ""), "?")
                    new.append({**b, "content": walk(b.get("content"), where, tkey)})
                    continue
                data = get_image(b)
                if data is None:
                    new.append(block)
                    continue
                rep = self._rewrite_image(data, key, where, seen, streams, fmt, stats)
                new.extend(rep if rep is not None else [block])
            return new

        for i, msg in enumerate(messages):
            m = msg if isinstance(msg, dict) else _as_dict(msg)
            role = m.get("role", "user")
            content = walk(m.get("content"), f"message {i + 1}", f"{role}")
            out.append({**m, "content": content} if content is not m.get("content") else msg)
        return out, stats


def _as_dict(obj: Any) -> dict:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(exclude_none=True)
    return dict(vars(obj))


# -- client wrappers ---------------------------------------------------------------


class _Create:
    def __init__(self, inner: Any, rw: Rewriter, fmt: str, owner: Any):
        self._inner, self._rw, self._fmt, self._owner = inner, rw, fmt, owner

    def create(self, **kw: Any) -> Any:
        if "messages" in kw:
            kw["messages"], self._owner.last_stats = self._rw.rewrite(kw["messages"], self._fmt)
        return self._inner.create(**kw)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _Ns:
    def __init__(self, **attrs: Any):
        self.__dict__.update(attrs)


class FovealClient:
    """Proxy whose message-creating calls go through the Rewriter."""

    def __init__(self, client: Any, memory: Any, fmt: str = "anthropic", **opts: Any):
        self._client = client
        self.rewriter = Rewriter(memory, **opts)
        self.last_stats: Stats | None = None
        if fmt == "openai":
            self.chat = _Ns(completions=_Create(client.chat.completions, self.rewriter, fmt, self))
        else:
            self.messages = _Create(client.messages, self.rewriter, fmt, self)
            if hasattr(client, "beta"):
                self.beta = _Ns(messages=_Create(client.beta.messages, self.rewriter, fmt, self))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def wrap(client: Any, memory: Any, fmt: str = "anthropic", **opts: Any) -> FovealClient:
    return FovealClient(client, memory, fmt, **opts)


# -- the look tool, for models to page detail back in ---------------------------------

LOOK_TOOL = {
    "name": "foveal_look",
    "description": "See a stored image again. asset is the id prefix shown in a [foveal: ...] "
    "note; region=[x0, y0, x1, y1] fractions zooms in; detail is 'glimpse' or 'full'.",
    "input_schema": {
        "type": "object",
        "properties": {
            "asset": {"type": "string"},
            "detail": {"type": "string", "enum": ["glimpse", "full"]},
            "region": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
        },
        "required": ["asset"],
        "additionalProperties": False,
    },
}


def run_look_tool(memory: Any, tool_input: dict) -> list[dict]:
    """Content for a foveal_look tool_result (Anthropic format)."""
    prefix = str(tool_input.get("asset", ""))
    asset_id = memory.store.find_prefix(prefix)
    if asset_id is None:
        return [{"type": "text", "text": f"no asset starts with {prefix!r}"}]
    v = memory.look(
        asset_id, region=tool_input.get("region"), detail=tool_input.get("detail", "full")
    )
    return [{"type": "text", "text": f"asset {asset_id[:12]} ({v.width}x{v.height}):"}, v.block()]
