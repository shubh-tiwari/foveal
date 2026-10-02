"""OpenRouter backend behind the Anthropic Messages interface the harness already speaks.

`OpenRouterClient().messages.create(**anthropic_style_kwargs)` converts the request to
OpenRouter's OpenAI-compatible chat format, calls it, and converts the reply back into an
Anthropic-shaped object (text / tool_use blocks, stop_reason, usage). The agents, the
foveal instrumentation and the evaluations then run unchanged on any OpenRouter model.

Conversions:
  * system prompt -> a system message; tools -> function tools (input_schema -> parameters)
  * image blocks -> image_url data URLs
  * tool_result blocks -> `tool` messages; images inside tool results (not allowed in tool
    messages) follow in one user message, labelled by tool call
  * the provider's assistant message (incl. reasoning_details) is kept on the reply and sent
    back unmodified on the next turn, as OpenRouter requires for reasoning with tools
  * output_config.effort -> reasoning.effort; Anthropic-only params are dropped
  * usage: prompt/completion tokens, cached tokens, and OpenRouter's exact `cost` (USD)

Reads OPENROUTER_API_KEY from the environment (e.g. loaded from .env).
"""

from __future__ import annotations

import json
import os
import time
from types import SimpleNamespace
from typing import Any

import httpx

URL = "https://openrouter.ai/api/v1/chat/completions"
STOP = {
    "tool_calls": "tool_use",
    "stop": "end_turn",
    "length": "max_tokens",
    "content_filter": "refusal",
}


def _get(o: Any, k: str, d: Any = None) -> Any:
    return o.get(k, d) if isinstance(o, dict) else getattr(o, k, d)


def _image_part(block: Any) -> dict | None:
    src = _get(block, "source") or {}
    if _get(src, "type") == "base64":
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{_get(src, 'media_type')};base64,{_get(src, 'data')}"},
        }
    if _get(src, "type") == "url":
        return {"type": "image_url", "image_url": {"url": _get(src, "url")}}
    return None


def _parts(content: Any) -> list[dict]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    out = []
    for b in content or []:
        t = _get(b, "type")
        if t == "text":
            out.append({"type": "text", "text": _get(b, "text", "")})
        elif t == "image":
            p = _image_part(b)
            if p:
                out.append(p)
    return out


def to_openai(system: Any, messages: list[Any]) -> list[dict]:
    out: list[dict] = []
    if system:
        out.append(
            {
                "role": "system",
                "content": system
                if isinstance(system, str)
                else " ".join(_get(b, "text", "") for b in system),
            }
        )
    for m in messages:
        role, content = _get(m, "role"), _get(m, "content")
        if role == "assistant":
            raw = next(
                (
                    _get(b, "message")
                    for b in content or []
                    if _get(b, "type") == "provider_message"
                ),
                None,
            )
            if raw is not None:  # replay the provider's own message, reasoning included
                out.append(raw)
                continue
            text = "".join(_get(b, "text", "") for b in content or [] if _get(b, "type") == "text")
            calls = [
                {
                    "id": _get(b, "id"),
                    "type": "function",
                    "function": {
                        "name": _get(b, "name"),
                        "arguments": json.dumps(_get(b, "input")),
                    },
                }
                for b in content or []
                if _get(b, "type") == "tool_use"
            ]
            msg: dict = {"role": "assistant", "content": text or None}
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
            continue
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue
        tool_images: list[dict] = []
        rest = []
        for b in content or []:
            if _get(b, "type") != "tool_result":
                rest.append(b)
                continue
            tc = _get(b, "content")
            parts = _parts(tc)
            texts = [p["text"] for p in parts if p["type"] == "text"]
            images = [p for p in parts if p["type"] == "image_url"]
            if images:
                texts.append(f"[{len(images)} image(s) attached in the next message]")
                tool_images.append(
                    {"type": "text", "text": f"Images from tool call {_get(b, 'tool_use_id')}:"}
                )
                tool_images += images
            body = "\n".join(texts) or ("(error)" if _get(b, "is_error") else "(empty)")
            out.append({"role": "tool", "tool_call_id": _get(b, "tool_use_id"), "content": body})
        user_parts = tool_images + _parts(rest)
        if user_parts:
            out.append({"role": "user", "content": user_parts})
    return out


def _tools(tools: list[dict] | None) -> list[dict] | None:
    if not tools:
        return None
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object"}),
            },
        }
        for t in tools
    ]


def from_openai(data: dict, model: str) -> SimpleNamespace:
    choice = data["choices"][0]
    msg = choice["message"]
    blocks: list[Any] = []
    if msg.get("content"):
        blocks.append(SimpleNamespace(type="text", text=msg["content"]))
    for tc in msg.get("tool_calls") or []:
        try:
            args = json.loads(tc["function"].get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {"_invalid_arguments": tc["function"].get("arguments")}
        blocks.append(
            SimpleNamespace(
                type="tool_use",
                id=tc["id"],
                name=tc["function"]["name"],
                input=args if isinstance(args, dict) else {},
            )
        )
    raw = {
        k: v
        for k, v in msg.items()
        if k in ("role", "content", "tool_calls", "reasoning_details") and v is not None
    }
    blocks.append(SimpleNamespace(type="provider_message", message=raw))
    u = data.get("usage") or {}
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    written = (u.get("prompt_tokens_details") or {}).get("cache_write_tokens") or 0
    usage = {
        "input_tokens": max(0, (u.get("prompt_tokens") or 0) - cached - written),
        "cache_read_input_tokens": cached,
        "cache_creation_input_tokens": written,
        "output_tokens": u.get("completion_tokens") or 0,
        "cost_usd": u.get("cost"),  # exact charge reported by OpenRouter
    }
    finish = choice.get("finish_reason") or "stop"
    stop = STOP.get(finish, "end_turn")
    if stop == "end_turn" and msg.get("tool_calls"):
        stop = "tool_use"
    return SimpleNamespace(
        model=data.get("model", model),
        content=blocks,
        stop_reason=stop,
        usage=SimpleNamespace(**usage, model_dump=lambda **_: dict(usage)),
    )


class _Messages:
    def __init__(self, client: OpenRouterClient):
        self._c = client

    def create(self, **kw: Any) -> SimpleNamespace:
        body: dict[str, Any] = {
            "model": kw["model"],
            "messages": to_openai(kw.get("system"), kw["messages"]),
            "max_tokens": kw.get("max_tokens", 4096),
        }
        tools = _tools(kw.get("tools"))
        if tools:
            body["tools"] = tools
        effort = (kw.get("output_config") or {}).get("effort")
        if effort:
            body["reasoning"] = {"effort": effort}
        return from_openai(self._c.post(body), kw["model"])


class OpenRouterClient:
    def __init__(self, api_key: str | None = None, timeout: float = 300.0, retries: int = 3):
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is not set (add it to .env)")
        self._http = httpx.Client(
            timeout=timeout, headers={"Authorization": f"Bearer {key}", "X-Title": "foveal-bench"}
        )
        self.retries = retries
        self.messages = _Messages(self)

    def post(self, body: dict) -> dict:
        for attempt in range(self.retries + 1):
            r = self._http.post(URL, json=body)
            if r.status_code == 200:
                data = r.json()
                if "error" in data:  # some upstream errors arrive with HTTP 200
                    raise RuntimeError(f"OpenRouter error: {data['error']}")
                return data
            if r.status_code in (408, 429, 500, 502, 503, 504) and attempt < self.retries:
                time.sleep(2**attempt * 2)
                continue
            raise RuntimeError(f"OpenRouter HTTP {r.status_code}: {r.text[:300]}")
        raise AssertionError("unreachable")
