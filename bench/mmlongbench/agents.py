"""Orchestrator + reader sub-agents for long-document QA.

The orchestrator never sees text extracted from the PDF - only its title, page count and
outline. It delegates reading to reader sub-agents (`ask_reader`, parallel allowed) and can
look at pages itself (`view_pages`). Readers start with the pages they were given and may
request more. Every history is append-only; images stay in context and are resent on every
call, as in a typical harness.
"""

from __future__ import annotations

import contextvars
import itertools
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from bench.mmlongbench.render import PageRenderer
from foveal.instrument import Tracer
from foveal.instrument.client import current_scope

FALLBACK_BETA = "server-side-fallback-2026-07-01"
MAX_PAGES_PER_CALL = 8


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class AgentConfig:
    orch_model: str = "claude-sonnet-5-5"
    reader_model: str = "claude-sonnet-5-5"
    effort: str = "medium"
    cache: bool = True
    fallbacks: bool = True
    max_orch_steps: int = 15
    max_reader_steps: int = 8
    max_tokens: int = 16000
    max_cost_usd: float = 6.0
    provider: str = "anthropic"  # anthropic | openrouter


ORCH_SYSTEM = """You answer one question about a long PDF document that you cannot read \
directly. You know its title, page count and (when available) its outline.

Tools:
- ask_reader(pages, instruction): a reader sub-agent looks at those pages (at most 8) and \
answers your instruction. Call several readers in parallel to cover different page ranges.
- view_pages(pages): look at pages yourself (at most 8 per call), e.g. to verify a reader's claim.
- final_answer(answer): submit your answer. Call it exactly once, when you are done.

Pages are numbered from 1. Give the final answer as short as possible: a number, a short \
phrase, or a JSON list for multi-part answers. If the document does not contain the answer, \
answer "Not answerable"."""

READER_SYSTEM = """You are a reader sub-agent. You are shown pages of a PDF document and an \
instruction from the orchestrator. Answer using only what the pages show. If you need other \
pages, call view_pages (at most 8 pages per call). Finish with a concise answer that cites \
page numbers, or say clearly that the pages do not contain the information."""

PAGES_SCHEMA = {
    "type": "array",
    "items": {"type": "integer"},
    "description": "1-indexed page numbers",
}

ORCH_TOOLS = [
    {
        "name": "ask_reader",
        "description": "Delegate reading of specific pages to a sub-agent.",
        "input_schema": {
            "type": "object",
            "properties": {"pages": PAGES_SCHEMA, "instruction": {"type": "string"}},
            "required": ["pages", "instruction"],
            "additionalProperties": False,
        },
    },
    {
        "name": "view_pages",
        "description": "Look at document pages yourself.",
        "input_schema": {
            "type": "object",
            "properties": {"pages": PAGES_SCHEMA},
            "required": ["pages"],
            "additionalProperties": False,
        },
    },
    {
        "name": "final_answer",
        "description": "Submit the final answer and end the task.",
        "input_schema": {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        },
    },
]

READER_TOOLS = [ORCH_TOOLS[1]]

LOOK_TOOL = {
    "name": "look",
    "description": (
        "See a page as an image. detail='glimpse' is a low-resolution thumbnail (about 256 "
        "tokens) for layout and finding things; detail='full' is full resolution. region="
        "[x0, y0, x1, y1] as fractions of the page (0-1) zooms into part of a page, e.g. one "
        "chart, at higher resolution. Prefer the text layer when it already answers."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "page": {"type": "integer", "description": "1-indexed page number"},
            "detail": {"type": "string", "enum": ["glimpse", "full"]},
            "region": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 4,
                "maxItems": 4,
                "description": "[x0, y0, x1, y1] fractions of page width/height",
            },
        },
        "required": ["page", "detail"],
        "additionalProperties": False,
    },
}

MEMORY_NOTE = """

Pages are shown as a short caption plus their text layer, not as images. Charts, figures, \
photos and visual layout are NOT in the text layer: call look(page, detail, region) to see \
them. Use a glimpse to find where something is on a page, then look at that region with \
detail='full' to read it."""

MEMORY_NOTE_ON_DEMAND = """

Pages are first shown as a short caption and a text preview, not as images. Call \
view_pages(pages) to read a page's full text layer when the preview is not enough. Charts, \
figures, photos and visual layout are NOT in the text layer: call look(page, detail, region) \
to see them. Use a glimpse to find where something is on a page, then look at that region \
with detail='full' to read it."""

NOTE_TOOL = {
    "name": "note",
    "description": (
        "Record a fact you established from a page so other agents and later questions can "
        "reuse it instead of re-reading. Be specific and self-contained (include the "
        "subject, number and unit). region=[x0, y0, x1, y1] fractions of the page where it is."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "page": {"type": "integer"},
            "claim": {"type": "string"},
            "region": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
            "seen_at": {"type": "string", "enum": ["text", "glimpse", "full"]},
        },
        "required": ["page", "claim"],
        "additionalProperties": False,
    },
}

RECALL_TOOL = {
    "name": "recall_notes",
    "description": "Search notes (facts) other agents recorded about this document.",
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
        "additionalProperties": False,
    },
}

FACTS_NOTE = """

Agents share notes. Earlier notes about this document are listed with the question; use \
them instead of re-reading pages when they answer it, and verify with look when unsure. \
When you establish a fact relevant to the question, record it with note(page, claim, region)."""

VIEW_PAGES_MEMORY = {
    **ORCH_TOOLS[1],
    "description": "Read pages' captions and full text layers (no images; use look for visuals).",
}


@dataclass
class TaskResult:
    answer: str | None
    orch_steps: int
    readers: int
    stop: str  # final_answer | refusal | max_steps | no_tool | budget | error
    notes: list[str] = field(default_factory=list)


def _is_high_tier(model: str) -> bool:
    return not model.startswith(
        ("claude-haiku", "claude-sonnet-4", "claude-opus-4-5", "claude-opus-4-6", "claude-3")
    )


class DocQA:
    def __init__(self, client: Any, tracer: Tracer, cfg: AgentConfig):
        self.client = client
        self.tracer = tracer
        self.cfg = cfg

    @staticmethod
    def _tools(doc: PageRenderer, role: str) -> tuple[str, list]:
        if doc.images:
            return (ORCH_SYSTEM, ORCH_TOOLS) if role == "orch" else (READER_SYSTEM, READER_TOOLS)
        note = MEMORY_NOTE_ON_DEMAND if getattr(doc, "text_on_demand", False) else MEMORY_NOTE
        facts = getattr(doc, "facts", False)
        if facts:
            note += FACTS_NOTE
        if role == "orch":
            tools = [ORCH_TOOLS[0], VIEW_PAGES_MEMORY, LOOK_TOOL]
            tools += [RECALL_TOOL, NOTE_TOOL] if facts else []
            return ORCH_SYSTEM + note, [*tools, ORCH_TOOLS[2]]
        return READER_SYSTEM + note, [VIEW_PAGES_MEMORY, LOOK_TOOL, *([NOTE_TOOL] if facts else [])]

    def _page_tool(self, doc: PageRenderer, u: Any) -> dict:
        """Result for view_pages / look / note / recall_notes tool calls (both roles)."""
        if u.name == "recall_notes":
            return {
                "type": "tool_result",
                "tool_use_id": u.id,
                "content": doc.notes(str(u.input.get("query", ""))),
            }
        if u.name == "note":
            page = u.input.get("page")
            if not isinstance(page, int) or not 1 <= page <= doc.page_count:
                return {
                    "type": "tool_result",
                    "tool_use_id": u.id,
                    "is_error": True,
                    "content": f"page must be an integer 1-{doc.page_count}",
                }
            s = current_scope()
            level = {"text": "L1", "glimpse": "L2", "full": "L3"}.get(
                u.input.get("seen_at", "full"), "L3"
            )
            try:
                msg = doc.note(
                    page,
                    str(u.input.get("claim", "")),
                    u.input.get("region"),
                    author=f"{s.task_id}/{s.agent_id}",
                    level=level,
                )
            except ValueError as e:
                return {
                    "type": "tool_result",
                    "tool_use_id": u.id,
                    "is_error": True,
                    "content": str(e),
                }
            return {"type": "tool_result", "tool_use_id": u.id, "content": msg}
        if u.name == "look":
            page = u.input.get("page")
            if not isinstance(page, int) or not 1 <= page <= doc.page_count:
                return {
                    "type": "tool_result",
                    "tool_use_id": u.id,
                    "is_error": True,
                    "content": f"page must be an integer 1-{doc.page_count}",
                }
            try:
                blocks = doc.look(page, u.input.get("region"), u.input.get("detail", "glimpse"))
            except ValueError as e:
                return {
                    "type": "tool_result",
                    "tool_use_id": u.id,
                    "is_error": True,
                    "content": str(e),
                }
            return {"type": "tool_result", "tool_use_id": u.id, "content": blocks}
        pg, err = self._valid_pages(u.input.get("pages"), doc.page_count)
        if err:
            return {"type": "tool_result", "tool_use_id": u.id, "content": err, "is_error": True}
        return {"type": "tool_result", "tool_use_id": u.id, "content": doc.read_blocks(pg)}

    # -- model call -------------------------------------------------------

    def _call(self, model: str, system: str, messages: list, tools: list) -> Any:
        if self.tracer.total_cost_usd >= self.cfg.max_cost_usd:
            raise BudgetExceeded(f"spent ${self.tracer.total_cost_usd:.2f}")
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": self.cfg.max_tokens,
            "system": system,
            "messages": messages,
            "tools": tools,
        }
        if self.cfg.provider == "openrouter":  # Anthropic-only params do not apply
            kwargs["output_config"] = {"effort": self.cfg.effort}
            return self.client.messages.create(**kwargs)
        betas: list[str] = []
        if _is_high_tier(model):
            kwargs["thinking"] = {"type": "adaptive"}
            kwargs["output_config"] = {"effort": self.cfg.effort}
        if self.cfg.cache:
            kwargs["cache_control"] = {"type": "ephemeral"}
        if self.cfg.fallbacks and _is_high_tier(model):
            betas.append(FALLBACK_BETA)
            kwargs["extra_body"] = {"fallbacks": "default"}
        if betas:
            kwargs["betas"] = betas
            return self.client.beta.messages.create(**kwargs)
        return self.client.messages.create(**kwargs)

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _valid_pages(pages: Any, n: int) -> tuple[list[int], str | None]:
        if not isinstance(pages, list) or not pages:
            return [], "pages must be a non-empty list of integers"
        clean = []
        for p in pages:
            if isinstance(p, int) and 1 <= p <= n and p not in clean:
                clean.append(p)
        if not clean:
            return [], f"no valid pages; the document has pages 1-{n}"
        if len(clean) > MAX_PAGES_PER_CALL:
            return [], f"at most {MAX_PAGES_PER_CALL} pages per call"
        return clean, None

    @staticmethod
    def _text(resp: Any) -> str:
        return "\n".join(b.text for b in resp.content if getattr(b, "type", None) == "text")

    # -- reader -----------------------------------------------------------

    def run_reader(
        self, doc: PageRenderer, agent_id: str, pages: list[int], instruction: str
    ) -> str:
        with self.tracer.scope(agent_id=agent_id):
            messages: list[dict] = [
                {
                    "role": "user",
                    "content": [
                        *doc.page_blocks(pages),
                        {
                            "type": "text",
                            "text": f"Document has {doc.page_count} pages.\n\n"
                            f"Instruction: {instruction}",
                        },
                    ],
                }
            ]
            system, tools = self._tools(doc, "reader")
            for _ in range(self.cfg.max_reader_steps):
                resp = self._call(self.cfg.reader_model, system, messages, tools)
                messages.append({"role": "assistant", "content": resp.content})
                if resp.stop_reason == "refusal":
                    return "[reader declined the request]"
                uses = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]
                if resp.stop_reason != "tool_use" or not uses:
                    return self._text(resp) or "[reader returned no text]"
                results = [self._page_tool(doc, u) for u in uses]
                messages.append({"role": "user", "content": results})
            return self._text(resp) or "[reader hit its step limit without an answer]"

    # -- orchestrator -----------------------------------------------------

    def run_task(self, doc: PageRenderer, question: str) -> TaskResult:
        reader_ids = itertools.count(1)
        id_lock = threading.Lock()
        toc = doc.toc()
        outline = (
            "\n".join(f"{'  ' * (lvl - 1)}- {title} (p. {page})" for lvl, title, page in toc[:80])
            or "(no outline available)"
        )
        messages: list[dict] = [
            {
                "role": "user",
                "content": (
                    f"Document: {doc.title()}\nPages: {doc.page_count}\nOutline:\n{outline}\n\n"
                    f"Question: {question}"
                    + (
                        "\n\nNotes from earlier work on this document (by other agents; "
                        f"verify with look if unsure):\n{doc.notes(question)}"
                        if getattr(doc, "facts", False)
                        else ""
                    )
                ),
            }
        ]
        readers = 0
        system, tools = self._tools(doc, "orch")
        with self.tracer.scope(agent_id="orch"):
            for step in range(1, self.cfg.max_orch_steps + 1):
                resp = self._call(self.cfg.orch_model, system, messages, tools)
                messages.append({"role": "assistant", "content": resp.content})
                if resp.stop_reason == "refusal":
                    return TaskResult(None, step, readers, "refusal")
                uses = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]
                final = next((u for u in uses if u.name == "final_answer"), None)
                if final is not None:
                    return TaskResult(str(final.input.get("answer")), step, readers, "final_answer")
                if not uses:
                    messages.append(
                        {"role": "user", "content": "Please call final_answer with your answer."}
                    )
                    continue

                def handle(u: Any) -> dict:
                    if u.name != "ask_reader":
                        return self._page_tool(doc, u)
                    pg, err = self._valid_pages(u.input.get("pages"), doc.page_count)
                    if err:
                        return {
                            "type": "tool_result",
                            "tool_use_id": u.id,
                            "content": err,
                            "is_error": True,
                        }
                    with id_lock:
                        rid = f"reader-{next(reader_ids)}"
                    answer = self.run_reader(doc, rid, pg, str(u.input.get("instruction", "")))
                    return {
                        "type": "tool_result",
                        "tool_use_id": u.id,
                        "content": f"[{rid}, pages {pg}] {answer}",
                    }

                n_readers = sum(1 for u in uses if u.name == "ask_reader")
                readers += n_readers
                if n_readers > 1:
                    with ThreadPoolExecutor(max_workers=min(8, len(uses))) as pool:
                        futs = [
                            pool.submit(contextvars.copy_context().run, handle, u) for u in uses
                        ]
                        results = [f.result() for f in futs]
                else:
                    results = [handle(u) for u in uses]
                messages.append({"role": "user", "content": results})
        return TaskResult(None, self.cfg.max_orch_steps, readers, "max_steps")
