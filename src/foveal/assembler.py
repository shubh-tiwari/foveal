"""Budget-aware, prompt-cache-aware context assembly.

Two decisions:

1. **Levels at insertion.** When an asset enters the context it gets a level - L0 handle,
   L1 text, L2 glimpse or L3 full - chosen by relevance to the current question and by the
   remaining token budget. Once inserted, a block never changes, so the history stays an
   append-only prefix that prompt caching (and preserved thinking) can reuse.

2. **Batch demotion (compaction).** Old high-level assets keep costing cache reads on every
   later call. Rewriting them to a lower level saves that, but costs one cache write of the
   whole rewritten prefix (and drops later thinking blocks on models that bind them to the
   history). `CacheModel.should_compact` rewrites only when the savings over the expected
   remaining calls exceed that one-off cost - and then demotes everything due at once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from foveal.facts import tokens as words
from foveal.instrument.pricing import CACHE_WRITE_MULT, prices_for

LEVEL_ORDER = ("L0", "L1", "L2", "L3")
TEXT_TOKENS_PER_CHAR = 0.27  # rough; English text averages ~3.7 chars per token


@dataclass
class Item:
    """One asset in a conversation."""

    asset_id: str
    added_at: int  # call index when it entered the context
    level: str = "L0"
    tokens: int = 0


@dataclass
class Context:
    blocks: list[dict[str, Any]]
    tokens: int
    levels: dict[str, str]


@dataclass
class CacheModel:
    """Expected cost of keeping a cached prefix versus rewriting it smaller."""

    model: str = "claude-sonnet-5-5"
    thinking_bound: bool = True  # model invalidates thinking blocks after edited history

    def keep_cost(self, prefix_tokens: int, calls_left: int) -> float:
        _, _, cread = prices_for(self.model)
        return prefix_tokens * cread * calls_left / 1e6

    def rewrite_cost(self, new_prefix_tokens: int, calls_left: int) -> float:
        inp, _, cread = prices_for(self.model)
        write = new_prefix_tokens * inp * CACHE_WRITE_MULT / 1e6
        reads = new_prefix_tokens * cread * max(0, calls_left - 1) / 1e6
        return write + reads

    def should_compact(
        self, prefix_tokens: int, compacted_tokens: int, calls_left: int, margin: float = 1.2
    ) -> bool:
        """Rewrite only if it is clearly cheaper over the remaining calls. With
        `thinking_bound`, require a larger margin: rewritten history drops the model's
        earlier reasoning, a quality cost the token price does not capture."""
        if compacted_tokens >= prefix_tokens or calls_left <= 0:
            return False
        m = margin * (1.5 if self.thinking_bound else 1.0)
        return self.keep_cost(prefix_tokens, calls_left) > m * self.rewrite_cost(
            compacted_tokens, calls_left
        )


@dataclass
class Assembler:
    """Chooses levels for assets under a token budget, using a foveal Memory for content."""

    memory: Any
    budget_tokens: int = 8000
    recent: int = 2  # the newest N assets are promoted at least to L2
    cache: CacheModel = field(default_factory=CacheModel)

    # -- costs ----------------------------------------------------------------

    def level_tokens(self, asset_id: str, level: str) -> int:
        a = self.memory.get(asset_id)
        if level == "L0":
            return 12 + int(len(a.caption or "") * TEXT_TOKENS_PER_CHAR)
        if level == "L1":
            return 12 + int(min(len(a.text or ""), 4000) * TEXT_TOKENS_PER_CHAR)
        detail = "glimpse" if level == "L2" else "full"
        return self.memory.look(asset_id, detail=detail).tokens + 12

    def relevance(self, asset_id: str, question: str) -> float:
        """Word overlap between the question and the asset's caption and text."""
        if not question:
            return 0.0
        a = self.memory.get(asset_id)
        q = set(words(question))
        doc = set(words((a.caption or "") + " " + (a.text or "")[:4000]))
        return len(q & doc) / (len(q) or 1)

    # -- decision 1: level at insertion --------------------------------------------

    def choose(
        self, asset_ids: list[str], question: str = "", used_tokens: int = 0
    ) -> dict[str, str]:
        """Levels for new assets: everyone starts at L0; upgrades go to the most relevant
        (and the most recent) first, while the budget lasts."""
        levels = dict.fromkeys(asset_ids, "L0")
        spent = used_tokens + sum(self.level_tokens(i, "L0") for i in asset_ids)
        order = sorted(
            range(len(asset_ids)), key=lambda k: (-self.relevance(asset_ids[k], question), -k)
        )
        recent = set(asset_ids[-self.recent :]) if self.recent else set()
        for k in order:
            aid = asset_ids[k]
            rel = self.relevance(aid, question)
            target = (
                "L3"
                if rel >= 0.5
                else "L2"
                if (rel >= 0.2 or aid in recent)
                else "L1"
                if rel > 0
                else "L0"
            )
            for lvl in LEVEL_ORDER[1 : LEVEL_ORDER.index(target) + 1]:
                extra = self.level_tokens(aid, lvl) - self.level_tokens(aid, levels[aid])
                if spent + extra > self.budget_tokens:
                    break
                spent += extra
                levels[aid] = lvl
        return levels

    def render(self, asset_id: str, level: str) -> list[dict[str, Any]]:
        a = self.memory.get(asset_id)
        if level in ("L0", "L1"):
            lv = ("L0",) if level == "L0" else ("L0", "L1")
            return [{"type": "text", "text": self.memory.describe(asset_id, levels=lv)}]
        view = self.memory.look(asset_id, detail="glimpse" if level == "L2" else "full")
        head = f"[asset {a.asset_id[:12]} | {level}] {a.caption or ''}".strip()
        return [{"type": "text", "text": head}, view.block()]

    def build(self, asset_ids: list[str], question: str = "") -> Context:
        levels = self.choose(asset_ids, question)
        blocks: list[dict[str, Any]] = []
        for aid in asset_ids:
            blocks += self.render(aid, levels[aid])
        total = sum(self.level_tokens(a, levels[a]) for a in asset_ids)
        return Context(blocks, total, levels)

    # -- decision 2: batch demotion -------------------------------------------------

    def plan_compaction(
        self, items: list[Item], calls_left: int, keep_recent: int = 3
    ) -> dict[str, str] | None:
        """Demotions to apply in one rewrite, or None if keeping the cache is cheaper.
        Items older than the `keep_recent` newest drop to L0."""
        if len(items) <= keep_recent:
            return None
        newest = {i.asset_id for i in sorted(items, key=lambda i: i.added_at)[-keep_recent:]}
        plan = {i.asset_id: "L0" for i in items if i.asset_id not in newest and i.level != "L0"}
        if not plan:
            return None
        prefix = sum(i.tokens for i in items)
        compacted = prefix - sum(
            i.tokens - self.level_tokens(i.asset_id, "L0") for i in items if i.asset_id in plan
        )
        return plan if self.cache.should_compact(prefix, compacted, calls_left) else None
