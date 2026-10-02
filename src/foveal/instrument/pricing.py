"""Per-model prices (USD per million tokens), first-party Claude API, cached 2026-09-25.

Cache writes (5-minute TTL) bill at 1.25x input; cache reads at the listed rate.
"""

from __future__ import annotations

from typing import Any

# model prefix -> (input, output, cache_read)
PRICES: dict[str, tuple[float, float, float]] = {
    "claude-fable-5-1": (10.0, 50.0, 0.25),
    "claude-fable-5": (10.0, 50.0, 1.0),
    "claude-opus-5-5": (4.0, 20.0, 0.20),
    "claude-opus-5": (5.0, 25.0, 0.50),
    "claude-opus-4-8": (5.0, 25.0, 0.50),
    "claude-opus-4-7": (5.0, 25.0, 0.50),
    "claude-opus-4-6": (5.0, 25.0, 0.50),
    "claude-sonnet-5-5": (2.0, 10.0, 0.20),
    "claude-sonnet-5": (2.0, 10.0, 0.20),
    "claude-sonnet-4-6": (3.0, 15.0, 0.30),
    "claude-haiku-4-5": (1.0, 5.0, 0.10),
}
CACHE_WRITE_MULT = 1.25


def prices_for(model: str) -> tuple[float, float, float]:
    for prefix in sorted(PRICES, key=len, reverse=True):
        if model.startswith(prefix):
            return PRICES[prefix]
    return PRICES["claude-opus-5-5"]


def call_cost_usd(model: str, usage: dict[str, Any]) -> float:
    inp, out, cread = prices_for(model)
    u = usage or {}
    return (
        (u.get("input_tokens") or 0) * inp
        + (u.get("cache_creation_input_tokens") or 0) * inp * CACHE_WRITE_MULT
        + (u.get("cache_read_input_tokens") or 0) * cread
        + (u.get("output_tokens") or 0) * out
    ) / 1e6
