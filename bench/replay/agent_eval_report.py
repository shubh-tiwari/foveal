"""Per-policy accuracy, image tokens and cost for an agent_eval run log."""

from __future__ import annotations

import sys
from collections import defaultdict

from foveal.instrument.records import read_jsonl


def summarize(path: str) -> str:
    recs = read_jsonl(path)
    acc: dict[str, list[float]] = defaultdict(list)
    elem: dict[str, list[float]] = defaultdict(list)
    tok: dict[str, int] = defaultdict(int)
    cost: dict[str, float] = defaultdict(float)
    for r in recs:
        if r.get("type") == "task":
            acc[r["policy"]].append(r["score"])
            elem[r["policy"]].append(float(r["element_ok"]))
        elif r.get("type") == "call":
            p = (r.get("meta") or {}).get("policy")
            tok[p] += sum((i.get("tokens") or i.get("est_tokens") or 0) for i in r["images"])
            cost[p] += (r.get("meta") or {}).get("cost_usd", 0.0)
    base = tok.get("full_history") or 1
    lines = [
        "| Policy | Steps | Element acc. | Element+op acc. | Image tokens | vs full | Cost |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for p in acc:
        n = len(acc[p])
        lines.append(
            f"| {p} | {n} | {sum(elem[p]) / n:.1%} | {sum(acc[p]) / n:.1%} | "
            f"{tok[p]:,} | {tok[p] / base - 1:+.0%} | ${cost[p]:.2f} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    print(summarize(sys.argv[1]))
