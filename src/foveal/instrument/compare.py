"""Compare a foveal run against a baseline run on the same tasks (Phase 1 gate)."""

from __future__ import annotations

from typing import Any

from foveal.instrument.metrics import compute


def _tasks(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {r["task_id"]: r for r in records if r.get("type") == "task" and "score" in r}


def compare(
    base: list[dict[str, Any]], new: list[dict[str, Any]], rescore: Any = None, ratio: float = 0.95
) -> tuple[str, bool]:
    """Markdown comparison. `rescore(gold, pred, fmt, question)` re-scores both runs alike."""
    sb, sn = compute(base), compute(new)
    tb, tn = _tasks(base), _tasks(new)
    shared = [t for t in tb if t in tn]

    def sc(r: dict[str, Any]) -> float:
        if rescore is None:
            return float(r["score"])
        return float(rescore(r["gold"], r["pred"], r["answer_format"], r.get("question", "")))

    base_ok = sum(sc(tb[t]) for t in shared)
    new_ok = sum(sc(tn[t]) for t in shared)
    agree = sum(sc(tb[t]) == sc(tn[t]) for t in shared)
    no_worse = sum(sc(tn[t]) >= sc(tb[t]) for t in shared)
    ok = bool(shared) and new_ok >= ratio * base_ok
    tok_b = sb["image_tokens_total"] + sb.get("ingest_image_tokens", 0)
    tok_n = sn["image_tokens_total"] + sn.get("ingest_image_tokens", 0)
    cost_b, cost_n = sb["cost_usd_with_ingest"], sn["cost_usd_with_ingest"]

    def delta(a: float, b: float) -> str:
        return f"{(b - a) / a:+.0%}" if a else "n/a"

    n = len(shared)
    lines = [
        "# foveal comparison",
        "",
        f"**Gate:** {'PASS' if ok else 'FAIL'}: success {new_ok:.0f}/{n} vs baseline "
        f"{base_ok:.0f}/{n} (needs >= {ratio:.0%} of baseline)",
        "",
        "| Metric | Baseline | foveal | Change |",
        "| --- | --- | --- | --- |",
        f"| Success | {base_ok:.0f}/{n} | {new_ok:.0f}/{n} | {new_ok - base_ok:+.0f} |",
        f"| Image tokens (tasks + ingest) | {tok_b:,} | {tok_n:,} | {delta(tok_b, tok_n)} |",
        f"| Ingest image tokens | {sb.get('ingest_image_tokens', 0):,} | "
        f"{sn.get('ingest_image_tokens', 0):,} | |",
        f"| Re-perception rate | {sb['reperception_rate']:.1%} | {sn['reperception_rate']:.1%} | |",
        f"| Cost incl. ingest | ${cost_b:.2f} | ${cost_n:.2f} | {delta(cost_b, cost_n)} |",
        f"| Model calls | {sb['n_calls']} | {sn['n_calls']} | |",
        f"| Same score as baseline | | {agree}/{n} | |",
        f"| No worse than baseline | | {no_worse}/{n} | |",
        "",
        "| Task | Base score | foveal score | Base img tokens | foveal img tokens |",
        "| --- | --- | --- | --- | --- |",
    ]
    bt = {t.task_id: t for t in sb["tasks"]}
    nt = {t.task_id: t for t in sn["tasks"]}
    for t in shared:
        lines.append(
            f"| {t} | {sc(tb[t]):.0f} | {sc(tn[t]):.0f} | "
            f"{bt[t].total if t in bt else 0:,} | {nt[t].total if t in nt else 0:,} |"
        )
    return "\n".join(lines) + "\n", ok
