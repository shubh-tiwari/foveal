"""Markdown summary and plots for a Phase 0 run."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from foveal.instrument.metrics import CATEGORIES, gate


def markdown(summary: dict[str, Any], gate_threshold: float = 0.40, keep_last: int = 3) -> str:
    s = summary
    _, verdict = gate(s, gate_threshold)

    def pct(x: float | None) -> str:
        return "n/a" if x is None else f"{x:.1%}"

    rows = [
        ("Tasks / model calls", f"{s['n_tasks']} / {s['n_calls']}"),
        ("Image tokens, total", f"{s['image_tokens_total']:,}"),
        ("Image tokens per task (median)", f"{s['image_tokens_per_task_median']:,.0f}"),
        ("**Re-perception rate**", f"**{pct(s['reperception_rate'])}**"),
        ("  same-agent resend", pct(s["same_agent_share"])),
        ("  cross-agent duplicate", pct(s["cross_agent_share"])),
        ("Near-duplicate share of first-seen (pHash)", pct(s["near_dup_share_of_first_seen"])),
        ("Cross-task share of first-seen (same doc)", pct(s["cross_task_share_of_first_seen"])),
        (f"Image tokens under keep-last-{keep_last}", f"{s[f'keep_last_{keep_last}_tokens']:,}"),
        (
            f"Re-perception left under keep-last-{keep_last}",
            pct(s[f"reperception_under_keep_last_{keep_last}"]),
        ),
        ("Repeated tokens served from cache (est.)", pct(s["repeated_tokens_cached_share"])),
        (
            "Repeated-image cost: no cache / est. actual",
            f"${s['repeated_cost_usd_no_cache']:.2f} / ${s['repeated_cost_usd_est_actual']:.2f}",
        ),
        ("Success rate", pct(s["success_rate"])),
        ("Run cost (from usage)", f"${s['cost_usd_total']:.2f}"),
        (
            "Ingest (captions): calls / image tokens / cost",
            f"{s.get('ingest_calls', 0)} / {s.get('ingest_image_tokens', 0):,} / "
            f"${s.get('ingest_cost_usd', 0.0):.2f}",
        ),
        ("Mean latency per call", f"{s['latency_s_per_call_mean']:.1f}s"),
    ]
    lines = [
        "# foveal Phase 0 report",
        "",
        f"**Gate:** {verdict}",
        "",
        "| Metric | Value |",
        "| --- | --- |",
    ]
    lines += [f"| {k} | {v} |" for k, v in rows]
    lines += [
        "",
        "## Image tokens by agent role",
        "",
        "| Role | first seen | same-agent | cross-agent |",
        "| --- | --- | --- | --- |",
    ]
    for role, d in sorted(s["per_agent_role"].items()):
        lines.append(f"| {role} | " + " | ".join(f"{d[c]:,}" for c in CATEGORIES) + " |")
    lines += [
        "",
        "## Per task",
        "",
        "| Task | Doc | Calls | Image tokens | Re-perception | Unique images | Score |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for t in s["tasks"]:
        lines.append(
            f"| {t.task_id} | {t.doc_id or ''} | {t.n_calls} | {t.total:,} | "
            f"{t.reperception_rate:.1%} | {t.unique_images} | "
            f"{'' if t.score is None else t.score} |"
        )
    return "\n".join(lines) + "\n"


def plot_cumulative(summary: dict[str, Any], out: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    tasks = [t for t in summary["tasks"] if t.cumulative]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    max_len = max((len(t.cumulative) for t in tasks), default=0)
    totals = np.full((len(tasks), max_len), np.nan)
    for i, t in enumerate(tasks):
        ys = [sum(c.values()) for _, c in t.cumulative]
        totals[i, : len(ys)] = ys
        ax1.plot(range(1, len(ys) + 1), ys, color="#9aa5b1", lw=0.8, alpha=0.7)
    if max_len:
        alive = (~np.isnan(totals)).sum(axis=0)
        span = int((alive >= min(3, len(tasks))).sum())  # only where >= 3 tasks still run
        med = np.nanmedian(totals[:, :span], axis=0)
        ax1.plot(range(1, span + 1), med, color="#1f4e79", lw=2.2, label="median task")
        ax1.legend(frameon=False)
    ax1.set_xlabel("model call within task (all agents)")
    ax1.set_ylabel("cumulative image tokens")
    ax1.set_title("Image tokens grow with every call")

    stack = np.zeros((len(CATEGORIES), max_len))
    for t in tasks:
        last = dict.fromkeys(CATEGORIES, 0)
        for j in range(max_len):
            if j < len(t.cumulative):
                last = t.cumulative[j][1]
            for k, c in enumerate(CATEGORIES):
                stack[k, j] += last[c]
    if len(tasks):
        stack /= len(tasks)
    labels = ["first seen", "same-agent resend", "cross-agent duplicate"]
    colors = ["#1f4e79", "#e07b39", "#c9a227"]
    ax2.stackplot(range(1, max_len + 1), stack, labels=labels, colors=colors, alpha=0.9)
    ax2.set_xlabel("model call within task (all agents)")
    ax2.set_ylabel("mean cumulative image tokens")
    ax2.set_title(f"Re-perception: {summary['reperception_rate']:.0%} of image tokens")
    ax2.legend(loc="upper left", frameon=False)
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
