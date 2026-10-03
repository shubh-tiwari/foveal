"""README figures, rendered from the run logs in light and dark variants.

    uv run python -m bench.figures            # writes docs/figures/<name>-{light,dark}.png

Every number comes from runs/*.jsonl or the replay results. Doc-QA comparisons use only
questions answered in both modes, and success uses one scorer for all runs (the cached LLM
judge with --judge). Colors are the dataviz reference palette, validated in both modes.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from foveal.instrument.metrics import CATEGORIES, compute  # noqa: E402
from foveal.instrument.records import read_jsonl  # noqa: E402

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "text2": "#52514e",
        "grid": "#e4e3df",
        "base": "#a8a7a1",
        "foveal": "#2a78d6",
        "facts": "#1baf7a",
        "warm": "#eb6834",
    },
    "dark": {
        "surface": "#1a1a19",
        "text": "#ffffff",
        "text2": "#c3c2b7",
        "grid": "#33332f",
        "base": "#6f6e69",
        "foveal": "#3987e5",
        "facts": "#199e70",
        "warm": "#d95926",
    },
}

DOCQA = [  # (label, full-page log, foveal log)
    ("Claude Sonnet 5.5", "runs/phase0-sonnet55.jsonl", "runs/phase1-memory-v2.jsonl"),
    ("Qwen3.7 Plus", "runs/qwen-16-images.jsonl", "runs/qwen-16-memory.jsonl"),
    ("Qwen3.7 Plus,\nfigure-heavy", "runs/qwen-p1s-images.jsonl", "runs/qwen-p1s-memory.jsonl"),
]
OUT = Path("docs/figures")


def style(t: dict) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": t["surface"],
            "axes.facecolor": t["surface"],
            "savefig.facecolor": t["surface"],
            "text.color": t["text"],
            "axes.labelcolor": t["text2"],
            "axes.edgecolor": t["grid"],
            "xtick.color": t["text2"],
            "ytick.color": t["text2"],
            "axes.titlecolor": t["text"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": t["grid"],
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "legend.frameon": False,
        }
    )


# -- data -----------------------------------------------------------------------------------


def _tasks(recs: list[dict]) -> dict[str, dict]:
    return {r["task_id"]: r for r in recs if r.get("type") == "task" and "score" in r}


def per_question(path: str, ids: set[str]) -> dict[str, float]:
    """Cost and image tokens per question over `ids`, with each document's one-off ingest
    (captioning) shared across that document's answered questions."""
    recs = read_jsonl(path)
    tasks = _tasks(recs)
    cost: dict[str, float] = defaultdict(float)
    img: dict[str, float] = defaultdict(float)
    ingest_cost: dict[str, float] = defaultdict(float)
    ingest_img: dict[str, float] = defaultdict(float)
    for c in recs:
        if c.get("type") != "call":
            continue
        tok = sum(
            (i.get("tokens") or i.get("est_tokens") or 0)
            for i in c["images"]
            if i.get("kind") == "image"
        )
        tid = str(c.get("task_id"))
        if tid.startswith("ingest:"):
            doc = tid.split(":", 1)[1]
            ingest_cost[doc] += c["meta"]["cost_usd"]
            ingest_img[doc] += tok
        else:
            cost[tid] += c["meta"]["cost_usd"]
            img[tid] += tok
    per_doc = defaultdict(int)
    for t in tasks.values():
        per_doc[t["doc_id"]] += 1
    n = len(ids)
    c_tot = sum(
        cost[i] + ingest_cost[tasks[i]["doc_id"]] / per_doc[tasks[i]["doc_id"]] for i in ids
    )
    i_tot = sum(img[i] + ingest_img[tasks[i]["doc_id"]] / per_doc[tasks[i]["doc_id"]] for i in ids)
    return {"cost": c_tot / n, "image_tokens": i_tot / n}


def success(path: str, ids: set[str], scorer: Any) -> float:
    t = _tasks(read_jsonl(path))
    return sum(
        scorer(t[i]["gold"], t[i]["pred"], t[i]["answer_format"], t[i].get("question", ""))
        for i in ids
    ) / len(ids)


def docqa_rows(scorer: Any) -> list[dict]:
    rows = []
    for model, base, fov in DOCQA:
        if not (Path(base).exists() and Path(fov).exists()):
            continue
        logs = [base, fov]
        ids = set.intersection(*(set(_tasks(read_jsonl(p))) for p in logs))
        for label, p in zip(["full pages", "foveal"], logs, strict=True):
            rows.append(
                {
                    "model": model,
                    "mode": label,
                    "n": len(ids),
                    "success": success(p, ids, scorer),
                    **per_question(p, ids),
                }
            )
    return rows


# -- charts ---------------------------------------------------------------------------------


def chart_docqa(rows: list[dict], t: dict, path: Path) -> None:
    models = list(dict.fromkeys(r["model"] for r in rows))
    modes = ["full pages", "foveal"]
    colors = {"full pages": t["base"], "foveal": t["foveal"]}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8))
    for ax, key, title, fmt in (
        (axes[0], "cost", "Cost per question (USD)", lambda v: f"${v:.3f}"),
        (axes[1], "image_tokens", "Image tokens per question", lambda v: f"{v / 1000:.0f}K"),
    ):
        w = 0.36
        for j, mode in enumerate(modes):
            xs, vs = [], []
            for k, m in enumerate(models):
                r = next((r for r in rows if r["model"] == m and r["mode"] == mode), None)
                if r:
                    xs.append(k + (j - 0.5) * w)
                    vs.append(r[key])
            bars = ax.bar(
                xs, vs, w, color=colors[mode], label=mode, edgecolor=t["surface"], linewidth=2
            )
            for k, (b, v) in enumerate(zip(bars, vs, strict=True)):
                label = fmt(v)
                if mode == "foveal":
                    base = next(
                        r for r in rows if r["model"] == models[k] and r["mode"] == "full pages"
                    )
                    label += f"\n{v / base[key] - 1:+.0%}"
                ax.annotate(
                    label,
                    (b.get_x() + b.get_width() / 2, v),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color=t["text"] if mode == "foveal" else t["text2"],
                    fontweight="bold" if mode == "foveal" else "normal",
                    xytext=(0, 3),
                    textcoords="offset points",
                )
        ax.set_xticks(
            range(len(models)),
            [f"{m}\n({next(r['n'] for r in rows if r['model'] == m)} questions)" for m in models],
        )
        ax.set_title(title, loc="left")
        ax.grid(False)  # every bar carries its value
        ax.margins(y=0.22)
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
    axes[0].legend(loc="upper right")
    fig.suptitle(
        "Long-document QA (MMLongBench-Doc): full page images vs foveal memory",
        x=0.01,
        ha="left",
        fontsize=11,
        color=t["text"],
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def chart_cost_success(rows: list[dict], t: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    colors = {"full pages": t["base"], "foveal": t["foveal"], "foveal + notes": t["facts"]}
    markers = {"Claude Sonnet 5.5": "o", "Qwen3.7 Plus": "D"}
    for m in dict.fromkeys(r["model"] for r in rows):
        mr = {r["mode"]: r for r in rows if r["model"] == m}
        b, f = mr["full pages"], mr["foveal"]
        ax.annotate(
            "",
            (f["cost"], f["success"] * 100),
            (b["cost"], b["success"] * 100),
            arrowprops={
                "arrowstyle": "->",
                "color": t["text2"],
                "lw": 1.2,
                "shrinkA": 7,
                "shrinkB": 7,
            },
        )
        for mode, r in mr.items():
            ax.scatter(
                r["cost"],
                r["success"] * 100,
                s=80,
                marker=markers.get(m, "o"),
                color=colors[mode],
                edgecolor=t["surface"],
                linewidth=2,
                zorder=3,
            )
            # below the point only when the arrow runs sideways into a right-hand label
            below = mode != "full pages" and abs(r["success"] - b["success"]) < 0.05
            ax.annotate(
                f"{m.split()[0]} · {mode}\n{r['success']:.0%} correct, ${r['cost']:.3f}/question",
                (r["cost"], r["success"] * 100),
                xytext=(0, -14) if below else (10, 0),
                ha="center" if below else "left",
                va="top" if below else "center",
                textcoords="offset points",
                fontsize=8.5,
                color=t["text2"],
            )
    ax.set_xlabel("cost per question, incl. one-off captioning (USD)")
    ax.set_ylabel("questions answered correctly (%)")
    ax.set_ylim(0, 105)
    ax.set_xlim(0, max(r["cost"] for r in rows) * 1.6)
    ax.set_title("Cost vs accuracy, same questions per model (one judge)", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


POLICY_LABELS = {
    "full_history": "all screenshots",
    "last_3": "last 3 only",
    "last_1": "current only",
    "foveal_diff": "foveal diffs (all history)",
    "foveal_window3": "foveal diffs (last 3)",
}


def replay_costs(path: Path) -> dict[str, float]:
    res = json.loads(path.read_text())
    out: dict[str, float] = defaultdict(float)
    for r in res:
        for p, v in r["cost"].items():
            out[p] += v
    return dict(out)


def chart_replay(t: dict, path: Path) -> None:
    sets = [
        ("Web agent, 113 Mind2Web trajectories", Path("runs/phase2-replay/results.json")),
        ("Screen recordings, a frame every 2 s", Path("runs/phase2-video-2s/results.json")),
    ]
    sets = [(n, p) for n, p in sets if p.exists()]
    order = ["full_history", "foveal_diff", "last_3", "foveal_window3", "last_1"]
    fig, axes = plt.subplots(1, len(sets), figsize=(9.6, 3.4), sharey=True)
    for ax, (name, p) in zip(axes if len(sets) > 1 else [axes], sets, strict=True):
        c = replay_costs(p)
        ys = list(range(len(order)))[::-1]
        vals = [c[o] for o in order]
        cols = [t["foveal"] if o.startswith("foveal") else t["base"] for o in order]
        bars = ax.barh(ys, vals, 0.62, color=cols, edgecolor=t["surface"], linewidth=2)
        for b, v, o in zip(bars, vals, order, strict=True):
            ax.annotate(
                f"${v:.2f}",
                (v, b.get_y() + b.get_height() / 2),
                va="center",
                xytext=(4, 0),
                textcoords="offset points",
                fontsize=9,
                color=t["text"] if o.startswith("foveal") else t["text2"],
                fontweight="bold" if o.startswith("foveal") else "normal",
            )
        ax.set_yticks(ys, [POLICY_LABELS[o] for o in order])
        ax.tick_params(axis="y", length=0)
        ax.set_title(name, loc="left", fontsize=10)
        ax.grid(False)
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)
        ax.margins(x=0.25)
    fig.suptitle(
        "Screenshot history: estimated cost per policy (Sonnet 5.5 prices, "
        "prompt caching where the prefix allows)",
        x=0.01,
        ha="left",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def chart_reperception(t: dict, path: Path) -> None:
    import numpy as np

    s = compute(read_jsonl("runs/phase0-sonnet55.jsonl"))
    tasks = [x for x in s["tasks"] if x.cumulative]
    max_len = max(len(x.cumulative) for x in tasks)
    stack = np.zeros((len(CATEGORIES), max_len))
    for x in tasks:
        last = dict.fromkeys(CATEGORIES, 0)
        for j in range(max_len):
            if j < len(x.cumulative):
                last = x.cumulative[j][1]
            for k, cat in enumerate(CATEGORIES):
                stack[k, j] += last[cat]
    stack /= len(tasks)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    labels = ["first time seen", "re-sent by the same agent", "re-read by another agent"]
    colors = [t["foveal"], t["warm"], t["base"]]
    xs = range(1, max_len + 1)
    ax.stackplot(
        xs, stack / 1000, colors=colors, labels=labels, edgecolor=t["surface"], linewidth=1.5
    )
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_xlim(1, max_len)
    ax.set_xlabel("model call within a question (all agents)")
    ax.set_ylabel("cumulative image tokens (thousands)")
    ax.set_title(
        f"Without foveal, {s['reperception_rate']:.0%} of image tokens are pixels already sent",
        loc="left",
    )
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", action="store_true", help="score with the cached LLM judge")
    ap.add_argument(
        "--all", action="store_true", help="also render the extra charts (not used in the README)"
    )
    args = ap.parse_args(argv)
    sys.path.insert(0, str(Path.cwd()))
    from bench.mmlongbench.score import score

    scorer: Any = score
    if args.judge:
        import anthropic
        from dotenv import load_dotenv

        from bench.mmlongbench.judge import Judge

        load_dotenv(".env")
        scorer = Judge(anthropic.Anthropic())
    rows = docqa_rows(scorer)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "docqa.json").write_text(json.dumps(rows, indent=1))
    for name, t in THEMES.items():
        style(t)
        chart_docqa(rows, t, OUT / f"docqa-{name}.png")  # the README chart
        if args.all:
            chart_cost_success(rows, t, OUT / f"cost-vs-accuracy-{name}.png")
            chart_replay(t, OUT / f"screens-{name}.png")
            chart_reperception(t, OUT / f"reperception-{name}.png")
    for r in rows:
        print(
            f"{r['model']:18} {r['mode']:15} n={r['n']:2} success {r['success']:.0%} "
            f"cost/q ${r['cost']:.4f} image tok/q {r['image_tokens']:,.0f}"
        )
    print(f"figures in {OUT}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
