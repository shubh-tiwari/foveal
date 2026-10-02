"""Offline replay: image tokens and cost of history policies on recorded trajectories.

No model calls. For each trajectory, at each step k, a policy decides which image content
the model call would contain:

  full_history   every screenshot so far, full (typical harness; append-only)
  last_1/last_3  only the most recent 1 or 3 screenshots (the common workaround)
  foveal_diff    first screenshot full, then only changed regions (+ scroll note) per step,
                 each diffed against the best of the last 5 frames; kept append-only, so it
                 carries the same information as full_history
  foveal_window3 the last 3 steps as one full frame + diffs (same information as last_3)

Cost uses Claude Sonnet 5.5 prices. Append-only policies get prompt caching (earlier content
read at the cache price, new content written at 1.25x); sliding windows change the prefix
every step, so their images are billed at the full input price.

    uv run python -m bench.replay.simulate --shards 1
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from bench.replay.mind2web import Trajectory, load
from foveal.diff import FrameDiff, describe, diff_frames, reconstruct
from foveal.instrument.pricing import CACHE_WRITE_MULT, prices_for
from foveal.instrument.tokens import estimate_image_tokens

MODEL = "claude-sonnet-5-5"
TEXT_PER_DIFF = 40  # rough text tokens for the diff description per step
POLICIES = ["full_history", "last_1", "last_3", "foveal_diff", "foveal_window3"]
REF_WINDOW = 5  # foveal_diff may diff against any of the last 5 frames


def frame_tokens(w: int, h: int) -> int:
    return estimate_image_tokens(w, h, MODEL)


def diff_tokens(d: FrameDiff, w: int, h: int) -> int:
    """Image + text tokens to transmit a diff instead of the frame."""
    if d.kind == "full":
        return frame_tokens(w, h)
    crops = sum(frame_tokens(b.w, b.h) for b in d.regions)
    return min(frame_tokens(w, h), crops) + TEXT_PER_DIFF


def replay(t: Trajectory) -> dict:
    w, h = t.steps[0].frame.size
    full = [frame_tokens(*s.frame.size) for s in t.steps]
    diffs: list[FrameDiff | None] = [None]
    dtok = [full[0]]  # best reference among the last REF_WINDOW frames
    dtok_prev = [full[0]]  # previous frame only (used by the 3-frame window)
    fidelity, target_ok = [], []
    for k in range(1, len(t.steps)):
        cur = t.steps[k]
        cands = []
        for j in range(max(0, k - REF_WINDOW), k):
            d = diff_frames(t.steps[j].frame, cur.frame)
            cands.append((diff_tokens(d, w, h), j, d))
        dtok_prev.append(cands[-1][0])
        tok, j, d = min(cands, key=lambda c: (c[0], -c[1]))
        prev = t.steps[j]
        diffs.append(d)
        dtok.append(tok)
        rec = np.asarray(reconstruct(prev.frame, cur.frame, d), dtype=np.int16)
        true = np.asarray(cur.frame, dtype=np.int16)
        err = np.abs(rec - true)
        fidelity.append(float((err.max(axis=2) > 48).mean()))  # share of visibly wrong pixels
        if cur.target is not None:
            x, y, bw, bh = cur.target
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(w, x + bw), min(h, y + bh)
            if x1 > x0 and y1 > y0:
                target_ok.append(float(err[y0:y1, x0:x1].mean()) < 6)

    n = len(t.steps)
    per_call = {p: [] for p in POLICIES}
    for k in range(n):
        per_call["full_history"].append(sum(full[: k + 1]))
        per_call["last_1"].append(full[k])
        per_call["last_3"].append(sum(full[max(0, k - 2) : k + 1]))
        per_call["foveal_diff"].append(sum(dtok[: k + 1]))
        j = max(0, k - 2)
        per_call["foveal_window3"].append(full[j] + sum(dtok_prev[j + 1 : k + 1]))
    inp, _, cread = prices_for(MODEL)
    cost = {}
    for p, calls in per_call.items():
        if p in ("full_history", "foveal_diff"):  # append-only: prefix cached
            c = 0.0
            for k, tok in enumerate(calls):
                new = tok - (calls[k - 1] if k else 0)
                c += (new * inp * CACHE_WRITE_MULT + (tok - new) * cread) / 1e6
            cost[p] = c
        else:
            cost[p] = sum(calls) * inp / 1e6
    return {
        "id": t.annotation_id,
        "website": t.website,
        "steps": n,
        "tokens": {p: sum(v) for p, v in per_call.items()},
        "cumulative": {p: list(np.cumsum(v)) for p, v in per_call.items()},
        "cost": cost,
        "diff_kinds": Counter(d.kind for d in diffs[1:]),
        "scrolls": sum(1 for d in diffs[1:] if d and d.scroll_dy),
        "fidelity_bad_px": fidelity,
        "target_ok": target_ok,
        "example": describe(diffs[1]) if n > 1 else "",
    }


def report(results: list[dict], out: Path, title: str = "Multimodal-Mind2Web") -> str:
    tot = {p: sum(r["tokens"][p] for r in results) for p in POLICIES}
    cost = {p: sum(r["cost"][p] for r in results) for p in POLICIES}
    kinds = sum((r["diff_kinds"] for r in results), Counter())
    n_diffs = sum(kinds.values())
    bad = [f for r in results for f in r["fidelity_bad_px"]]
    tgt = [x for r in results for x in r["target_ok"]]
    steps = sum(r["steps"] for r in results)
    base = tot["full_history"]
    lines = [
        f"# foveal Phase 2 offline replay: {title}",
        "",
        f"{len(results)} trajectories, {steps} steps, priced as {MODEL}. No model calls.",
        "",
        "| Policy | Same info as | Image tokens | vs full history | Est. cost |",
        "| --- | --- | --- | --- | --- |",
    ]
    info = {
        "full_history": "everything",
        "last_1": "current only",
        "last_3": "last 3",
        "foveal_diff": "everything",
        "foveal_window3": "last 3",
    }
    for p in POLICIES:
        lines.append(
            f"| {p} | {info[p]} | {tot[p]:,} | {tot[p] / base - 1:+.0%} | ${cost[p]:.2f} |"
        )
    lines += [
        "",
        "## Diff engine",
        "",
        f"- Step-to-step diffs: {n_diffs}. Identical {kinds['identical']}, "
        f"partial {kinds['partial']}, full {kinds['full']}. "
        f"Scroll detected in {sum(r['scrolls'] for r in results)}.",
        f"- Reconstruction fidelity: median share of visibly wrong pixels "
        f"{np.median(bad) if bad else 0:.3%}, worst {max(bad) if bad else 0:.2%}.",
        f"- Target element pixel-exact after reconstruction: {np.mean(tgt) if tgt else 0:.1%} "
        f"of {len(tgt)} steps.",
        f'- Example description: "{next((r["example"] for r in results if r["example"]), "")}"',
    ]
    md = "\n".join(lines) + "\n"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(md)
    _plot(results, out / "cumulative_tokens.png", title)
    (out / "results.json").write_text(
        json.dumps(
            [{k: v for k, v in r.items() if k != "cumulative"} for r in results],
            default=str,
            indent=1,
        )
    )
    return md


def _plot(results: list[dict], path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {
        "full_history": "#9aa5b1",
        "last_1": "#c9a227",
        "last_3": "#e07b39",
        "foveal_diff": "#1f4e79",
        "foveal_window3": "#4a90c2",
    }
    max_len = max(r["steps"] for r in results)
    for p in POLICIES:
        ys = []
        for k in range(max_len):
            vals = [r["cumulative"][p][k] for r in results if r["steps"] > k]
            if len(vals) < 3:
                break
            ys.append(float(np.median(vals)))
        ax.plot(range(1, len(ys) + 1), ys, label=p, color=colors[p], lw=2.2)
    ax.set_xlabel("agent step")
    ax.set_ylabel("cumulative image tokens (median trajectory)")
    ax.set_title(f"{title}: image tokens by history policy", fontsize=10)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="mind2web", choices=["mind2web", "video"])
    ap.add_argument("--every-s", type=float, default=2.0, help="video: seconds per frame")
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--min-steps", type=int, default=3)
    ap.add_argument("--out", default="runs/phase2-replay")
    args = ap.parse_args(argv)
    if args.source == "video":
        from bench.replay import videos

        trajs = videos.load(args.every_s)
    else:
        trajs = load(args.shards, args.min_steps)
    print(f"{len(trajs)} trajectories, {sum(len(t.steps) for t in trajs)} steps")
    results = [replay(t) for t in trajs]
    title = (
        f"screen recordings, one frame every {args.every_s:g}s"
        if args.source == "video"
        else "Multimodal-Mind2Web"
    )
    print(report(results, Path(args.out), title))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
