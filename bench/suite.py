"""One entry point for foveal's evaluations.

    uv run python -m bench.suite --report                 # $0: report from existing logs
    uv run python -m bench.suite --dry-run                # $0: every run with fake clients
    uv run python -m bench.suite --run p3-facts --budget 3   # live; refuses if caps > budget

Each live run has its own spending cap; the suite refuses to start runs whose caps add up to
more than --budget. The report re-scores every doc-QA run with the same scorer (or the
cached LLM judge with --judge) and plots cost against success.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from foveal.instrument.metrics import compute
from foveal.instrument.records import read_jsonl

PY = [sys.executable, "-m"]
MM = [*PY, "bench.mmlongbench.run"]
FRESH = [
    "--docs",
    "8",
    "--per-doc",
    "3",
    "--evidence",
    "Chart,Table,Figure",
    "--exclude-docs-from",
    "runs/phase0-sonnet55.jsonl",
]

# name -> (cap in USD, commands). Caps are passed to each command as --max-cost-usd.
RUNS: dict[str, tuple[float, list[list[str]]]] = {
    "p1-scale": (
        14.0,
        [
            [*MM, *FRESH, "--run-id", "p1-scale-images", "--max-cost-usd", "9"],
            [
                *MM,
                *FRESH,
                "--run-id",
                "p1-scale-memory",
                "--mode",
                "memory",
                "--store",
                ".foveal",
                "--max-cost-usd",
                "5",
            ],
        ],
    ),
    "p2-live": (
        5.0,
        [
            [
                *PY,
                "bench.replay.agent_eval",
                "--steps",
                "150",
                "--run-id",
                "p2-live",
                "--max-cost-usd",
                "5",
            ]
        ],
    ),
    "p3-facts": (
        3.0,
        [
            [
                *MM,
                "--mode",
                "memory",
                "--facts",
                "--docs",
                "4",
                "--per-doc",
                "4",
                "--run-id",
                "p3-facts",
                "--store",
                ".foveal-p3",
                "--max-cost-usd",
                "3",
            ]
        ],
    ),
}

# Doc-QA runs on the same 16 questions, for the cost/success chart: label -> log
DOCQA = {
    "full pages (Phase 0)": "runs/phase0-sonnet55.jsonl",
    "memory v1 (Phase 1)": "runs/phase1-memory-sonnet55.jsonl",
    "memory v2 (Phase 1 tuned)": "runs/phase1-memory-v2.jsonl",
    "memory + facts (Phase 3)": "runs/p3-facts.jsonl",
}


def _scorer(judge: bool):
    sys.path.insert(0, str(Path.cwd()))
    from bench.mmlongbench.score import score

    if not judge:
        return score
    import anthropic
    from dotenv import load_dotenv

    from bench.mmlongbench.judge import Judge

    load_dotenv()
    return Judge(anthropic.Anthropic())  # cached verdicts are free; new ones cost ~$0.001


def report(judge: bool, out: Path) -> str:
    sc = _scorer(judge)
    rows = []
    for label, path in DOCQA.items():
        if not Path(path).exists():
            continue
        recs = read_jsonl(path)
        s = compute(recs)
        tasks = [r for r in recs if r.get("type") == "task" and "score" in r]
        ok = sum(sc(r["gold"], r["pred"], r["answer_format"], r.get("question", "")) for r in tasks)
        n = len(tasks)
        rows.append(
            (
                label,
                ok / n if n else 0.0,
                n,
                s["cost_usd_with_ingest"],
                s["image_tokens_total"] + s.get("ingest_image_tokens", 0),
            )
        )
    lines = [
        "# foveal evaluation suite",
        "",
        f"Doc QA on MMLongBench-Doc (Claude Sonnet 5.5); success by "
        f"{'the cached LLM judge' if judge else 'the rule-based scorer'}.",
        "",
        "| Configuration | Success | Tasks | Cost incl. ingest | Image tokens |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| {lbl} | {acc:.0%} | {n} | ${cost:.2f} | {tok:,} |" for lbl, acc, n, cost, tok in rows
    ]
    for name in ("phase2-replay", "phase2-video-2s"):
        p = Path("runs") / name / "report.md"
        if p.exists():
            body = p.read_text().split("\n", 2)[2]
            lines += ["", f"## Offline replay: {name}", "", body.split("## Diff engine")[0].strip()]
    live = Path("runs/p2-live.jsonl")
    if live.exists():
        from bench.replay.agent_eval_report import summarize

        lines += ["", "## Web-agent next action (Mind2Web)", "", summarize(str(live))]
    pending = [
        n
        for n, (_, cmds) in RUNS.items()
        if not all(Path("runs", c[c.index("--run-id") + 1] + ".jsonl").exists() for c in cmds)
    ]
    lines += [
        "",
        "Pending live runs: "
        + (", ".join(f"{n} (cap ${RUNS[n][0]:.0f})" for n in pending) or "none"),
    ]
    md = "\n".join(lines) + "\n"
    out.mkdir(parents=True, exist_ok=True)
    (out / "suite_report.md").write_text(md)
    if rows:
        _pareto(rows, out / "cost_vs_success.png")
    return md


def _pareto(rows: list[tuple], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for lbl, acc, _, cost, _ in rows:
        ax.scatter(cost, acc * 100, s=70, color="#9aa5b1" if "full" in lbl else "#1f4e79")
        ax.annotate(lbl, (cost, acc * 100), textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.set_xlabel("cost per run of 16 questions, incl. captioning (USD)")
    ax.set_ylabel("success (%)")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 100)  # full scale: small samples must not look like big gaps
    ax.set_title("Doc QA: cost vs success", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--run", nargs="*", default=None, help=f"live runs: {', '.join(RUNS)}")
    ap.add_argument("--budget", type=float, default=0.0)
    ap.add_argument("--out", default="runs/suite")
    args = ap.parse_args(argv)

    if args.dry_run:
        for name, (_, cmds) in RUNS.items():
            for cmd in cmds:
                c = [x for x in cmd]
                i = c.index("--run-id")
                c[i + 1] = "dry-" + c[i + 1]
                if "--store" in c:
                    c[c.index("--store") + 1] = ".foveal-dry"
                for flag in ("--evidence", "--exclude-docs-from"):  # need the real dataset
                    if flag in c:
                        j = c.index(flag)
                        del c[j : j + 2]
                print(f"[dry] {name}: {' '.join(c[2:])}")
                subprocess.run([*c, "--dry-run"], check=True, capture_output=True)
        print("all runs completed in dry-run mode")
    if args.run is not None:
        names = args.run or list(RUNS)
        unknown = set(names) - set(RUNS)
        if unknown:
            ap.error(f"unknown runs: {unknown}")
        total = sum(RUNS[n][0] for n in names)
        if total > args.budget:
            print(
                f"refusing: caps for {names} add up to ${total:.0f}, "
                f"over --budget ${args.budget:.0f}"
            )
            return 2
        for n in names:
            for cmd in RUNS[n][1]:
                print(f"[live] {n}: {' '.join(cmd[2:])}")
                subprocess.run(cmd, check=True)
    if args.report or not (args.dry_run or args.run is not None):
        print(report(args.judge, Path(args.out)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
