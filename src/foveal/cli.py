"""`foveal analyze runs/<id>.jsonl` - Phase 0 metrics, gate verdict and plots."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from foveal.instrument.metrics import compute, gate
from foveal.instrument.records import read_jsonl


def analyze(
    path: str, out_dir: str | None, keep_last: int, gate_threshold: float, plot: bool = True
) -> int:
    from foveal.instrument.report import markdown, plot_cumulative

    records = read_jsonl(path)
    summary = compute(records, keep_last=keep_last)
    out = Path(out_dir) if out_dir else Path(path).with_suffix("")
    out.mkdir(parents=True, exist_ok=True)
    md = markdown(summary, gate_threshold, keep_last)
    (out / "report.md").write_text(md)
    print(md)
    if plot:
        png = plot_cumulative(summary, out / "cumulative_image_tokens.png")
        print(f"plot: {png}")
    print(f"report: {out / 'report.md'}")
    ok, _ = gate(summary, gate_threshold)
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="foveal")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze", help="compute Phase 0 metrics from a run log")
    a.add_argument("log")
    a.add_argument("--out", default=None)
    a.add_argument("--keep-last", type=int, default=3)
    a.add_argument("--gate", type=float, default=0.40)
    a.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "analyze":
        analyze(args.log, args.out, args.keep_last, args.gate, plot=not args.no_plot)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
