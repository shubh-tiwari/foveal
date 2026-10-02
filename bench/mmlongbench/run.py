"""Phase 0 run: MMLongBench-Doc tasks through the orchestrator harness, fully instrumented.

uv run python -m bench.mmlongbench.run --dry-run                 # offline, free
uv run python -m bench.mmlongbench.run --docs 1 --per-doc 1      # one live task
uv run python -m bench.mmlongbench.run --docs 4 --per-doc 4 --max-cost-usd 6
    # cheaper still: --reader-model claude-haiku-4-5 (standard-res tier, $1/$5)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from bench.mmlongbench.agents import AgentConfig, BudgetExceeded, DocQA
from bench.mmlongbench.data import select_tasks, synthetic_tasks
from bench.mmlongbench.render import PageRenderer
from bench.mmlongbench.score import score
from foveal.instrument import InstrumentedAnthropic, JsonlSink, TokenCounter, Tracer


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=4)
    ap.add_argument("--per-doc", type=int, default=4)
    ap.add_argument("--min-pages", type=int, default=10)
    ap.add_argument("--max-pages", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--orch-model", default="claude-sonnet-5-5")
    ap.add_argument("--reader-model", default="claude-sonnet-5-5")
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--no-cache", action="store_true", help="disable prompt caching")
    ap.add_argument("--no-fallbacks", action="store_true")
    ap.add_argument(
        "--no-count-tokens",
        action="store_true",
        help="skip count_tokens ground truth; use the estimate only",
    )
    ap.add_argument("--long-edge", type=int, default=1568)
    ap.add_argument("--max-cost-usd", type=float, default=6.0)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--dry-run", action="store_true", help="fake client + synthetic PDFs")
    args = ap.parse_args(argv)

    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S") + ("-dry" if args.dry_run else "")
    log = Path(args.runs_dir) / f"{run_id}.jsonl"

    if args.dry_run:
        from bench.mmlongbench.fake import FakeAnthropic

        raw = FakeAnthropic()
        tasks = synthetic_tasks(
            Path(".cache/synthetic"), n_docs=min(args.docs, 2), per_doc=min(args.per_doc, 2)
        )
    else:
        import anthropic
        from dotenv import load_dotenv

        load_dotenv()  # ANTHROPIC_API_KEY from .env (never printed)
        raw = anthropic.Anthropic()
        tasks = select_tasks(args.docs, args.per_doc, args.min_pages, args.max_pages, args.seed)

    counter = (
        None
        if args.no_count_tokens
        else TokenCounter(
            raw,
            ".cache/foveal_tokens_dry.sqlite" if args.dry_run else ".cache/foveal_tokens.sqlite",
        )
    )
    tracer = Tracer(run_id, JsonlSink(log), counter)
    client = InstrumentedAnthropic(raw, tracer)
    cfg = AgentConfig(
        orch_model=args.orch_model,
        reader_model=args.reader_model,
        effort=args.effort,
        cache=not args.no_cache,
        fallbacks=not args.no_fallbacks,
        max_cost_usd=args.max_cost_usd,
    )
    qa = DocQA(client, tracer, cfg)
    tracer.log_event(type="run", config=vars(args), n_tasks=len(tasks))

    renderers: dict[str, PageRenderer] = {}
    print(f"run {run_id}: {len(tasks)} tasks -> {log}")
    for i, t in enumerate(tasks, 1):
        doc = renderers.setdefault(t.doc_id, PageRenderer(t.pdf_path, long_edge=args.long_edge))
        t0 = time.perf_counter()
        with tracer.scope(task_id=t.task_id, doc_id=t.doc_id):
            try:
                res = qa.run_task(doc, t.question)
            except BudgetExceeded as e:
                print(f"budget reached ({e}); stopping")
                tracer.log_event(type="task", status="budget", question=t.question)
                break
            except Exception as e:  # keep the run going; the error is logged
                print(f"[{i}] {t.task_id}: error {e!r}")
                tracer.log_event(type="task", status="error", error=repr(e), question=t.question)
                continue
            s = score(t.answer, res.answer, t.answer_format)
            tracer.log_event(
                type="task",
                status=res.stop,
                question=t.question,
                gold=t.answer,
                pred=res.answer,
                answer_format=t.answer_format,
                score=s,
                evidence_pages=t.evidence_pages,
                orch_steps=res.orch_steps,
                readers=res.readers,
                wall_s=time.perf_counter() - t0,
            )
        print(
            f"[{i}/{len(tasks)}] {t.task_id}: {res.stop} pred={res.answer!r} "
            f"gold={t.answer!r} score={s} spent=${tracer.total_cost_usd:.2f}"
        )

    print(
        f"done. total cost ${tracer.total_cost_usd:.2f}. analyze with:\n"
        f"  uv run foveal analyze {log}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
