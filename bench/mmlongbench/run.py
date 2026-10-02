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
from bench.mmlongbench.render import MemoryPages, PageRenderer
from bench.mmlongbench.score import score
from foveal.facts import RedisFacts
from foveal.instrument import InstrumentedAnthropic, JsonlSink, TokenCounter, Tracer
from foveal.memory import Captioner, Memory


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=4)
    ap.add_argument("--per-doc", type=int, default=4)
    ap.add_argument("--min-pages", type=int, default=10)
    ap.add_argument("--max-pages", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--evidence",
        default=None,
        help="comma list of evidence types to keep, e.g. Chart,Table,Figure",
    )
    ap.add_argument(
        "--scanned-only", action="store_true", help="only documents with (almost) no text layer"
    )
    ap.add_argument(
        "--exclude-docs-from",
        default=None,
        help="run log whose documents to skip (fresh tasks for a scale-up)",
    )
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
    ap.add_argument(
        "--mode",
        default="images",
        choices=["images", "memory"],
        help="images: full page images (baseline); memory: foveal L0/L1 + look()",
    )
    ap.add_argument(
        "--provider",
        default="anthropic",
        choices=["anthropic", "openrouter"],
        help="openrouter: any OpenRouter model id, e.g. google/gemini-3.8-flash",
    )
    ap.add_argument(
        "--caption-model",
        default=None,
        help="default: claude-haiku-4-5 (anthropic) or the reader model (openrouter)",
    )
    ap.add_argument(
        "--caption-image-tokens",
        type=int,
        default=384,
        help="downscale pages to this many visual tokens before captioning (0 = full page)",
    )
    ap.add_argument("--redis-url", default=None, help="share facts via Redis instead of SQLite")
    ap.add_argument(
        "--facts",
        action="store_true",
        help="memory mode: agents write notes; later questions recall them",
    )
    ap.add_argument(
        "--text-on-demand",
        action="store_true",
        help="memory mode: show caption + text preview; full text via view_pages",
    )
    ap.add_argument("--store", default=None, help="foveal store dir (default .foveal[-dry])")
    ap.add_argument("--dry-run", action="store_true", help="fake client + synthetic PDFs")
    args = ap.parse_args(argv)

    run_id = args.run_id or (
        time.strftime("%Y%m%d-%H%M%S") + f"-{args.mode}" + ("-dry" if args.dry_run else "")
    )
    log = Path(args.runs_dir) / f"{run_id}.jsonl"

    if args.dry_run:
        from bench.mmlongbench.fake import FakeAnthropic

        raw = FakeAnthropic()
        tasks = synthetic_tasks(
            Path(".cache/synthetic"), n_docs=min(args.docs, 2), per_doc=min(args.per_doc, 2)
        )
    else:
        from dotenv import load_dotenv

        load_dotenv()  # API keys from .env (never printed)
        if args.provider == "openrouter":
            from bench.openrouter import OpenRouterClient

            raw = OpenRouterClient()
            args.no_count_tokens = True  # count_tokens is an Anthropic endpoint
        else:
            import anthropic

            raw = anthropic.Anthropic()
        exclude = None
        if args.exclude_docs_from:
            from foveal.instrument.records import read_jsonl

            exclude = {
                r["doc_id"]
                for r in read_jsonl(args.exclude_docs_from)
                if r.get("type") == "task" and r.get("doc_id")
            }
        tasks = select_tasks(
            args.docs,
            args.per_doc,
            args.min_pages,
            args.max_pages,
            args.seed,
            evidence=set(args.evidence.split(",")) if args.evidence else None,
            scanned_only=args.scanned_only,
            exclude_docs=exclude,
        )

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
        provider=args.provider,
    )
    qa = DocQA(client, tracer, cfg)
    tracer.log_event(type="run", config=vars(args), n_tasks=len(tasks))

    caption_model = args.caption_model or (
        args.reader_model if args.provider == "openrouter" else "claude-haiku-4-5"
    )
    memory = None
    if args.mode == "memory":
        store = args.store or (".foveal-dry" if args.dry_run else ".foveal")
        memory = Memory(
            store,
            captioner=Captioner(
                client,
                model=caption_model,
                image_tokens=args.caption_image_tokens or None,
                request={"output_config": {"effort": "none"}}
                if args.provider == "openrouter"
                else None,
            ),
            model=args.reader_model,
            long_edge=args.long_edge,
            facts=RedisFacts(args.redis_url) if args.redis_url else None,
        )

    renderers: dict[str, PageRenderer] = {}

    def open_doc(t):  # noqa: ANN001
        if t.doc_id not in renderers:
            if memory is None:
                renderers[t.doc_id] = PageRenderer(t.pdf_path, long_edge=args.long_edge)
            else:  # perceive every page once, up front; captions are logged as "ingest"
                with tracer.scope(task_id=f"ingest:{t.doc_id}", agent_id="ingest", doc_id=t.doc_id):
                    renderers[t.doc_id] = MemoryPages(
                        t.pdf_path,
                        memory,
                        text_on_demand=args.text_on_demand,
                        facts=args.facts,
                        long_edge=args.long_edge,
                    )
        return renderers[t.doc_id]

    print(f"run {run_id} ({args.mode}): {len(tasks)} tasks -> {log}")
    for i, t in enumerate(tasks, 1):
        doc = open_doc(t)
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
