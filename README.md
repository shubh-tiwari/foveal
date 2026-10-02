# foveal

Multimodal working memory for agents: perceive each image once, keep cheap references in
context, zoom into detail only when needed, and share what was learned across sub-agents.

> Status: **Phase 0 (measurement).** The memory layer isn't built yet. This repo currently
> measures how many image tokens agent harnesses spend re-sending pixels they already sent.

## Phase 0 result

On 16 MMLongBench-Doc questions answered by Claude Sonnet 5.5 (an orchestrator plus reader
sub-agents), **62.3% of all image tokens were pages already sent earlier in the same task**.
That passes the 40% gate. Prompt caching already served about 74% of those repeats. See
[ROADMAP.md](ROADMAP.md) for the full numbers.

![Cumulative image tokens per model call](docs/phase0_cumulative_image_tokens.png)

## Phase 0: measure re-perception

`foveal.instrument` wraps the Anthropic client without changing any request. It logs every
image block in every call (sha256, pHash, size, token cost, position) together with `usage`,
and writes one JSONL line per call:

```python
import anthropic
from foveal.instrument import InstrumentedAnthropic, JsonlSink, TokenCounter, Tracer

raw = anthropic.Anthropic()
tracer = Tracer("run1", JsonlSink("runs/run1.jsonl"), TokenCounter(raw))
client = InstrumentedAnthropic(raw, tracer)
with tracer.scope(task_id="q1", agent_id="orch"):
    client.messages.create(...)
```

The test harness is a long-document QA orchestrator with reader sub-agents, run on
[MMLongBench-Doc](https://huggingface.co/datasets/yubo2333/MMLongBench-Doc):

```sh
uv sync --all-extras
uv run python -m bench.mmlongbench.run --dry-run                   # offline, free
uv run python -m bench.mmlongbench.run --docs 1 --per-doc 1        # one live task
uv run python -m bench.mmlongbench.run --docs 4 --per-doc 4 --max-cost-usd 6
uv run foveal analyze runs/<run_id>.jsonl                          # report.md + plot
```

The default model is Claude Sonnet 5.5 (`claude-sonnet-5-5`) for both the orchestrator and the
readers. Use `--orch-model` and `--reader-model` to change them. For example,
`--reader-model claude-haiku-4-5` sends reader pages at standard resolution for less.

The report covers:

- **Re-perception rate:** the share of image tokens whose exact image was already sent in
  the same task. It is split into same-agent resends and cross-agent duplicates.
- **Near-duplicate and cross-task shares:** how much of the first-seen traffic a memory
  layer could still reuse.
- **Image tokens under keep-last-3:** the common workaround, simulated from the same logs.
- **Cache-adjusted cost:** how much of the repeated traffic prompt caching already discounts.
- **Gate:** PASS when re-perception is at least 40%.

Image token estimates follow the Vision docs: `⌈w/28⌉·⌈h/28⌉` after downscaling to the
model's tier limits. Ground truth comes from `messages.count_tokens`, cached per image hash.

## Roadmap

Phase 0 (measure) is in progress. Phases 1–4 build the asset store and L0–L3 ladder, diff
sync, shared facts, and the budget-aware assembler with MCP and SDK interfaces. See
[ROADMAP.md](ROADMAP.md) for each phase's scope and gate.
