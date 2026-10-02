# foveal roadmap

The plan is 5 gated phases over about 14 part-time weeks. Each phase ends with a gate: a
measured result that decides whether the next phase is worth building. Gate thresholds are
starting proposals. Revisit them once Phase 0 numbers exist.

| Phase | Weeks | Builds | Gate to pass |
| --- | --- | --- | --- |
| 0. Measure | 1–2 | Instrumentation, orchestrator harness, re-perception report | Re-perception ≥ 40% of image tokens on a real harness |
| 1. Asset store + ladder | 3–5 | `Memory`, `ingest()`, L0–L3 levels, `look()` | Answers from L0/L1 + `look()` match full images on ≥ 95% of tasks |
| 2. Diff sync | 6–8 | Version chains, changed-region diffs, frame dedup | Image tokens on screen/video tasks fall ≥ 50% vs keep-last-3, with no drop in success |
| 3. Shared facts | 9–11 | Facts with provenance, invalidation, subscriptions, multi-process store | Cross-agent duplicates fall ≥ 80% and stale-action rate falls vs baseline |
| 4. Assembler + interfaces + eval | 12–14 | Budget-aware context assembly, MCP server, SDK middleware, benchmark suite, report | Cost–success Pareto curves beat all baselines on ≥ 2 workloads |

## Phase 0 — Measure (in progress)

Built so far: `foveal.instrument`, the MMLongBench-Doc orchestrator harness, and
`foveal analyze`.

- [x] Package scaffold, 0.0.1 build
- [x] Wrapper that logs image hashes, token counts and usage per call and per agent
- [x] Image token estimate checked against `count_tokens` (within 0.2%)
- [x] Full 16-task run on Claude Sonnet 5.5 (2 Oct 2026, $4.06): **gate PASS**
  - Re-perception 62.3% of 2.50M image tokens: 44.3% same-agent resend, 18.0% cross-agent
  - Median task 50.5%; excluding the largest task, 52.9%
  - Keep-last-3 saves only 6%, because resends sit inside multi-page tool results
  - Prompt caching already serves about 74% of repeated tokens, so repeat cost is $3.12
    without caching and about $1.05 with it
  - Cross-task: 65% of first-seen tokens were pages already read for an earlier question on
    the same document, which is the Phase 3 shared-facts opportunity
  - Success 10/16
- [ ] Register `foveal` on PyPI and create the GitHub repo (owner action)

**If the gate fails:** document QA with a well-split orchestrator may re-send very little. Before
stopping, measure a longer-horizon workload where history accumulates:

1. Make readers multi-turn by default (follow-up questions to the same reader).
2. Run a computer-use or browser loop (screenshot each step, e.g. a WebArena slice).
3. Report the same metrics with prompt caching on and off. Caching may already make
   resends cheap, which shrinks foveal's dollar savings even when the token savings are real.

## Phase 1 — Asset store and representation ladder (in progress)

Goal: perceive each input once, and keep only cheap representations in context.

- [x] `Memory` and `Store` (SQLite plus content-addressed blobs), with `ingest`, `describe`, `look`
- [x] L0 captions from Claude Haiku 4.5, L1 PDF text or optional OCR, L2 thumbnail at most
  256 tokens, L3 full image or a zoomed PDF re-render
- [x] Harness `--mode memory` and `foveal compare`, with captioning billed as ingest
- [x] 16-task comparison against the Phase 0 baseline (2 Oct 2026): **gate PASS**
  - Success is 10/16 for both, judged by Claude Haiku 4.5 identically for both runs. Every task
    got the same verdict.
  - Image tokens: 2.50M → 389K (-84%). That includes 258K one-off caption tokens.
    Task-time image tokens: 2.50M → 131K (-95%).
  - Cost: $4.06 → $1.53 (-62%), including $0.2 of captioning
  - Several questions were answered from L0/L1 text alone, with no image sent.
  - Caveats: the sample is small (16 tasks), and these PDFs have text layers. Scanned
    documents need OCR for L1, and figure-heavy questions lean harder on `look()`.
  - The rule-based scorer alone gives 9 vs 10. Its miss was a correct "the document mentions
    no cooler" answer to a "Not answerable" question, so the LLM judge is the reported score.
- [ ] Scale up: all 135 documents (or a stratified 40), plus a scanned/figure-heavy slice
- Known limit: images returned by `look()` stay in history and are resent. Dropping them means
  editing earlier turns, which breaks the prompt cache and invalidates preserved thinking.
  That belongs to the Phase 4 assembler.

- `Asset` and `Memory` data model, with a content-addressed store (SQLite + files on disk)
- `ingest(source)` for images, PDF pages and screenshots, with sha256 dedup
- Levels:
  - **L0:** a one-line caption from a cheap model, cached by hash
  - **L1:** text structure: OCR (Tesseract first), PDF text layer, accessibility tree
  - **L2:** a thumbnail at a fixed token budget, e.g. ≤ 256 tokens
  - **L3:** the full image, or crops via `look(asset_id, region, detail)`
- Re-run the Phase 0 harness with readers that receive L0/L1 plus a `look()` tool instead of
  raw pages. Measure image tokens and success against the Phase 0 baseline.

## Phase 2 — Diff-based sync for continuous observations

Goal: consecutive screenshots and video frames cost only what changed.

- Asset versioning: a new screenshot of the same source becomes `version + 1`
- A diff engine: pHash/SSIM gate, then changed-region boxes; accessibility-tree diff where
  available
- `diff_since(asset_id, version)` returns text ("unchanged except the dialog at top right")
  plus crops of the changed regions
- Video: frame sampling with PyAV, plus near-duplicate frame collapse
- Benchmarks: a computer-use or web slice (OSWorld or WebArena) and a Video-MME slice

## Phase 3 — Shared, synced fact memory

Goal: one agent's perception becomes every agent's knowledge, and stale facts are never acted on.

- `Fact` records: claim, asset, version, region, level seen, confidence, author
- `write_fact`, `recall_facts` (embedding + keyword search) and verification by zooming into the
  cited region
- Invalidation: a new asset version marks dependent facts `stale`
- `subscribe(asset_id, region)`: notify on change instead of polling
- Redis backend for multi-process orchestrators
- New metric: stale-action rate, the share of actions based on an outdated asset version

## Phase 4 — Budget-aware assembler, interfaces and evaluation

Goal: a drop-in library that builds each model call under a token budget, plus the evidence that it works.

- Context assembler: promote relevant or recent assets, demote old ones, all under `budget_tokens`
- **Prompt-cache-aware demotion:** model the trade-off between keeping a cached prefix and
  rewriting history, and demote in batches. No published work on this was found.
  On models that run the preserved-thinking check, rewriting earlier turns invalidates thinking
  blocks, so the assembler must prefer append-only strategies there.
- MCP server: `ingest`, `look`, `diff_since`, `recall_facts`, `write_fact`, `subscribe`
- SDK middleware: rewrite outgoing history for Anthropic and OpenAI-compatible clients
  (OpenRouter fits here for cross-provider results)
- Evaluation:
  - Baselines: full history, keep-last-N (1/3/5), text summaries, ReVision-style dropping
  - Workloads: document QA, computer use, video
  - Metrics: success, image tokens, cost, latency, re-perception, stale-action rate
  - Results reported as cost–success Pareto curves
- Deliverables: PyPI release, MCP server, technical report or workshop paper
