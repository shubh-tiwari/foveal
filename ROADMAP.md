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
- [x] Cost tuning (2 Oct 2026, $1.40 run). Captioning from a 384-token downscale cut caption
  cost 66% ($0.29 → $0.10) with no accuracy loss, and is now the default. Text-on-demand
  saved nothing: agents fetched the full text for most pages anyway, adding 15 calls. It
  stays available as `--text-on-demand`, off by default.
  - Against the full-image baseline: success 11/16 vs 10/16 (same judge), -93% image tokens,
    -66% cost ($4.06 → $1.40).
  - The remaining cost is mostly agent overhead: every call re-sends the system prompt, tool
    definitions and history. Caption cost is $0.002 per page.
- [x] OCR fallback for scanned pages (Tesseract, only for new pages with almost no text layer)
- [x] Task slices: `--evidence Chart,Table,Figure`, `--scanned-only`,
  `--exclude-docs-from <run>` (fresh documents)
- [ ] **Scale-up run (needs budget approval, ~$8–14):** a fresh figure-heavy slice in both
  modes, then the judged compare:
  ```sh
  X="--docs 8 --per-doc 3 --evidence Chart,Table,Figure --exclude-docs-from runs/phase0-sonnet55.jsonl --max-cost-usd 8"
  uv run python -m bench.mmlongbench.run $X --run-id p1-scale-images
  uv run python -m bench.mmlongbench.run $X --run-id p1-scale-memory --mode memory --store .foveal
  uv run foveal compare runs/p1-scale-images.jsonl runs/p1-scale-memory.jsonl --judge
  ```
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

## Phase 2 — Diff-based sync for continuous observations (built; live eval pending)

Goal: consecutive screenshots and video frames cost only what changed.

- [x] `foveal.diff`: tile-based change detection that tolerates JPEG noise, vertical-scroll
  detection, merged change boxes, reconstruction and a text description
- [x] Offline replay on Multimodal-Mind2Web (2 Oct 2026, $0, no model calls): 113 web-agent
  trajectories, 725 steps, 1280x720 viewports. See `docs/phase2_replay_mind2web.md`.
  - Lossless: reconstructed frames are pixel-exact, and so is every action target (579 of 579).
  - Tokens: foveal_diff, with the same information as full history, is -21% vs full history.
    foveal_window3 is -15% vs keep-last-3. **The token gate (-50% vs keep-last-3) is not met
    on this workload.** 63% of steps load a new page or jump far down one, so most frames
    really are new.
  - Cost is where it wins. Keep-last-N changes the prompt prefix every step, so nothing is
    cached. Append-only histories cache. foveal_diff keeps all history for $2.17 versus
    $4.39 for keep-last-3 (-51%) and $2.81 for full history (-23%).
- [x] Diff engine hardening: scroll inside the band between a sticky header and footer, and a
  robust row-error measure, so a toast or changed widget doesn't hide a scroll
- [x] Video replay (2 Oct 2026, $0): 9 screen recordings of flight booking on desktop and
  phone (HumynLabs, CC-BY-4.0), one frame every 2 s (507 frames) or 5 s (205 frames)
  - 2 s: foveal_diff is -43% tokens and -43% cost vs full history. foveal_window3 is -25%
    vs keep-last-3.
  - 5 s: -23% vs full history. Frames further apart share less.
  - Even at 2 s, about half the frames are mostly new content (page loads, momentum
    scrolling, typing). Tuning thresholds gained only 2 points, so the defaults stay strict
    for fidelity: at most 0.27% of pixels wrong in the worst frame.
- [x] `Memory.ingest_frame(image, source)` stores versioned streams: an unchanged frame
  creates no version, each new version stores its diff, and content is deduplicated across
  streams. Also `latest`, `version`, `history`, and `diff_since(source, v)`, which returns
  text plus exact crops computed directly between the two frames.
- [x] Live-ready web-agent evaluation, `bench/replay/agent_eval.py`: Mind2Web next-action
  prediction (task, previous actions, about 8 on-screen candidates) under full_history,
  last_3 and foveal_diff. The cache breakpoint sits after the screenshots, so append-only
  policies get real cache hits. Tested with dry runs only.
- [ ] **Live evaluation (needs budget approval, ~$3–5):**
  `uv run python -m bench.replay.agent_eval --steps 150 --max-cost-usd 5`, then
  `python -m bench.replay.agent_eval_report runs/<id>.jsonl`.
  Gate: foveal_diff accuracy within 1 point of full_history at lower cost.

- Asset versioning: a new screenshot of the same source becomes `version + 1`
- A diff engine: pHash/SSIM gate, then changed-region boxes; accessibility-tree diff where
  available
- `diff_since(asset_id, version)` returns text ("unchanged except the dialog at top right")
  plus crops of the changed regions
- Video: frame sampling with PyAV, plus near-duplicate frame collapse
- Benchmarks: a computer-use or web slice (OSWorld or WebArena) and a Video-MME slice

## Phase 3 — Shared, synced fact memory (built; live eval pending)

Goal: one agent's perception becomes every agent's knowledge, and stale facts are never acted on.

- [x] `foveal.facts.Fact`: claim, asset, stream version, pixel region, level seen, confidence,
  author, status
- [x] `Memory.write_fact`, `recall_facts` (BM25 over claims, weighted by confidence, plus an
  optional embedding hook), and `verify_fact`, which zooms into the cited region
- [x] **Region-aware invalidation.** On a new stream version, a fact whose region is untouched
  carries forward to the new version, shifted by any scroll and left alone inside a sticky
  header. A fact whose region changed, scrolled out of view, or that has no region becomes
  `stale` with a reason, and its author gets a `stale` event.
- [x] Subscriptions: `subscribe(agent, source, region)` watches a fixed screen area, and
  `poll(agent)` returns `changed` / `stale` events
- [x] Backends behind one interface: `SQLiteFacts` (WAL, the store's own file; tested with a
  second process polling events) and `RedisFacts` (JSON facts with set indexes; events in a
  stream for polling and also published on a channel). Tests run against a throwaway
  `redis-server`.
- [x] Harness `--mode memory --facts` (optionally with `--redis-url`): readers and the
  orchestrator get `note`, the orchestrator gets `recall_notes`, and every question starts
  with the top fresh notes from earlier work on the same document. Tested with dry runs only.
- [ ] **Live evaluation (needs budget approval, ~$2–3):** the same 16 questions with
  `--mode memory --facts`, compared with the Phase 1 memory run (`foveal compare --judge`).
  Gate: same success, fewer image/text tokens on later questions per document
  (cross-question repeats were 65% of first-seen pages in Phase 0).
  ```sh
  uv run python -m bench.mmlongbench.run --mode memory --facts --docs 4 --per-doc 4 \
      --run-id p3-facts --store .foveal-p3 --max-cost-usd 3
  uv run foveal compare runs/phase1-memory-v2.jsonl runs/p3-facts.jsonl --judge
  ```
- Not yet: stale-action rate on a live screen workload, which needs a live computer-use or
  browser loop (Phase 4 evaluation).

## Phase 4 — Assembler, interfaces and evaluation (built; live evals pending)

- [x] `foveal.Assembler`: levels chosen at insertion under `budget_tokens`, by question
  relevance and recency. `CacheModel.should_compact` weighs the cache reads saved over the
  remaining calls against the one-off rewrite, with an extra margin on models that bind
  thinking to history. `plan_compaction` demotes everything due in one batch.
- [x] SDK middleware `foveal.wrap(client, memory, fmt="anthropic"|"openai")`: exact repeats
  become references, and a new screenshot from the same tool becomes a diff. The rewrite is
  prefix-stable (tested), so caching and preserved thinking keep working. `LOOK_TOOL` and
  `run_look_tool` page images back in.
- [x] MCP server `foveal-mcp` (MCP Python SDK 2.x, `MCPServer`) with ingest, ingest_frame,
  describe, look, diff_since, recall_facts, write_fact, verify_fact, subscribe and poll.
  Tested in-process and with a real stdio handshake.
- [x] Evaluation suite `bench/suite.py`: a $0 report and cost/success chart from existing
  logs (judge verdicts cached), dry runs of every evaluation, and live runs that refuse to
  start if their caps exceed `--budget`
- [x] Technical report draft: `docs/technical_report.md`
- [ ] Live evaluations (pending budget): `p1-scale` ($14), `p2-live` ($5), `p3-facts` ($3)
- [x] OpenRouter backend (`bench/openrouter.py`, `--provider openrouter`): Anthropic-shaped
  requests and replies over OpenRouter's chat API. Images in tool results move to a user
  message, reasoning_details are replayed unmodified, and the exact cost comes from
  `usage.cost`. Pricing check (2 Oct 2026): Claude costs the same on OpenRouter as direct.
  Repriced from our logs, the memory run would cost about $0.31 on Qwen3.7 Plus and $0.52 on
  Gemini 3.8 Flash, against $1.41 on Sonnet 5.5.
- [ ] Cross-provider smoke test (Qwen3.7 Plus, Gemini 3.8 Flash), then pending runs on them
- [ ] Stale-action rate on a live screen loop
- [ ] PyPI release (deferred by the owner)
