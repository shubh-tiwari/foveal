# Doc QA: Qwen3.7 Plus (OpenRouter), full page images vs foveal memory

**Gate:** FAIL: success 8/10 vs baseline 10/10 (needs >= 95% of baseline)

| Metric | Baseline | foveal | Change |
| --- | --- | --- | --- |
| Success | 10/10 | 8/10 | -2 |
| Image tokens (tasks + ingest) | 3,858,890 | 161,284 | -96% |
| Ingest image tokens | 0 | 32,704 | |
| Re-perception rate | 86.4% | 78.9% | |
| Cost incl. ingest | $0.66 | $0.29 | -56% |
| Model calls | 127 | 169 | |
| Same score as baseline | | 8/10 | |
| No worse than baseline | | 8/10 | |

| Task | Base score | foveal score | Base img tokens | foveal img tokens |
| --- | --- | --- | --- | --- |
| Campaign_038_Introducing_AC_Whitepaper_v#q0 | 1 | 0 | 112,000 | 0 |
| Campaign_038_Introducing_AC_Whitepaper_v#q1 | 1 | 1 | 13,440 | 22,400 |
| Campaign_038_Introducing_AC_Whitepaper_v#q2 | 1 | 1 | 11,200 | 22,400 |
| Campaign_038_Introducing_AC_Whitepaper_v#q3 | 1 | 0 | 22,400 | 22,400 |
| reportq32015-151009093138-lva1-app6891_9#q0 | 1 | 1 | 371,690 | 6,820 |
| reportq32015-151009093138-lva1-app6891_9#q1 | 1 | 1 | 254,045 | 27,280 |
| reportq32015-151009093138-lva1-app6891_9#q2 | 1 | 1 | 173,910 | 3,410 |
| reportq32015-151009093138-lva1-app6891_9#q3 | 1 | 1 | 272,800 | 8,525 |
| formwork-150318073913-conversion-gate01_#q0 | 1 | 1 | 2,161,940 | 8,525 |
| formwork-150318073913-conversion-gate01_#q1 | 1 | 1 | 465,465 | 6,820 |
