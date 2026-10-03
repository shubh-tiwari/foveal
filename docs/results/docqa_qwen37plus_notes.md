# Doc QA: Qwen3.7 Plus, foveal memory (baseline) vs foveal memory + shared notes

**Gate:** FAIL: success 8/13 vs baseline 9/13 (needs >= 95% of baseline)

| Metric | Baseline | foveal | Change |
| --- | --- | --- | --- |
| Success | 9/13 | 8/13 | -1 |
| Image tokens (tasks + ingest) | 233,774 | 369,383 | +58% |
| Ingest image tokens | 47,412 | 60,034 | |
| Re-perception rate | 74.1% | 79.1% | |
| Cost incl. ingest | $0.42 | $0.49 | +17% |
| Model calls | 232 | 254 | |
| Same score as baseline | | 12/13 | |
| No worse than baseline | | 12/13 | |

| Task | Base score | foveal score | Base img tokens | foveal img tokens |
| --- | --- | --- | --- | --- |
| Campaign_038_Introducing_AC_Whitepaper_v#q0 | 0 | 0 | 0 | 6,720 |
| Campaign_038_Introducing_AC_Whitepaper_v#q1 | 1 | 1 | 22,400 | 31,229 |
| Campaign_038_Introducing_AC_Whitepaper_v#q2 | 1 | 0 | 22,400 | 2,734 |
| Campaign_038_Introducing_AC_Whitepaper_v#q3 | 0 | 0 | 22,400 | 22,400 |
| reportq32015-151009093138-lva1-app6891_9#q0 | 1 | 1 | 6,820 | 27,280 |
| reportq32015-151009093138-lva1-app6891_9#q1 | 1 | 1 | 27,280 | 44,330 |
| reportq32015-151009093138-lva1-app6891_9#q2 | 1 | 1 | 3,410 | 0 |
| reportq32015-151009093138-lva1-app6891_9#q3 | 1 | 1 | 8,525 | 8,525 |
| formwork-150318073913-conversion-gate01_#q0 | 1 | 1 | 8,525 | 1,705 |
| formwork-150318073913-conversion-gate01_#q1 | 1 | 1 | 6,820 | 35,805 |
| formwork-150318073913-conversion-gate01_#q2 | 1 | 1 | 6,820 | 20,460 |
| formwork-150318073913-conversion-gate01_#q3 | 0 | 0 | 46,035 | 103,233 |
| 11-21-16-Updated-Post-Election-Release#q0 | 0 | 0 | 4,928 | 4,928 |
