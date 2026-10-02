# Doc QA: Claude Sonnet 5.5, full page images vs foveal memory

**Gate:** PASS: success 10/16 vs baseline 10/16 (needs >= 95% of baseline)

| Metric | Baseline | foveal | Change |
| --- | --- | --- | --- |
| Success | 10/16 | 10/16 | +0 |
| Image tokens (tasks + ingest) | 2,500,798 | 183,193 | -93% |
| Ingest image tokens | 0 | 60,702 | |
| Re-perception rate | 62.3% | 41.6% | |
| Cost incl. ingest | $4.06 | $1.40 | -66% |
| Model calls | 136 | 155 | |
| Same score as baseline | | 14/16 | |
| No worse than baseline | | 15/16 | |

| Task | Base score | foveal score | Base img tokens | foveal img tokens |
| --- | --- | --- | --- | --- |
| Campaign_038_Introducing_AC_Whitepaper_v#q0 | 0 | 0 | 62,804 | 0 |
| Campaign_038_Introducing_AC_Whitepaper_v#q1 | 1 | 1 | 17,944 | 6,729 |
| Campaign_038_Introducing_AC_Whitepaper_v#q2 | 0 | 0 | 2,243 | 2,243 |
| Campaign_038_Introducing_AC_Whitepaper_v#q3 | 0 | 1 | 6,729 | 6,729 |
| reportq32015-151009093138-lva1-app6891_9#q0 | 1 | 1 | 174,216 | 18,100 |
| reportq32015-151009093138-lva1-app6891_9#q1 | 1 | 1 | 40,992 | 3,416 |
| reportq32015-151009093138-lva1-app6891_9#q2 | 1 | 1 | 237,412 | 12,365 |
| reportq32015-151009093138-lva1-app6891_9#q3 | 1 | 1 | 213,500 | 0 |
| formwork-150318073913-conversion-gate01_#q0 | 1 | 1 | 300,608 | 21,314 |
| formwork-150318073913-conversion-gate01_#q1 | 1 | 1 | 83,692 | 14,761 |
| formwork-150318073913-conversion-gate01_#q2 | 1 | 0 | 170,800 | 13,664 |
| formwork-150318073913-conversion-gate01_#q3 | 0 | 0 | 109,312 | 15,348 |
| 11-21-16-Updated-Post-Election-Release#q0 | 0 | 0 | 120,883 | 2,467 |
| 11-21-16-Updated-Post-Election-Release#q1 | 1 | 1 | 737,633 | 0 |
| 11-21-16-Updated-Post-Election-Release#q2 | 0 | 0 | 111,015 | 0 |
| 11-21-16-Updated-Post-Election-Release#q3 | 1 | 1 | 111,015 | 5,355 |
