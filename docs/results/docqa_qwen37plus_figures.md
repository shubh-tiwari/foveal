# Doc QA, figure/table/chart questions: Qwen3.7 Plus, full page images vs foveal memory

**Gate:** PASS: success 7/8 vs baseline 7/8 (needs >= 95% of baseline)

| Metric | Baseline | foveal | Change |
| --- | --- | --- | --- |
| Success | 7/8 | 7/8 | +0 |
| Image tokens (tasks + ingest) | 1,522,416 | 293,166 | -81% |
| Ingest image tokens | 0 | 38,148 | |
| Re-perception rate | 79.1% | 60.3% | |
| Cost incl. ingest | $0.33 | $0.22 | -34% |
| Model calls | 71 | 85 | |
| Same score as baseline | | 6/8 | |
| No worse than baseline | | 7/8 | |

| Task | Base score | foveal score | Base img tokens | foveal img tokens |
| --- | --- | --- | --- | --- |
| guojixueshengshenghuozhinanyingwen9.1#q0 | 1 | 0 | 675,024 | 78,679 |
| guojixueshengshenghuozhinanyingwen9.1#q1 | 1 | 1 | 108,192 | 57,535 |
| 2312.10997v5#q0 | 1 | 1 | 285,824 | 9,520 |
| 2312.10997v5#q1 | 1 | 1 | 9,856 | 2,464 |
| Independents-Report#q0 | 1 | 1 | 251,328 | 4,928 |
| Independents-Report#q1 | 0 | 1 | 172,480 | 82,180 |
| 8dfc21ec151fb9d3578fc32d5c4e5df9#q0 | 1 | 1 | 4,928 | 4,928 |
| 8dfc21ec151fb9d3578fc32d5c4e5df9#q1 | 1 | 1 | 14,784 | 14,784 |
