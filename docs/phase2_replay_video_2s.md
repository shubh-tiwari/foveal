# foveal Phase 2 offline replay: screen recordings, one frame every 2s

9 trajectories, 507 steps, priced as claude-sonnet-5-5. No model calls.

| Policy | Same info as | Image tokens | vs full history | Est. cost |
| --- | --- | --- | --- | --- |
| full_history | everything | 15,711,456 | +0% | $4.23 |
| last_1 | current only | 474,924 | -97% | $0.95 |
| last_3 | last 3 | 1,400,622 | -91% | $2.80 |
| foveal_diff | everything | 8,911,799 | -43% | $2.42 |
| foveal_window3 | last 3 | 1,046,724 | -93% | $2.09 |

## Diff engine

- Step-to-step diffs: 498. Identical 13, partial 239, full 246. Scroll detected in 14.
- Reconstruction fidelity: median share of visibly wrong pixels 0.000%, worst 0.27%.
- Target element pixel-exact after reconstruction: 0.0% of 0 steps.
- Example description: "Changed substantially; full frame follows."
