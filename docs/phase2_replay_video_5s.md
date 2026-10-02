# foveal Phase 2 offline replay: screen recordings, one frame every 5s

9 trajectories, 205 steps, priced as claude-sonnet-5-5. No model calls.

| Policy | Same info as | Image tokens | vs full history | Est. cost |
| --- | --- | --- | --- | --- |
| full_history | everything | 2,605,768 | +0% | $0.96 |
| last_1 | current only | 191,420 | -93% | $0.38 |
| last_3 | last 3 | 550,110 | -79% | $1.10 |
| foveal_diff | everything | 1,999,383 | -23% | $0.74 |
| foveal_window3 | last 3 | 472,014 | -82% | $0.94 |

## Diff engine

- Step-to-step diffs: 196. Identical 0, partial 64, full 132. Scroll detected in 2.
- Reconstruction fidelity: median share of visibly wrong pixels 0.000%, worst 0.02%.
- Target element pixel-exact after reconstruction: 0.0% of 0 steps.
- Example description: "Changed substantially; full frame follows."
