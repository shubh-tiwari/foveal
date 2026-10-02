# foveal Phase 2 offline replay: Multimodal-Mind2Web

113 trajectories, 725 steps, priced as claude-sonnet-5-5. No model calls.

| Policy | Same info as | Image tokens | vs full history | Est. cost |
| --- | --- | --- | --- | --- |
| full_history | everything | 4,054,440 | +0% | $2.81 |
| last_1 | current only | 867,100 | -79% | $1.73 |
| last_3 | last 3 | 2,195,856 | -46% | $4.39 |
| foveal_diff | everything | 3,209,685 | -21% | $2.16 |
| foveal_window3 | last 3 | 1,852,836 | -54% | $3.71 |

## Diff engine

- Step-to-step diffs: 612. Identical 6, partial 225, full 381. Scroll detected in 85.
- Reconstruction fidelity: median share of visibly wrong pixels 0.000%, worst 0.01%.
- Target element pixel-exact after reconstruction: 100.0% of 579 steps.
- Example description: "Changed substantially; full frame follows."
