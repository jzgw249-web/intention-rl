# GATE 1: Naive intention reward loitering diagnostic

Run date: 2026-08-13  
Environment: `MiniGrid-Empty-8x8-v0`  
Training seed: 42  
Training steps: 30,000 per condition  
Evaluation: 30 deterministic episodes on the native, unshaped environment

## Final-policy evaluation

| Condition | success_rate | native_return | episode_length | Action distribution |
|---|---:|---:|---:|---|
| Intention | 0.00 | 0.000 +/- 0.000 | 256.0 +/- 0.0 | left 50%, right 50% |
| Sparse | 0.00 | 0.000 +/- 0.000 | 256.0 +/- 0.0 | forward 100% |

The intention policy never leaves `(1,1)`. It alternates `right` and `left`,
switching between directions 0 and 1 until the 256-step time limit. This
directly confirms the loitering failure mode predicted in the task brief.

The sparse final deterministic policy moves from `(1,1)` to `(6,1)`, then
continues choosing `forward` into the wall until truncation. It therefore also
fails under final deterministic evaluation at 30,000 steps.

## Training episodes

| Condition | Episodes | success_rate | native_return | episode_length |
|---|---:|---:|---:|---:|
| Intention, all | 123 | 0.057 | 0.028 +/- 0.119 | 249.7 +/- 27.8 |
| Intention, last 30 | 30 | 0.00 | 0.000 +/- 0.000 | 256.0 +/- 0.0 |
| Sparse, all | 267 | 0.824 | 0.580 +/- 0.332 | 114.7 +/- 86.0 |
| Sparse, last 30 | 30 | 1.00 | 0.809 (mean) | 54.3 +/- 56.6 |

## Interpretation and caveat

The central loitering hypothesis is confirmed for the naive intention reward:
the converged policy deliberately avoids progress and collects shaping reward
through in-place rotations.

The brief's secondary expectation that the saved sparse policy would be the
successful one is not confirmed by this 30,000-step deterministic checkpoint.
Training rollouts show that the stochastic sparse policy was succeeding,
including all of its last 30 completed training episodes, while deterministic
action selection collapses to `forward`. Subsequent evaluation work should
therefore report whether policy inference is deterministic or stochastic and
should not infer final policy quality solely from training rollouts.

Raw CSV, trajectories, and model files are retained locally under
`gate1_diagnostics/` and are not part of the historical `results_summary.txt`.
