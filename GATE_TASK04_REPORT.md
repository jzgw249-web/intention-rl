# GATE Task 04 Report

## Scope and protocol

- Branch: `fix/eval-pipeline`; no push and no change to `master`.
- Environment: `MiniGrid-Empty-Random-6x6-v0` (`max_steps=144`).
- Conditions: six; training seeds: 42--49 (`n=8` per condition).
- Requested training budget: 40,000 steps. Stable-Baselines3 finishes the
  2,048-step PPO rollout, so every run ends at 40,960 environment steps.
- Native, unshaped stochastic evaluation every 2,000 requested steps, plus the
  initial and final evaluations; 30 fixed evaluation seeds per checkpoint.
- Reported standard deviations use the eight training-seed aggregates
  (`ddof=0`). No hyperparameter scan was run.

All 48 models and all 144 metric CSV files are present. Every metric file has
22 rows (0, each 2k checkpoint through 40k, and the final 40,960 checkpoint).

## Aggregate results

| Condition | First >=50% success (steps) | Reached | Final success | Final native return | Final episode length |
|---|---:|---:|---:|---:|---:|
| sparse | 5,750 +/- 1,854 | 8/8 | 0.996 +/- 0.011 | 0.879 +/- 0.053 | 19.26 +/- 8.29 |
| dense | 7,250 +/- 1,984 | 8/8 | 0.967 +/- 0.029 | 0.699 +/- 0.030 | 47.63 +/- 4.84 |
| intention_naive | 4,000 +/- 0* | 1/8 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 144.00 +/- 0.00 |
| intention_pb | 5,750 +/- 1,854 | 8/8 | 0.975 +/- 0.028 | 0.761 +/- 0.097 | 37.88 +/- 15.20 |
| potential_dist | 5,750 +/- 1,199 | 8/8 | 0.996 +/- 0.011 | 0.869 +/- 0.034 | 20.86 +/- 5.28 |
| intention_pb_shifted | 6,000 +/- 1,732 | 8/8 | 0.996 +/- 0.011 | 0.843 +/- 0.069 | 25.13 +/- 10.91 |

`*` The naive first-crossing statistic is based only on seed 47. The other
seven seeds never reached 50%, so 4,000 must not be interpreted as an `n=8`
mean.

## First-50% checkpoint by training seed

| Condition | 42 | 43 | 44 | 45 | 46 | 47 | 48 | 49 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sparse | 4k | 4k | 6k | 4k | 6k | 10k | 6k | 6k |
| dense | 6k | 10k | 8k | 6k | 4k | 10k | 8k | 6k |
| intention_naive | -- | -- | -- | -- | -- | 4k | -- | -- |
| intention_pb | 4k | 8k | 4k | 4k | 8k | 6k | 4k | 8k |
| potential_dist | 6k | 4k | 4k | 6k | 6k | 8k | 6k | 6k |
| intention_pb_shifted | 8k | 4k | 4k | 8k | 6k | 8k | 6k | 4k |

The machine-readable versions are in `task04_results/experiment_summary.csv`
and `task04_results/first_50_by_seed.csv`.

## GATE conclusions

### Distance potential versus sparse

The acceleration ratio is
`mean sparse crossing / mean distance crossing = 5750 / 5750 = 1.00x`.
The distance potential therefore did not improve the measured sample
efficiency in this environment. It reduced the between-seed spread of the
crossing checkpoint, but final success, native return, and episode length were
also close to sparse. This is consistent with the task brief's alternative
interpretation: Random-6x6 may leave little acceleration headroom at a 2k
measurement grid because sparse already crosses at 4k--10k.

### Shifted versus raw intention potential

Raw crossed at 5.75k +/- 1.85k; shifted crossed at 6.00k +/- 1.73k. Their
per-seed checkpoints overlap, and the 250-step difference in the means is below
one per-seed evaluation interval. Shifted has better final return and shorter
episodes than raw, but this run does not isolate a decisive A-versus-B cause.
In particular, removing the raw terminal penalty did not produce earlier
crossing than sparse. The evidence therefore does not support the terminal
penalty as the sole or dominant explanation; the cosine potential's lack of
progress information remains the more plausible limitation, while the current
resolution and `n=8` do not justify a stronger causal claim.

### Naive intention reward in the random environment

The naive reward still collapses. Only seed 47 briefly crossed 50% at 4k; all
eight seeds finish with zero success, zero native return, and the full 144-step
episode length. Randomizing start position and direction therefore does not
remove the failure. This supports the explanation that an always-positive
shaping reward can favor reward collection without task completion, rather
than the old fixed diagonal geometry being the necessary cause.

### Is the 2k resolution sufficient?

It is materially better than the previous 10k grid. Non-naive crossings occupy
4k, 6k, 8k, and 10k checkpoints, so the observations are no longer all crushed
into a single checkpoint. However, most sparse and distance crossings remain
at 4k or 6k, so the design cannot resolve a sub-2k advantage. It is sufficient
to reject a large distance-potential speedup in this run, not to rule out a
small one.

## Task-brief issues and implementation choices

1. The brief says both that all potentials should be in `[-1, 0]` and that the
   existing raw `intention_pb`, `Phi=(cos+1)/2`, should remain unchanged. Those
   requirements conflict. Raw was kept unchanged in `[0, 1]` for a valid
   historical comparison; shifted and distance potentials use `[-1, 0]`.
2. Thirty fixed environment seeds do not guarantee 30 unique task instances.
   For seeds 20000--20029, only 21 unique `(start position, direction)` pairs
   occur. The evaluation is still more diverse than fixed Empty-8x8, but the
   report does not call these 30 independent layouts/tasks.
3. Empty-Random-6x6 randomizes start and direction but retains the same empty
   layout and fixed goal. This provides state diversity, not layout diversity.
4. The PPO rollout granularity makes the actual final checkpoint 40,960 rather
   than exactly 40,000. The requested evaluation checkpoints through 40,000 are
   present, and the extra final checkpoint is reported rather than hidden.

## Video artifact

`task04_results/normal_navigation.mp4` records an eight-action successful
trajectory from the distance-potential seed-43 model on environment seed 20000.
`task04_results/loitering_vs_navigation.mp4` concatenates the existing naive
loitering clip (first 16.06 s) with this normal navigation clip (last 4.50 s).
The combined file is 256x256, 16 fps, 20.56 s, and decodes to 329 frames.

## Git-history note

Commit `a3dbf20` was the first task-03 implementation, built around the earlier
FourRooms/BFS interpretation. The task specification was then revised to the
minimal safeguarded Empty-8x8 experiment while the same filename remained in
use. Commit `a23d210` deliberately reverted `a3dbf20` before the revised work
was implemented, preserving an auditable history instead of silently rewriting
the earlier commit. The resulting task-03 state was committed as `383bdd7`.

## Stop condition

GATE reached. No lambda scan and no FourRooms run was started.
