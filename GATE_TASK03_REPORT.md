# Task 03 GATE report: pipeline rebuild and calibration

Run date: 2026-08-14

## Pipeline changes

- Evaluation always uses the native environment without reward shaping.
- `success_rate`, `native_return`, and `episode_length` are written to separate
  CSV files. Stochastic-policy metrics are primary; deterministic metrics are
  retained in explicitly named columns.
- Evaluation uses the same fixed 30 episode seeds across conditions and
  training seeds.
- Observations now contain the flattened 7x7x3 image plus a 4-way direction
  one-hot vector (151 dimensions); the environment passes `check_env`.
- The environment id is configurable, with FourRooms as the default.
- Geometric, optimal-path, and wrong-direction potential wrappers share PPO's
  gamma and use normalized layout-specific potentials.
- The reusable BFS module computes distances and optimal actions on the
  `(position, direction)` state graph using left, right, and forward actions.

## Potential and boundary validation

Across 20 FourRooms layouts:

| Potential | Observed range |
|---|---|
| Geometric | [-1.0, 0.0] |
| Optimal path | [-1.0, 0.0] |
| Wrong direction | [0.0, 1.0] |

All three potential wrappers passed explicit one-step tests for both endings:
true termination sets successor potential to zero; time-limit truncation uses
the actual successor potential.

## Five-condition smoke test

Requested budget was 5,000 steps. Stable-Baselines3 completed the rollout at
6,144 steps because PPO uses a 2,048-step rollout buffer. The table reports the
exact 5,000-step evaluation checkpoint, using 30 stochastic evaluation episodes.

| Condition | success_rate | native_return | episode_length |
|---|---:|---:|---:|
| A Sparse | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 100.0 +/- 0.0 |
| B Geometric | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 100.0 +/- 0.0 |
| C Optimal path | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 100.0 +/- 0.0 |
| D Wrong direction | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 100.0 +/- 0.0 |
| E Naive intention | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 100.0 +/- 0.0 |

All five conditions trained, evaluated, saved models, and produced three plots.
The zero success at this deliberately short budget is not interpreted as a
condition comparison.

## Sparse FourRooms calibration (seed 42)

The stochastic-policy success curve at selected checkpoints was:

| Steps | success_rate |
|---:|---:|
| 0 | 0.067 |
| 75,000 | 0.100 |
| 125,000 | 0.167 |
| 200,000 | 0.200 |
| 275,000 | 0.333 |
| 350,000 | 0.333 |
| 400,000 | 0.433 |
| 450,000 | 0.400 |
| 500,000 | 0.300 |
| 501,760 | 0.433 |

The baseline first became non-zero at 75,000 steps after a zero checkpoint at
50,000. It never reached 50% within the requested 500,000-step budget. The
highest measured checkpoint was 0.433 at 400,000 and 501,760 steps. At the final
rollout boundary, native return was 0.318 and episode length was 69.5 steps.

FourRooms should not be replaced solely under the brief's fallback rule because
the baseline clearly learns above chance. However, 500,000 steps does not yield
a stable 50% crossing. For D2, use at least 500,000 steps per run and treat
"first reaches 50%" as potentially censored. If compute allows, 750,000 steps is
the safer budget; otherwise retain FourRooms and report runs that do not cross
50% as not reached rather than switching to the easier environment after seeing
the result.

## Independent review of the terminal rule

For an episodic MDP with an absorbing terminal state, the potential-based
telescoping argument requires `Phi(terminal)=0`, so true `terminated` transitions
use zero successor potential. A `truncated` transition is a time-limit cut of a
nonterminal process, so its actual successor potential must be retained. This
matches the task brief and the automated tests.

There is one theoretical scope caveat: policy invariance is a theorem for the
underlying MDP state representation. Here the policy receives a partial 151-D
observation while the shaping potential uses hidden global position and goal
information. Therefore the implementation has the correct potential form on
the underlying MiniGrid state, but policy invariance should not be claimed
without qualification for the induced POMDP with function approximation.

## Additional artifact

`record_video.py` now supports historical channel-first image models and the new
151-D observation models. The existing Gate 1 intention model was rendered as a
257-frame loitering video under `gate1_diagnostics/intention_loitering.mp4`.
