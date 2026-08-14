# Intention-Guided Reward Shaping for Sample-Efficient Reinforcement Learning

This project tests whether a low-cost directional intention signal can improve
sample efficiency in sparse-reward reinforcement learning, and documents a
failure mode caused by a naive always-positive shaping reward.

## Conditions

- `sparse`: native sparse reward only.
- `intention_pb`: potential-based intention shaping,
  `lambda * (gamma * Phi(s') - Phi(s))`.
- `dense`: the original dense negative-distance reward.
- `intention_naive`: the original always-positive intention reward.
- `intention`: backward-compatible alias for `intention_naive`.

Evaluation always uses the unshaped native environment. Success rate, native
return, and episode length are saved separately. Stochastic-policy evaluation
is the primary metric; deterministic results are retained as a secondary view.

## Setup

```bash
python -m venv .venv312
.venv312\Scripts\activate
pip install -r requirements.txt
```

## Run one experiment

```bash
python train.py --wrapper intention_pb --env-id MiniGrid-Empty-8x8-v0 --seed 42 --total-timesteps 100000
```

## Run the full safeguarded experiment

```bash
python run_experiments.py --max-workers 4
python analyze_results.py
```

Historical claims in `results_summary.txt` are retained as project evidence but
must not be treated as valid native-success results. See `GATE1_REPORT.md` for
the corrected loitering diagnosis.
