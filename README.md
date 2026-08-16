# Intention-Guided Reward Shaping for Sample-Efficient Reinforcement Learning

This project tests whether low-cost reward-shaping signals improve sample
efficiency in sparse-reward reinforcement learning. It also documents the
failure mode caused by a naive always-positive intention reward.

## Task-05 conditions

- `sparse`: native sparse reward only.
- `potential_geo`: potential-based normalized negative Euclidean distance.
- `potential_bfs`: potential-based normalized exact left/right/forward
  shortest-path distance on the current FourRooms layout.
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
python train.py --wrapper potential_geo --seed 42
```

The task-05 setup is `MiniGrid-FourRooms-v0`, a 500,000-step calibrated budget,
evaluation every 10,000 steps, and 30 fixed evaluation tasks. The primary
summary is normalized success-rate AUC through the planned 500,000-step budget.

## Run the calibrated matrix

```bash
python run_experiments.py --total-timesteps 500000 --eval-freq 10000
python analyze_results.py
```

Historical claims in `results_summary.txt` are retained as project evidence but
must not be treated as valid native-success results. See
`GATE_TASK05_REPORT.md` for the FourRooms experiment.
