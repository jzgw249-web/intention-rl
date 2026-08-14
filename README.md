# Intention-Guided Reward Shaping for Sample-Efficient Reinforcement Learning

This project tests whether low-cost reward-shaping signals improve sample
efficiency in sparse-reward reinforcement learning. It also documents the
failure mode caused by a naive always-positive intention reward.

## Conditions

- `sparse`: native sparse reward only.
- `dense`: the original dense negative-distance reward.
- `intention_naive`: the original always-positive intention reward.
- `intention_pb`: potential-based raw intention shaping with
  `Phi = (cos + 1) / 2`.
- `potential_dist`: potential-based shaping with normalized negative distance.
- `intention_pb_shifted`: potential-based shifted intention shaping with
  `Phi = (cos - 1) / 2`.
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
python train.py --wrapper potential_dist --seed 42
```

The default task-04 setup is `MiniGrid-Empty-Random-6x6-v0`, 40,000 requested
training steps, evaluation every 2,000 steps, and 30 fixed evaluation seeds.

## Run the full task-04 experiment

```bash
python run_experiments.py --max-workers 4
python analyze_results.py
```

Historical claims in `results_summary.txt` are retained as project evidence but
must not be treated as valid native-success results. See
`GATE_TASK03_REVISED_REPORT.md` and `GATE_TASK04_REPORT.md` for the safeguarded
evaluations.
