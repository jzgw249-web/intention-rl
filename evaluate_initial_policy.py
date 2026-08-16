import argparse
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

from train import (
    DEFAULT_ENV_ID,
    EVALUATION_POLICY_SEED,
    build_model,
    evaluate_native,
    make_env,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--eval-seed-base", type=int, default=30_000)
    parser.add_argument("--output", default="task05_calibration/initial_policy.json")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env = DummyVecEnv([make_env(DEFAULT_ENV_ID, "sparse", args.seed, training=True)])
    model = build_model(env, args.seed)
    try:
        np.random.seed(EVALUATION_POLICY_SEED)
        torch.manual_seed(EVALUATION_POLICY_SEED)
        metrics = evaluate_native(
            model,
            DEFAULT_ENV_ID,
            range(args.eval_seed_base, args.eval_seed_base + args.eval_episodes),
            deterministic=False,
        )
    finally:
        env.close()

    payload = {
        "training_steps": 0,
        "policy_seed": args.seed,
        "evaluation_policy_seed": EVALUATION_POLICY_SEED,
        "evaluation_seeds": list(range(
            args.eval_seed_base,
            args.eval_seed_base + args.eval_episodes,
        )),
        "metrics": {
            name: {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "values": values.tolist(),
            }
            for name, values in metrics.items()
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        name: {"mean": row["mean"], "std": row["std"]}
        for name, row in payload["metrics"].items()
    }, indent=2))


if __name__ == "__main__":
    main()
