import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


CONDITIONS = ("sparse", "geo", "optimal", "wrong", "intention")
LABELS = {
    "sparse": "A Sparse",
    "geo": "B Geometric",
    "optimal": "C Optimal path",
    "wrong": "D Wrong direction",
    "intention": "E Naive intention",
}


def find_experiment(log_dir, env_slug, condition, seed):
    return Path(log_dir) / f"{env_slug}_{condition}_seed{seed}" / "evaluation"


def load_metric(log_dir, env_slug, conditions, seeds, metric):
    frames = []
    for condition in conditions:
        for seed in seeds:
            path = find_experiment(log_dir, env_slug, condition, seed) / f"{metric}.csv"
            if not path.exists():
                continue
            frame = pd.read_csv(path)
            frame["condition"] = condition
            frame["seed"] = seed
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No {metric}.csv files found")
    return pd.concat(frames, ignore_index=True)


def plot_metric(frame, metric, output_path):
    plt.figure(figsize=(9, 5.5))
    for condition, group in frame.groupby("condition"):
        summary = group.groupby("timesteps")["stochastic_mean"].agg(["mean", "std", "count"])
        std = summary["std"].fillna(0.0)
        steps = summary.index.to_numpy()
        mean = summary["mean"].to_numpy()
        plt.plot(steps, mean, marker="o", label=LABELS.get(condition, condition))
        plt.fill_between(steps, mean - std.to_numpy(), mean + std.to_numpy(), alpha=0.15)
    plt.xlabel("Training steps")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(f"Native evaluation: {metric} (stochastic policy)")
    if metric == "success_rate":
        plt.ylim(-0.02, 1.02)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--env-slug", default="fourrooms")
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--output-dir", default="plots")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for metric in ("success_rate", "native_return", "episode_length"):
        frame = load_metric(args.log_dir, args.env_slug, args.conditions, args.seeds, metric)
        plot_metric(frame, metric, output_dir / f"{metric}.png")


if __name__ == "__main__":
    main()
