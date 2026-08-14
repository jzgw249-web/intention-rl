import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONDITIONS = ("sparse", "intention_pb", "dense", "intention_naive")
LABELS = {
    "sparse": "Sparse",
    "intention_pb": "Intention (potential-based)",
    "dense": "Dense distance (original)",
    "intention_naive": "Intention (naive)",
}


def evaluation_dir(log_dir, env_slug, condition, seed):
    return Path(log_dir) / f"{env_slug}_{condition}_seed{seed}" / "evaluation"


def load_metric(log_dir, env_slug, conditions, seeds, metric):
    frames = []
    for condition in conditions:
        for seed in seeds:
            path = evaluation_dir(log_dir, env_slug, condition, seed) / f"{metric}.csv"
            frame = pd.read_csv(path)
            frame["condition"] = condition
            frame["seed"] = seed
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def plot_metric(frame, metric, output_path, n_seeds):
    plt.figure(figsize=(9.5, 5.8))
    for condition in CONDITIONS:
        group = frame[frame.condition == condition]
        summary = group.groupby("timesteps")["stochastic_mean"].agg(["mean", "std"])
        steps = summary.index.to_numpy()
        mean = summary["mean"].to_numpy()
        std = summary["std"].fillna(0).to_numpy()
        plt.plot(steps, mean, marker="o", linewidth=2, label=LABELS[condition])
        plt.fill_between(steps, mean - std, mean + std, alpha=0.18)
    plt.xlabel("Training steps")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(f"Native evaluation: {metric} (mean +/- std, n={n_seeds} seeds)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def summarize(success, native_return, episode_length, seeds):
    rows = []
    first_crossings = {}
    for condition in CONDITIONS:
        crossings = []
        final_successes = []
        final_returns = []
        final_lengths = []
        for seed in seeds:
            success_seed = success[(success.condition == condition) & (success.seed == seed)].sort_values("timesteps")
            crossed = success_seed[success_seed.stochastic_mean >= 0.5]
            if not crossed.empty:
                crossings.append(float(crossed.iloc[0].timesteps))
            final_successes.append(float(success_seed.iloc[-1].stochastic_mean))
            return_seed = native_return[(native_return.condition == condition) & (native_return.seed == seed)].sort_values("timesteps")
            length_seed = episode_length[(episode_length.condition == condition) & (episode_length.seed == seed)].sort_values("timesteps")
            final_returns.append(float(return_seed.iloc[-1].stochastic_mean))
            final_lengths.append(float(length_seed.iloc[-1].stochastic_mean))
        first_crossings[condition] = crossings
        rows.append({
            "condition": condition,
            "first_50_mean_steps": np.mean(crossings) if crossings else np.nan,
            "first_50_std_steps": np.std(crossings) if crossings else np.nan,
            "first_50_reached_seeds": len(crossings),
            "final_success_mean": np.mean(final_successes),
            "final_success_std": np.std(final_successes),
            "final_native_return_mean": np.mean(final_returns),
            "final_native_return_std": np.std(final_returns),
            "final_episode_length_mean": np.mean(final_lengths),
            "final_episode_length_std": np.std(final_lengths),
            "n_seeds": len(seeds),
        })
    table = pd.DataFrame(rows)
    sparse = first_crossings["sparse"]
    potential = first_crossings["intention_pb"]
    acceleration = np.mean(sparse) / np.mean(potential) if sparse and potential else np.nan
    return table, acceleration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="main_logs")
    parser.add_argument("--env-slug", default="empty-8x8")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 50)))
    parser.add_argument("--output-dir", default="main_results")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = {}
    for metric in ("success_rate", "native_return", "episode_length"):
        frames[metric] = load_metric(args.log_dir, args.env_slug, CONDITIONS, args.seeds, metric)
        plot_metric(frames[metric], metric, output_dir / f"{metric}.png", len(args.seeds))
    table, acceleration = summarize(
        frames["success_rate"], frames["native_return"], frames["episode_length"], args.seeds
    )
    table.to_csv(output_dir / "experiment_summary.csv", index=False)
    (output_dir / "acceleration.txt").write_text(
        "not reached\n" if np.isnan(acceleration) else f"{acceleration:.6f}\n",
        encoding="utf-8",
    )
    print(table.to_string(index=False))
    print("acceleration", acceleration)


if __name__ == "__main__":
    main()
