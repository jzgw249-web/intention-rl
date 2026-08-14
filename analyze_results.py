import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONDITIONS = (
    "sparse", "dense", "intention_naive", "intention_pb",
    "potential_dist", "intention_pb_shifted",
)
LABELS = {
    "sparse": "Sparse",
    "dense": "Dense (original)",
    "intention_naive": "Intention naive",
    "intention_pb": "Intention PB raw",
    "potential_dist": "Distance potential",
    "intention_pb_shifted": "Intention PB shifted",
}


def evaluation_dir(log_dir, env_slug, condition, seed):
    return Path(log_dir) / f"{env_slug}_{condition}_seed{seed}" / "evaluation"


def load_metric(log_dir, env_slug, seeds, metric):
    frames = []
    for condition in CONDITIONS:
        for seed in seeds:
            frame = pd.read_csv(evaluation_dir(log_dir, env_slug, condition, seed) / f"{metric}.csv")
            frame["condition"] = condition
            frame["seed"] = seed
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def plot_metric(frame, metric, path, n_seeds):
    plt.figure(figsize=(10.5, 6.2))
    for condition in CONDITIONS:
        group = frame[frame.condition == condition]
        summary = group.groupby("timesteps")["stochastic_mean"].agg(["mean", "std"])
        steps = summary.index.to_numpy()
        mean = summary["mean"].to_numpy()
        std = summary["std"].fillna(0).to_numpy()
        plt.plot(steps, mean, linewidth=2, label=LABELS[condition])
        plt.fill_between(steps, mean - std, mean + std, alpha=0.15)
    plt.xlabel("Training steps")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(f"Random-6x6 native evaluation (mean +/- std, n={n_seeds})")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def summarize(success, native_return, episode_length, seeds):
    summary_rows = []
    crossing_rows = []
    crossings_by_condition = {}
    for condition in CONDITIONS:
        crossings = []
        final_successes, final_returns, final_lengths = [], [], []
        for seed in seeds:
            success_seed = success[(success.condition == condition) & (success.seed == seed)].sort_values("timesteps")
            crossed = success_seed[success_seed.stochastic_mean >= 0.5]
            crossing = float(crossed.iloc[0].timesteps) if not crossed.empty else np.nan
            crossings.append(crossing)
            crossing_rows.append({"condition": condition, "seed": seed, "first_50_steps": crossing})
            final_successes.append(float(success_seed.iloc[-1].stochastic_mean))
            return_seed = native_return[(native_return.condition == condition) & (native_return.seed == seed)].sort_values("timesteps")
            length_seed = episode_length[(episode_length.condition == condition) & (episode_length.seed == seed)].sort_values("timesteps")
            final_returns.append(float(return_seed.iloc[-1].stochastic_mean))
            final_lengths.append(float(length_seed.iloc[-1].stochastic_mean))
        finite = np.asarray([value for value in crossings if not np.isnan(value)])
        crossings_by_condition[condition] = finite
        summary_rows.append({
            "condition": condition,
            "first_50_mean_steps": finite.mean() if len(finite) else np.nan,
            "first_50_std_steps": finite.std() if len(finite) else np.nan,
            "first_50_reached_seeds": len(finite),
            "final_success_mean": np.mean(final_successes),
            "final_success_std": np.std(final_successes),
            "final_native_return_mean": np.mean(final_returns),
            "final_native_return_std": np.std(final_returns),
            "final_episode_length_mean": np.mean(final_lengths),
            "final_episode_length_std": np.std(final_lengths),
            "n_seeds": len(seeds),
        })
    sparse = crossings_by_condition["sparse"]
    distance = crossings_by_condition["potential_dist"]
    acceleration = sparse.mean() / distance.mean() if len(sparse) and len(distance) else np.nan
    return pd.DataFrame(summary_rows), pd.DataFrame(crossing_rows), acceleration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="task04_logs")
    parser.add_argument("--env-slug", default="empty-random-6x6")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 50)))
    parser.add_argument("--output-dir", default="task04_results")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = {}
    for metric in ("success_rate", "native_return", "episode_length"):
        frames[metric] = load_metric(args.log_dir, args.env_slug, args.seeds, metric)
        plot_metric(frames[metric], metric, output_dir / f"{metric}.png", len(args.seeds))
    summary, crossings, acceleration = summarize(
        frames["success_rate"], frames["native_return"], frames["episode_length"], args.seeds
    )
    summary.to_csv(output_dir / "experiment_summary.csv", index=False)
    crossings.to_csv(output_dir / "first_50_by_seed.csv", index=False)
    (output_dir / "acceleration.txt").write_text(
        "not reached\n" if np.isnan(acceleration) else f"{acceleration:.6f}\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(crossings.pivot(index="condition", columns="seed", values="first_50_steps"))
    print("distance acceleration", acceleration)


if __name__ == "__main__":
    main()
