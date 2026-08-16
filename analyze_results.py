import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONDITIONS = ("sparse", "potential_geo", "potential_bfs", "intention_naive")
LABELS = {
    "sparse": "Sparse",
    "potential_geo": "Geometric potential",
    "potential_bfs": "BFS potential",
    "intention_naive": "Intention naive",
}
METRICS = ("success_rate", "native_return", "episode_length")


def evaluation_dir(log_dir, condition, seed):
    return Path(log_dir) / f"fourrooms_{condition}_seed{seed}" / "evaluation"


def load_metric(log_dir, seeds, metric, budget):
    frames = []
    for condition in CONDITIONS:
        for seed in seeds:
            path = evaluation_dir(log_dir, condition, seed) / f"{metric}.csv"
            frame = pd.read_csv(path)
            frame = frame[frame.timesteps <= budget].copy()
            frame["condition"] = condition
            frame["seed"] = seed
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def plot_metric(frame, metric, output_path, n_seeds):
    plt.figure(figsize=(10.5, 6.2))
    for condition in CONDITIONS:
        group = frame[frame.condition == condition]
        pivot = group.pivot(index="timesteps", columns="seed", values="stochastic_mean")
        steps = pivot.index.to_numpy()
        mean = pivot.mean(axis=1).to_numpy()
        std = pivot.std(axis=1, ddof=0).to_numpy()
        plt.plot(steps, mean, linewidth=2, label=LABELS[condition])
        plt.fill_between(steps, mean - std, mean + std, alpha=0.16)
    plt.xlabel("Training steps")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(f"FourRooms native evaluation (mean +/- std, n={n_seeds})")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def first_crossing(frame, threshold):
    crossed = frame[frame.stochastic_mean >= threshold]
    return float(crossed.iloc[0].timesteps) if not crossed.empty else np.nan


def summarize(frames, seeds, budget):
    rows = []
    per_seed = []
    success = frames["success_rate"]
    for condition in CONDITIONS:
        for seed in seeds:
            success_seed = success[
                (success.condition == condition) & (success.seed == seed)
            ].sort_values("timesteps")
            steps = success_seed.timesteps.to_numpy(dtype=float)
            values = success_seed.stochastic_mean.to_numpy(dtype=float)
            row = {
                "condition": condition,
                "seed": seed,
                "normalized_auc": float(np.trapezoid(values, steps) / budget),
                "first_50_steps": first_crossing(success_seed, 0.5),
                "first_90_steps": first_crossing(success_seed, 0.9),
                "t0_success": float(success_seed.iloc[0].stochastic_mean),
                "final_success": float(success_seed.iloc[-1].stochastic_mean),
            }
            for metric in ("native_return", "episode_length"):
                metric_seed = frames[metric][
                    (frames[metric].condition == condition)
                    & (frames[metric].seed == seed)
                ].sort_values("timesteps")
                row[f"final_{metric}"] = float(metric_seed.iloc[-1].stochastic_mean)
            per_seed.append(row)

    per_seed_frame = pd.DataFrame(per_seed)
    for condition in CONDITIONS:
        group = per_seed_frame[per_seed_frame.condition == condition]
        row = {"condition": condition, "n_seeds": len(group)}
        for column in (
            "normalized_auc", "t0_success", "final_success",
            "final_native_return", "final_episode_length",
        ):
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_std"] = float(group[column].std(ddof=0))
        for name in ("first_50_steps", "first_90_steps"):
            finite = group[name].dropna()
            row[f"{name}_mean"] = float(finite.mean()) if len(finite) else np.nan
            row[f"{name}_std"] = float(finite.std(ddof=0)) if len(finite) else np.nan
            row[f"{name}_reached"] = int(len(finite))
        rows.append(row)
    return pd.DataFrame(rows), per_seed_frame


def paired_row(per_seed, left, right, label):
    left_values = per_seed[per_seed.condition == left].set_index("seed")["normalized_auc"]
    right_values = per_seed[per_seed.condition == right].set_index("seed")["normalized_auc"]
    delta = left_values - right_values
    mean_delta = float(delta.mean())
    paired_se = float(delta.std(ddof=1) / np.sqrt(len(delta)))
    t_value = mean_delta / paired_se if paired_se else np.nan
    same_sign = int((delta > 0).sum() if mean_delta > 0 else (delta < 0).sum())
    row = {
        "comparison": label,
        "mean_auc_delta": mean_delta,
        "paired_se": paired_se,
        "t_value": t_value,
        "same_sign_seeds": same_sign,
        "n_seeds": len(delta),
    }
    delta_rows = [
        {"comparison": label, "seed": int(seed), "auc_delta": float(value)}
        for seed, value in delta.items()
    ]
    return row, delta_rows


def paired_comparisons(per_seed):
    rows = []
    delta_rows = []
    for condition in ("potential_geo", "potential_bfs", "intention_naive"):
        row, deltas = paired_row(
            per_seed, condition, "sparse", f"{condition} - sparse"
        )
        rows.append(row)
        delta_rows.extend(deltas)
    row, deltas = paired_row(
        per_seed, "potential_bfs", "potential_geo", "potential_bfs - potential_geo"
    )
    rows.append(row)
    delta_rows.extend(deltas)
    return pd.DataFrame(rows), pd.DataFrame(delta_rows)


def calibration_summary(log_dir):
    rows = []
    for seed in (42, 43):
        path = evaluation_dir(log_dir, "sparse", seed) / "success_rate.csv"
        frame = pd.read_csv(path).sort_values("timesteps")
        rows.append({
            "seed": seed,
            "t0_success": float(frame.iloc[0].stochastic_mean),
            "max_success": float(frame.stochastic_mean.max()),
            "max_success_step": int(frame.loc[frame.stochastic_mean.idxmax(), "timesteps"]),
            "first_50_steps": first_crossing(frame, 0.5),
            "first_90_steps": first_crossing(frame, 0.9),
            "final_step": int(frame.iloc[-1].timesteps),
            "final_success": float(frame.iloc[-1].stochastic_mean),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="task05_logs")
    parser.add_argument("--calibration-log-dir", default="task05_calibration_logs")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 50)))
    parser.add_argument("--budget", type=int, default=500_000)
    parser.add_argument("--output-dir", default="task05_results")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = {
        metric: load_metric(args.log_dir, args.seeds, metric, args.budget)
        for metric in METRICS
    }
    for metric, frame in frames.items():
        plot_metric(
            frame, metric, output_dir / f"{metric}.png", len(args.seeds)
        )
    summary, per_seed = summarize(frames, args.seeds, args.budget)
    paired, paired_by_seed = paired_comparisons(per_seed)
    calibration = calibration_summary(args.calibration_log_dir)
    summary.to_csv(output_dir / "experiment_summary.csv", index=False)
    per_seed.to_csv(output_dir / "per_seed_metrics.csv", index=False)
    paired.to_csv(output_dir / "paired_auc.csv", index=False)
    paired_by_seed.to_csv(output_dir / "paired_auc_by_seed.csv", index=False)
    calibration.to_csv(output_dir / "calibration_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(paired.to_string(index=False))
    print(calibration.to_string(index=False))


if __name__ == "__main__":
    main()
