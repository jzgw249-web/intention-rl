"""Cross-environment summary: a correctly shaped potential never accelerates.

Plots the paired per-seed effect of the distance potential relative to the sparse
baseline in all three environments, annotated with how often the goal is visible in
the agent's observation.  Harm appears only in the environment where the potential's
arguments are almost never observable.

Note the potential is named `potential_dist` in the Empty-Random-6x6 round and
`potential_geo` afterwards; it is the same wrapper.
"""

import csv
import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

SEEDS = tuple(range(42, 50))
OUT_DIR = Path("task07_results")

# (label, log dir, baseline dir pattern, shaped dir pattern, budget, goal visibility)
ENVIRONMENTS = [
    ("Empty-Random-6x6\n(random start)", "task04_logs",
     "empty-random-6x6_sparse_seed{s}", "empty-random-6x6_potential_dist_seed{s}",
     40_000, 56.2),
    ("Empty-8x8\n(fixed start and goal)", "task07_logs",
     "empty-8x8_sparse_seed{s}", "empty-8x8_potential_geo_seed{s}",
     100_000, 27.0),
    ("FourRooms\n(random start, goal, layout)", "task05_logs",
     "fourrooms_sparse_seed{s}", "fourrooms_potential_geo_seed{s}",
     500_000, 9.4),
]


def auc(log_dir, pattern, seed, budget):
    path = Path(log_dir) / pattern.format(s=seed) / "evaluation" / "success_rate.csv"
    steps, values = [], []
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            step = float(row["timesteps"])
            if step <= budget:
                steps.append(step)
                values.append(float(row["stochastic_mean"]))
    return float(np.trapezoid(values, steps) / budget)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    labels, means, errors, visibility, same_sign = [], [], [], [], []
    print(f"{'environment':<34}{'ΔAUC':>10}{'SE':>9}{'t':>8}{'same sign':>12}")
    print("-" * 73)
    for label, log_dir, base_pat, shaped_pat, budget, vis in ENVIRONMENTS:
        base = np.array([auc(log_dir, base_pat, s, budget) for s in SEEDS])
        shaped = np.array([auc(log_dir, shaped_pat, s, budget) for s in SEEDS])
        diff = shaped - base
        se = diff.std(ddof=1) / np.sqrt(diff.size)
        n_same = int((diff > 0).sum()) if diff.mean() > 0 else int((diff < 0).sum())
        labels.append(label)
        means.append(diff.mean())
        errors.append(se)
        visibility.append(vis)
        same_sign.append(n_same)
        flat = label.replace("\n", " ")
        print(f"{flat:<34}{diff.mean():>+10.4f}{se:>9.4f}"
              f"{diff.mean()/se:>+8.2f}{n_same:>9}/{len(SEEDS)}")

    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    colours = ["#4a7fb5" if m >= 0 else "#c0504d" for m in means]
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=errors, capsize=5, color=colours, width=0.55,
           edgecolor="#333333", linewidth=0.7, error_kw={"elinewidth": 1.1})
    ax.axhline(0, color="#333333", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("$\\Delta$AUC vs sparse baseline\n(paired by seed)", fontsize=10)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)

    # Value labels always sit above the bar, so a negative bar's label does not
    # collide with the visibility annotation that runs along the bottom.
    span = max(means) - min(means)
    top = max(means) + 0.34 * span
    bottom = min(means) - 0.34 * span
    for xi, (m, e, vis, n) in enumerate(zip(means, errors, visibility, same_sign)):
        label_y = (m + e + 0.05 * span) if m >= 0 else 0.05 * span
        ax.text(xi, label_y, f"{m:+.4f}\n{n}/8 seeds",
                ha="center", va="bottom", fontsize=9)
        ax.text(xi, bottom + 0.03 * span, f"goal visible\n{vis:.1f}% of steps",
                ha="center", va="bottom", fontsize=8.5, color="#555555")
    ax.set_ylim(bottom, top)
    ax.set_title("A correctly shaped potential never accelerates learning;\n"
                 "it only hurts where its arguments are unobservable",
                 fontsize=10.5, pad=8)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"environment_summary.{extension}",
                    dpi=200, bbox_inches="tight")
    print(f"\nfigure -> {OUT_DIR / 'environment_summary.png'}")


if __name__ == "__main__":
    main()
