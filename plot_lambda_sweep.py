"""Dose-response: how the shaping coefficient controls the harm on FourRooms.

Two endpoints come free from task-05: lambda=0 is the `sparse` baseline and
lambda=1 is `potential_geo` at its default coefficient.  The remaining values were
run separately into `lambda_logs/`.

Every point is the paired per-seed difference from the baseline, so seed-level
luck cancels.  With n=8 we report the mean, its standard error, and how many seeds
share the sign -- no significance claims.
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

BUDGET = 500_000
SEEDS = tuple(range(42, 50))
OUT_DIR = Path("lambda_results")

# lambda -> (log root, directory pattern).  0 and 1 reuse the task-05 runs.
POINTS = {
    0.0: ("task05_logs", "fourrooms_sparse_seed{s}"),
    0.1: ("lambda_logs", "fourrooms_potential_geo_lam0p1_seed{s}"),
    0.25: ("lambda_logs", "fourrooms_potential_geo_lam0p25_seed{s}"),
    0.5: ("lambda_logs", "fourrooms_potential_geo_lam0p5_seed{s}"),
    1.0: ("task05_logs", "fourrooms_potential_geo_seed{s}"),
    2.0: ("lambda_logs", "fourrooms_potential_geo_lam2_seed{s}"),
}


def auc(log_root, pattern, seed):
    path = Path(log_root) / pattern.format(s=seed) / "evaluation" / "success_rate.csv"
    steps, values = [], []
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            step = float(row["timesteps"])
            if step <= BUDGET:
                steps.append(step)
                values.append(float(row["stochastic_mean"]))
    return float(np.trapezoid(values, steps) / BUDGET)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline = np.array([auc(*POINTS[0.0], s) for s in SEEDS])

    lambdas, deltas, errors, signs, aucs = [], [], [], [], []
    rows = ["lambda,auc_mean,auc_std,delta_auc,se,t,same_sign_seeds,n"]
    print(f"{'λ':>6}{'AUC':>10}{'ΔAUC':>11}{'SE':>9}{'t':>8}{'same sign':>12}")
    print("-" * 58)
    for lam in sorted(POINTS):
        root, pattern = POINTS[lam]
        missing = [s for s in SEEDS
                   if not (Path(root) / pattern.format(s=s) /
                           "evaluation" / "success_rate.csv").exists()]
        if missing:
            print(f"{lam:>6.2f}   incomplete -- missing seeds {missing}")
            continue
        values = np.array([auc(root, pattern, s) for s in SEEDS])
        diff = values - baseline
        se = diff.std(ddof=1) / np.sqrt(diff.size) if lam else 0.0
        t = diff.mean() / se if se else 0.0
        same = (int((diff < 0).sum()) if diff.mean() < 0
                else int((diff > 0).sum())) if lam else len(SEEDS)
        lambdas.append(lam)
        aucs.append(values.mean())
        deltas.append(diff.mean())
        errors.append(se)
        signs.append(same)
        label = "baseline" if lam == 0 else f"{same}/{len(SEEDS)}"
        print(f"{lam:>6.2f}{values.mean():>10.4f}{diff.mean():>+11.4f}"
              f"{se:>9.4f}{t:>+8.2f}{label:>12}")
        rows.append(f"{lam},{values.mean():.6f},{values.std():.6f},{diff.mean():.6f},"
                    f"{se:.6f},{t:.4f},{same},{len(SEEDS)}")
    (OUT_DIR / "lambda_sweep.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    lambdas = np.asarray(lambdas)
    deltas = np.asarray(deltas)
    errors = np.asarray(errors)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.axhline(0, color="#333333", linewidth=1.0, zorder=1)
    # Shade the region where the harm is not separable from zero at n=8.
    safe = lambdas[np.abs(deltas) <= 2 * np.maximum(errors, 1e-9)]
    if safe.size:
        ax.axvspan(-0.05, safe.max() + 0.08, color="#4a7fb5", alpha=0.07, zorder=0)
        ax.text(safe.max() / 2 if safe.max() else 0.1,
                min(deltas) * 0.92,
                "harm not separable\nfrom zero", ha="center", va="center",
                fontsize=8.5, color="#3a6a9a")
    ax.errorbar(lambdas, deltas, yerr=errors, marker="o", markersize=6.5,
                capsize=4, linewidth=1.9, color="#c0504d",
                markerfacecolor="white", markeredgewidth=1.8, zorder=3)
    for lam, d, e, n in zip(lambdas, deltas, errors, signs):
        if lam == 0:
            continue
        ax.annotate(f"{n}/8", (lam, d - e), textcoords="offset points",
                    xytext=(0, -13), ha="center", fontsize=8.5, color="#444444")
    ax.set_xlabel("Shaping coefficient  $\\lambda$", fontsize=11)
    ax.set_ylabel("$\\Delta$AUC vs sparse baseline\n(paired by seed)", fontsize=10.5)
    ax.set_xticks(lambdas)
    ax.set_xticklabels([f"{v:g}" for v in lambdas], fontsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)
    ax.set_xlim(-0.08, lambdas.max() + 0.12)
    ax.set_title("MiniGrid-FourRooms: the harm is controlled by the shaping magnitude\n"
                 "($\\lambda=0$ is the baseline by definition; annotations give seeds "
                 "sharing the sign)", fontsize=10.5, pad=9)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"lambda_sweep.{extension}", dpi=200, bbox_inches="tight")
    print(f"\nfigure -> {OUT_DIR / 'lambda_sweep.png'}")
    print(f"table  -> {OUT_DIR / 'lambda_sweep.csv'}")


if __name__ == "__main__":
    main()
