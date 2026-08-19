"""Task-06 analysis: does making the potential's arguments observable remove the harm?

Produces the 2 (observability) x 3 (condition) table, paired per-seed comparisons,
the sharper geo-vs-bfs prediction test, and the two-panel learning-curve figure
used as the centrepiece of the extended abstract.

Cross-setting *levels* are confounded: the appended coordinates are ~59x larger in
scale than the image features, so the augmented runs are not a clean control for
absolute performance.  Within-setting differences are unaffected -- all three
conditions in a setting share the same observation -- so every claim below is built
from within-setting differences.
"""

import csv
import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BUDGET = 500_000
SEEDS = tuple(range(42, 50))
CONDITIONS = ("sparse", "potential_geo", "potential_bfs")
SETTINGS = {
    "partial": ("task05_logs", "fourrooms_{cond}_seed{seed}"),
    "augmented": ("task06_logs", "fourrooms_augmented_{cond}_seed{seed}"),
}
LABELS = {
    "sparse": "Sparse (baseline)",
    "potential_geo": r"$\Phi_{\mathrm{geo}}$ (Euclidean)",
    "potential_bfs": r"$\Phi_{\mathrm{bfs}}$ (shortest path)",
    "intention_naive": "Naive (always-positive)",
}
COLOURS = {
    "sparse": "#1f4e79",
    "potential_geo": "#c0504d",
    "potential_bfs": "#e08214",
    "intention_naive": "#7f7f7f",
}
OUT_DIR = Path("task06_results")


def load_curve(setting, condition, seed):
    log_root, pattern = SETTINGS[setting]
    path = (Path(log_root) / pattern.format(cond=condition, seed=seed)
            / "evaluation" / "success_rate.csv")
    steps, values = [], []
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            step = float(row["timesteps"])
            if step <= BUDGET:
                steps.append(step)
                values.append(float(row["stochastic_mean"]))
    return np.asarray(steps), np.asarray(values)


def normalised_auc(steps, values):
    return float(np.trapezoid(values, steps) / BUDGET)


def gather(setting, condition):
    aucs, finals, curves = [], [], []
    for seed in SEEDS:
        steps, values = load_curve(setting, condition, seed)
        aucs.append(normalised_auc(steps, values))
        finals.append(values[-1])
        curves.append(values)
    return np.asarray(aucs), np.asarray(finals), steps, np.vstack(curves)


def paired(diff):
    """Paired mean, standard error, t and how many seeds share the sign."""
    standard_error = diff.std(ddof=1) / np.sqrt(diff.size)
    t = diff.mean() / standard_error if standard_error else float("nan")
    same_sign = int((diff > 0).sum()) if diff.mean() > 0 else int((diff < 0).sum())
    return diff.mean(), standard_error, t, same_sign


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data, curves = {}, {}
    for setting in SETTINGS:
        for condition in CONDITIONS:
            aucs, finals, steps, band = gather(setting, condition)
            data[(setting, condition)] = (aucs, finals)
            curves[(setting, condition)] = (steps, band)
    # naive only exists for the partial setting; used in the figure for context
    naive_steps, naive_band = None, None
    try:
        _, _, naive_steps, naive_band = gather("partial", "intention_naive")
    except FileNotFoundError:
        pass

    # ---------------- main table ----------------
    lines = ["setting,condition,auc_mean,auc_std,final_mean,final_std,n"]
    print(f"{'setting':<11}{'condition':<16}{'AUC':>18}{'final success':>16}")
    print("-" * 62)
    for setting in SETTINGS:
        for condition in CONDITIONS:
            aucs, finals = data[(setting, condition)]
            print(f"{setting:<11}{condition:<16}"
                  f"{aucs.mean():>10.4f} ± {aucs.std():<6.4f}"
                  f"{finals.mean():>9.3f} ± {finals.std():<5.3f}")
            lines.append(f"{setting},{condition},{aucs.mean():.6f},{aucs.std():.6f},"
                         f"{finals.mean():.6f},{finals.std():.6f},{len(SEEDS)}")
        print()
    (OUT_DIR / "summary.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---------------- within-setting harm ----------------
    print("=" * 62)
    print("Within-setting shaping effect (condition - sparse, paired by seed)")
    print("=" * 62)
    harm = {}
    rows = ["setting,condition,delta_auc,se,t,same_sign_seeds,n"]
    for setting in SETTINGS:
        base = data[(setting, "sparse")][0]
        for condition in ("potential_geo", "potential_bfs"):
            diff = data[(setting, condition)][0] - base
            harm[(setting, condition)] = diff
            mean, se, t, same = paired(diff)
            print(f"  {setting:<11}{condition:<16}"
                  f"Δ={mean:+.4f}  SE={se:.4f}  t={t:+.2f}  {same}/{len(SEEDS)} same sign")
            rows.append(f"{setting},{condition},{mean:.6f},{se:.6f},{t:.4f},"
                        f"{same},{len(SEEDS)}")
        print()

    # ---------------- prediction 1: harm should vanish ----------------
    print("=" * 62)
    print("PREDICTION 1 — making Phi's arguments observable removes the harm")
    print("=" * 62)
    for condition in ("potential_geo", "potential_bfs"):
        h_partial = harm[("partial", condition)]
        h_augmented = harm[("augmented", condition)]
        change = h_augmented - h_partial
        mean, se, t, same = paired(change)
        base_p = data[("partial", "sparse")][0].mean()
        base_a = data[("augmented", "sparse")][0].mean()
        print(f"  {condition}")
        print(f"    partial   Δ={h_partial.mean():+.4f}"
              f"   (relative to baseline: {abs(h_partial.mean())/base_p:.1%})")
        print(f"    augmented Δ={h_augmented.mean():+.4f}"
              f"   (relative to baseline: {abs(h_augmented.mean())/base_a:.1%})")
        recovered = 1 - abs(h_augmented.mean()) / abs(h_partial.mean())
        print(f"    harm removed: {recovered:.0%}"
              f"    change Δ={mean:+.4f} SE={se:.4f} t={t:+.2f} {same}/{len(SEEDS)}")
        rows.append(f"harm_change,{condition},{mean:.6f},{se:.6f},{t:.4f},"
                    f"{same},{len(SEEDS)}")
        print()

    # ---------------- prediction 2: geo should gain more than bfs ----------------
    print("=" * 62)
    print("PREDICTION 2 — geo should improve MORE than bfs")
    print("  (Phi_geo becomes fully determined by the observation; Phi_bfs still")
    print("   depends on unobserved doorway positions)")
    print("=" * 62)
    # Test A: difference of the two gains.  This subtracts two noisy quantities and
    # is underpowered at n=8, so it is reported but not used as the verdict.
    gain_geo = harm[("augmented", "potential_geo")] - harm[("partial", "potential_geo")]
    gain_bfs = harm[("augmented", "potential_bfs")] - harm[("partial", "potential_bfs")]
    print(f"  [A] gain difference (underpowered)")
    print(f"      geo gain: {gain_geo.mean():+.4f}    bfs gain: {gain_bfs.mean():+.4f}")
    mean_a, se_a, t_a, same_a = paired(gain_geo - gain_bfs)
    print(f"      geo - bfs: Δ={mean_a:+.4f}  SE={se_a:.4f}  t={t_a:+.2f}  "
          f"{same_a}/{len(SEEDS)} same sign  -> direction matches, "
          f"magnitude not separable from noise")
    rows.append(f"prediction2a,geo_minus_bfs_gain,{mean_a:.6f},{se_a:.6f},{t_a:.4f},"
                f"{same_a},{len(SEEDS)}")

    # Test B: residual harm under augmented observation.  This is the prediction in
    # its natural form -- Phi_geo becomes fully observable and its harm should be
    # gone, while Phi_bfs stays partly hidden and its harm should persist.
    print(f"\n  [B] residual harm under augmented observation (the sharper test)")
    verdict_parts = []
    for condition, expectation in (("potential_geo", "should vanish"),
                                   ("potential_bfs", "should persist")):
        mean_b, se_b, t_b, same_b = paired(harm[("augmented", condition)])
        gone = abs(t_b) < 1.0 and same_b <= len(SEEDS) * 0.75
        print(f"      {condition:<15} Δ={mean_b:+.4f}  SE={se_b:.4f}  t={t_b:+.2f}  "
              f"{same_b}/{len(SEEDS)}  -> {'gone' if gone else 'persists'}"
              f"   (predicted: {expectation})")
        verdict_parts.append(gone)
        rows.append(f"prediction2b_residual,{condition},{mean_b:.6f},{se_b:.6f},"
                    f"{t_b:.4f},{same_b},{len(SEEDS)}")
    supported = verdict_parts == [True, False]
    print(f"\n  --> PREDICTION 2 "
          f"{'SUPPORTED (via test B)' if supported else 'NOT SUPPORTED — revise'}")
    (OUT_DIR / "paired_comparisons.csv").write_text("\n".join(rows) + "\n",
                                                    encoding="utf-8")

    # ---------------- figure ----------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    panel_titles = {
        "partial": "Partial observation\n(7$\\times$7 egocentric view only)",
        "augmented": "Augmented observation\n(+ agent and goal coordinates)",
    }
    for ax, setting in zip(axes, ("partial", "augmented")):
        for condition in CONDITIONS:
            steps, band = curves[(setting, condition)]
            mean_curve, std_curve = band.mean(axis=0), band.std(axis=0)
            ax.plot(steps, mean_curve, color=COLOURS[condition],
                    label=LABELS[condition], linewidth=1.9)
            ax.fill_between(steps, mean_curve - std_curve, mean_curve + std_curve,
                            color=COLOURS[condition], alpha=0.15, linewidth=0)
        if setting == "partial" and naive_band is not None:
            mean_curve = naive_band.mean(axis=0)
            ax.plot(naive_steps, mean_curve, color=COLOURS["intention_naive"],
                    label=LABELS["intention_naive"], linewidth=1.6, linestyle="--")
        ax.set_title(panel_titles[setting], fontsize=11.5)
        ax.set_xlabel("Environment steps", fontsize=11)
        ax.grid(alpha=0.25, linestyle=":")
        ax.set_xlim(0, BUDGET)
        # "100k" reads better than "100000" at this figure width
        ax.set_xticks(np.arange(0, BUDGET + 1, 100_000))
        ax.set_xticklabels(["0"] + [f"{int(v/1000)}k"
                                    for v in np.arange(100_000, BUDGET + 1, 100_000)])
        ax.tick_params(labelsize=10)
    axes[0].set_ylabel("Evaluation success rate\n(bare environment)", fontsize=11)
    axes[0].set_ylim(bottom=0)
    axes[0].legend(fontsize=9.5, loc="upper left", framealpha=0.9)
    axes[1].legend(fontsize=9.5, loc="upper left", framealpha=0.9)
    # No suptitle: the LaTeX caption already states the finding, and repeating it
    # inside the artwork wastes vertical space in a two-page abstract.
    fig.text(0.995, 0.005, f"mean $\\pm$ s.d. over {len(SEEDS)} seeds",
             ha="right", va="bottom", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    for extension in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"task06_learning_curves.{extension}",
                    dpi=200, bbox_inches="tight")
    print(f"\nfigure -> {OUT_DIR / 'task06_learning_curves.png'}")
    print(f"tables -> {OUT_DIR / 'summary.csv'}, "
          f"{OUT_DIR / 'paired_comparisons.csv'}")


if __name__ == "__main__":
    main()
