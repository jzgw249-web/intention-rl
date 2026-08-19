"""Regenerate every report statistic that is recoverable from archived evaluation logs.

This is deliberately read-only with respect to logs and existing result directories.
It writes only RESULTS.md and results_all.csv in the repository root.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SEEDS = tuple(range(42, 50))
ROWS: list[dict[str, object]] = []


def add(section: str, metric: str, value: object, *, source: str, status: str = "reproduced", **extra):
    row = {"section": section, "metric": metric, "value": value, "status": status,
           "source": source}
    row.update(extra)
    ROWS.append(row)
    return row


def curve(log_root: str, pattern: str, seed: int, metric: str, budget: int):
    path = ROOT / log_root / pattern.format(seed=seed) / "evaluation" / f"{metric}.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        records = [r for r in csv.DictReader(handle) if float(r["timesteps"]) <= budget]
    if not records:
        raise ValueError(f"no records at or below budget in {path}")
    steps = np.asarray([float(r["timesteps"]) for r in records])
    values = np.asarray([float(r["stochastic_mean"]) for r in records])
    return steps, values


def aucs(log_root: str, pattern: str, budget: int):
    result = []
    for seed in SEEDS:
        steps, values = curve(log_root, pattern, seed, "success_rate", budget)
        result.append(float(np.trapezoid(values, steps) / budget))
    return np.asarray(result)


def paired(diff: np.ndarray):
    mean = float(diff.mean())
    se = float(diff.std(ddof=1) / math.sqrt(diff.size))
    t = mean / se if se else 0.0
    same = int((diff > 0).sum()) if mean > 0 else int((diff < 0).sum())
    return mean, se, t, same


def add_comparison(section: str, metric: str, left: np.ndarray, right: np.ndarray, source: str):
    mean, se, t, same = paired(left - right)
    return add(section, metric, mean, source=source, se=se, t=t,
               same_sign_seeds=same, n=len(SEEDS))


def add_auc(section: str, metric: str, values: np.ndarray, source: str):
    return add(section, metric, float(values.mean()), source=source,
               std=float(values.std(ddof=0)), n=len(values))


def failure_mode_i():
    cos = max(abs(5 / math.sqrt(50)), abs(5 / math.sqrt(50)))
    prediction = 0.5 * (cos + 1) / 2 * 256
    add("failure_mode_i", "closed_form_naive_return", prediction,
        source="analytic: 0.5*(max discrete cosine+1)/2*256", cosine=cos)

    specs = (
        ("Empty-8x8", "main_logs", "empty-8x8_intention_naive_seed{seed}", 100_000),
        ("Empty-Random-6x6", "task04_logs", "empty-random-6x6_intention_naive_seed{seed}", 40_000),
        ("FourRooms", "task05_logs", "fourrooms_intention_naive_seed{seed}", 500_000),
    )
    for env, root, pattern, budget in specs:
        src = f"{root}/{pattern}/evaluation"
        for metric in ("success_rate", "episode_length"):
            finals = np.asarray([curve(root, pattern, s, metric, budget)[1][-1] for s in SEEDS])
            add("failure_mode_i", f"{env}.{metric}.final", float(finals.mean()),
                source=src, std=float(finals.std(ddof=0)), n=len(SEEDS))


def failure_mode_ii():
    root, budget = "task07_logs", 100_000
    patterns = {
        "sparse": "empty-8x8_sparse_seed{seed}",
        "potential_geo": "empty-8x8_potential_geo_seed{seed}",
        "intention_pb_shifted": "empty-8x8_intention_pb_shifted_seed{seed}",
        "intention_pb": "empty-8x8_intention_pb_seed{seed}",
    }
    data = {name: aucs(root, pat, budget) for name, pat in patterns.items()}
    for name, values in data.items():
        add_auc("failure_mode_ii.empty8", f"{name}.auc", values,
                f"{root}/{patterns[name]}/evaluation/success_rate.csv")
        if name != "sparse":
            add_comparison("failure_mode_ii.empty8", f"{name}-sparse", values,
                           data["sparse"], f"{root}/.../success_rate.csv")

    raw = aucs("task04_logs", "empty-random-6x6_intention_pb_seed{seed}", 40_000)
    shifted = aucs("task04_logs", "empty-random-6x6_intention_pb_shifted_seed{seed}", 40_000)
    sparse = aucs("task04_logs", "empty-random-6x6_sparse_seed{seed}", 40_000)
    row = add_comparison("failure_mode_ii.empty_random6", "shifted-raw", shifted, raw,
                         "task04_logs/.../evaluation/success_rate.csv")
    row["harm_recovered_fraction"] = float((shifted - raw).mean() / (sparse - raw).mean())


def main_results():
    specs = (
        ("Empty-8x8", "task07_logs", "empty-8x8_sparse_seed{seed}",
         "empty-8x8_potential_geo_seed{seed}", 100_000),
        ("Empty-Random-6x6", "task04_logs", "empty-random-6x6_sparse_seed{seed}",
         "empty-random-6x6_potential_dist_seed{seed}", 40_000),
        ("FourRooms", "task05_logs", "fourrooms_sparse_seed{seed}",
         "fourrooms_potential_geo_seed{seed}", 500_000),
    )
    for env, root, sparse_pat, geo_pat, budget in specs:
        add_comparison("main_result", f"{env}.potential_geo-sparse",
                       aucs(root, geo_pat, budget), aucs(root, sparse_pat, budget),
                       f"{root}/.../evaluation/success_rate.csv")


def causal_test():
    settings = {
        "partial": ("task05_logs", "fourrooms_{condition}_seed{seed}"),
        "augmented": ("task06_logs", "fourrooms_augmented_{condition}_seed{seed}"),
    }
    harms = {}
    for setting, (root, pattern) in settings.items():
        sparse = aucs(root, pattern.format(condition="sparse", seed="{seed}"), 500_000)
        for condition in ("potential_geo", "potential_bfs"):
            shaped = aucs(root, pattern.format(condition=condition, seed="{seed}"), 500_000)
            harms[(setting, condition)] = shaped - sparse
            add_comparison("causal_test", f"{setting}.{condition}-sparse", shaped, sparse,
                           f"{root}/.../evaluation/success_rate.csv")
    for condition in ("potential_geo", "potential_bfs"):
        partial, augmented = harms[("partial", condition)], harms[("augmented", condition)]
        row = add_comparison("causal_test", f"{condition}.harm_change", augmented, partial,
                             "task05_logs + task06_logs")
        row["harm_removed_fraction"] = float(1 - abs(augmented.mean()) / abs(partial.mean()))


def lambda_sweep():
    points = {
        0.0: ("task05_logs", "fourrooms_sparse_seed{seed}"),
        0.1: ("lambda_logs", "fourrooms_potential_geo_lam0p1_seed{seed}"),
        0.25: ("lambda_logs", "fourrooms_potential_geo_lam0p25_seed{seed}"),
        0.5: ("lambda_logs", "fourrooms_potential_geo_lam0p5_seed{seed}"),
        1.0: ("task05_logs", "fourrooms_potential_geo_seed{seed}"),
        2.0: ("lambda_logs", "fourrooms_potential_geo_lam2_seed{seed}"),
    }
    base = aucs(*points[0.0], 500_000)
    for lam, (root, pattern) in points.items():
        values = aucs(root, pattern, 500_000)
        add_auc("lambda_sweep", f"lambda={lam:g}.auc", values,
                f"{root}/{pattern}/evaluation/success_rate.csv")
        add_comparison("lambda_sweep", f"lambda={lam:g}.delta", values, base,
                       f"{root}/... vs task05 sparse")


def auxiliary():
    stats_path = ROOT / "analysis" / "observability_stats.csv"
    if not stats_path.exists():
        raise FileNotFoundError(
            f"{stats_path} is missing; run analysis/observability_stats.py first"
        )
    with stats_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            published = row["published_value"]
            # Empty-8x8 random success is a new supplemental result and has no
            # published comparator; all eight previously published rows do.
            add("auxiliary", f"{row['metric']}.{row['env']}", float(row["value"]),
                source="analysis/observability_stats.csv", status="reproduced",
                n_episodes=int(row["n_episodes"]),
                n_observations=int(row["n_observations"]),
                published=(float(published) if published else "not_published"),
                abs_diff=(float(row["abs_diff"]) if row["abs_diff"] else "n/a"),
                agrees=(row["agrees"].lower() == "true" if row["agrees"] else "n/a"))

    official_roots = ("main_logs", "task04_logs", "task05_logs", "task06_logs",
                      "task07_logs", "lambda_logs")
    official = sum(len([p for p in (ROOT / d).iterdir() if p.is_dir()]) for d in official_roots)
    all_log_roots = [p for p in ROOT.iterdir() if p.is_dir() and p.name.endswith("logs")]
    all_logged = sum(len([p for p in d.iterdir() if p.is_dir()]) for d in all_log_roots)
    add("auxiliary", "training_runs.official_analysis", official,
        source="run-directory count in six official log roots")
    add("auxiliary", "training_runs.all_log_directories", all_logged,
        source="run-directory count in every repository directory ending in 'logs'")


def document_audit():
    expected = {
        "完整方案说明_定稿.pdf": "generated by build_scheme_pdf.py; PDF text audit is external",
        "extended_abstract_2page.tex": "text inspected",
        "extended_abstract.tex": "text inspected",
        "POSTER_PLAN.md": "text inspected",
    }
    for name, note in expected.items():
        path = ROOT.parent / name
        add("document_audit", name, "present" if path.exists() else "missing",
            source=str(path), note=note)

    # Differences found by checking the PDF's extracted text and the three text sources.
    # These are intentionally reported, not repaired.
    differences = (
        ("closed_form_rounding", "109.26 in all four documents; exact formula gives "
         "109.254834, which rounds to 109.25 at two decimals. 109.26 results from "
         "rounding the per-step value to 0.4268 before multiplying."),
        ("task04_shift_delta", "+0.0674 in the PDF and both abstracts; archived logs "
         "give +0.0684375 (display +0.0684). t=+3.14, 8/8 and 81% still agree."),
        ("empty_random_geo_delta", "+0.0041 in both abstracts; archived logs give "
         "+0.0041667, whose four-decimal display is +0.0042."),
        ("training_count", "task-08 says approximately 235; six formal log roots contain "
         "208 runs and all *logs roots contain 231 run directories."),
    )
    for metric, note in differences:
        add("document_difference", metric, "mismatch", source="four-document audit", note=note)


def fmt(value):
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_outputs():
    columns = ["section", "metric", "value", "status", "source", "std", "se", "t",
               "same_sign_seeds", "n", "cosine", "harm_recovered_fraction",
               "harm_removed_fraction", "note", "n_episodes", "n_observations",
               "published", "abs_diff", "agrees"]
    with (ROOT / "results_all.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ROWS)

    lines = ["# Reproduced results", "", "Generated by `python reproduce_all.py`.", "",
             "AUC is trapezoidal integration of bare-environment evaluation success rate, "
             "divided by the stated training budget. Standard deviations use `ddof=0`; paired "
             "SE and t use seed-level differences with `ddof=1`.", ""]
    current = None
    for row in ROWS:
        if row["section"] != current:
            current = str(row["section"])
            lines.extend([f"## {current}", "",
                          "| metric | value | extra | status | source |",
                          "|---|---:|---|---|---|"])
        extras = []
        for key in ("std", "se", "t", "same_sign_seeds", "n",
                    "harm_recovered_fraction", "harm_removed_fraction", "note",
                    "n_episodes", "n_observations", "published", "abs_diff", "agrees"):
            if key in row:
                extras.append(f"{key}={fmt(row[key])}")
        lines.append(f"| {row['metric']} | {fmt(row['value'])} | {'; '.join(extras)} | "
                     f"{row['status']} | {row['source']} |")
    lines.extend(["", "## Reproducibility gaps and document differences", "",
                  "- The six formal experiment groups contain 208 runs. The approximate total "
                  "of 235 cannot be reproduced literally: all directories ending in `logs` contain "
                  f"{next(r['value'] for r in ROWS if r['metric']=='training_runs.all_log_directories')} "
                  "run directories; additional Gate-1 diagnostic runs live outside those roots.",
                  "- Goal visibility, initial-potential variance, and random-policy success rates "
                  "are regenerated by `analysis/observability_stats.py`; agreement with the "
                  "published values is recorded per row in the auxiliary table.",
                  "- Exact numeric/rounding differences against the four documents are listed in "
                  "the `document_difference` table above. No document was modified.", ""])
    (ROOT / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    failure_mode_i()
    failure_mode_ii()
    main_results()
    causal_test()
    lambda_sweep()
    auxiliary()
    document_audit()
    write_outputs()
    print(f"wrote {ROOT / 'RESULTS.md'}")
    print(f"wrote {ROOT / 'results_all.csv'}")


if __name__ == "__main__":
    main()
