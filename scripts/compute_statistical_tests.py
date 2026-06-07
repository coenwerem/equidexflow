#!/usr/bin/env python3
"""Compute BCa bootstrap CIs, Wilcoxon signed-rank tests, and Cohen's d_z
for the EquiDexFlow 4-variant ablation on 81 paired test objects.

Outputs:
  outputs/paper_results/equidex/statistical_analysis.yaml
  outputs/paper_results/equidex/statistical_analysis_latex.tex  (ready-to-paste table)

Usage:
  ~/ResearchProjects/MuJoCoDex/.venv/bin/python \
    ~/ResearchProjects/frogger/scripts/equidex/compute_statistical_tests.py
"""
from __future__ import annotations

import csv
import itertools
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import bootstrap, wilcoxon

RESULTS_DIR = Path.home() / "ResearchProjects/frogger/outputs/paper_results/equidex/equidex_results"
OUT_DIR = RESULTS_DIR.parent
N_RESAMPLES = 10000
ALPHA = 0.05

VARIANTS = {
    "Full":        "equidexflow_leap_full_fc_reach_flow",
    "GeomOnly":    "equidexflow_leap_geom_only_fc_reach_flow",
    "PoseOnly":    "equidexflow_leap_pose_only_fc_reach_flow",
    "ContactOnly": "equidexflow_leap_contact_only_fc_reach_flow",
}


def load_rollout(variant_dir: str) -> dict[str, np.ndarray]:
    path = RESULTS_DIR / variant_dir / "rollout.csv"
    top1, top3 = [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            top1.append(float(row["top1_score"]))
            top3.append(float(row["top3_score"]))
    return {"top1": np.array(top1), "top3": np.array(top3)}


def load_forces(variant_dir: str) -> dict[str, float]:
    path = RESULTS_DIR / variant_dir / "forces.csv"
    out = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["finger"] == "friction_violation_rate":
                out["friction_viol"] = float(row["mag_error_N"])
            elif row["finger"] == "wrench_balance_residual_Nm":
                out["wrench_res"] = float(row["mag_error_N"])
    return out


def load_contacts(variant_dir: str) -> float:
    path = RESULTS_DIR / variant_dir / "contacts.csv"
    means = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            means.append(float(row["mean_error_m"]))
    return float(np.mean(means))


def bca_ci(data: np.ndarray, alpha: float = ALPHA) -> tuple[float, float, float]:
    """BCa bootstrap 95% CI on the mean. Returns (mean, ci_low, ci_high)."""
    mean_val = float(data.mean())
    res = bootstrap(
        (data,),
        statistic=lambda x, axis: np.mean(x, axis=axis),
        n_resamples=N_RESAMPLES,
        method="BCa",
        confidence_level=1 - alpha,
        random_state=42,
    )
    return mean_val, float(res.confidence_interval.low), float(res.confidence_interval.high)


def cohens_dz(x: np.ndarray, y: np.ndarray) -> float:
    """Paired-sample Cohen's d_z = mean(diff) / std(diff)."""
    diff = x - y
    return float(diff.mean() / diff.std(ddof=1))


def holm_correct(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni correction for multiple comparisons."""
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    corrected = [0.0] * n
    cummax = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adjusted = p * (n - rank)
        cummax = max(cummax, adjusted)
        corrected[orig_idx] = min(cummax, 1.0)
    return corrected


def main():
    print("=== Loading per-object data ===")
    data = {}
    for name, vdir in VARIANTS.items():
        rollout = load_rollout(vdir)
        forces = load_forces(vdir)
        contact_err = load_contacts(vdir)
        data[name] = {
            "top1": rollout["top1"],
            "top3": rollout["top3"],
            "contact_err": contact_err,
            "wrench_res": forces["wrench_res"],
            "friction_viol": forces["friction_viol"],
        }
        print(f"  {name}: {len(rollout['top1'])} objects, "
              f"Top-1 mean={rollout['top1'].mean():.3f}")

    # --- BCa Bootstrap CIs ---
    print("\n=== BCa Bootstrap 95% CIs (n_resamples={}) ===".format(N_RESAMPLES))
    ci_results = {}
    for name in VARIANTS:
        ci_results[name] = {}
        for metric in ["top1", "top3"]:
            mean, lo, hi = bca_ci(data[name][metric])
            ci_results[name][metric] = {
                "mean": round(mean, 4),
                "ci_low": round(lo, 4),
                "ci_high": round(hi, 4),
                "ci_str": f"{mean:.3f} [{lo:.3f}, {hi:.3f}]",
            }
            print(f"  {name} {metric}: {mean:.4f} [{lo:.4f}, {hi:.4f}]")

    # --- Pairwise Wilcoxon + Cohen's d_z ---
    print("\n=== Pairwise Wilcoxon Signed-Rank Tests ===")
    variant_names = list(VARIANTS.keys())
    pairs = list(itertools.combinations(variant_names, 2))

    pairwise_results = []
    raw_pvalues = []

    for v1, v2 in pairs:
        for metric in ["top1", "top3"]:
            stat, p = wilcoxon(data[v1][metric], data[v2][metric], alternative="two-sided")
            dz = cohens_dz(data[v1][metric], data[v2][metric])
            raw_pvalues.append(p)
            pairwise_results.append({
                "pair": f"{v1} vs {v2}",
                "metric": metric,
                "wilcoxon_stat": round(float(stat), 2),
                "p_raw": round(float(p), 6),
                "cohens_dz": round(dz, 4),
                "mean_diff": round(float(data[v1][metric].mean() - data[v2][metric].mean()), 4),
            })

    corrected = holm_correct(raw_pvalues)
    for i, r in enumerate(pairwise_results):
        r["p_holm"] = round(corrected[i], 6)
        sig = "***" if corrected[i] < 0.001 else "**" if corrected[i] < 0.01 else "*" if corrected[i] < 0.05 else "ns"
        r["significance"] = sig
        print(f"  {r['pair']:30s} {r['metric']:5s}  "
              f"p_holm={r['p_holm']:.6f} {sig:4s}  "
              f"d_z={r['cohens_dz']:+.3f}  "
              f"delta={r['mean_diff']:+.4f}")

    # --- Summary ---
    output = {
        "method": "BCa bootstrap CIs (R=10000) + Wilcoxon signed-rank (Holm-corrected) + Cohen's d_z",
        "n_objects": 81,
        "pairing": "per-object (same 81 test objects across all variants)",
        "note": "CIs are over the test set, not over training seeds. Multi-seed retraining was infeasible.",
        "bootstrap_cis": ci_results,
        "pairwise_tests": pairwise_results,
    }

    out_yaml = OUT_DIR / "statistical_analysis.yaml"
    with open(out_yaml, "w") as f:
        yaml.dump(output, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print(f"\n  wrote {out_yaml}")

    # --- LaTeX table ---
    tex_lines = [
        r"\begin{table}[ht]",
        r"\caption{Pairwise statistical comparisons on 81 test objects (Wilcoxon signed-rank, Holm-corrected).}",
        r"\label{tab:pairwise_tests}",
        r"\centering\small",
        r"\begin{tabular}{@{}llrrrl@{}}",
        r"\toprule",
        r"Comparison & Metric & $\Delta$ & $d_z$ & $p_{\mathrm{Holm}}$ & Sig. \\",
        r"\midrule",
    ]
    for r in pairwise_results:
        tex_lines.append(
            f"  {r['pair']} & {r['metric']} & "
            f"{r['mean_diff']:+.3f} & {r['cohens_dz']:+.3f} & "
            f"{r['p_holm']:.4f} & {r['significance']} \\\\"
        )
    tex_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    out_tex = OUT_DIR / "statistical_analysis_latex.tex"
    with open(out_tex, "w") as f:
        f.write("\n".join(tex_lines) + "\n")
    print(f"  wrote {out_tex}")

    print("\nDone.")


if __name__ == "__main__":
    main()
