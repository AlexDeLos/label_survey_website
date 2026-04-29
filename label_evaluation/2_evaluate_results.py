"""
2_evaluate_results.py
---------------------
Run this script after collecting survey evaluations to analyse inter-rater
agreement and label accuracy.

Usage
-----
    python 2_evaluate_results.py

Requirements
------------
    pip install pandas matplotlib seaborn scikit-learn scipy

Output
------
    Six interactive matplotlib figures:
      1. Overall accuracy distribution across all labels
      2. Per-label-category accuracy rates (heatmap)
      3. Per-study accuracy summary
      4. Per-sample agreement (how often did all reviewers agree?)
      5. Inter-rater agreement — pairwise Cohen's Kappa per label category
      6. Fleiss' Kappa per label category (multi-rater)
"""

import json
import os
import sys
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import cohen_kappa_score
from scipy.stats import chi2

module_dir = "./"
sys.path.append(module_dir)

from constants import RNA_USED  # noqa: E402

# ══════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════

RESULTS_FILE = f"./label_evaluation/data/evaluation_results_{'RNA' if RNA_USED else 'MA'}.csv"

# Map raw scores to an ordinal integer (used for Kappa calculations)
SCORE_ORDER   = {"Correct": 2, "Mostly Correct": 1, "Incorrect": 0}
SCORE_PALETTE = {"Correct": "#4caf50", "Mostly Correct": "#ff9800", "Incorrect": "#f44336"}

sns.set_theme(style="whitegrid", font_scale=1.1)

# ══════════════════════════════════════════════════════════════════
#  DATA LOADING & PREPARATION
# ══════════════════════════════════════════════════════════════════

def load_and_explode(results_file: str) -> pd.DataFrame:
    """
    Load results CSV and explode the label_scores JSON column so that
    each row represents one (username, sample_id, label_category, score).
    """
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")

    df = pd.read_csv(results_file)
    print(f"Loaded {len(df)} evaluations from {len(df['username'].unique())} reviewer(s) "
          f"covering {len(df['sample_id'].unique())} unique samples.")

    rows = []
    for _, row in df.iterrows():
        try:
            scores = json.loads(row["label_scores"])
        except (json.JSONDecodeError, TypeError):
            continue
        for category, score in scores.items():
            rows.append({
                "username":   row["username"],
                "study_id":   row["study_id"],
                "sample_id":  row["sample_id"],
                "category":   category,
                "score":      score,
                "score_int":  SCORE_ORDER.get(score, -1),
            })

    exploded = pd.DataFrame(rows)
    exploded = exploded[exploded["score_int"] >= 0]   # drop any malformed entries
    return exploded


# ══════════════════════════════════════════════════════════════════
#  FLEISS' KAPPA (manual implementation — no extra dependencies)
# ══════════════════════════════════════════════════════════════════

def fleiss_kappa(ratings_matrix: np.ndarray) -> float:
    """
    Compute Fleiss' Kappa for a ratings matrix of shape (n_subjects, n_categories).
    Each entry [i, j] is the number of raters who assigned category j to subject i.
    Returns kappa (float) or NaN if undefined.
    """
    n, k = ratings_matrix.shape
    n_raters = int(ratings_matrix[0].sum())
    if n_raters < 2 or n == 0:
        return float("nan")

    # Proportion of all assignments to each category
    p_j = ratings_matrix.sum(axis=0) / (n * n_raters)

    # Per-subject agreement
    P_i = ((ratings_matrix ** 2).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    P_bar  = P_i.mean()
    Pe_bar = (p_j ** 2).sum()

    if Pe_bar == 1.0:
        return float("nan")
    return (P_bar - Pe_bar) / (1 - Pe_bar)


def compute_fleiss_kappa_per_category(exploded: pd.DataFrame) -> pd.Series:
    """Return a Series of Fleiss' Kappa keyed by label category."""
    categories   = exploded["category"].unique()
    score_levels = sorted(SCORE_ORDER.values())   # [0, 1, 2]
    kappas = {}

    for cat in categories:
        sub = exploded[exploded["category"] == cat]
        # Only include samples that have been rated by ≥ 2 reviewers
        counts = sub.groupby("sample_id")["username"].count()
        valid_samples = counts[counts >= 2].index
        sub = sub[sub["sample_id"].isin(valid_samples)]

        if sub.empty:
            kappas[cat] = float("nan")
            continue

        # Build ratings matrix: rows = samples, cols = score levels
        pivot = sub.pivot_table(index="sample_id", columns="score_int", aggfunc="size", fill_value=0)
        # Ensure all score columns exist
        for lvl in score_levels:
            if lvl not in pivot.columns:
                pivot[lvl] = 0
        matrix = pivot[score_levels].values.astype(float)
        kappas[cat] = fleiss_kappa(matrix)

    return pd.Series(kappas)


# ══════════════════════════════════════════════════════════════════
#  PLOT HELPERS
# ══════════════════════════════════════════════════════════════════

def _kappa_color(k: float) -> str:
    """Traffic-light colour for a Kappa value."""
    if np.isnan(k):   return "#bdbdbd"
    if k >= 0.61:     return "#4caf50"
    if k >= 0.41:     return "#ff9800"
    return "#f44336"


def _add_kappa_legend(ax):
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4caf50", label="Substantial / Almost perfect (κ ≥ 0.61)"),
        Patch(facecolor="#ff9800", label="Moderate (0.41 – 0.60)"),
        Patch(facecolor="#f44336", label="Fair / Slight / Poor (< 0.41)"),
        Patch(facecolor="#bdbdbd", label="Insufficient data"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)


# ══════════════════════════════════════════════════════════════════
#  FIGURE 1 — Overall accuracy distribution
# ══════════════════════════════════════════════════════════════════

def plot_overall_accuracy(exploded: pd.DataFrame):
    counts = exploded["score"].value_counts().reindex(SCORE_ORDER.keys(), fill_value=0)
    pcts   = counts / counts.sum() * 100

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(pcts.index, pcts.values,
                  color=[SCORE_PALETTE[s] for s in pcts.index], edgecolor="white", width=0.5)
    for bar, pct in zip(bars, pcts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=11)

    ax.set_title("Overall Accuracy Distribution (all labels, all reviewers)", fontsize=13)
    ax.set_ylabel("Percentage of evaluations (%)")
    ax.set_ylim(0, pcts.max() * 1.15)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════
#  FIGURE 2 — Per-label-category accuracy heatmap
# ══════════════════════════════════════════════════════════════════

def plot_per_category_heatmap(exploded: pd.DataFrame):
    cats  = exploded["category"].unique()
    order = list(SCORE_ORDER.keys())   # Correct, Mostly Correct, Incorrect

    matrix = pd.DataFrame(index=cats, columns=order, dtype=float)
    for cat in cats:
        sub   = exploded[exploded["category"] == cat]
        total = len(sub)
        for s in order:
            matrix.loc[cat, s] = (sub["score"] == s).sum() / total * 100 if total else 0

    matrix = matrix.sort_values("Correct", ascending=False)

    fig, ax = plt.subplots(figsize=(9, max(4, len(cats) * 0.45 + 1)))
    sns.heatmap(matrix.astype(float), annot=True, fmt=".1f", cmap="RdYlGn",
                linewidths=0.5, ax=ax, vmin=0, vmax=100,
                cbar_kws={"label": "% of evaluations"})
    ax.set_title("Accuracy Rate (%) per Label Category", fontsize=13)
    ax.set_xlabel("")
    ax.set_ylabel("Label category")
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════
#  FIGURE 3 — Per-study accuracy summary
# ══════════════════════════════════════════════════════════════════

def plot_per_study(exploded: pd.DataFrame):
    order  = list(SCORE_ORDER.keys())
    groups = exploded.groupby(["study_id", "score"]).size().unstack(fill_value=0)
    for s in order:
        if s not in groups.columns:
            groups[s] = 0
    groups = groups[order]
    pcts   = groups.div(groups.sum(axis=1), axis=0) * 100
    pcts   = pcts.sort_values("Correct", ascending=True)

    fig, ax = plt.subplots(figsize=(9, max(4, len(pcts) * 0.5 + 1)))
    left = np.zeros(len(pcts))
    for s in order:
        ax.barh(pcts.index, pcts[s], left=left,
                color=SCORE_PALETTE[s], label=s, edgecolor="white")
        left += pcts[s].values

    ax.set_title("Accuracy Distribution per Study", fontsize=13)
    ax.set_xlabel("Percentage of evaluations (%)")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend(loc="lower right")
    ax.set_xlim(0, 100)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════
#  FIGURE 4 — Per-sample agreement
# ══════════════════════════════════════════════════════════════════

def plot_per_sample_agreement(exploded: pd.DataFrame):
    """
    For each (sample_id, category) pair that has ≥ 2 evaluations,
    compute whether all reviewers agreed (full agreement) or not.
    Summarise as a distribution.
    """
    rows = []
    for (sample_id, cat), sub in exploded.groupby(["sample_id", "category"]):
        if len(sub) < 2:
            continue
        unique_scores = sub["score"].nunique()
        rows.append({
            "sample_id":  sample_id,
            "category":   cat,
            "n_raters":   len(sub),
            "full_agree": unique_scores == 1,
        })

    if not rows:
        print("Not enough multi-rated samples to plot agreement.")
        return None

    agree_df   = pd.DataFrame(rows)
    # Per-sample: fraction of categories where all raters agreed
    sample_agg = agree_df.groupby("sample_id")["full_agree"].mean() * 100

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(sample_agg, bins=20, range=(0, 100),
            color="#5c6bc0", edgecolor="white")
    ax.axvline(sample_agg.mean(), color="#f44336", linestyle="--",
               label=f"Mean: {sample_agg.mean():.1f}%")
    ax.set_title("Per-Sample Full Agreement Rate\n(% of label categories where all reviewers agreed)", fontsize=12)
    ax.set_xlabel("Full agreement rate (%)")
    ax.set_ylabel("Number of samples")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend()
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════
#  FIGURE 5 — Pairwise Cohen's Kappa per category
# ══════════════════════════════════════════════════════════════════

def plot_pairwise_kappa(exploded: pd.DataFrame):
    reviewers = exploded["username"].unique()
    pairs     = list(itertools.combinations(reviewers, 2))
    cats      = sorted(exploded["category"].unique())

    if not pairs:
        print("Need ≥ 2 reviewers to compute pairwise Kappa.")
        return None

    kappa_data = pd.DataFrame(index=cats, columns=[f"{a} vs {b}" for a, b in pairs], dtype=float)

    for cat in cats:
        sub = exploded[exploded["category"] == cat]
        pivot = sub.pivot_table(index="sample_id", columns="username",
                                values="score_int", aggfunc="first")
        for (a, b) in pairs:
            col = f"{a} vs {b}"
            if a not in pivot.columns or b not in pivot.columns:
                kappa_data.loc[cat, col] = np.nan
                continue
            shared = pivot[[a, b]].dropna()
            if len(shared) < 2 or shared[a].nunique() < 2:
                kappa_data.loc[cat, col] = np.nan
                continue
            try:
                kappa_data.loc[cat, col] = cohen_kappa_score(
                    shared[a].astype(int), shared[b].astype(int)
                )
            except Exception:
                kappa_data.loc[cat, col] = np.nan

    kappa_data = kappa_data.astype(float)

    fig, ax = plt.subplots(figsize=(max(6, len(pairs) * 2.5), max(4, len(cats) * 0.45 + 1)))
    colors = [[_kappa_color(v) for v in row] for row in kappa_data.values]
    sns.heatmap(kappa_data, annot=True, fmt=".2f", ax=ax,
                linewidths=0.5, cmap="RdYlGn", vmin=-1, vmax=1,
                cbar_kws={"label": "Cohen's κ"})
    ax.set_title("Pairwise Cohen's Kappa per Label Category", fontsize=13)
    ax.set_xlabel("Reviewer pair")
    ax.set_ylabel("Label category")
    _add_kappa_legend(ax)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════
#  FIGURE 6 — Fleiss' Kappa per category
# ══════════════════════════════════════════════════════════════════

def plot_fleiss_kappa(exploded: pd.DataFrame):
    kappas = compute_fleiss_kappa_per_category(exploded).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(9, max(4, len(kappas) * 0.45 + 1)))
    colors = [_kappa_color(k) for k in kappas.values]
    bars   = ax.barh(kappas.index, kappas.values, color=colors, edgecolor="white")

    for bar, val in zip(bars, kappas.values):
        if not np.isnan(val):
            ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}", va="center", fontsize=9)

    ax.axvline(0,    color="black",   linewidth=0.8, linestyle="--")
    ax.axvline(0.41, color="#ff9800", linewidth=1,   linestyle=":", label="Moderate threshold (0.41)")
    ax.axvline(0.61, color="#4caf50", linewidth=1,   linestyle=":", label="Substantial threshold (0.61)")
    ax.set_title("Fleiss' Kappa per Label Category\n(multi-rater agreement)", fontsize=13)
    ax.set_xlabel("Fleiss' κ")
    ax.set_xlim(-1, 1.15)
    _add_kappa_legend(ax)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════
#  SUMMARY PRINTOUT
# ══════════════════════════════════════════════════════════════════

def print_summary(exploded: pd.DataFrame):
    print("\n" + "═" * 60)
    print("  EVALUATION SUMMARY")
    print("═" * 60)

    total = len(exploded)
    for score in SCORE_ORDER:
        n   = (exploded["score"] == score).sum()
        pct = n / total * 100
        print(f"  {score:<18} {n:>5}  ({pct:.1f}%)")

    print(f"\n  Total label evaluations : {total}")
    print(f"  Unique samples evaluated: {exploded['sample_id'].nunique()}")
    print(f"  Unique label categories : {exploded['category'].nunique()}")
    print(f"  Reviewers               : {', '.join(exploded['username'].unique())}")

    kappas = compute_fleiss_kappa_per_category(exploded).dropna()
    if not kappas.empty:
        print(f"\n  Fleiss' κ  —  mean: {kappas.mean():.3f}  "
              f"min: {kappas.min():.3f}  max: {kappas.max():.3f}")
        worst = kappas.idxmin()
        best  = kappas.idxmax()
        print(f"    Lowest agreement : '{worst}'  (κ = {kappas[worst]:.3f})")
        print(f"    Highest agreement: '{best}'  (κ = {kappas[best]:.3f})")
    print("═" * 60 + "\n")


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    exploded = load_and_explode(RESULTS_FILE)
    print_summary(exploded)

    figs = []

    print("Generating Figure 1: Overall accuracy distribution...")
    figs.append(("Overall Accuracy", plot_overall_accuracy(exploded)))

    print("Generating Figure 2: Per-category accuracy heatmap...")
    figs.append(("Per-Category Heatmap", plot_per_category_heatmap(exploded)))

    print("Generating Figure 3: Per-study accuracy...")
    figs.append(("Per-Study Accuracy", plot_per_study(exploded)))

    print("Generating Figure 4: Per-sample agreement...")
    fig4 = plot_per_sample_agreement(exploded)
    if fig4:
        figs.append(("Per-Sample Agreement", fig4))

    print("Generating Figure 5: Pairwise Cohen's Kappa...")
    fig5 = plot_pairwise_kappa(exploded)
    if fig5:
        figs.append(("Pairwise Kappa", fig5))

    print("Generating Figure 6: Fleiss' Kappa per category...")
    figs.append(("Fleiss Kappa", plot_fleiss_kappa(exploded)))

    print(f"\nShowing {len(figs)} figures — close each window to proceed to the next.\n")
    for title, fig in figs:
        fig.canvas.manager.set_window_title(title)
        plt.figure(fig.number)
        plt.show(block=True)


if __name__ == "__main__":
    main()
