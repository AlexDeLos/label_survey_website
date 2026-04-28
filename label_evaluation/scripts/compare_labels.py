"""
compare_labels.py
-----------------
Compare two directories of GEO label JSON files (one file per study,
keyed GSM → {axis: value}).

Public functions
----------------
compare_labels(dir1, dir2, output_dir)
    High-level statistics + per-category divergence breakdown.
    For 'treatment' specifically, divergences are split into:
      • val_only     – treatment type changed, intensity identical
      • intensity_only – same treatment type(s), only intensity changed
      • both_changed – type AND intensity both changed

analyze_divergence_patterns(dir1, dir2, output_dir)
    Top-N value swaps per category + study-level conflict distribution.

All figures are saved as SVGs under `output_dir`.

Usage
-----
    python compare_labels.py  \
        new_storage/labels/TULIP_1.2_RNA_old/5.0 \
        new_storage/labels/TULIP_1.2_RNA/5.0 \
        outputs/label_comparison
"""

import argparse
import os
from collections import Counter, defaultdict
from pathlib import Path

import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns


# =============================================================================
# Canonicalization helpers
# =============================================================================

def canonicalize(v):
    """Convert any label value to a stable, comparable form."""
    if isinstance(v, list):
        if len(v) > 0 and isinstance(v[0], dict):
            return tuple(sorted([tuple(sorted(d.items())) for d in v]))
        return tuple(sorted([str(i).lower().strip() for i in v]))
    return str(v).lower().strip()


def _extract_treatment_parts(raw_value) -> tuple[frozenset[str], frozenset[tuple[str, int]]]:
    """
    Decompose a raw treatment value into its canonical val-set and
    (val, intensity) pair-set so that value-only vs intensity-only
    divergences can be distinguished.

    Parameters
    ----------
    raw_value : list[dict] | list[str] | str | None
        The raw value stored in the label JSON for the 'treatment' axis.

    Returns
    -------
    val_set : frozenset[str]
        Set of treatment type strings (lowercased), e.g. {'chemical', 'heat'}.
    pair_set : frozenset[tuple[str, int]]
        Set of (val, intensity) pairs.
    """
    if not isinstance(raw_value, list):
        s = str(raw_value).lower().strip() if raw_value is not None else "unspecified"
        return frozenset([s]), frozenset([(s, -1)])

    if len(raw_value) == 0:
        return frozenset(["unspecified"]), frozenset([("unspecified", -1)])

    if isinstance(raw_value[0], dict):
        vals, pairs = [], []
        for d in raw_value:
            v   = str(d.get("val", "unspecified")).lower().strip()
            ity = int(d.get("intensity", -1))
            vals.append(v)
            pairs.append((v, ity))
        return frozenset(vals), frozenset(pairs)

    # Flat string list (no intensity info)
    vals = [str(x).lower().strip() for x in raw_value]
    return frozenset(vals), frozenset([(v, -1) for v in vals])


def _classify_treatment_divergence(
    raw1, raw2
) -> str | None:
    """
    Return the divergence class for two treatment raw values, or None if equal.

    Classes
    -------
    'intensity_only'  – same treatment type(s), only intensity changed
    'val_only'        – treatment type changed, intensity same (or absent)
    'both_changed'    – type AND intensity both changed
    """
    val_set1, pair_set1 = _extract_treatment_parts(raw1)
    val_set2, pair_set2 = _extract_treatment_parts(raw2)

    if val_set1 == val_set2 and pair_set1 == pair_set2:
        return None   # identical

    val_changed = val_set1 != val_set2
    int_changed  = pair_set1 != pair_set2

    if val_changed and int_changed:
        return "both_changed"
    if val_changed:
        return "val_only"
    return "intensity_only"


# =============================================================================
# Core loading helper
# =============================================================================

def _load_common(dir1: str, dir2: str):
    """
    Yield (gse_id, gsm_id, sample_dict_1, sample_dict_2) for all samples
    present in both directories.
    """
    path1, path2 = Path(dir1), Path(dir2)
    common_files = (
        {f.name for f in path1.glob("*.json")}
        & {f.name for f in path2.glob("*.json")}
    )
    for filename in sorted(common_files):
        with open(path1 / filename) as f1, open(path2 / filename) as f2:
            d1, d2 = json.load(f1), json.load(f2)
        gse_id = filename.replace(".json", "")
        for gsm in sorted(set(d1.keys()) & set(d2.keys())):
            yield gse_id, gsm, d1[gsm], d2[gsm]


# =============================================================================
# Plotting helpers
# =============================================================================

sns.set_theme(style="whitegrid", context="talk")
_PALETTE = sns.color_palette("Set2")


def _save(fig, output_dir: str, name: str):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def _plot_category_divergence_bar(
    category_diffs: Counter,
    total_samples: int,
    output_dir: str,
):
    """Bar chart: per-category divergence rate (%)."""
    cats   = [c for c, _ in category_diffs.most_common()]
    counts = [category_diffs[c] for c in cats]
    rates  = [100 * n / total_samples for n in counts]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(cats[::-1], rates[::-1], color=_PALETTE[0])
    ax.bar_label(bars, labels=[f"{r:.1f}%" for r in rates[::-1]],
                 padding=4, fontsize=11)
    ax.set_xlabel("Samples with divergence (%)")
    ax.set_title("Per-Category Divergence Rate\n(% of shared samples that differ)")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_xlim(0, max(rates) * 1.18)
    fig.tight_layout()
    _save(fig, output_dir, "01_category_divergence_rates.svg")


def _plot_treatment_split(
    treatment_split: Counter,
    output_dir: str,
):
    """Pie + bar chart for treatment divergence sub-types."""
    labels = {
        "intensity_only": "Intensity only",
        "val_only":        "Value (type) only",
        "both_changed":    "Both changed",
    }
    sizes  = [treatment_split.get(k, 0) for k in labels]
    total  = sum(sizes)
    if total == 0:
        print("  No treatment divergences to plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Pie ---
    wedge_colors = [_PALETTE[1], _PALETTE[2], _PALETTE[3]]
    axes[0].pie(
        sizes,
        labels=[f"{labels[k]}\n({v})" for k, v in zip(labels, sizes)],
        colors=wedge_colors,
        autopct="%1.1f%%",
        startangle=140,
        textprops={"fontsize": 12},
    )
    axes[0].set_title("Treatment Divergence Sub-types")

    # --- Bar (absolute counts) ---
    bar_labels = list(labels.values())
    bars = axes[1].bar(bar_labels, sizes, color=wedge_colors)
    axes[1].bar_label(bars, padding=4, fontsize=11)
    axes[1].set_ylabel("Number of divergent samples")
    axes[1].set_title("Treatment Divergence Breakdown")
    axes[1].set_ylim(0, max(sizes) * 1.2)

    fig.suptitle(
        f"Treatment divergences: {total} total  "
        f"| {treatment_split.get('intensity_only', 0)} intensity-only  "
        f"| {treatment_split.get('val_only', 0)} value-only  "
        f"| {treatment_split.get('both_changed', 0)} both",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    _save(fig, output_dir, "02_treatment_divergence_split.svg")


def _plot_study_conflict_distribution(
    study_df: pd.DataFrame,
    output_dir: str,
):
    """Histogram of per-study conflict rates + scatter coloured by n_samples."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Histogram ---
    axes[0].hist(
        study_df["Conflict_Rate"],
        bins=20, color=_PALETTE[0], edgecolor="white",
    )
    axes[0].axvline(0.5, color="red", linestyle="--", label="50% threshold")
    axes[0].set_xlabel("Conflict rate")
    axes[0].set_ylabel("Number of studies")
    axes[0].set_title("Distribution of Per-Study Conflict Rates")
    axes[0].legend()

    # --- Scatter: conflict rate vs study size, sized by n_samples ---
    sc = axes[1].scatter(
        study_df["Samples"],
        study_df["Conflict_Rate"] * 100,
        c=study_df["Conflict_Rate"],
        cmap="RdYlGn_r",
        s=study_df["Samples"].clip(upper=200) + 20,
        alpha=0.7,
        edgecolors="grey",
        linewidths=0.4,
    )
    plt.colorbar(sc, ax=axes[1], label="Conflict rate")
    axes[1].axhline(50, color="red", linestyle="--", alpha=0.6)
    axes[1].set_xlabel("Study size (# samples)")
    axes[1].set_ylabel("Conflict rate (%)")
    axes[1].set_title("Conflict Rate vs Study Size")

    # Annotate the five worst studies
    top5 = study_df.nlargest(5, "Conflict_Rate")
    for _, row in top5.iterrows():
        axes[1].annotate(
            row["GSE"],
            xy=(row["Samples"], row["Conflict_Rate"] * 100),
            xytext=(6, 0), textcoords="offset points",
            fontsize=9, color="darkred",
        )

    fig.tight_layout()
    _save(fig, output_dir, "03_study_conflict_distribution.svg")


def _plot_top_swaps(
    global_swaps: dict[str, Counter],
    output_dir: str,
    top_n: int = 10,
):
    """
    Horizontal bar charts showing the most common value swaps per category.
    Categories are plotted in separate subplots stacked vertically.
    """
    cats = [c for c in global_swaps if global_swaps[c]]
    if not cats:
        return

    n_cols = min(2, len(cats))
    n_rows = int(np.ceil(len(cats) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(13 * n_cols, 4 * n_rows + 1),
                             squeeze=False)

    for idx, cat in enumerate(cats):
        ax = axes[idx // n_cols][idx % n_cols]
        top = global_swaps[cat].most_common(top_n)
        if not top:
            ax.axis("off")
            continue

        labels = [f"{_fmt_val(v1)}  →  {_fmt_val(v2)}" for (v1, v2), _ in top]
        counts = [cnt for _, cnt in top]

        bars = ax.barh(labels[::-1], counts[::-1], color=_PALETTE[idx % len(_PALETTE)])
        ax.bar_label(bars, padding=3, fontsize=10)
        ax.set_xlabel("Occurrences")
        ax.set_title(f"{cat.capitalize()} — top {top_n} swaps")
        ax.set_xlim(0, max(counts) * 1.2)

    # Hide empty subplots
    for idx in range(len(cats), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    fig.suptitle("Most Common Label Value Swaps (dir1 → dir2)", fontsize=15, y=1.01)
    fig.tight_layout()
    _save(fig, output_dir, "04_top_value_swaps.svg")


def _plot_treatment_swap_breakdown(
    treatment_swaps: dict[str, Counter],
    output_dir: str,
    top_n: int = 10,
):
    """
    Separate bar charts for treatment swaps split by divergence sub-type:
    intensity_only, val_only, both_changed.
    """
    sub_types = {
        "intensity_only": "Intensity-only swaps (same type, different intensity)",
        "val_only":       "Value-only swaps (type changed, intensity same/absent)",
        "both_changed":   "Both type and intensity changed",
    }
    has_data = {k: bool(treatment_swaps.get(k)) for k in sub_types}
    if not any(has_data.values()):
        return

    n_plots = sum(has_data.values())
    fig, axes = plt.subplots(1, n_plots,
                             figsize=(10 * n_plots, 5),
                             squeeze=False)

    col = 0
    for key, title in sub_types.items():
        if not has_data[key]:
            continue
        ax = axes[0][col]
        col += 1
        top = treatment_swaps[key].most_common(top_n)
        labels = [f"{_fmt_val(v1)}  →  {_fmt_val(v2)}" for (v1, v2), _ in top]
        counts = [cnt for _, cnt in top]

        bars = ax.barh(labels[::-1], counts[::-1],
                       color=_PALETTE[(col - 1) % len(_PALETTE)])
        ax.bar_label(bars, padding=3, fontsize=10)
        ax.set_xlabel("Occurrences")
        ax.set_title(title, fontsize=12)
        ax.set_xlim(0, max(counts) * 1.2)

    fig.suptitle("Treatment Swaps by Sub-type (dir1 → dir2)", fontsize=14, y=1.02)
    fig.tight_layout()
    _save(fig, output_dir, "05_treatment_swaps_by_subtype.svg")


def _fmt_val(v) -> str:
    """Compact string representation for a canonicalized label value."""
    if isinstance(v, tuple):
        # Treatment dict tuples: (('intensity', 2), ('val', 'heat'))
        parts = []
        for item in v:
            if isinstance(item, tuple) and len(item) == 2:
                parts.append(f"{item[0]}={item[1]}")
            else:
                parts.append(str(item))
        return "(" + ", ".join(parts) + ")"
    return str(v)


def _fmt_treatment_swap(raw1, raw2) -> str:
    """Human-readable summary of a treatment swap for the top-swaps counter."""
    val1, pair1 = _extract_treatment_parts(raw1)
    val2, pair2 = _extract_treatment_parts(raw2)
    # Represent as sorted tuple of val strings for readability
    return (
        "+".join(sorted(val1)) + " " + _fmt_intensity(pair1),
        "+".join(sorted(val2)) + " " + _fmt_intensity(pair2),
    )


def _fmt_intensity(pair_set: frozenset[tuple[str, int]]) -> str:
    intensities = sorted({p[1] for p in pair_set if p[1] != -1})
    if not intensities:
        return ""
    return "[i=" + ",".join(str(i) for i in intensities) + "]"


# =============================================================================
# Main analysis functions
# =============================================================================

def compare_labels(dir1: str, dir2: str, output_dir: str = "outputs/label_comparison") -> pd.DataFrame:
    """
    Compute per-category divergence statistics, with treatment split into
    intensity-only, value-only, and both-changed sub-categories.

    Prints a summary table and saves figures.

    Parameters
    ----------
    dir1, dir2 : str
        Paths to the two label directories (each contains one JSON per study).
    output_dir : str
        Directory where figures are saved.

    Returns
    -------
    pd.DataFrame
        One row per study with columns:
        GSE, Samples, Diff_Samples, Conflict_Rate,
        plus per-category difference counts.
    """
    category_diffs: Counter = Counter()
    treatment_split: Counter = Counter()
    # Per-study accumulation
    study_records: list[dict] = []

    # We accumulate per-study data in a temporary dict
    study_accum: dict[str, dict] = defaultdict(
        lambda: {"samples": 0, "diff_samples": set(), "cat_counts": Counter()}
    )

    for gse_id, gsm_id, s1, s2 in _load_common(dir1, dir2):
        rec = study_accum[gse_id]
        rec["samples"] += 1
        sample_has_diff = False

        categories = set(s1.keys()) | set(s2.keys())
        for cat in categories:
            raw1 = s1.get(cat)
            raw2 = s2.get(cat)

            if cat == "treatment":
                cls = _classify_treatment_divergence(raw1, raw2)
                if cls is not None:
                    category_diffs["treatment"] += 1
                    treatment_split[cls] += 1
                    rec["cat_counts"]["treatment"] += 1
                    sample_has_diff = True
            else:
                v1, v2 = canonicalize(raw1), canonicalize(raw2)
                if v1 != v2 and v1 != "unspecified":
                    category_diffs[cat] += 1
                    rec["cat_counts"][cat] += 1
                    sample_has_diff = True

        if sample_has_diff:
            rec["diff_samples"].add(gsm_id)

    # Flatten study accumulator into records
    for gse_id, rec in study_accum.items():
        n_samples = rec["samples"]
        n_diff    = len(rec["diff_samples"])
        row = {
            "GSE": gse_id,
            "Samples": n_samples,
            "Diff_Samples": n_diff,
            "Conflict_Rate": n_diff / n_samples if n_samples else 0.0,
        }
        row.update({f"{cat}_Diffs": cnt for cat, cnt in rec["cat_counts"].items()})
        study_records.append(row)

    study_df = pd.DataFrame(study_records).fillna(0)
    total_samples = int(study_df["Samples"].sum())
    total_diff    = int(study_df["Diff_Samples"].sum())

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("LABEL COMPARISON STATISTICS")
    print("=" * 50)
    print(f"Files processed            : {len(study_records)}")
    print(f"Total shared samples       : {total_samples:,}")
    print(f"Samples with ≥1 difference : {total_diff:,}")
    if total_samples:
        print(f"Overall divergence rate    : {100 * total_diff / total_samples:.2f}%")

    print("\nDifferences by category (ranked):")
    for cat, count in category_diffs.most_common():
        print(f"  {cat:25}: {count:5}  ({100 * count / total_samples:.2f}%)")

    print("\nTreatment divergence sub-types:")
    total_t = sum(treatment_split.values())
    for sub, label in [
        ("intensity_only", "Intensity only (same type)"),
        ("val_only",       "Value only (type changed) "),
        ("both_changed",   "Both type + intensity     "),
    ]:
        n = treatment_split.get(sub, 0)
        pct_of_t = 100 * n / total_t if total_t else 0
        pct_of_s = 100 * n / total_samples if total_samples else 0
        print(f"  {label}: {n:4}  ({pct_of_t:.1f}% of treatment diffs | {pct_of_s:.2f}% of all samples)")

    high_conflict = int((study_df["Conflict_Rate"] > 0.5).sum())
    zero_conflict = int((study_df["Conflict_Rate"] == 0).sum())
    print(f"\nStudies with >50% conflict : {high_conflict}")
    print(f"Studies with   0% conflict : {zero_conflict}")

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    print("\nGenerating figures...")
    _plot_category_divergence_bar(category_diffs, total_samples, output_dir)
    _plot_treatment_split(treatment_split, output_dir)
    _plot_study_conflict_distribution(study_df, output_dir)

    return study_df


def analyze_divergence_patterns(
    dir1: str,
    dir2: str,
    output_dir: str = "outputs/label_comparison",
    top_n: int = 10,
) -> None:
    """
    Collect the most common per-category value swaps (dir1 → dir2),
    with treatment swaps further split by divergence sub-type.

    Prints a summary and saves figures.

    Parameters
    ----------
    dir1, dir2 : str
        Paths to the two label directories.
    output_dir : str
        Directory where figures are saved.
    top_n : int
        How many swaps to report / plot per category.
    """
    # global_swaps[cat][(v1, v2)] = count
    global_swaps: dict[str, Counter] = defaultdict(Counter)

    # treatment_swaps[sub_type][(readable_from, readable_to)] = count
    treatment_swaps: dict[str, Counter] = {
        "intensity_only": Counter(),
        "val_only":        Counter(),
        "both_changed":    Counter(),
    }

    # Study-level data for conflict-rate calculation
    study_diff: Counter = Counter()
    study_total: Counter = Counter()

    for gse_id, gsm_id, s1, s2 in _load_common(dir1, dir2):
        study_total[gse_id] += 1
        sample_diff = False

        for cat in ["tissue", "treatment", "modification",
                    "developmental_stage", "ecotype", "medium"]:
            raw1 = s1.get(cat)
            raw2 = s2.get(cat)

            if cat == "treatment":
                cls = _classify_treatment_divergence(raw1, raw2)
                if cls is not None:
                    readable = _fmt_treatment_swap(raw1, raw2)
                    global_swaps["treatment"][(canonicalize(raw1), canonicalize(raw2))] += 1
                    treatment_swaps[cls][readable] += 1
                    sample_diff = True
            else:
                v1, v2 = canonicalize(raw1), canonicalize(raw2)
                if v1 != v2 and v1 != "unspecified":
                    global_swaps[cat][(v1, v2)] += 1
                    sample_diff = True

        if sample_diff:
            study_diff[gse_id] += 1

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    study_conflict = {
        gse: study_diff[gse] / study_total[gse]
        for gse in study_total
    }
    conflict_df = pd.DataFrame([
        {"GSE": gse, "Samples": study_total[gse], "Conflict_Rate": rate}
        for gse, rate in study_conflict.items()
    ])

    print("\n--- TOP 5 MOST CONFLICTED STUDIES ---")
    print(
        conflict_df.sort_values("Conflict_Rate", ascending=False)
        .head(5)[["GSE", "Samples", "Conflict_Rate"]]
        .to_string(index=False)
    )

    for cat in ["tissue", "treatment"]:
        print(f"\n--- TOP {top_n} COMMON {cat.upper()} DIVERGENCES ---")
        for (v1, v2), count in global_swaps[cat].most_common(top_n):
            print(f"  {_fmt_val(v1):40}  →  {_fmt_val(v2):40}  ({count}×)")

    print("\n--- TREATMENT SWAPS BY SUB-TYPE ---")
    for sub_type, label in [
        ("intensity_only", "Intensity-only"),
        ("val_only",        "Value-only    "),
        ("both_changed",    "Both changed  "),
    ]:
        print(f"\n  {label} — top {top_n}:")
        for (from_str, to_str), count in treatment_swaps[sub_type].most_common(top_n):
            print(f"    {from_str:35}  →  {to_str:35}  ({count}×)")

    high_conflict = sum(1 for r in study_conflict.values() if r > 0.5)
    zero_conflict  = sum(1 for r in study_conflict.values() if r == 0)
    print(f"\n--- DISTRIBUTION SUMMARY ---")
    print(f"Studies with >50% conflict : {high_conflict}")
    print(f"Studies with   0% conflict : {zero_conflict}")

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    print("\nGenerating figures...")
    _plot_top_swaps(global_swaps, output_dir, top_n=top_n)
    _plot_treatment_swap_breakdown(treatment_swaps, output_dir, top_n=top_n)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare label JSONs in two directories."
    )
    parser.add_argument("-dir2",default="new_storage/labels/TULIP_1.2/5.0", help="Path to the first label directory")
    parser.add_argument("-dir1",default="new_storage/labels/TULIP_1.2/5.0_old", help="Path to the second label directory")
    parser.add_argument(
        "--output-dir", "-o",
        default="outputs/label_comparison",
        help="Directory for output figures (default: outputs/label_comparison)",
    )
    parser.add_argument(
        "--top-n", "-n",
        type=int, default=10,
        help="Number of top swaps to report/plot (default: 10)",
    )
    args = parser.parse_args()

    study_df = compare_labels(args.dir1, args.dir2, output_dir=args.output_dir)
    analyze_divergence_patterns(
        args.dir1, args.dir2,
        output_dir=args.output_dir,
        top_n=args.top_n,
    )
    # compare_labels("new_storage/labels/TULIP_1.2_RNA_old/5.0", "new_storage/labels/TULIP_1.2_RNA/5.0")
    # analyze_divergence_patterns("new_storage/labels/TULIP_1.2_RNA_old/5.0", "new_storage/labels/TULIP_1.2_RNA/5.0")
