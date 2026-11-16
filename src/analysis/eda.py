"""Exploratory data analysis: statistics, figures and the dataset analysis report.

Every number written to the report is computed from the downloaded data. EDA
is descriptive and run on the full dataset; nothing computed here is used to
fit a model (feature selection is repeated on the training split only).
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu

from config import settings
from src.features.feature_selection import (
    find_correlated_pairs,
    leakage_assessment,
    mutual_information_ranking,
)
from src.utils.logging_config import get_logger

logger = get_logger("analysis.eda")

PALETTE = {0: "#4C78A8", 1: "#E45756"}
LABELS = {0: "Normal (0)", 1: "Shill (1)"}


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def class_difference_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Mann-Whitney U test and rank-biserial effect size per feature."""
    rows = []
    normal = df[df[settings.TARGET] == 0]
    shill = df[df[settings.TARGET] == 1]
    for f in settings.BEHAVIOURAL_FEATURES:
        u, p = mannwhitneyu(shill[f], normal[f], alternative="two-sided")
        effect = 2 * u / (len(shill) * len(normal)) - 1  # rank-biserial correlation
        rows.append(
            {
                "feature": f,
                "mean_normal": normal[f].mean(),
                "mean_shill": shill[f].mean(),
                "median_normal": normal[f].median(),
                "median_shill": shill[f].median(),
                "mann_whitney_p": p,
                "rank_biserial": effect,
            }
        )
    return pd.DataFrame(rows).sort_values("rank_biserial", key=np.abs, ascending=False)


def group_structure(df: pd.DataFrame) -> dict:
    """Describe how records are organised by auction and bidder."""
    by_auction = df.groupby(settings.AUCTION_ID)[settings.TARGET].agg(["mean", "size"])
    by_bidder = df.groupby(settings.BIDDER_ID)[settings.TARGET].agg(["mean", "size"])
    auction_level = ["Auction_Bids", "Starting_Price_Average", "Auction_Duration"]
    bidder_level = ["Bidder_Tendency", "Winning_Ratio"]
    const_auction = {
        c: float((df.groupby(settings.AUCTION_ID)[c].nunique() == 1).mean()) for c in auction_level
    }
    multi_bidder = df.groupby(settings.BIDDER_ID).filter(lambda g: len(g) > 1)
    const_bidder = {
        c: float((multi_bidder.groupby(settings.BIDDER_ID)[c].nunique() == 1).mean())
        for c in bidder_level
    }
    return {
        "n_auctions": int(by_auction.shape[0]),
        "n_bidder_ids": int(by_bidder.shape[0]),
        "records_per_auction": by_auction["size"].describe().round(2).to_dict(),
        "records_per_bidder_id": by_bidder["size"].describe().round(2).to_dict(),
        "auctions_with_shill": int((by_auction["mean"] > 0).sum()),
        "auctions_mixed_classes": int(((by_auction["mean"] > 0) & (by_auction["mean"] < 1)).sum()),
        "bidder_ids_with_shill": int((by_bidder["mean"] > 0).sum()),
        "bidder_ids_mixed_classes": int(((by_bidder["mean"] > 0) & (by_bidder["mean"] < 1)).sum()),
        "auction_feature_constant_within_auction": const_auction,
        "bidder_feature_constant_within_bidder_id": const_bidder,
        "masked_bidder_id_example": df[settings.BIDDER_ID].iloc[0],
    }


def compute_eda_summary(df: pd.DataFrame) -> dict:
    """Compute all EDA statistics as a JSON-serialisable dictionary."""
    feats = settings.BEHAVIOURAL_FEATURES
    counts = df[settings.TARGET].value_counts().sort_index()
    corr = df[feats + [settings.TARGET]].corr(method="spearman")
    pearson = df[feats + [settings.TARGET]].corr(method="pearson")
    so_table = pd.crosstab(df["Successive_Outbidding"], df[settings.TARGET])
    summary = {
        "shape": list(df.shape),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_values": {c: int(v) for c, v in df.isna().sum().items()},
        "total_missing": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_rows_excluding_record_id": int(df.drop(columns=[settings.RECORD_ID]).duplicated().sum()),
        "duplicate_feature_vectors": int(df[feats].duplicated().sum()),
        "conflicting_feature_vectors": int(
            (df.groupby(feats)[settings.TARGET].nunique() > 1).sum()
        ),
        "unique_values": {c: int(v) for c, v in df.nunique().items()},
        "describe": df[feats].describe().T.round(4).to_dict(orient="index"),
        "class_counts": {int(k): int(v) for k, v in counts.items()},
        "class_share": {int(k): float(v / len(df)) for k, v in counts.items()},
        "imbalance_ratio": float(counts.max() / counts.min()),
        "spearman_with_target": corr[settings.TARGET].drop(settings.TARGET).round(4).to_dict(),
        "pearson_with_target": pearson[settings.TARGET].drop(settings.TARGET).round(4).to_dict(),
        "correlated_pairs": find_correlated_pairs(df[feats]).to_dict(orient="records"),
        "last_equals_early_share": float((df["Last_Bidding"] == df["Early_Bidding"]).mean()),
        "successive_outbidding_crosstab": {
            str(k): {int(c): int(v) for c, v in row.items()} for k, row in so_table.iterrows()
        },
        "mutual_information_full_data": mutual_information_ranking(
            df[feats], df[settings.TARGET]
        ).round(4).to_dict(),
        "class_difference_tests": class_difference_tests(df).round(6).to_dict(orient="records"),
        "group_structure": group_structure(df),
        "leakage_assessment": leakage_assessment().to_dict(orient="records"),
    }
    return summary


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _save(fig, name: str) -> str:
    path = settings.FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return str(path.relative_to(settings.PROJECT_ROOT))


def make_eda_figures(df: pd.DataFrame) -> list[str]:
    """Create and save all EDA figures; return their relative paths."""
    sns.set_theme(style="whitegrid")
    feats = settings.BEHAVIOURAL_FEATURES
    plot_df = df.assign(Label=df[settings.TARGET].map(LABELS))
    pal = {LABELS[k]: v for k, v in PALETTE.items()}
    paths = []

    # Class distribution
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df[settings.TARGET].value_counts().sort_index()
    bars = ax.bar([LABELS[i] for i in counts.index], counts.values, color=[PALETTE[i] for i in counts.index])
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v} ({v / len(df):.1%})", ha="center", va="bottom")
    ax.set_title("Target class distribution")
    ax.set_ylabel("Records")
    paths.append(_save(fig, "eda_class_distribution.png"))

    # Feature distributions by class
    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    for ax, f in zip(axes.ravel(), feats):
        sns.histplot(plot_df, x=f, hue="Label", stat="density", common_norm=False,
                     bins=30, palette=pal, element="step", ax=ax)
        ax.set_title(f)
    fig.suptitle("Feature distributions by class (density, per-class normalised)")
    paths.append(_save(fig, "eda_feature_distributions.png"))

    # Box plots
    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    for ax, f in zip(axes.ravel(), feats):
        sns.boxplot(plot_df, x="Label", y=f, hue="Label", palette=pal, ax=ax, legend=False)
        ax.set_title(f)
        ax.set_xlabel("")
    fig.suptitle("Behavioural features: normal vs shill records")
    paths.append(_save(fig, "eda_boxplots_by_class.png"))

    # Correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df[feats + [settings.TARGET]].corr(method="spearman")
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, ax=ax, square=True)
    ax.set_title("Spearman correlation matrix")
    paths.append(_save(fig, "eda_correlation_heatmap.png"))

    # Successive outbidding vs class
    fig, ax = plt.subplots(figsize=(6, 4))
    ct = pd.crosstab(df["Successive_Outbidding"], df[settings.TARGET])
    ct.plot(kind="bar", ax=ax, color=[PALETTE[0], PALETTE[1]])
    ax.set_title("Successive_Outbidding value vs class")
    ax.set_xlabel("Successive_Outbidding")
    ax.set_ylabel("Records")
    ax.legend([LABELS[0], LABELS[1]])
    ax.tick_params(axis="x", rotation=0)
    paths.append(_save(fig, "eda_successive_outbidding.png"))

    # Group structure
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    by_auction = df.groupby(settings.AUCTION_ID)[settings.TARGET].agg(["mean", "size"])
    axes[0].hist(by_auction["size"], bins=26, color="#72B7B2")
    axes[0].set_title("Records (bidders) per auction")
    axes[0].set_xlabel("Records")
    axes[1].hist(by_auction["mean"], bins=20, color="#E45756")
    axes[1].set_title("Share of shill records within each auction")
    axes[1].set_xlabel("Shill share")
    paths.append(_save(fig, "eda_group_structure.png"))

    # Pairwise scatter of strongest features
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.scatterplot(plot_df.sample(frac=1, random_state=settings.RANDOM_SEED), x="Bidding_Ratio",
                    y="Winning_Ratio", hue="Label", palette=pal, alpha=0.5, s=14, ax=ax)
    ax.set_title("Bidding_Ratio vs Winning_Ratio")
    paths.append(_save(fig, "eda_bidding_vs_winning.png"))
    return paths


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _md_table(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_dataset_report(summary: dict, figures: list[str]) -> None:
    """Write ``reports/dataset_analysis_report.md`` from computed statistics."""
    s = summary
    gs = s["group_structure"]
    tests = pd.DataFrame(s["class_difference_tests"])
    # p-values below double precision underflow to exactly 0.
    tests["mann_whitney_p"] = tests["mann_whitney_p"].map(lambda p: "< 1e-300" if p == 0 else f"{p:.2e}")
    desc = pd.DataFrame(s["describe"]).T.reset_index().rename(columns={"index": "feature"})
    desc = desc[["feature", "mean", "std", "min", "50%", "max"]]
    uniq = pd.DataFrame(list(s["unique_values"].items()), columns=["column", "unique_values"])
    mi = pd.DataFrame(list(s["mutual_information_full_data"].items()), columns=["feature", "mutual_information"])
    pairs = pd.DataFrame(s["correlated_pairs"]) if s["correlated_pairs"] else pd.DataFrame(
        columns=["feature_a", "feature_b", "abs_spearman"])
    leak = pd.DataFrame(s["leakage_assessment"])
    so = s["successive_outbidding_crosstab"]
    so_rows = pd.DataFrame(
        [{"Successive_Outbidding": k, "normal": v.get(0, 0), "shill": v.get(1, 0)} for k, v in so.items()]
    )

    text = f"""# Dataset Analysis Report - UCI Shill Bidding Dataset (ID 562)

*Generated automatically by `src/analysis/eda.py` from the official UCI download.
All values below are computed from the data.*

Source: {settings.UCI_DATASET_URL} - DOI {settings.UCI_DOI} - licence {settings.DATASET_LICENSE}.

## 1. Structure

* Shape: **{s['shape'][0]} rows x {s['shape'][1]} columns**
* Missing values: **{s['total_missing']}**
* Exact duplicate rows: **{s['duplicate_rows']}** (excluding Record_ID: {s['duplicate_rows_excluding_record_id']})
* Duplicate behavioural feature vectors: **{s['duplicate_feature_vectors']}**, of which
  **{s['conflicting_feature_vectors']}** distinct vectors carry conflicting labels.
* `Bidder_ID` is a masked string (e.g. `{gs['masked_bidder_id_example']}`); all other columns are numeric.

{_md_table(uniq)}

## 2. Target distribution and imbalance

| Class | Records | Share |
|---|---|---|
| 0 - Normal | {s['class_counts'][0]} | {s['class_share'][0]:.2%} |
| 1 - Shill | {s['class_counts'][1]} | {s['class_share'][1]:.2%} |

Imbalance ratio (majority : minority) = **{s['imbalance_ratio']:.2f} : 1**. The dataset is
moderately imbalanced, so accuracy is not an adequate metric (always predicting
"normal" already yields {s['class_share'][0]:.1%} accuracy). Precision, recall, F1 and
PR-AUC for the shill class are used instead, and class weighting is evaluated.

## 3. Descriptive statistics

{_md_table(desc)}

Eight behavioural features are already normalised to [0, 1]; `Auction_Duration` is in days
with only {s['unique_values']['Auction_Duration']} distinct values.

## 4. Behavioural differences between classes

Mann-Whitney U test (two-sided) and rank-biserial effect size (positive = higher in shill records):

{_md_table(tests[['feature', 'mean_normal', 'mean_shill', 'median_normal', 'median_shill', 'mann_whitney_p', 'rank_biserial']])}

**Encoding direction.** Shill records have *higher* values of `Winning_Ratio`, `Last_Bidding` and
`Starting_Price_Average`, although the UCI descriptions say shill bidders rarely win, become inactive
near the end and that their sellers use *low* starting prices. This suggests every behavioural
feature is encoded as a suspicion score (higher = more shill-like) rather than as the raw quantity.
The UCI page does not publish the formulas, so this is an observation from the data, not a
documented fact; features should be interpreted as "suspicion indicators" rather than raw ratios.

### Successive_Outbidding is close to a deterministic indicator

{_md_table(so_rows)}

Only {so_rows.loc[so_rows.Successive_Outbidding == '0.0', 'shill'].sum() if '0.0' in so else 'n/a'} shill records have
`Successive_Outbidding = 0`. A single-rule baseline ("flag if Successive_Outbidding > 0") is
therefore evaluated alongside the ML models to measure what the models add.

## 5. Correlation and redundancy

Spearman correlation with the target:

{_md_table(pd.DataFrame(list(s['spearman_with_target'].items()), columns=['feature', 'spearman_rho']))}

Feature pairs with |Spearman| >= {settings.CORRELATION_THRESHOLD}:

{_md_table(pairs) if len(pairs) else '_None._'}

`Last_Bidding` equals `Early_Bidding` exactly in **{s['last_equals_early_share']:.1%}** of records
(bidders who placed a single bid have identical first and last bid times).

Mutual information with the target (full data, descriptive only):

{_md_table(mi)}

## 6. Group structure and implications for evaluation

* {gs['n_auctions']} auctions and {gs['n_bidder_ids']} distinct (masked) bidder IDs.
* Records per auction: median {gs['records_per_auction']['50%']}, max {gs['records_per_auction']['max']}.
* {gs['auctions_with_shill']} auctions contain at least one shill record; {gs['auctions_mixed_classes']} of them also
  contain normal records.
* Auction-level features are constant within an auction (share of auctions with a single value):
  {', '.join(f"{k}: {v:.0%}" for k, v in gs['auction_feature_constant_within_auction'].items())}.
  A row-level split therefore places records of the same auction in train and test.
* Bidder-level features are expected to be constant per bidder, but among masked bidder IDs with
  more than one record they are constant in only
  {', '.join(f"{k}: {v:.0%}" for k, v in gs['bidder_feature_constant_within_bidder_id'].items())} of cases.
  Either the masking (only the first and last characters are kept) merges different real bidders
  under one ID, or these features are computed per auction; in both cases a masked ID cannot be
  trusted as a unique bidder key, so **Bidder_ID-grouped validation is only approximate**.
  Auction_ID is the reliable grouping key.

## 7. Temporal availability / leakage assessment

{_md_table(leak)}

Several features (`Last_Bidding`, `Auction_Bids`, `Winning_Ratio`, final `Bidding_Ratio`) are only
known after an auction has ended. The model is therefore a **retrospective** detector suitable for
post-auction investigation, not real-time prevention.

## 8. Figures

""" + "\n".join(f"![{p}](../{p.replace(chr(92), '/')})" for p in figures) + "\n"
    settings.DATASET_REPORT_MD.write_text(text, encoding="utf-8")


def run_eda(df: pd.DataFrame) -> dict:
    """Run the full EDA, persist the summary JSON, figures and Markdown report."""
    settings.ensure_directories()
    summary = compute_eda_summary(df)
    figures = make_eda_figures(df)
    summary["figures"] = figures
    settings.EDA_SUMMARY_FILE.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    write_dataset_report(summary, figures)
    logger.info("EDA complete: %d figures, report at %s", len(figures), settings.DATASET_REPORT_MD)
    return summary
