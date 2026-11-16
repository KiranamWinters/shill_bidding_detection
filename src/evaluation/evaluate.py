"""Evaluation metrics, threshold optimisation and evaluation figures."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from config import settings


def compute_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    """Threshold-dependent and threshold-free metrics for the shill class."""
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "f2": fbeta_score(y_true, pred, beta=2, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "brier": brier_score_loss(y_true, proba),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "false_positive_rate": fp / (fp + tn) if (fp + tn) else 0.0,
        "flagged_share": float(pred.mean()),
    }


def text_classification_report(y_true, proba, threshold: float) -> str:
    pred = (np.asarray(proba) >= threshold).astype(int)
    return classification_report(y_true, pred, target_names=["Normal", "Shill"], digits=4, zero_division=0)


def optimise_threshold(
    y_true, proba,
    beta: float = settings.THRESHOLD_BETA,
    min_precision: float = settings.MIN_PRECISION_FOR_THRESHOLD,
) -> tuple[float, pd.DataFrame]:
    """Choose a decision threshold on validation (out-of-fold) predictions.

    Criterion: maximise F-beta (beta=2 weights recall twice as much as
    precision, because a missed shill bidder is costlier than a review of a
    normal one) among thresholds that keep precision >= ``min_precision`` so
    the investigation queue stays manageable. Falls back to plain F-beta
    maximisation if no threshold reaches the precision floor.

    Returns the chosen threshold and the full threshold sweep table.
    """
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    rows = []
    for t in grid:
        pred = (proba >= t).astype(int)
        rows.append({
            "threshold": t,
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0),
            "f1": f1_score(y_true, pred, zero_division=0),
            "fbeta": fbeta_score(y_true, pred, beta=beta, zero_division=0),
            "flagged": int(pred.sum()),
        })
    sweep = pd.DataFrame(rows)
    eligible = sweep[sweep["precision"] >= min_precision]
    pool = eligible if len(eligible) else sweep
    best_score = pool["fbeta"].max()
    # Among (near-)ties choose the threshold closest to 0.5 for stability.
    ties = pool[pool["fbeta"] >= best_score - 1e-9]
    chosen = float(ties.iloc[(ties["threshold"] - 0.5).abs().argsort()].iloc[0]["threshold"])
    return chosen, sweep


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _save(fig, name: str) -> str:
    path = settings.FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return str(path.relative_to(settings.PROJECT_ROOT))


def plot_confusion_matrices(results: dict[str, dict], name: str = "eval_confusion_matrices.png") -> str:
    """Grid of test-set confusion matrices (one per model)."""
    n = len(results)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.6 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (model, m) in zip(axes, results.items()):
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Normal", "Shill"], yticklabels=["Normal", "Shill"])
        ax.set_title(f"{model}\n(threshold {m['threshold']:.2f})", fontsize=9)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    for ax in axes[n:]:
        ax.axis("off")
    return _save(fig, name)


def plot_roc_pr(curves: dict[str, dict], prefix: str = "eval") -> list[str]:
    """Overlayed ROC and precision-recall curves on the test set."""
    paths = []
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for model, c in curves.items():
        ax.plot(c["fpr"], c["tpr"], label=f"{model} (AUC={c['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title("ROC curves - held-out test auctions")
    ax.legend(fontsize=7, loc="lower right")
    paths.append(_save(fig, f"{prefix}_roc_curves.png"))

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for model, c in curves.items():
        ax.plot(c["recall"], c["precision"], label=f"{model} (AP={c['pr_auc']:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curves - held-out test auctions")
    ax.legend(fontsize=7, loc="lower left")
    paths.append(_save(fig, f"{prefix}_pr_curves.png"))
    return paths


def curve_data(y_true, proba) -> dict:
    """ROC and PR curve points (for plots and the dashboard)."""
    fpr, tpr, _ = roc_curve(y_true, proba)
    precision, recall, _ = precision_recall_curve(y_true, proba)
    return {
        "fpr": fpr.tolist(), "tpr": tpr.tolist(),
        "precision": precision.tolist(), "recall": recall.tolist(),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
    }


def plot_model_comparison(table: pd.DataFrame, name: str = "eval_model_comparison.png") -> str:
    """Grouped bar chart of key test metrics per model."""
    metrics = ["precision", "recall", "f1", "pr_auc", "roc_auc"]
    long = table.melt(id_vars="model", value_vars=metrics, var_name="metric", value_name="score")
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(long, x="model", y="score", hue="metric", ax=ax)
    lo = max(0.0, long["score"].min() - 0.05)
    ax.set_ylim(lo, 1.0)
    ax.set_title("Model comparison on held-out test auctions (shill class)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=25)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    ax.legend(ncol=5, fontsize=8, loc="lower right")
    return _save(fig, name)


def plot_threshold_sweep(sweep: pd.DataFrame, chosen: float, model: str,
                         name: str = "eval_threshold_optimisation.png") -> str:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for col in ("precision", "recall", "f1", "fbeta"):
        ax.plot(sweep["threshold"], sweep[col], label=col if col != "fbeta" else f"F{settings.THRESHOLD_BETA:g}")
    ax.axvline(chosen, color="k", ls="--", lw=1, label=f"chosen = {chosen:.2f}")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score (out-of-fold, training auctions)")
    ax.set_title(f"Threshold optimisation - {model}")
    ax.legend(fontsize=8)
    return _save(fig, name)


def plot_validation_study(study: pd.DataFrame, name: str = "eval_validation_strategy.png") -> str:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.barplot(study, x="model", y="pr_auc_mean", hue="cv_strategy", ax=ax)
    ax.set_ylim(max(0, study["pr_auc_mean"].min() - 0.05), 1.0)
    ax.set_ylabel("CV PR-AUC (training data)")
    ax.set_title("Effect of the validation strategy on estimated performance")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=15)
    return _save(fig, name)
