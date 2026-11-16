"""Global and local model explanations.

* Global: permutation importance (model-agnostic, on held-out test auctions),
  native importances (tree impurity / absolute standardised coefficients) and
  mean |SHAP| values.
* Local: SHAP values for tree-based and linear models; a model-agnostic
  occlusion analysis (replace one feature by its training median and measure
  the change in probability) for any other model.

Feature attributions describe how the *model* uses each behaviour. They are
associations learned from labelled data, not proof of causation or intent.
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from config import settings
from src.utils.logging_config import get_logger

logger = get_logger("explainability")

TREE_MODELS = (
    "DecisionTreeClassifier", "RandomForestClassifier", "ExtraTreesClassifier",
    "GradientBoostingClassifier", "HistGradientBoostingClassifier",
)
LINEAR_MODELS = ("LogisticRegression",)

# Plain-language reading of a high (shill-like) value of each indicator.
BEHAVIOUR_PHRASES = {
    "Bidder_Tendency": "the bidder concentrates on auctions of only a few sellers",
    "Bidding_Ratio": "the bidder placed an unusually large share of the bids in this auction",
    "Successive_Outbidding": "the bidder repeatedly outbid their own winning bid",
    "Last_Bidding": "the bidder's last-bidding indicator is high (inactivity pattern near the auction end)",
    "Auction_Bids": "the auction attracted more bids than concurrent auctions",
    "Starting_Price_Average": "the auction's starting-price indicator is high",
    "Early_Bidding": "the bidder's early-bidding indicator is high",
    "Winning_Ratio": "the bidder's winning-ratio indicator is high (pattern associated with shill records)",
    "Auction_Duration": "the length of this auction (a contextual auction attribute)",
}


def _final_estimator(pipeline):
    return pipeline.named_steps["model"]


def _transform(pipeline, X: pd.DataFrame) -> pd.DataFrame:
    return pipeline[:-1].transform(X)


class ModelExplainer:
    """Explains predictions of a fitted model bundle."""

    def __init__(self, bundle: dict):
        self.pipeline = bundle["pipeline"]
        self.features = list(bundle["features"])
        self.background = bundle["background"][self.features]
        self.reference = bundle["reference_stats"]
        self.model_name = type(_final_estimator(self.pipeline)).__name__
        self._shap_explainer = None
        self._shap_ok = None

    # ------------------------------------------------------------------ SHAP
    @property
    def method(self) -> str:
        return "SHAP" if self._init_shap() else "occlusion"

    def _init_shap(self) -> bool:
        if self._shap_ok is not None:
            return self._shap_ok
        try:
            import shap

            model = _final_estimator(self.pipeline)
            bg = _transform(self.pipeline, self.background)
            if self.model_name in TREE_MODELS:
                self._shap_explainer = shap.TreeExplainer(model)
            elif self.model_name in LINEAR_MODELS:
                self._shap_explainer = shap.LinearExplainer(model, bg)
            else:
                self._shap_ok = False
                return False
            self._shap_ok = True
        except Exception as exc:  # SHAP is optional; fall back gracefully
            logger.warning("SHAP unavailable for %s: %s", self.model_name, exc)
            self._shap_ok = False
        return self._shap_ok

    def shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """SHAP values for the shill class, shape (n_records, n_features)."""
        if not self._init_shap():
            raise RuntimeError("SHAP is not available for this model.")
        Xt = _transform(self.pipeline, X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            values = self._shap_explainer.shap_values(Xt)
        if isinstance(values, list):  # older SHAP: one array per class
            values = values[1]
        values = np.asarray(values)
        if values.ndim == 3:  # (n, features, classes)
            values = values[:, :, 1]
        return values

    # ------------------------------------------------------------- occlusion
    def occlusion(self, X: pd.DataFrame) -> np.ndarray:
        """Change in shill probability when each feature is set to its training median."""
        base = self.pipeline.predict_proba(X)[:, 1]
        out = np.zeros((len(X), len(self.features)))
        for j, f in enumerate(self.features):
            Xm = X.copy()
            Xm[f] = self.reference["median"][f]
            out[:, j] = base - self.pipeline.predict_proba(Xm)[:, 1]
        return out

    # ----------------------------------------------------------------- local
    def explain_record(self, record: pd.DataFrame, top_k: int = 5) -> dict:
        """Explain a single record (one-row DataFrame)."""
        X = record[self.features].astype(float)
        proba = float(self.pipeline.predict_proba(X)[0, 1])
        if self._init_shap():
            contrib = self.shap_values(X)[0]
            method = "SHAP"
            unit = "SHAP value (model output units; positive = pushes towards shill)"
        else:
            contrib = self.occlusion(X)[0]
            method = "occlusion"
            unit = "change in shill probability vs. the training median"
        table = pd.DataFrame({
            "feature": self.features,
            "value": X.iloc[0].values,
            "contribution": contrib,
            "normal_median": [self.reference["median_normal"][f] for f in self.features],
            "shill_median": [self.reference["median_shill"][f] for f in self.features],
        })
        table["direction"] = np.where(table["contribution"] > 0, "towards shill", "towards normal")
        table = table.reindex(table["contribution"].abs().sort_values(ascending=False).index)
        return {
            "probability": proba,
            "method": method,
            "unit": unit,
            "contributions": table.reset_index(drop=True),
            "narrative": self.narrative(table, proba, top_k),
        }

    @staticmethod
    def narrative(table: pd.DataFrame, proba: float, top_k: int = 3) -> list[str]:
        """Plain-language sentences describing the strongest contributions."""
        lines = []
        up = table[table["contribution"] > 0].head(top_k)
        down = table[table["contribution"] < 0].head(2)
        for _, r in up.iterrows():
            phrase = BEHAVIOUR_PHRASES.get(r["feature"], r["feature"])
            lines.append(
                f"Raised the suspicion score: {phrase} - {r['feature']} = {r['value']:.3f} "
                f"(typical normal {r['normal_median']:.3f}, typical shill {r['shill_median']:.3f})."
            )
        for _, r in down.iterrows():
            lines.append(
                f"Lowered the suspicion score: {r['feature']} = {r['value']:.3f} is closer to normal "
                f"behaviour (typical normal {r['normal_median']:.3f})."
            )
        if not lines:
            lines.append(f"No single behaviour dominated this prediction (probability {proba:.3f}).")
        lines.append(
            "These are statistical associations learned by the model; a flag indicates behaviour "
            "that warrants investigation, not proof of shill bidding."
        )
        return lines


# --------------------------------------------------------------------- global
def native_importance(pipeline, features: list[str]) -> pd.Series | None:
    """Impurity importance (trees) or |coefficient| on standardised inputs (linear)."""
    model = _final_estimator(pipeline)
    if hasattr(model, "feature_importances_"):
        return pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    if hasattr(model, "coef_"):
        return pd.Series(np.abs(model.coef_[0]), index=features).sort_values(ascending=False)
    return None


def global_permutation_importance(pipeline, X: pd.DataFrame, y: pd.Series, features: list[str]) -> pd.DataFrame:
    """Permutation importance (drop in average precision) on held-out data."""
    result = permutation_importance(
        pipeline, X[features], y, scoring="average_precision", n_repeats=15,
        random_state=settings.RANDOM_SEED, n_jobs=-1,
    )
    return pd.DataFrame({
        "feature": features,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)


def make_global_figures(explainer: ModelExplainer, X_test: pd.DataFrame, perm: pd.DataFrame,
                        native: pd.Series | None, model_label: str) -> tuple[list[str], dict]:
    """Save permutation / native / SHAP importance figures. Returns paths and mean |SHAP|."""
    paths = []
    fig, ax = plt.subplots(figsize=(7, 4.5))
    p = perm.sort_values("importance_mean")
    ax.barh(p["feature"], p["importance_mean"], xerr=p["importance_std"], color="#4C78A8")
    ax.set_xlabel("Mean decrease in average precision when permuted")
    ax.set_title(f"Permutation importance (test auctions) - {model_label}")
    fig.tight_layout()
    path = settings.FIGURES_DIR / "explain_permutation_importance.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    paths.append(str(path.relative_to(settings.PROJECT_ROOT)))

    if native is not None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        n = native.sort_values()
        ax.barh(n.index, n.values, color="#72B7B2")
        ax.set_title(f"Model-native feature importance - {model_label}")
        fig.tight_layout()
        path = settings.FIGURES_DIR / "explain_native_importance.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        paths.append(str(path.relative_to(settings.PROJECT_ROOT)))

    mean_abs_shap = {}
    if explainer.method == "SHAP":
        try:
            import shap

            sample = X_test.sample(min(len(X_test), 600), random_state=settings.RANDOM_SEED)
            values = explainer.shap_values(sample)
            mean_abs_shap = dict(zip(explainer.features, np.abs(values).mean(axis=0).round(5).tolist()))
            plt.figure()
            shap.summary_plot(values, _transform(explainer.pipeline, sample), show=False, max_display=12)
            plt.title(f"SHAP summary - {model_label}")
            plt.tight_layout()
            path = settings.FIGURES_DIR / "explain_shap_summary.png"
            plt.savefig(path, dpi=130, bbox_inches="tight")
            plt.close("all")
            paths.append(str(path.relative_to(settings.PROJECT_ROOT)))
        except Exception as exc:
            logger.warning("SHAP summary plot failed: %s", exc)
    return paths, mean_abs_shap
