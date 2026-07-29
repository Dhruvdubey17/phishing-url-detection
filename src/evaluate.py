"""Evaluation of the trained pipelines on the class-balanced test split.

Run as ``python -m src.evaluate``. Writes ``reports/metrics.json``,
``reports/model_comparison.csv`` and the confusion-matrix, ROC and
feature-importance figures.

All headline numbers come from the 50/50 test split built by
``models.make_splits``. On a balanced split, accuracy is directly comparable to
F1 and a coin flip scores 50%, so none of the figures are inflated by a
dominant class.
"""

from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.data_loader import load_xy
from src.models import BUILDERS, MODELS_DIR, REPORTS_DIR, make_splits


def score(y_true, y_pred, y_proba) -> dict[str, float]:
    """Standard binary metrics, with phishing (label 1) as the positive class."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def _plot_confusion(matrices: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(1, len(matrices), figsize=(5.5 * len(matrices), 4.8))
    for ax, (name, cm) in zip(np.atleast_1d(axes), matrices.items()):
        ax.imshow(cm, cmap="Blues")
        total = cm.sum()
        for (i, j), v in np.ndenumerate(cm):
            ax.text(
                j, i, f"{v:,}\n{v / total * 100:.2f}%",
                ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black",
            )
        ax.set_title(name.replace("_", " ").title())
        ax.set_xlabel("predicted")
        ax.set_ylabel("actual")
        ax.set_xticks([0, 1], ["legitimate", "phishing"])
        ax.set_yticks([0, 1], ["legitimate", "phishing"])
    fig.suptitle("Confusion matrices — class-balanced test split")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "confusion_matrices.png", dpi=150)
    plt.close(fig)


def _plot_roc(curves: dict[str, tuple[np.ndarray, np.ndarray, float]]) -> None:
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for name, (fpr, tpr, auc) in curves.items():
        ax.plot(fpr, tpr, label=f"{name.replace('_', ' ').title()} (AUC = {auc:.5f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="chance")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC — class-balanced test split")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "roc_curves.png", dpi=150)
    plt.close(fig)


def _plot_importance(importances: dict[str, pd.Series], top_n: int = 20) -> None:
    fig, axes = plt.subplots(1, len(importances), figsize=(7.5 * len(importances), 6.5))
    for ax, (name, imp) in zip(np.atleast_1d(axes), importances.items()):
        top = imp.sort_values(ascending=False).head(top_n).iloc[::-1]
        ax.barh(top.index, top.to_numpy(), color="#2b6cb0")
        ax.set_title(f"{name.replace('_', ' ').title()} — top {top_n} features")
        ax.set_xlabel("importance")
        ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    X, y = load_xy()
    X_test, y_test = make_splits(X, y)["test"]
    print(f"test split: {len(X_test):,} rows, {y_test.mean() * 100:.2f}% phishing\n")

    metrics, matrices, curves, importances = {}, {}, {}, {}
    for name in BUILDERS:
        path = MODELS_DIR / f"{name}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing — run `python -m src.train` first")
        model = joblib.load(path)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics[name] = score(y_test, y_pred, y_proba)
        matrices[name] = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        curves[name] = (fpr, tpr, metrics[name]["roc_auc"])
        importances[name] = pd.Series(
            model.named_steps["clf"].feature_importances_, index=X.columns
        )

        print(f"=== {name}")
        for k, v in metrics[name].items():
            print(f"    {k:9s} = {v:.5f}")
        tn, fp, fn, tp = matrices[name].ravel()
        print(f"    confusion  tn={tn:,} fp={fp:,} fn={fn:,} tp={tp:,}\n")

    _plot_confusion(matrices)
    _plot_roc(curves)
    _plot_importance(importances)

    (REPORTS_DIR / "metrics.json").write_text(
        json.dumps(
            {
                "test_split": {"n": len(X_test), "phishing_fraction": float(y_test.mean())},
                "models": metrics,
                "confusion_matrices": {k: v.tolist() for k, v in matrices.items()},
            },
            indent=2,
        )
    )

    comparison = pd.DataFrame(metrics).T.round(5)
    comparison.index.name = "model"
    comparison.to_csv(REPORTS_DIR / "model_comparison.csv")
    pd.DataFrame(importances).round(6).rename_axis("feature").to_csv(
        REPORTS_DIR / "feature_importance.csv"
    )

    print(comparison.to_string())
    print(f"\nwrote metrics.json, model_comparison.csv, feature_importance.csv and 3 figures to reports/")


if __name__ == "__main__":
    main()
