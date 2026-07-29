"""Hyperparameter search and final training for both classifiers.

Run as ``python -m src.train``. Writes fitted pipelines to ``models/`` and the
chosen hyperparameters to ``reports/best_params.json``.

The search runs on a stratified subsample of the training split rather than all
811k rows. Ranking a dozen candidate configurations does not need the full
corpus — the ordering is stable well before that — and the subsample keeps the
whole search to a few minutes. The winning configuration is then refit on the
complete training split, so the persisted model has seen everything.
"""

from __future__ import annotations

import json
import time

import joblib
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split

from src.data_loader import load_xy
from src.models import (
    BUILDERS,
    MODELS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    SEARCH_SPACES,
    make_splits,
)
from src.evaluate import score

TUNE_SAMPLE = 150_000
SEARCH_ITERATIONS = 15
CV_FOLDS = 3


def tune(name: str, X_train, y_train) -> dict:
    """Randomized search over ``SEARCH_SPACES[name]``, scored by F1."""
    if len(X_train) > TUNE_SAMPLE:
        X_s, _, y_s, _ = train_test_split(
            X_train,
            y_train,
            train_size=TUNE_SAMPLE,
            stratify=y_train,
            random_state=RANDOM_STATE,
        )
    else:
        X_s, y_s = X_train, y_train

    search = RandomizedSearchCV(
        estimator=BUILDERS[name](),
        param_distributions=SEARCH_SPACES[name],
        n_iter=SEARCH_ITERATIONS,
        scoring="f1",
        cv=StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE,
        n_jobs=1,  # the estimators already use every core
        verbose=1,
        refit=False,
    )
    print(f"\n[{name}] searching {SEARCH_ITERATIONS} configs on {len(X_s):,} rows ...")
    started = time.perf_counter()
    search.fit(X_s, y_s)
    print(f"[{name}] search took {time.perf_counter() - started:.0f}s")
    print(f"[{name}] best CV F1 = {search.best_score_:.5f}")

    best = {k.removeprefix("clf__"): v for k, v in search.best_params_.items()}
    print(f"[{name}] best params = {json.dumps(best, default=str)}")
    return best


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    X, y = load_xy()
    splits = make_splits(X, y)
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]

    print(f"train={len(X_train):,}  val={len(X_val):,}  test={len(splits['test'][0]):,}")
    print(f"train class balance: {y_train.mean() * 100:.2f}% phishing (imbalanced, as collected)")
    print(f"test  class balance: {splits['test'][1].mean() * 100:.2f}% phishing (balanced by design)")

    summary = {}
    for name in BUILDERS:
        best = tune(name, X_train, y_train)

        print(f"[{name}] refitting on all {len(X_train):,} training rows ...")
        started = time.perf_counter()
        model = BUILDERS[name](**best)
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - started

        val = score(y_val, model.predict(X_val), model.predict_proba(X_val)[:, 1])
        print(f"[{name}] validation F1 = {val['f1']:.5f}  ROC-AUC = {val['roc_auc']:.5f}")

        path = MODELS_DIR / f"{name}.joblib"
        joblib.dump(model, path, compress=3)
        summary[name] = {
            "best_params": best,
            "fit_seconds": round(fit_seconds, 1),
            "validation": val,
            "artifact": str(path.relative_to(MODELS_DIR.parent)),
        }
        print(f"[{name}] saved {path.relative_to(MODELS_DIR.parent)}")

    summary["_meta"] = {
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(splits["test"][0]),
        "n_features": X.shape[1],
        "random_state": RANDOM_STATE,
    }
    out = REPORTS_DIR / "best_params.json"
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote {out.relative_to(REPORTS_DIR.parent)}")


if __name__ == "__main__":
    main()
