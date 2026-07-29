"""Model definitions, data splitting and hyperparameter search spaces.

Both classifiers are wrapped in a ``Pipeline``. There is no scaler in it: both
are axis-aligned tree ensembles, whose splits are invariant to any monotone
rescaling of a feature, so standardizing would cost time and change nothing. The
Pipeline is kept anyway because it fixes the interface of the persisted artifact
(a single object that goes from raw feature frame to prediction) and because any
preprocessing added later lands inside the cross-validation fold, where it
cannot leak statistics from validation rows into training.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

RANDOM_STATE = 42
TEST_PER_CLASS = 50_000
VAL_FRACTION = 0.10


def build_rf(**overrides) -> Pipeline:
    """Random forest: strong on tabular features, and its variance across
    bootstrap samples makes it a fair baseline for the boosted comparison."""
    params = dict(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    params.update(overrides)
    return Pipeline([("clf", RandomForestClassifier(**params))])


def build_xgb(**overrides) -> Pipeline:
    """Gradient-boosted trees. ``hist`` is used because exact split search over
    a million rows is needlessly slow for a 35-column matrix."""
    params = dict(
        n_estimators=400,
        max_depth=8,
        learning_rate=0.2,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=1,
        gamma=0.0,
        tree_method="hist",
        eval_metric="logloss",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    params.update(overrides)
    return Pipeline([("clf", XGBClassifier(**params))])


BUILDERS = {"random_forest": build_rf, "xgboost": build_xgb}

# Prefixed with the pipeline step name so these feed RandomizedSearchCV directly.
SEARCH_SPACES = {
    "random_forest": {
        "clf__n_estimators": randint(150, 400),
        "clf__max_depth": [None, 20, 30, 40],
        "clf__min_samples_split": randint(2, 12),
        "clf__min_samples_leaf": randint(1, 5),
        "clf__max_features": ["sqrt", "log2", 0.4],
    },
    "xgboost": {
        "clf__n_estimators": randint(200, 600),
        "clf__max_depth": randint(6, 14),
        "clf__learning_rate": loguniform(0.03, 0.4),
        "clf__subsample": uniform(0.6, 0.4),
        "clf__colsample_bytree": uniform(0.6, 0.4),
        "clf__min_child_weight": randint(1, 10),
        "clf__gamma": uniform(0.0, 0.4),
    },
}


def make_splits(
    X: pd.DataFrame, y: pd.Series, random_state: int = RANDOM_STATE
) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """Carve the corpus into train / validation / class-balanced test.

    The test set is built first and holds exactly ``TEST_PER_CLASS`` rows of each
    label, so headline precision, recall and F1 are read off a 50/50 split and
    are not flattered by whichever class happens to dominate the corpus. What is
    left over keeps its natural 55/45 imbalance for training, which is mild
    enough that neither resampling nor class weights are warranted; the
    validation split is stratified so it mirrors the training distribution.

    Splitting is a deterministic function of ``random_state``, so training and
    evaluation reconstruct an identical test set without shipping index files.
    """
    rng = np.random.RandomState(random_state)
    per_class_pools = [np.flatnonzero(y.to_numpy() == label) for label in (0, 1)]
    # Never let the test split swallow the corpus: on a small input (the tests
    # use a few thousand synthetic rows) a fixed 50k per class would leave
    # nothing to train on.
    per_class = min(TEST_PER_CLASS, *(len(pool) // 2 for pool in per_class_pools))
    test_idx = np.concatenate([rng.permutation(pool)[:per_class] for pool in per_class_pools])
    mask = np.zeros(len(y), dtype=bool)
    mask[test_idx] = True

    X_test, y_test = X[mask], y[mask]
    X_rest, y_rest = X[~mask], y[~mask]

    X_train, X_val, y_train, y_val = train_test_split(
        X_rest,
        y_rest,
        test_size=VAL_FRACTION,
        stratify=y_rest,
        random_state=random_state,
    )
    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }
