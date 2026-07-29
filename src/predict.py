"""Inference entry point.

Scoring a URL must apply exactly the same canonicalization the training corpus
went through. Skipping it would feed the model a distribution it never saw — a
raw ``https://`` prefix, for instance, sets ``uses_https`` to 1 when every
training row had 0. This module is the only supported way to go from raw URL
strings to a prediction.

    python -m src.predict "paypal.secure-login.xyz/webscr" "github.com/torvalds"
"""

from __future__ import annotations

import sys

import joblib
import numpy as np
import pandas as pd

from src.data_loader import canonicalize
from src.features import extract_features_frame
from src.models import MODELS_DIR

DEFAULT_MODEL = "xgboost"


def load_model(name: str = DEFAULT_MODEL):
    """Load a persisted pipeline from ``models/``."""
    path = MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run `python -m src.train` first")
    return joblib.load(path)


def featurize(urls) -> pd.DataFrame:
    """Canonicalize raw URLs and extract the model's feature matrix."""
    series = pd.Series(list(urls), dtype="object")
    return extract_features_frame(canonicalize(series).fillna(""), n_jobs=1)


def predict_urls(model, urls) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(labels, phishing_probabilities)`` for an iterable of raw URLs."""
    X = featurize(urls)
    return model.predict(X), model.predict_proba(X)[:, 1]


def main(argv: list[str]) -> None:
    if not argv:
        print(__doc__)
        raise SystemExit(1)
    model = load_model()
    labels, probs = predict_urls(model, argv)
    for url, label, prob in zip(argv, labels, probs):
        verdict = "PHISHING  " if label == 1 else "legitimate"
        print(f"{verdict} p={prob:.4f}  {url}")


if __name__ == "__main__":
    main(sys.argv[1:])
