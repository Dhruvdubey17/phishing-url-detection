"""End-to-end integration tests over a small slice of the real corpus.

The unit tests check pieces in isolation; these check that the pieces still fit
together — the same path a caller takes from a raw URL string to a prediction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_loader import PROCESSED_PATH, canonicalize
from src.features import FEATURE_NAMES, extract_features_frame
from src.models import BUILDERS, MODELS_DIR, TEST_PER_CLASS, make_splits
from src.predict import featurize, load_model, predict_urls

SAMPLE_SIZE = 2_000


@pytest.fixture(scope="module")
def sample() -> pd.DataFrame:
    if not PROCESSED_PATH.exists():
        pytest.skip("data/processed/dataset.csv not found — run `python -m src.data_loader`")
    df = pd.read_csv(PROCESSED_PATH, dtype={"url": str, "label": "int8"}, nrows=50_000)
    return df.sample(SAMPLE_SIZE, random_state=0).reset_index(drop=True)


def test_raw_slice_flows_through_feature_extraction(sample):
    X = extract_features_frame(sample["url"], n_jobs=1)
    assert X.shape == (len(sample), len(FEATURE_NAMES))
    assert list(X.columns) == list(FEATURE_NAMES)


def test_no_nan_or_inf_leaks_through(sample):
    X = extract_features_frame(sample["url"], n_jobs=1)
    values = X.to_numpy()
    assert not np.isnan(values).any(), "NaN reached the feature matrix"
    assert np.isfinite(values).all(), "inf reached the feature matrix"


def test_features_are_all_numeric(sample):
    X = extract_features_frame(sample["url"], n_jobs=1)
    assert all(np.issubdtype(dtype, np.floating) for dtype in X.dtypes)


def test_canonicalize_is_idempotent(sample):
    once = canonicalize(sample["url"])
    twice = canonicalize(once)
    pd.testing.assert_series_equal(once, twice)


def test_featurize_matches_direct_extraction_on_canonical_input(sample):
    urls = sample["url"].tolist()
    direct = extract_features_frame(pd.Series(urls), n_jobs=1)
    through_predict = featurize(urls)
    # Already-canonical URLs must survive the inference path unchanged.
    np.testing.assert_allclose(through_predict.to_numpy(), direct.to_numpy(), rtol=1e-6)


def test_splits_are_disjoint_and_test_is_balanced():
    n = 40_000
    X = pd.DataFrame(
        np.random.RandomState(0).rand(n, len(FEATURE_NAMES)), columns=list(FEATURE_NAMES)
    )
    y = pd.Series(np.random.RandomState(1).randint(0, 2, n), dtype="int8")
    splits = make_splits(X, y)

    indices = {k: set(v[0].index) for k, v in splits.items()}
    assert not indices["train"] & indices["test"]
    assert not indices["train"] & indices["val"]
    assert not indices["val"] & indices["test"]
    assert sum(len(v) for v in indices.values()) == n

    y_test = splits["test"][1]
    assert (y_test == 0).sum() == (y_test == 1).sum(), "test split must be class-balanced"


def test_splits_are_reproducible():
    n = 5_000
    X = pd.DataFrame(
        np.random.RandomState(2).rand(n, len(FEATURE_NAMES)), columns=list(FEATURE_NAMES)
    )
    y = pd.Series(np.random.RandomState(3).randint(0, 2, n), dtype="int8")
    first = make_splits(X, y)["test"][0].index
    second = make_splits(X, y)["test"][0].index
    np.testing.assert_array_equal(first, second)


@pytest.mark.artifacts
@pytest.mark.parametrize("name", list(BUILDERS))
def test_end_to_end_slice_predicts_without_error(sample, name):
    if not (MODELS_DIR / f"{name}.joblib").exists():
        pytest.skip(f"{name}.joblib not found — run `python -m src.train`")
    model = load_model(name)
    labels, probs = predict_urls(model, sample["url"].tolist())

    assert labels.shape == (len(sample),)
    assert np.isfinite(probs).all()
    assert set(np.unique(labels)) <= {0, 1}
    accuracy = (labels == sample["label"].to_numpy()).mean()
    assert accuracy > 0.75, f"{name} scored {accuracy:.3f} on a random corpus slice"


@pytest.mark.artifacts
def test_raw_uncanonicalized_urls_do_not_break_inference():
    """Inference must accept URLs in the wild form, with scheme and trailing slash."""
    if not (MODELS_DIR / "xgboost.joblib").exists():
        pytest.skip("xgboost.joblib not found — run `python -m src.train`")
    model = load_model("xgboost")
    raw = ["HTTPS://WWW.Example.COM/Path/", "http://1.2.3.4:99/a", "", "no-scheme.com"]
    labels, probs = predict_urls(model, raw)
    assert labels.shape == (len(raw),)
    assert np.isfinite(probs).all()
