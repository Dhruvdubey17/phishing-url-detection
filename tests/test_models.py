"""Tests for the persisted model artifacts.

These require trained models. Run ``python -m src.train`` first; without the
artifacts the whole module skips rather than failing, so a fresh clone can still
run the rest of the suite.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from src.features import FEATURE_NAMES
from src.models import BUILDERS, MODELS_DIR
from src.predict import featurize, load_model, predict_urls

pytestmark = pytest.mark.artifacts

MODEL_NAMES = list(BUILDERS)

# Hosts chosen to be unambiguous: major sites nobody would flag, against
# hostnames combining an impersonated brand, a throwaway TLD and credential
# bait. Kept small and clear-cut on purpose.
LEGITIMATE = [
    "https://www.google.com/search?q=weather",
    "https://en.wikipedia.org/wiki/Phishing",
    "https://github.com/torvalds/linux",
    "https://www.bbc.co.uk/news",
    "https://stackoverflow.com/questions/tagged/python",
]
PHISHING = [
    "http://paypal-secure-login.verify-account.top/webscr?cmd=login",
    "http://appleid.apple.com.confirm-account.xyz/signin/update",
    "http://192.168.44.21:8080/secure/banking/login.php?account=verify",
    "http://update-your-account-now.duckdns.org/login/confirm",
    "http://facebook-security-alert.000webhostapp.com/verify/password",
]


@pytest.fixture(scope="module", params=MODEL_NAMES)
def model(request) -> Pipeline:
    path = MODELS_DIR / f"{request.param}.joblib"
    if not path.exists():
        pytest.skip(f"{path.name} not found — run `python -m src.train`")
    return load_model(request.param)


def test_artifact_loads_as_pipeline(model):
    assert isinstance(model, Pipeline)
    assert "clf" in model.named_steps


def test_model_expects_the_full_feature_schema(model):
    assert model.named_steps["clf"].n_features_in_ == len(FEATURE_NAMES)


def test_predict_shape_and_domain(model):
    urls = LEGITIMATE + PHISHING
    labels, probs = predict_urls(model, urls)
    assert labels.shape == (len(urls),)
    assert probs.shape == (len(urls),)
    assert set(np.unique(labels)) <= {0, 1}
    assert ((probs >= 0) & (probs <= 1)).all()


def test_predict_proba_rows_sum_to_one(model):
    proba = model.predict_proba(featurize(LEGITIMATE + PHISHING))
    assert proba.shape == (len(LEGITIMATE) + len(PHISHING), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-6)


def test_probability_agrees_with_hard_label(model):
    labels, probs = predict_urls(model, LEGITIMATE + PHISHING)
    np.testing.assert_array_equal(labels, (probs >= 0.5).astype(labels.dtype))


def test_obvious_examples_are_classified_correctly(model):
    labels, probs = predict_urls(model, LEGITIMATE + PHISHING)
    truth = np.array([0] * len(LEGITIMATE) + [1] * len(PHISHING))
    wrong = [
        f"{url} -> p={p:.3f} (expected {t})"
        for url, pred, p, t in zip(LEGITIMATE + PHISHING, labels, probs, truth)
        if pred != t
    ]
    assert not wrong, "misclassified obvious cases:\n" + "\n".join(wrong)


def test_prediction_is_deterministic(model):
    first, _ = predict_urls(model, PHISHING)
    second, _ = predict_urls(model, PHISHING)
    np.testing.assert_array_equal(first, second)


def test_phishing_scores_higher_than_legitimate_on_average(model):
    _, legit = predict_urls(model, LEGITIMATE)
    _, phish = predict_urls(model, PHISHING)
    assert phish.mean() > legit.mean()


def test_single_url_and_batch_agree(model):
    urls = LEGITIMATE + PHISHING
    batch, _ = predict_urls(model, urls)
    singles = np.concatenate([predict_urls(model, [u])[0] for u in urls])
    np.testing.assert_array_equal(batch, singles)
