"""Dataset acquisition, canonicalization and merging.

Source: `flwrlabs/fed-phishing-urls` on the Hugging Face Hub (Apache-2.0), which
aggregates two public phishing URL corpora: `ealvaradob/phishing-dataset`
(urls.json) and `kmack/Phishing_urls`. We use only the `url` and `label`
columns; the `client_id` column exists for federated-learning experiments and is
irrelevant here.

The two upstream corpora were collected differently, and that difference leaks
into the raw strings: 33.6% of phishing URLs carry an explicit `scheme://`
prefix versus 10.7% of benign ones. A single rule on "does this string contain
'://'" classifies the raw corpus with 61.3% accuracy while saying nothing at all
about phishing. `canonicalize` removes that artifact so the models are forced to
learn from the parts of the URL that actually matter.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_PATH = ROOT / "data" / "processed" / "dataset.csv"
FEATURES_PATH = ROOT / "data" / "processed" / "features.parquet"

_HF_BASE = "https://huggingface.co/datasets/flwrlabs/fed-phishing-urls/resolve/main/data"
_RAW_FILES = ("train-00000-of-00001.parquet", "test-00000-of-00001.parquet")


def download_raw() -> None:
    """Fetch the parquet shards into data/raw/ if they are not already there."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name in _RAW_FILES:
        dest = RAW_DIR / name
        if dest.exists():
            continue
        print(f"downloading {name} ...")
        urllib.request.urlretrieve(f"{_HF_BASE}/{name}", dest)


def canonicalize(urls: pd.Series) -> pd.Series:
    """Strip collection artifacts so they cannot be mistaken for phishing signal.

    Three normalizations, all vectorized:

    1. Drop the ``scheme://`` prefix. Whether a scheme survived collection is a
       property of the upstream feed, not of the URL, and it is by far the
       strongest artifact in the corpus.
    2. Drop trailing slashes. ``example.com/`` and ``example.com`` are the same
       resource, but the upstream feeds disagreed on which form to store. All of
       them go, not just one: a handful of rows end in ``//``, and stripping a
       single slash would leave the artifact in place for exactly those rows.
    3. Lowercase the host only. Hostnames are case-insensitive per RFC 3986, so
       this loses nothing; path and query case is preserved because it is real
       signal (benign wiki-style URLs genuinely use CamelCase paths).
    """
    s = urls.astype("string").str.strip()
    s = s.str.replace(r"^[A-Za-z][A-Za-z0-9+.\-]*://", "", regex=True)
    s = s.str.replace(r"/+$", "", regex=True)
    host = s.str.split("/", n=1).str[0].str.lower()
    rest = s.str.split("/", n=1).str[1]
    return host + rest.radd("/").fillna("")


def load_raw() -> pd.DataFrame:
    """Read and concatenate the raw parquet shards into one [url, label] frame."""
    download_raw()
    frames = [pd.read_parquet(RAW_DIR / name, columns=["url", "label"]) for name in _RAW_FILES]
    return pd.concat(frames, ignore_index=True)


def build_dataset(save: bool = True) -> pd.DataFrame:
    """Canonicalize, drop duplicates and conflicting labels, and persist the result.

    A URL that appears under both labels after canonicalization is ambiguous and
    is dropped entirely rather than resolved arbitrarily; keeping it would put
    the same feature vector on both sides of the decision boundary.
    """
    raw = load_raw()
    df = pd.DataFrame({"url": canonicalize(raw["url"]), "label": raw["label"].astype("int8")})
    df = df[df["url"].str.len() > 0].dropna(subset=["url"])

    before = len(df)
    df = df.drop_duplicates(subset=["url", "label"])
    deduped = before - len(df)

    conflicting = df["url"].duplicated(keep=False)
    n_conflicting = int(conflicting.sum())
    df = df[~conflicting].reset_index(drop=True)
    df["url"] = df["url"].astype(str)

    print(f"raw rows                : {len(raw):,}")
    print(f"exact duplicates dropped: {deduped:,}")
    print(f"label conflicts dropped : {n_conflicting:,}")
    print(f"final rows              : {len(df):,}")
    counts = df["label"].value_counts().sort_index()
    for label, n in counts.items():
        name = "phishing" if label == 1 else "legitimate"
        print(f"  label={label} ({name:10s}): {n:>9,}  ({n / len(df) * 100:.2f}%)")

    if save:
        PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(PROCESSED_PATH, index=False)
        print(f"wrote {PROCESSED_PATH.relative_to(ROOT)}")
    return df


def load_dataset() -> pd.DataFrame:
    """Load the processed dataset, building it first if it is missing."""
    if not PROCESSED_PATH.exists():
        return build_dataset()
    return pd.read_csv(PROCESSED_PATH, dtype={"url": str, "label": "int8"})


def load_xy(rebuild: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    """Return the feature matrix and label vector, caching features to parquet.

    Extraction over a million URLs takes about a minute, and training, ablation
    and evaluation all need the same matrix, so it is computed once and reused.
    """
    from src.features import FEATURE_NAMES, extract_features_frame

    df = load_dataset()
    y = df["label"].astype("int8")

    if FEATURES_PATH.exists() and not rebuild:
        cached = pd.read_parquet(FEATURES_PATH)
        if len(cached) == len(df) and list(cached.columns) == list(FEATURE_NAMES):
            return cached, y
        print("feature cache is stale, rebuilding")

    print(f"extracting {len(FEATURE_NAMES)} features from {len(df):,} URLs ...")
    X = extract_features_frame(df["url"], n_jobs=1)
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    X.to_parquet(FEATURES_PATH, index=False)
    print(f"wrote {FEATURES_PATH.relative_to(ROOT)}")
    return X, y


if __name__ == "__main__":
    build_dataset()
    load_xy(rebuild=True)
