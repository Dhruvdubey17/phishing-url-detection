"""Ablation studies: which feature families matter, and how much data is enough.

Run as ``python -m src.ablation``. Writes ``reports/ablation_results.csv`` and
``reports/learning_curve.png``.

Two sweeps:

*Feature groups* — each family on its own, and each family removed from the full
set. Training in isolation shows how far a family gets alone; removing it shows
how much of that is unique rather than duplicated by the other families. Both
are needed, because correlated families each look strong alone and yet cost
nothing to drop.

*Training size* — 10 / 25 / 50 / 100% of the training split, to show whether the
corpus is large enough that more data would stop helping.

*Split strategy* — the same model measured on a random split and on a
domain-disjoint one, where no registrable domain appears on both sides. A random
split lets the model recognise domains it has already seen, which is a fair test
of "have I seen this before" and an unfair one of "is this new thing phishing".
The gap between the two is the part of the score that comes from memorising
domains rather than reading URL structure.

Configurations in the first two studies are retrained from scratch and scored on
the same class-balanced test split used for the headline metrics, so the numbers
sit on the same scale as ``reports/metrics.json``.
"""

from __future__ import annotations

import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

import numpy as np

from src.data_loader import load_dataset, load_xy
from src.evaluate import score
from src.features import FEATURE_GROUPS, _registered_domain
from src.models import BUILDERS, RANDOM_STATE, REPORTS_DIR, TEST_PER_CLASS, make_splits

SIZE_FRACTIONS = (0.10, 0.25, 0.50, 1.00)
DISJOINT_DOMAIN_FRACTION = 0.15


def domain_disjoint_split(urls, y, random_state: int = RANDOM_STATE):
    """Split by registrable domain, then balance the held-out side.

    Every URL under a given domain lands wholly in train or wholly in test, so a
    model cannot score a test row by recalling another row from the same domain.
    """
    registered = urls.str.split("/").str[0].str.split(".").map(_registered_domain)
    rng = np.random.RandomState(random_state)

    # numpy cannot shuffle a pandas arrow-backed array in place.
    domains = np.asarray(registered.unique(), dtype=object)
    rng.shuffle(domains)
    held_out = set(domains[: int(len(domains) * DISJOINT_DOMAIN_FRACTION)])
    in_test = registered.isin(held_out).to_numpy()

    labels = y.to_numpy()
    pools = [np.flatnonzero(in_test & (labels == k)) for k in (0, 1)]
    per_class = min(TEST_PER_CLASS, *(len(p) for p in pools))
    test_idx = np.concatenate([rng.permutation(p)[:per_class] for p in pools])

    mask = np.zeros(len(labels), dtype=bool)
    mask[test_idx] = True
    return np.flatnonzero(~in_test), np.flatnonzero(mask)


def _best_params() -> dict[str, dict]:
    """Reuse the tuned hyperparameters so ablation differences come from the
    ablated variable, not from a second, unrelated search."""
    path = REPORTS_DIR / "best_params.json"
    if not path.exists():
        print("best_params.json not found — falling back to builder defaults")
        return {name: {} for name in BUILDERS}
    saved = json.loads(path.read_text())
    return {name: saved.get(name, {}).get("best_params", {}) for name in BUILDERS}


def _run(name: str, params: dict, X_tr, y_tr, X_te, y_te) -> dict:
    started = time.perf_counter()
    model = BUILDERS[name](**params)
    model.fit(X_tr, y_tr)
    result = score(y_te, model.predict(X_te), model.predict_proba(X_te)[:, 1])
    result["fit_seconds"] = round(time.perf_counter() - started, 1)
    return result


def feature_configs() -> dict[str, list[str]]:
    """All-features, one-family-only, and leave-one-family-out column sets."""
    everything = [c for group in FEATURE_GROUPS.values() for c in group]
    configs = {"all_features": everything}
    for group, cols in FEATURE_GROUPS.items():
        configs[f"only_{group}"] = list(cols)
    for group in FEATURE_GROUPS:
        configs[f"without_{group}"] = [c for g, cols in FEATURE_GROUPS.items() if g != group for c in cols]
    return configs


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    X, y = load_xy()
    splits = make_splits(X, y)
    X_train, y_train = splits["train"]
    X_test, y_test = splits["test"]
    params = _best_params()
    rows = []

    print("=== feature-group ablation ===")
    for config, cols in feature_configs().items():
        for name in BUILDERS:
            res = _run(name, params[name], X_train[cols], y_train, X_test[cols], y_test)
            rows.append(
                {"study": "feature_group", "config": config, "model": name,
                 "n_features": len(cols), "train_rows": len(X_train), **res}
            )
            print(f"{config:22s} {name:14s} n_feat={len(cols):2d}  "
                  f"F1={res['f1']:.5f}  AUC={res['roc_auc']:.5f}  ({res['fit_seconds']}s)")

    print("\n=== training-size ablation (all features) ===")
    for fraction in SIZE_FRACTIONS:
        if fraction < 1.0:
            X_sub, _, y_sub, _ = train_test_split(
                X_train, y_train, train_size=fraction,
                stratify=y_train, random_state=RANDOM_STATE,
            )
        else:
            X_sub, y_sub = X_train, y_train
        for name in BUILDERS:
            res = _run(name, params[name], X_sub, y_sub, X_test, y_test)
            rows.append(
                {"study": "training_size", "config": f"{int(fraction * 100)}pct", "model": name,
                 "n_features": X.shape[1], "train_rows": len(X_sub), **res}
            )
            print(f"{int(fraction * 100):3d}% ({len(X_sub):>7,} rows) {name:14s} "
                  f"F1={res['f1']:.5f}  AUC={res['roc_auc']:.5f}  ({res['fit_seconds']}s)")

    print("\n=== split-strategy ablation (all features) ===")
    urls = load_dataset()["url"]
    dj_train, dj_test = domain_disjoint_split(urls, y)
    for name in BUILDERS:
        random_f1 = next(
            r["f1"] for r in rows
            if r["study"] == "training_size" and r["config"] == "100pct" and r["model"] == name
        )
        res = _run(
            name, params[name],
            X.iloc[dj_train], y.iloc[dj_train], X.iloc[dj_test], y.iloc[dj_test],
        )
        rows.append(
            {"study": "split_strategy", "config": "domain_disjoint", "model": name,
             "n_features": X.shape[1], "train_rows": len(dj_train), **res}
        )
        print(f"domain_disjoint  {name:14s} F1={res['f1']:.5f}  AUC={res['roc_auc']:.5f}  "
              f"(random split F1={random_f1:.5f}, gap={random_f1 - res['f1']:+.5f})")

    results = pd.DataFrame(rows)
    out = REPORTS_DIR / "ablation_results.csv"
    results.round(5).to_csv(out, index=False)
    print(f"\nwrote {out.relative_to(REPORTS_DIR.parent)}")

    curve = results[results["study"] == "training_size"]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for name, part in curve.groupby("model"):
        ax.plot(part["train_rows"], part["f1"], marker="o", label=name.replace("_", " ").title())
    ax.set_xscale("log")
    ax.set_xlabel("training rows (log scale)")
    ax.set_ylabel("F1 on balanced test split")
    ax.set_title("Learning curve")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "learning_curve.png", dpi=150)
    plt.close(fig)
    print(f"wrote reports/learning_curve.png")


if __name__ == "__main__":
    main()
