# Phishing URL Detection

Classifies a URL as phishing or legitimate from the URL string alone — no page
fetch, no DNS, no WHOIS, no reputation lookup. Everything the model sees is
derived by parsing and measuring the string itself.

The constraint is the point. A detector that needs only the string can run
inline on a mail gateway or a proxy, in front of the click rather than after it,
and it cannot be defeated by an attacker who simply takes the landing page down
before anyone inspects it. The cost is that a phishing page hosted on a
compromised legitimate domain is invisible to it — see [Limitations](#limitations).

Two gradient-free tree ensembles are trained and compared: a random forest and
XGBoost, both tuned by randomized search with cross-validation, both scored on a
class-balanced held-out split.

## Results

Headline metrics, measured on a held-out split of 100,000 URLs balanced exactly
50/50 between the classes. Phishing is the positive class.

| model | accuracy | precision | recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest | 0.8945 | 0.9222 | 0.8617 | 0.8909 | 0.9689 |
| **XGBoost** | **0.8980** | **0.9345** | 0.8561 | **0.8936** | **0.9724** |

Confusion matrices on the same split (50,000 legitimate / 50,000 phishing):

| model | true neg | false pos | false neg | true pos |
|---|---|---|---|---|
| Random Forest | 46,363 | 3,637 | 6,913 | 43,087 |
| XGBoost | 46,999 | 3,001 | 7,195 | 42,805 |

XGBoost wins on precision and AUC; the random forest catches slightly more
phishing at the cost of ~600 more false positives. For a mail gateway, where a
false positive blocks a legitimate message, XGBoost's higher precision is the
better trade, and it is the default in `src/predict.py`.

Both models are noticeably more precise than sensitive: they are conservative,
and the errors concentrate in phishing URLs that are structurally
indistinguishable from ordinary ones. Raising recall means moving the decision
threshold and giving up precision; `predict_proba` is exposed so callers can pick
their own operating point rather than being stuck with 0.5.

Generated artifacts live in `reports/`: `metrics.json`, `model_comparison.csv`,
`feature_importance.csv`, `confusion_matrices.png`, `roc_curves.png`,
`feature_importance.png`, `ablation_results.csv`, `learning_curve.png`.

## Dataset

**Source:** [`flwrlabs/fed-phishing-urls`](https://huggingface.co/datasets/flwrlabs/fed-phishing-urls)
on the Hugging Face Hub. **License: Apache-2.0.** It aggregates two public
corpora of real, in-the-wild URLs:

- [`ealvaradob/phishing-dataset`](https://huggingface.co/datasets/ealvaradob/phishing-dataset) (`urls.json`)
- [`kmack/Phishing_urls`](https://huggingface.co/datasets/kmack/Phishing_urls)

This is real collected data, not synthetic. Only the `url` and `label` columns
are used; the upstream `client_id` column exists for federated-learning
experiments and is irrelevant here.

| stage | rows |
|---|---|
| raw (train + test shards) | 1,132,306 |
| exact duplicates removed | −130,479 |
| conflicting labels removed | −54 |
| **final** | **1,001,773** |

Final class balance: **551,867 legitimate (55.09%)** and **449,906 phishing (44.91%)**.

A URL that survived canonicalization under both labels is genuinely ambiguous,
so it is dropped rather than resolved arbitrarily — keeping it would place an
identical feature vector on both sides of the decision boundary.

### Canonicalization, and why it costs accuracy on paper

The two upstream corpora were collected differently, and that difference leaked
into the raw strings. Before any cleaning:

| trivial rule | phishing | legitimate | accuracy of that rule alone |
|---|---|---|---|
| string contains `://` | 33.6% | 10.7% | **61.3%** |
| string ends with `/` | 33.9% | 23.5% | 55.1% |

The majority-class baseline is 55.1%. So "does this string carry a scheme
prefix" alone classifies the raw corpus at 61.3% — while saying nothing
whatsoever about phishing. It records which upstream feed a row came from.

`data_loader.canonicalize` removes these artifacts before anything else runs:

1. **Strip the `scheme://` prefix.** Whether a scheme survived collection is a
   property of the feed, not of the URL.
2. **Strip trailing slashes** — all of them, since a handful of rows end in `//`
   and removing only one would leave the artifact in place for exactly those.
3. **Lowercase the host only.** Hostnames are case-insensitive per RFC 3986, so
   nothing is lost; path and query case is preserved because it is real signal
   (benign wiki-style URLs genuinely use CamelCase paths).

After canonicalization the scheme rule collapses to 55.2% — the baseline. This
deliberately gives up roughly 11 points of "free" accuracy. The reported numbers
are lower than they would otherwise be, and they measure phishing rather than
data provenance. Canonicalization also exposed 130,479 duplicates that scheme
and slash differences had been hiding.

One consequence is visible in `reports/feature_importance.csv`: `uses_https` has
an importance of exactly **0.0**. Every training row has the scheme stripped, so
the feature is constant. It is kept because it is computed correctly for callers
who pass real URLs at inference time, and reporting it as dead weight here is
more useful than quietly dropping it.

## Features

47 features in three families, all computed from the string. `FEATURE_GROUPS` in
`src/features.py` defines the families, and the ablation study measures each
one's contribution.

### Lexical (20) — size and character composition

| feature | rationale |
|---|---|
| `url_length`, `hostname_length`, `path_length`, `query_length` | Phishing URLs pad length to push the real host out of a truncated display. |
| `num_digits`, `num_letters`, `num_special_chars` | Generated hostnames skew toward digits and punctuation. |
| `num_dots`, `num_hyphens`, `num_slashes` | Cheap structural proxies that survive obfuscation. |
| `num_subdomains`, `num_dots_in_domain` | Deep nesting puts a trusted-looking word in front of a domain the attacker owns. |
| `has_at_symbol` | `@` makes everything left of it userinfo, hiding the real host behind a familiar brand. |
| `has_hyphen_in_domain` | `paypal-secure.com` style composition. |
| `tld_length` | Long TLDs skew toward the cheap new gTLDs. |
| `host_digit_count`, `host_hyphen_count` | Digits and hyphens in the host specifically, where they matter more than in the path. |
| `num_host_tokens`, `longest_host_token` | Token shape separates generated names from chosen ones. |
| `subdomain_length` | Where `secure-login-yourbank-verify` gets parked in front of an attacker-owned domain. |

### Structural (13) — the shape of the parsed URL

| feature | rationale |
|---|---|
| `uses_https` | Scheme, when present. Constant on this corpus; see above. |
| `has_port` | An explicit port suggests a service that is not standard hosting. |
| `has_ip_host` | A literal address means no domain registered and no certificate name to match. |
| `num_path_segments`, `longest_path_segment` | Deep or padded paths. |
| `num_query_params` | Credential-harvest URLs carry campaign and target parameters. |
| `has_fragment`, `has_userinfo` | Both are used to push the real host out of view. |
| `has_double_slash_redirect` | `site.com/redirect//evil.com`, the classic open-redirect/path-confusion trick. |
| `path_to_url_ratio` | Distinguishes bare hostnames from deep paths. |
| `has_file_extension`, `has_executable_extension` | A link landing directly on an installer or archive is malware delivery. |
| `num_encoded_chars` | Percent-encoding hides keywords from naive scanners. |

### Suspicion heuristics (14) — known tradecraft

| feature | rationale |
|---|---|
| `is_shortener` | Shorteners hide the destination. Pattern match against known hosts, no network call. |
| `num_suspicious_keywords` | `secure`, `verify`, `login`, `account`, `confirm`… — words that provoke credential entry. |
| `has_suspicious_tld` | TLDs that are cheap or free and consistently over-represented in abuse feeds. |
| `brand_domain_mismatch` | The brand appears in the URL but is not the registrable domain: `paypal.secure-login.xyz` trips it, `paypal.com/login` does not. |
| `is_free_hosting` | Free hosting and dynamic DNS, where the registrable domain belongs to the provider and identifies nobody. |
| `tld_rank` | TLD position among the common web; the long tail shares one bucket. |
| `digit_letter_ratio`, `vowel_ratio_domain` | Composition ratios that separate words from generated strings. |
| `domain_entropy`, `url_entropy` | Algorithmically generated names spread characters more evenly than names a human chose. |
| `longest_consecutive_run` | Runs like `aaaaaa` or `------` mark generated or padded hostnames. |
| `longest_consonant_run` | Unpronounceable clusters are the clearest cheap gibberish signal. |
| `has_hex_token` | Long hex strings are session tokens and generated identifiers. |
| `has_punycode` | `xn--` encodes non-ASCII, how homograph lookalikes reach the wire. |

### Implementation notes

`extract_features(url) -> dict` is the single implementation; `extract_features_frame`
batches it and the test suite asserts the two agree, so there is no second copy
of the logic to drift. Extraction runs over the full 1,001,773-row corpus in
about 58 seconds and is cached to `data/processed/features.parquet`.

Parsing is done by hand rather than with `urllib.parse`, for two reasons:
`urlsplit` reads a scheme-less `example.com:8080/x` as having scheme
`example.com`, and its `.port`/`.hostname` accessors raise `ValueError` on
malformed input this corpus contains plenty of. Feature extraction must never
raise — every input, including `""`, `None` and megabyte-long junk, returns a
complete feature dict.

Registrable-domain extraction honours multi-label public suffixes, so
`www.bbc.co.uk` resolves to `bbc.co.uk` rather than `co.uk`. Without this every
domain-level feature is computed on a string the registrant never chose. The
bundled suffix list covers the multi-label suffixes present in this corpus, not
the full public suffix list — see [Limitations](#limitations).

## Models

Both classifiers are wrapped in a scikit-learn `Pipeline`. There is deliberately
**no scaler**: both are axis-aligned tree ensembles whose splits are invariant to
any monotone rescaling, so standardizing would cost time and change nothing. The
`Pipeline` is kept because it fixes the interface of the persisted artifact and
because any preprocessing added later lands inside the cross-validation fold,
where it cannot leak validation statistics into training.

### Splits

| split | rows | phishing |
|---|---|---|
| train | 811,595 | 44.35% |
| validation | 90,178 | 44.35% |
| test | 100,000 | 50.00% |

The test split is carved out first and holds exactly 50,000 rows of each label,
so headline precision, recall and F1 are read off a 50/50 split where a coin flip
scores 50% and accuracy is directly comparable to F1.

What remains keeps its natural 55/45 imbalance for training. That ratio is mild
enough that resampling and class weights are not warranted — neither improved
validation F1 in testing, and both cost calibration. The validation split is
stratified so it mirrors the training distribution.

Splitting is a deterministic function of the random seed, so training and
evaluation reconstruct an identical test set without shipping index files.

### Tuning

`RandomizedSearchCV`, 15 configurations, 3-fold stratified CV, scored by F1. The
search runs on a stratified 150,000-row subsample of the training split: ranking
a dozen candidates does not need all 811k rows, the ordering is stable well
before that, and the subsample keeps each search to roughly 8 minutes. The
winning configuration is then refit on the complete training split.

Selected hyperparameters (also in `reports/best_params.json`):

| | Random Forest | XGBoost |
|---|---|---|
| n_estimators | 364 | 544 |
| max_depth | None | 12 |
| learning_rate | — | 0.0432 |
| min_samples_split / min_child_weight | 4 | 1 |
| min_samples_leaf | 2 | — |
| max_features / colsample_bytree | 0.4 | 0.712 |
| subsample | — | 0.995 |
| gamma | — | 0.217 |

## Ablation studies

`python -m src.ablation` runs three sweeps, every configuration retrained from
scratch and scored on the same class-balanced test split as the headline
metrics. Full output in `reports/ablation_results.csv`.

### Feature groups

Each family alone, and each family removed. Both views are needed: correlated
families each look strong in isolation and yet cost nothing to drop.

| configuration | features | RF F1 | XGB F1 | XGB ROC-AUC |
|---|---|---|---|---|
| all features | 47 | 0.8909 | **0.8936** | 0.9724 |
| only lexical | 20 | 0.8427 | 0.8428 | 0.9429 |
| only structural | 13 | 0.7904 | 0.7822 | 0.8299 |
| only suspicion | 14 | 0.8473 | 0.8406 | 0.9424 |
| without lexical | 27 | 0.8804 | 0.8787 | 0.9650 |
| without structural | 34 | 0.8892 | 0.8917 | 0.9714 |
| without suspicion | 33 | 0.8463 | 0.8458 | 0.9456 |

**The suspicion heuristics carry the model.** Removing them costs 4.8 F1 points
(0.8936 → 0.8458), by far the largest drop. This matches the feature
importances, where `num_suspicious_keywords`, `has_suspicious_tld` and
`brand_domain_mismatch` are the top three for XGBoost and account for ~41% of
its total importance. Hand-written domain knowledge about phishing tradecraft
beats generic string statistics here.

**The structural family is nearly redundant.** It is the weakest alone (0.7822)
and removing it costs 0.2 points — almost all of its signal is already carried
by the other two. Its features are mostly binary and fire very rarely:
`has_userinfo` on 0.003% of rows, `has_punycode` on 0.065%, `has_port` on 0.189%,
`has_fragment` on 0.204%. A split has almost nothing to work with. The family
earns its place only marginally.

**No family is sufficient alone.** The best single family reaches 0.847 against
0.894 for the full set, so the three are genuinely complementary rather than
three views of one signal.

### Training set size

| training rows | RF F1 | XGB F1 | XGB ROC-AUC |
|---|---|---|---|
| 81,159 (10%) | 0.8715 | 0.8793 | 0.9646 |
| 202,898 (25%) | 0.8800 | 0.8865 | 0.9686 |
| 405,797 (50%) | 0.8858 | 0.8902 | 0.9707 |
| 811,595 (100%) | 0.8909 | 0.8936 | 0.9724 |

Returns are steadily diminishing: F1 rises roughly logarithmically with corpus
size. A tenfold increase in training data buys 1.4 F1 points, and the final
doubling (406k → 812k rows) buys 0.34. Extrapolating that trend, another
tenfold — 8 million URLs — would be worth on the order of one more point.

So **more data of the same kind is not the binding constraint**; the ceiling is
what 47 numeric features can express about a URL string. Plotted in
`reports/learning_curve.png` (note the zoomed y-axis — the whole range spans
about two F1 points).

### Split strategy — how much is memorization

The headline split is random, so a domain in the test set usually also appears
in training. The domain-disjoint split holds out entire registrable domains, so
every test URL sits on a domain the model has never seen.

| split | RF F1 | XGB F1 | XGB ROC-AUC |
|---|---|---|---|
| random (headline) | 0.8909 | 0.8936 | 0.9724 |
| domain-disjoint | 0.8406 | 0.8500 | 0.9298 |
| **gap** | **−0.0503** | **−0.0436** | −0.0426 |

**About 4–5 F1 points of the headline number come from recognizing domains the
model has already seen.** That is a legitimate capability — recognizing
known-bad infrastructure is useful — but it is not the same as judging an
unfamiliar URL, and the two should not be reported as one number.

The honest expectation for a URL on a domain the model has never encountered is
**F1 ≈ 0.85, ROC-AUC ≈ 0.93**, not the headline figure. Any URL-only detector
evaluated on a random split is quoting a number that includes this effect,
whether or not it is measured.

## Reproducing

Requires Python 3.10+ (developed on 3.12).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.data_loader          # download, canonicalize, cache features (~3 min)
python -m src.train                # tune and fit both models (~20 min)
python -m src.evaluate             # metrics, comparison table, figures
python -m src.ablation             # ablation studies and learning curve
pytest -v                          # full suite
```

Each step is independently re-runnable. `src.data_loader` downloads the parquet
shards into `data/raw/` on first run and skips the download afterwards; the
feature cache in `data/processed/` is rebuilt automatically if it goes stale.
Large generated files are gitignored and reproduced by the commands above.

Scoring URLs with a trained model:

```bash
python -m src.predict "paypal.secure-login.xyz/webscr" "github.com/torvalds/linux"
```

```
PHISHING   p=0.9999  paypal.secure-login.xyz/webscr
legitimate p=0.1427  github.com/torvalds/linux
```

In code:

```python
from src.predict import load_model, predict_urls

model = load_model("xgboost")            # or "random_forest"
labels, probabilities = predict_urls(model, ["http://example.com/login"])
```

Always go through `src.predict`. It applies the same canonicalization the
training corpus went through; feeding a raw `https://` URL straight to the model
would set `uses_https` to 1 when every training row had 0.

## Layout

```
src/data_loader.py   download, canonicalize, dedupe, cache features
src/features.py      the 47 features and batch extraction
src/models.py        pipelines, search spaces, split construction
src/train.py         randomized search and final fit
src/evaluate.py      metrics, comparison table, figures
src/ablation.py      feature-group, training-size and split-strategy studies
src/predict.py       inference entry point
tests/               feature unit tests, model artifact tests, integration tests
reports/             generated metrics, tables and figures
```

## Limitations

**A compromised legitimate domain is invisible to this approach.** A phishing
page served from a hacked WordPress install on an ordinary company domain has a
perfectly ordinary URL. No amount of string analysis will catch it. This is the
fundamental ceiling of URL-only detection, and it is why this belongs in a
layered system rather than on its own.

**Much of the achievable signal is domain identity, not URL structure.** A
classifier given only the registrable domain and allowed to memorize the corpus
would reach about 98% accuracy, because labels concentrate hard by domain —
`duckdns.org` is 99.98% phishing in this data, `wikipedia.org` is ~0%. That is
recall of known-bad infrastructure, not generalization. The
[split-strategy ablation](#ablation-studies) measures the gap directly by holding
out entire domains.

**The blocklists age.** `SUSPICIOUS_TLDS`, `SHORTENERS`, `FREE_HOSTING` and
`TARGETED_BRANDS` are static and reflect tradecraft at the time of writing. They
need periodic review; a campaign on a TLD or provider not on these lists loses
those features' contribution.

**The bundled public suffix list is partial.** It covers the multi-label
suffixes present in this corpus, not the full list. An unusual suffix outside it
resolves to the wrong registrable domain and degrades the domain-level features
for that URL.

**The corpus is a snapshot.** Phishing URL conventions drift as campaigns and
registrars change. Performance on URLs collected today will be lower than these
numbers, and the model needs periodic retraining against a fresh feed.

**Benign phishing-adjacent URLs are hard.** Security vendors, researchers and
URL scanners publish links that look exactly like the phishing they describe.
These are a real share of the false positives.

**The label is binary and the corpus mixes intents.** Credential harvesting and
malware delivery both sit on the phishing side. `has_executable_extension`
carries real weight partly because of this, so the model is better described as
a malicious-URL detector than a phishing-specific one.
