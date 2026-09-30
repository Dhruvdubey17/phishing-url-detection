# Phishing URL Detection

This project classifies a URL as phishing or legitimate by looking at the URL
string and nothing else. It never fetches the page, queries DNS, runs WHOIS, or
checks a reputation service. Every feature comes from parsing the string.

I kept it string-only on purpose. A detector like that can sit inline on a mail
gateway or proxy and make its call before anyone clicks, and taking the landing
page offline doesn't help an attacker hide from it. The downside is that it
can't catch a phishing page hosted on a legitimate domain that's been
compromised. More on that under [Limitations](#limitations).

I trained two tree ensembles, a random forest and XGBoost, tuned both with
randomized search and cross-validation, and scored them on a class-balanced
held-out split.

## Results

These numbers come from a held-out split of 100,000 URLs, exactly half phishing
and half legitimate. Phishing is the positive class.

| model | accuracy | precision | recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest | 0.8945 | 0.9222 | 0.8617 | 0.8909 | 0.9689 |
| XGBoost | 0.8980 | 0.9345 | 0.8561 | 0.8936 | 0.9724 |

Confusion matrices on the same split (50,000 legitimate, 50,000 phishing):

| model | true neg | false pos | false neg | true pos |
|---|---|---|---|---|
| Random Forest | 46,363 | 3,637 | 6,913 | 43,087 |
| XGBoost | 46,999 | 3,001 | 7,195 | 42,805 |

XGBoost comes out ahead on precision and AUC. The random forest catches a bit
more phishing, but it also flags about 600 more legitimate URLs. On a mail
gateway every false positive is a blocked legitimate message, so I went with
XGBoost as the default in `src/predict.py`.

Both models are more precise than they are sensitive. Most of what they miss is
phishing that looks structurally like any other URL. You can get more recall by
lowering the decision threshold if you're willing to lose some precision, which
is why `predict_proba` is exposed instead of a hard 0.5 cutoff.

Everything generated lands in `reports/`: `metrics.json`,
`model_comparison.csv`, `feature_importance.csv`, `confusion_matrices.png`,
`roc_curves.png`, `feature_importance.png`, `ablation_results.csv`,
`learning_curve.png`.

## Dataset

The data is [`flwrlabs/fed-phishing-urls`](https://huggingface.co/datasets/flwrlabs/fed-phishing-urls)
from the Hugging Face Hub (Apache-2.0). It merges two public collections of real
URLs gathered in the wild, so none of it is synthetic:

- [`ealvaradob/phishing-dataset`](https://huggingface.co/datasets/ealvaradob/phishing-dataset) (`urls.json`)
- [`kmack/Phishing_urls`](https://huggingface.co/datasets/kmack/Phishing_urls)

I only use the `url` and `label` columns. There's also a `client_id` column for
federated-learning experiments, which this project doesn't need.

| stage | rows |
|---|---|
| raw (train + test shards) | 1,132,306 |
| exact duplicates removed | −130,479 |
| conflicting labels removed | −54 |
| final | 1,001,773 |

That leaves 551,867 legitimate URLs (55.09%) and 449,906 phishing (44.91%).

The 54 conflicting rows are URLs that still showed up under both labels after
canonicalization. There's no principled way to pick a label for them, and
keeping them would put identical feature vectors on both sides of the decision
boundary, so they're dropped.

### Canonicalization (and why it makes the numbers look worse)

The two source collections were gathered differently, and you can see it in the
raw strings. Before any cleaning:

| trivial rule | phishing | legitimate | accuracy of that rule alone |
|---|---|---|---|
| string contains `://` | 33.6% | 10.7% | 61.3% |
| string ends with `/` | 33.9% | 23.5% | 55.1% |

Always guessing the majority class gets 55.1%. Just checking for a scheme
prefix gets 61.3% on the raw data, and that check knows nothing about phishing.
It's picking up which of the two feeds a row came from.

`data_loader.canonicalize` cleans this up before anything else touches the data:

1. It strips the `scheme://` prefix, since whether a scheme survived collection
   says something about the feed and nothing about the URL.
2. It strips every trailing slash. A few rows end in `//`, and removing just one
   slash would leave the artifact in exactly those rows.
3. It lowercases the host and leaves the rest alone. Hostnames are
   case-insensitive under RFC 3986, so that's lossless. Path and query case is
   real signal (benign wiki-style URLs use CamelCase paths), so it stays.

Afterwards the scheme rule scores 55.2%, which is baseline. That throws away
roughly 11 points of accuracy the model could have gotten for free from feed
differences, so the numbers in this README are lower than they could have been.
I'd rather report the lower number. Canonicalizing also exposed 130,479
duplicates that had been hidden behind scheme and slash differences.

One visible side effect: `uses_https` has an importance of exactly 0.0 in
`reports/feature_importance.csv`. With the scheme stripped from every training
row, it's a constant. I left it in anyway. It works correctly when someone
passes a real URL at inference time, and I think it's more useful to show it
as dead weight here than to quietly remove it.

## Features

There are 47 features, split into three families. `FEATURE_GROUPS` in
`src/features.py` defines which feature belongs where, and the ablation study
below measures how much each family contributes.

### Lexical (20): size and character composition

| feature | rationale |
|---|---|
| `url_length`, `hostname_length`, `path_length`, `query_length` | Phishing URLs pad length to push the real host out of a truncated display. |
| `num_digits`, `num_letters`, `num_special_chars` | Generated hostnames skew toward digits and punctuation. |
| `num_dots`, `num_hyphens`, `num_slashes` | Cheap structural proxies that survive obfuscation. |
| `num_subdomains`, `num_dots_in_domain` | Deep nesting puts a trusted-looking word in front of a domain the attacker owns. |
| `has_at_symbol` | `@` turns everything to its left into userinfo, so a familiar brand can sit in front of the real host. |
| `has_hyphen_in_domain` | Names like `paypal-secure.com`. |
| `tld_length` | Long TLDs tend to be the cheap new gTLDs. |
| `host_digit_count`, `host_hyphen_count` | Digits and hyphens count for more in the host than in the path. |
| `num_host_tokens`, `longest_host_token` | Token shape separates generated names from chosen ones. |
| `subdomain_length` | Where something like `secure-login-yourbank-verify` gets parked in front of an attacker-owned domain. |

### Structural (13): the shape of the parsed URL

| feature | rationale |
|---|---|
| `uses_https` | The scheme, when there is one. Constant on this corpus (see above). |
| `has_port` | An explicit port usually means something other than normal web hosting. |
| `has_ip_host` | A bare IP means no registered domain and no certificate name to check. |
| `num_path_segments`, `longest_path_segment` | Deep or padded paths. |
| `num_query_params` | Credential-harvest URLs tend to carry campaign and target parameters. |
| `has_fragment`, `has_userinfo` | Both can push the real host out of view. |
| `has_double_slash_redirect` | `site.com/redirect//evil.com`, the old open-redirect and path-confusion trick. |
| `path_to_url_ratio` | Tells a bare hostname apart from a deep path. |
| `has_file_extension`, `has_executable_extension` | A link straight to an installer or archive is malware delivery. |
| `num_encoded_chars` | Percent-encoding can hide keywords from naive scanners. |

### Suspicion heuristics (14): known tradecraft

| feature | rationale |
|---|---|
| `is_shortener` | Shorteners hide the destination. This matches against a list of known shortener hosts and makes no network call. |
| `num_suspicious_keywords` | Words like `secure`, `verify`, `login`, `account`, and `confirm` that push people to type in credentials. |
| `has_suspicious_tld` | Cheap or free TLDs that turn up far more often in abuse feeds. |
| `brand_domain_mismatch` | A brand name appears in the URL but isn't the registrable domain. `paypal.secure-login.xyz` trips it and `paypal.com/login` doesn't. |
| `is_free_hosting` | Free hosting and dynamic DNS. The registrable domain belongs to the provider, so it tells you nothing about who runs the site. |
| `tld_rank` | Where the TLD ranks among common TLDs. The long tail all shares one bucket. |
| `digit_letter_ratio`, `vowel_ratio_domain` | Ratios that tell real words from generated strings. |
| `domain_entropy`, `url_entropy` | Generated names spread their characters more evenly than names a person picked. |
| `longest_consecutive_run` | Runs like `aaaaaa` or `------` show up in generated or padded hostnames. |
| `longest_consonant_run` | Long unpronounceable clusters are the clearest cheap signal of gibberish. |
| `has_hex_token` | Long hex strings are usually session tokens or generated IDs. |
| `has_punycode` | `xn--` is how non-ASCII hostnames are encoded, which is how homograph lookalikes get through. |

### Implementation notes

All the feature logic lives in `extract_features(url) -> dict`.
`extract_features_frame` is just a batched wrapper, and a test checks that the
two agree, so there's only one copy of the logic. Running it over all 1,001,773
rows takes about 58 seconds, and the result is cached to
`data/processed/features.parquet`.

I wrote the URL parser by hand instead of using `urllib.parse`. `urlsplit`
decides that a scheme-less `example.com:8080/x` has the scheme `example.com`,
and its `.port` and `.hostname` accessors raise `ValueError` on malformed input,
of which this corpus has a lot. Feature extraction is never allowed to raise.
Any input at all, including `""`, `None`, or a megabyte of junk, gets a full
feature dict back.

Registrable-domain extraction understands multi-label public suffixes, so
`www.bbc.co.uk` becomes `bbc.co.uk` and not `co.uk`. Without that, every
domain-level feature would be looking at a string the owner never chose. The
bundled suffix list only covers the multi-label suffixes that appear in this
corpus, not the whole public suffix list (see [Limitations](#limitations)).

## Models

Both classifiers sit inside a scikit-learn `Pipeline`, and there's no scaler in
it. Tree ensembles split on one feature at a time, and those splits don't change
under monotone rescaling, so scaling would just burn time. I kept the `Pipeline`
anyway because it gives the saved model a fixed interface, and if I ever add
preprocessing it'll run inside each cross-validation fold, where it can't leak
validation statistics into training.

### Splits

| split | rows | phishing |
|---|---|---|
| train | 811,595 | 44.35% |
| validation | 90,178 | 44.35% |
| test | 100,000 | 50.00% |

The test split gets carved out first, with exactly 50,000 rows of each label. A
coin flip scores 50% on it, and accuracy lines up directly with F1.

Training keeps the natural 55/45 split. That's mild enough that resampling and
class weights weren't worth it: neither improved validation F1 when I tested
them, and both made calibration worse. The validation split is stratified to
match training.

The split depends only on the random seed, so training and evaluation rebuild
the same test set on their own and I don't have to ship index files around.

### Tuning

Tuning is `RandomizedSearchCV` over 15 configurations, with 3-fold stratified CV
scored on F1. The search runs on a stratified 150,000-row sample of the training
data. You don't need all 811k rows to rank a dozen candidates (the ranking
settles well before that), and the smaller sample keeps each search to about 8
minutes. The best configuration then gets refit on the full training split.

Selected hyperparameters (also in `reports/best_params.json`):

| | Random Forest | XGBoost |
|---|---|---|
| n_estimators | 364 | 544 |
| max_depth | None | 12 |
| learning_rate | n/a | 0.0432 |
| min_samples_split / min_child_weight | 4 | 1 |
| min_samples_leaf | 2 | n/a |
| max_features / colsample_bytree | 0.4 | 0.712 |
| subsample | n/a | 0.995 |
| gamma | n/a | 0.217 |

## Ablation studies

`python -m src.ablation` runs three sweeps. Each configuration is retrained from
scratch and scored on the same balanced test split as the main results. The full
output is in `reports/ablation_results.csv`.

### Feature groups

I tried each family on its own and also removed each one from the full set. You
need both, because two correlated families can each look strong alone while
either one could be dropped without losing anything.

| configuration | features | RF F1 | XGB F1 | XGB ROC-AUC |
|---|---|---|---|---|
| all features | 47 | 0.8909 | 0.8936 | 0.9724 |
| only lexical | 20 | 0.8427 | 0.8428 | 0.9429 |
| only structural | 13 | 0.7904 | 0.7822 | 0.8299 |
| only suspicion | 14 | 0.8473 | 0.8406 | 0.9424 |
| without lexical | 27 | 0.8804 | 0.8787 | 0.9650 |
| without structural | 34 | 0.8892 | 0.8917 | 0.9714 |
| without suspicion | 33 | 0.8463 | 0.8458 | 0.9456 |

The suspicion heuristics matter most by a wide margin. Taking them out drops
XGBoost's F1 by 4.8 points, from 0.8936 to 0.8458. The feature importances say
the same thing: `num_suspicious_keywords`, `has_suspicious_tld`, and
`brand_domain_mismatch` are XGBoost's top three and together account for about
41% of its importance. On this data, hand-written rules about how phishers
operate beat generic string statistics.

The structural family barely pulls its weight. It's the weakest on its own
(0.7822), and removing it costs 0.2 points, which means the other two families
already cover most of what it knows. Most of its features are binary and hardly
ever fire: `has_userinfo` on 0.003% of rows, `has_punycode` on 0.065%,
`has_port` on 0.189%, `has_fragment` on 0.204%. The trees don't have much to
split on there.

No family works well alone, though. The best single family gets 0.847, against
0.894 with everything, so each one brings something the others don't.

### Training set size

| training rows | RF F1 | XGB F1 | XGB ROC-AUC |
|---|---|---|---|
| 81,159 (10%) | 0.8715 | 0.8793 | 0.9646 |
| 202,898 (25%) | 0.8800 | 0.8865 | 0.9686 |
| 405,797 (50%) | 0.8858 | 0.8902 | 0.9707 |
| 811,595 (100%) | 0.8909 | 0.8936 | 0.9724 |

More data helps less and less. F1 grows roughly with the log of the training set
size: going from 10% to 100% of the data buys 1.4 F1 points, and the last
doubling (406k to 812k rows) buys 0.34. If that trend holds, ten times more data
(8 million URLs) would get you about one more point.

So more of the same data won't move things much. The bottleneck is how much 47
numeric features can capture about a URL string. The curve is in
`reports/learning_curve.png`. Watch the y-axis there, since the whole plot spans
only about two F1 points.

### Split strategy: how much is memorization

With a random split, most domains in the test set also appear in training. The
domain-disjoint split holds out entire registrable domains, so the model has
never seen the domain of any test URL.

| split | RF F1 | XGB F1 | XGB ROC-AUC |
|---|---|---|---|
| random (main) | 0.8909 | 0.8936 | 0.9724 |
| domain-disjoint | 0.8406 | 0.8500 | 0.9298 |
| gap | −0.0503 | −0.0436 | −0.0426 |

Around 4 to 5 F1 points of the main result come from the model recognizing
domains it has already seen. Recognizing known-bad infrastructure is useful,
but it's a different job from judging a URL you've never seen, and I don't
think the two should be blended into one number.

On a domain the model hasn't seen before, expect an F1 of about 0.85 and a
ROC-AUC of about 0.93. Any URL-only detector scored on a random split has this
effect baked into its number, whether anyone measured it or not.

## Reproducing

You need Python 3.10 or newer (I developed on 3.12).

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

Each step can be rerun by itself. `src.data_loader` downloads the parquet shards
into `data/raw/` the first time and skips the download after that. The feature
cache in `data/processed/` rebuilds automatically when it's out of date. Large
generated files are gitignored, and the commands above regenerate them.

To score some URLs with a trained model:

```bash
python -m src.predict "paypal.secure-login.xyz/webscr" "github.com/torvalds/linux"
```

```
PHISHING   p=0.9999  paypal.secure-login.xyz/webscr
legitimate p=0.1427  github.com/torvalds/linux
```

Or from Python:

```python
from src.predict import load_model, predict_urls

model = load_model("xgboost")            # or "random_forest"
labels, probabilities = predict_urls(model, ["http://example.com/login"])
```

Use `src.predict` rather than calling the model directly. It runs the same
canonicalization the training data went through. Skip that and a raw
`https://` URL gets `uses_https` set to 1, a value the model never saw in
training.

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

The biggest one: a phishing page on a compromised legitimate domain is invisible
to this. If someone hacks a WordPress site on an ordinary company domain and
hosts a phishing page there, the URL looks ordinary too, and no amount of string
analysis will flag it. Every URL-only detector has this problem, and it's why
something like this should be one layer in a larger system.

A lot of the signal is really about which domain a URL is on. A model that only
saw the registrable domain and just memorized the corpus would hit about 98%
accuracy, because labels cluster so heavily by domain. In this data
`duckdns.org` is 99.98% phishing and `wikipedia.org` is about 0%. That's
remembering known-bad infrastructure, which isn't the same as generalizing. The
[split-strategy ablation](#ablation-studies) measures how big that gap is by
holding out whole domains.

The lists in `SUSPICIOUS_TLDS`, `SHORTENERS`, `FREE_HOSTING`, and
`TARGETED_BRANDS` are hardcoded and reflect what attackers were doing when I
wrote them. They'll need updating, and a campaign on a TLD or provider that
isn't listed won't trigger those features.

The bundled public suffix list only covers the multi-label suffixes found in
this corpus. A URL with some other multi-label suffix gets the wrong registrable
domain, and its domain-level features are off as a result.

The dataset is a snapshot in time. Phishing URLs change as campaigns and
registrars change, so expect worse performance on URLs collected today, and plan
to retrain on fresh data now and then.

Some legitimate URLs look exactly like phishing. Security vendors, researchers,
and URL scanners publish links that mimic the attacks they're writing about,
and those make up a real share of the false positives.

Finally, the labels are just phishing or legitimate, and "phishing" here covers
both credential harvesting and malware delivery. That's part of why
`has_executable_extension` matters as much as it does. Strictly speaking, this
is a malicious-URL detector more than a phishing detector.
