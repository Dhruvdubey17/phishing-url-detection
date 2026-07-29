# Learning Guide: Phishing URL Detection

This document teaches you everything underlying this project, starting from
zero machine learning knowledge. It assumes you know Python (functions,
classes, pandas basics) and nothing else. By the end you should be able to
explain, from first principles, every decision this codebase makes — and
answer interview questions that go beyond it.

Read it in order the first time. After that, use the table of contents to
jump to whatever you're being quizzed on.

## Table of contents

1. [What machine learning classification actually is](#1-what-machine-learning-classification-actually-is)
2. [Train / validation / test splits](#2-train--validation--test-splits)
3. [The data pipeline: sourcing, cleaning, leakage](#3-the-data-pipeline-sourcing-cleaning-leakage)
4. [Feature engineering: turning a string into numbers](#4-feature-engineering-turning-a-string-into-numbers)
5. [Decision trees, the building block](#5-decision-trees-the-building-block)
6. [Random Forest](#6-random-forest)
7. [Gradient boosting and XGBoost](#7-gradient-boosting-and-xgboost)
8. [Random Forest vs XGBoost](#8-random-forest-vs-xgboost)
9. [Evaluation metrics](#9-evaluation-metrics)
10. [Hyperparameters and tuning](#10-hyperparameters-and-tuning)
11. [Cross-validation](#11-cross-validation)
12. [Data leakage, generalization, and the domain-disjoint experiment](#12-data-leakage-generalization-and-the-domain-disjoint-experiment)
13. [Software engineering choices in this codebase](#13-software-engineering-choices-in-this-codebase)
14. [Testing strategy](#14-testing-strategy)
15. [How this project's numbers were actually produced](#15-how-this-projects-numbers-were-actually-produced)
16. [Rapid-fire interview Q&A](#16-rapid-fire-interview-qa)

---

## 1. What machine learning classification actually is

### The core idea

You have a bunch of examples. Each example has some **features** (measurable
properties) and a **label** (the answer you want to predict). You want a
function that, given the features of a *new* example it has never seen, guesses
the label correctly.

In this project:
- **Example** = one URL, like `paypal.secure-login.xyz/webscr`
- **Features** = 47 numbers computed from that string (length, entropy, whether
  it contains "verify", etc.)
- **Label** = `1` (phishing) or `0` (legitimate)
- **Goal** = given a brand-new URL, predict whether it's phishing

This is **binary classification** — exactly two possible labels. (Contrast
with *regression*, where you predict a continuous number like a house price,
or *multiclass classification*, where you pick one of many categories.)

### Why not just write `if/else` rules?

You could write `if "verify" in url: return "phishing"`. That's actually
*exactly* what one of our features does (`num_suspicious_keywords`). The
difference between that and machine learning is:

- A hand-written rule is a **guess** about what matters, made by a human.
- A trained model **learns the weighting and combination** of many such
  signals from data — it might discover that `num_suspicious_keywords` matters
  a lot when `has_suspicious_tld` is also true, but barely at all otherwise.
  That's an *interaction* between features, and finding interactions among 47
  features by hand is not practical.

This project's approach is a hybrid: **humans hand-designed the features**
(the 47 numbers), and **the model learns how to combine them**. This is called
"feature engineering + a classical ML model," as opposed to *deep learning*,
where the model would learn features directly from raw text/pixels/audio. We
chose the classical approach deliberately — see [Section 8](#8-random-forest-vs-xgboost)
for why tree ensembles fit tabular numeric data like this well.

### Supervised learning

This whole setup — learn from (features, label) pairs — is called
**supervised learning**. The "supervision" is the label; someone (in this
case, whoever built the source datasets) already tagged each URL as phishing
or not. Contrast with **unsupervised learning** (finding structure in
unlabeled data, e.g. clustering) and **reinforcement learning** (learning by
trial and error against a reward signal, e.g. game-playing agents). Everything
in this project is supervised.

---

## 2. Train / validation / test splits

### Why you can't just train on everything and check accuracy on the same data

If you train a model on 100 examples and then check how well it predicts
those *same* 100 examples, you're not measuring whether it learned anything
general — you're measuring whether it **memorized**. A sufficiently flexible
model (e.g., a lookup table) gets 100% "accuracy" this way while being useless
on new data. This is **overfitting**: the model fits noise and quirks specific
to the training examples rather than the underlying pattern.

The opposite failure is **underfitting**: the model is too simple to capture
even the real pattern (e.g., trying to fit a straight line to obviously curved
data). This tension is called the **bias-variance tradeoff**:

- **Bias** = error from wrong assumptions (model too simple → underfitting)
- **Variance** = error from sensitivity to the specific training data (model
  too flexible → overfitting, different training samples give wildly
  different models)

You want the sweet spot. You can't find it by looking at training performance
alone — training performance keeps improving as you add flexibility, right
past the point where it stops generalizing. You need to check performance on
data the model never touched during training.

### The three-way split used in this project

`src/models.py::make_splits` divides the 1,001,773 URLs into three disjoint
sets:

| split | rows | purpose |
|---|---|---|
| **train** | 811,595 | the model fits its parameters to this data |
| **validation** | 90,178 | used *during development* to compare hyperparameter choices, without touching test |
| **test** | 100,000 (balanced 50/50) | touched exactly once, at the end, to report the final number |

Why three and not two?

If you only had train/test, and you tried 20 different hyperparameter
settings, checking each one against the test set, you'd eventually find a
setting that happens to do well on that *specific* test set by luck — this is
called **test set leakage through repeated evaluation**, and it makes your
final reported number optimistic. It's the same overfitting problem, just one
level up: instead of overfitting the model to the training data, you overfit
your *choice of hyperparameters* to the test data.

The validation set is the "practice test" you can check as many times as you
like while tuning. The test set is the "real exam" — evaluate on it once,
report that number, and don't go back and change anything based on what it
shows. (In `src/train.py`, hyperparameter search is measured with cross-validated
F1 on a training subsample, and a single validation check confirms the refit
model; `src/evaluate.py` is the only place the test set is touched.)

### Why this project's test split is artificially balanced 50/50

The raw corpus is 55% legitimate / 45% phishing — not extreme, but not even.
If you evaluate metrics like accuracy on an imbalanced test set, a lazy model
that always predicts the majority class gets a misleadingly high score (e.g.,
55% accuracy for doing nothing). By forcing the test set to exactly 50,000
of each label, `make_splits` guarantees:

- Accuracy and F1 are actually comparable — a coin flip scores 50%, not 55%.
- Every 1% of accuracy has the same meaning regardless of which class it comes
  from.

The **training** set is deliberately left imbalanced (44.35% phishing) because
that mismatch (55/45) is mild enough that artificially balancing it (e.g. by
discarding excess legitimate rows, or duplicating phishing rows) would have
thrown away real data or introduced duplicate-row artifacts, for no measured
benefit. This project actually tested that (rather than assuming it) — see
`src/train.py`'s corpus imbalance report.

### Stratification

`make_splits` uses `train_test_split(..., stratify=y_rest, ...)` for the
train/validation split. **Stratified** splitting means: preserve the same
class ratio in each resulting subset as in the original data. Without it, a
plain random split *usually* keeps roughly the right ratio by chance, but
stratification guarantees it exactly — important when a class is rare, since
a non-stratified split could accidentally starve one split of an entire class.

---

## 3. The data pipeline: sourcing, cleaning, leakage

### Where the data came from

`src/data_loader.py` pulls from `flwrlabs/fed-phishing-urls` on Hugging Face —
a dataset of 1,132,306 real URLs (not synthetic), licensed Apache-2.0, itself
built by combining two other public phishing-URL datasets. Each row is just a
URL string and a 0/1 label.

Two file formats matter here:
- **Parquet**: a columnar binary storage format. Unlike CSV (plain text, one
  row per line, must be re-parsed character by character), Parquet stores each
  column's values together, in a compact binary encoding, with the data type
  recorded in the file. This makes it much faster to read/write and it never
  has to guess whether a column is a number or a string. This project caches
  extracted features to `data/processed/features.parquet` for this reason —
  extracting 47 features from a million URLs takes ~58 seconds, and every
  script that needs the feature matrix (train, evaluate, ablation) would
  otherwise redo that work.
- **CSV**: plain text, human-readable, universally supported, but slower and
  loses type information (everything is a string until re-parsed). Used here
  for `data/processed/dataset.csv` because it's small (URL + one integer) and
  benefits from being diffable/inspectable.

### Deduplication

`build_dataset()` drops exact duplicate `(url, label)` pairs, and separately
drops any URL that appears under **both** labels (label conflicts) — 54 such
rows. Why drop conflicts instead of picking one label? Because if the exact
same feature vector appears in the training data tagged as both phishing and
legitimate, the model is being asked to predict two different answers for the
same input — mathematically impossible to do perfectly, and arbitrarily
resolving it (e.g. "majority wins") would inject a decision no one actually
verified.

### Canonicalization: the most important cleaning step

This is the part of the project most likely to come up in an interview,
because it's a textbook example of **collection artifacts / spurious
correlation**, not textbook ML theory.

The two upstream datasets that got merged into this corpus were **scraped
differently**. One kept URLs with their `https://` prefix intact more often;
the other stripped it more often. This is an accident of how each dataset's
authors collected their data — it has nothing to do with phishing.

But statistically, in the raw merged corpus:
- 33.6% of phishing URLs still had a `scheme://` prefix
- only 10.7% of legitimate URLs did

That's a huge difference — and it means a trivial rule ("does this string
contain `://`") classifies the raw corpus at **61.3% accuracy** by itself,
despite encoding zero real information about phishing. It's purely detecting
*which upstream dataset a row came from*. This is called a **spurious
correlation** or a **confound**: a feature that correlates with the label for
a reason unrelated to the actual thing you're trying to predict.

If you train a model on data containing this artifact, it will happily learn
to exploit it — cheaply boosting its measured accuracy while learning nothing
transferable. The moment you point it at a URL from a different source (a
production traffic stream, say), the artifact is gone and that portion of the
model's apparent skill evaporates.

**The fix** (`canonicalize()` in `src/data_loader.py`):
1. Strip `scheme://` from every URL, so no row carries this signal at all.
2. Strip **all** trailing slashes (not just one — some rows end in `//`, and
   removing only one slash would leave the artifact present for exactly those
   rows; this was actually a bug this project's own tests caught).
3. Lowercase the **host** portion only, leaving path/query case untouched.

Why only lowercase the host? Per **RFC 3986** (the spec for URI syntax), the
*scheme* and *host* of a URL are case-insensitive — `EXAMPLE.com` and
`example.com` are the same server. But the *path* is not necessarily
case-insensitive (it depends on the server), and in this data, legitimate
URLs genuinely contain meaningful mixed-case paths (e.g. Wikipedia-style
`/wiki/CamelCase_Title`). Lowercasing the whole string would destroy that real
signal along with fixing nothing.

**How to detect an artifact like this in general** (useful interview
knowledge): compute, for a candidate feature, the accuracy of predicting the
label using *that feature alone* — literally `y_true == some_rule`. If a
single crude rule scores far above the majority-class baseline, investigate
why before trusting anything downstream. This project did exactly that (see
the diagnostic in `data_loader.py`'s docstring and the README's canonicalization
section) — comparing against the 55.1% majority baseline is what exposed the
artifact.

### Result of canonicalization

After cleaning: 1,001,773 rows (551,867 legitimate / 449,906 phishing), and
the same "does it contain `://`" rule now scores 55.2% — essentially the
baseline, confirming the artifact is gone. It also surfaced 130,479 duplicate
rows that had been hidden by inconsistent scheme/slash formatting (e.g. the
same URL appearing once with `https://` and once without, previously counted
as two distinct rows).

---

## 4. Feature engineering: turning a string into numbers

Models don't understand strings. Every feature has to be a number. Feature
engineering is the craft of deciding *which* numbers to compute from raw
data, chosen because a domain expert believes they correlate with the label.

### Anatomy of a URL (you need this to understand the structural features)

Per RFC 3986, a URL breaks down like this:

```
  https://user:pass@sub.example.com:8443/a/b?x=1&y=2#top
  \___/   \______/ \_______________/ \__/\___/\_____/\_/
 scheme  userinfo        host        port  path query fragment
```

- **scheme** — the protocol (`http`, `https`, `ftp`, ...)
- **userinfo** — optional `user:password@` before the host. Rarely used
  legitimately on the modern web; historically abused to disguise the real
  host, since everything before the final unescaped `@` is userinfo and
  everything after is the actual host. `paypal.com@evil.com` goes to
  `evil.com`, not `paypal.com`.
- **host** — the domain name or IP address
- **port** — optional, explicit like `:8443`
- **path** — the `/a/b` part identifying a resource on that host
- **query** — the `?x=1&y=2` part, key=value pairs after `?`
- **fragment** — the `#top` part, client-side only, never sent to the server

`src/features.py::_parse` implements this splitting by hand rather than using
Python's built-in `urllib.parse`. Why? Because `urllib.parse.urlsplit` makes
assumptions that don't hold for messy real-world/adversarial input:
- It reads `example.com:8080/x` (no `//`) as **scheme = "example.com"**,
  port-looking text and all — wrong, since there's no `://`.
- Its `.hostname`/`.port` properties **raise `ValueError`** on some malformed
  input, and this corpus has plenty of malformed URLs (attackers don't
  produce RFC-compliant strings, and legitimate scraped data has typos).

Since feature extraction must **never throw an exception** — a single bad row
would crash extraction over a million rows — the parser is written to fail
soft, splitting on `://`, `@`, `:`, `/`, `?`, `#` directly with `str.partition`
and regexes, always producing *some* answer rather than raising.

### Registrable domain and the Public Suffix List problem

A recurring need: given a host like `www.bbc.co.uk`, what's the "real" domain
someone registered? Naively, "last two dot-separated labels" gives `co.uk` —
wrong. `co.uk` is a **public suffix**: a suffix under which *anyone* can
register a name (like `.com` or `.co.uk`), analogous to a TLD but sometimes
multi-part. The actual registered name is `bbc.co.uk` (suffix + one more
label).

The **Public Suffix List (PSL)**, maintained by Mozilla, is the authoritative
answer to "which suffixes are public." This project bundles a partial list of
the multi-part suffixes (`co.uk`, `com.au`, `co.jp`, etc.) that actually
appear in the corpus, in `MULTI_PART_SUFFIXES` — not the full PSL, which has
thousands of entries and needs ongoing maintenance. `_registered_domain()`
checks whether the last two labels match a known multi-part suffix; if so, it
takes three labels instead of two.

Getting this wrong corrupts every domain-level feature: without it,
`paypal.co.uk/signin` would compute its "registered domain" as `co.uk`, see
that "paypal" (a brand name) appears in the URL but not in the (wrong)
registered domain, and wrongly flag `brand_domain_mismatch`.

### Shannon entropy (the math you'll be asked to explain)

Several features (`domain_entropy`, `url_entropy`) measure the **information-theoretic randomness** of a string. Shannon entropy for a string is:

```
H(s) = - Σ p(c) · log2(p(c))     for each unique character c in s
```

where `p(c)` is the fraction of characters in `s` equal to `c`. Intuition:
- If a string is **one repeated character** (`"aaaaaa"`), there's only one
  possible character at each position, so entropy is 0 bits — completely
  predictable.
- If a string uses many different characters in roughly equal proportion
  (like a random hash `"x7fk2q9zvb"`), entropy is high — each character
  carries more "surprise."
- Human-chosen domain names (words, brand names) tend to sit in the middle:
  more entropy than `"aaaa"`, but noticeably less than a machine-generated
  random string, because natural language has structure (some letters are
  much more common than others, certain letter pairs recur).

Why this matters for phishing detection: algorithmically generated hostnames
— seen in domain-generation algorithms (DGAs) used by some malware, or
throwaway subdomains like `x7fk2q9.example.top` — have higher entropy than
names a person picked. `_entropy()` in `features.py` implements exactly the
formula above using `collections.Counter` to get character frequencies.

Related, simpler heuristics for the same underlying idea:
- `longest_consecutive_run` — longest run of one repeated character (low
  entropy signal, cheap to compute).
- `longest_consonant_run` — longest run of consecutive consonants. Real words
  alternate vowels and consonants because human languages are pronounceable;
  a run like `bcdfghjklm` is a strong "this wasn't typed by a person" signal,
  and is cheaper than entropy while catching a different failure mode
  (entropy can be fooled by strings with balanced but structured character
  use; unpronounceability can't).
- `vowel_ratio_domain` — proportion of vowels among the letters. Natural
  language sits in a predictable vowel/consonant ratio; a domain far outside
  that range is suspicious.

### The three feature families (and why grouping matters)

`FEATURE_GROUPS` in `features.py` partitions all 47 features into three
non-overlapping sets. This isn't just organizational — it's what makes the
[ablation study](#15-how-this-projects-numbers-were-actually-produced) possible: you can only measure "how much does
family X contribute" if X is cleanly separable from the others.

**Lexical (20 features)** — properties of the string's *characters*, without
needing to know what any of them mean structurally: lengths (`url_length`,
`hostname_length`, `path_length`, `query_length`), counts (`num_digits`,
`num_letters`, `num_special_chars`, `num_dots`, `num_hyphens`, `num_slashes`),
counts specific to the host (`host_digit_count`, `host_hyphen_count`,
`num_host_tokens`, `longest_host_token`, `subdomain_length`), and a few
targeted binary signals (`has_at_symbol`, `has_hyphen_in_domain`,
`tld_length`, `num_subdomains`, `num_dots_in_domain`).

**Structural (13 features)** — properties that require *parsing* the URL
into its components: `uses_https` (scheme), `has_port`, `has_ip_host`,
path-segment counts, query-parameter counts, `has_fragment`, `has_userinfo`,
`has_double_slash_redirect` (a `//` inside the path — an open-redirect
trick: `site.com/redirect//evil.com`), `path_to_url_ratio`,
`has_file_extension`/`has_executable_extension` (does the path end in
`.exe`/`.zip`/etc. — a malware-delivery signal rather than a
credential-phishing one), and `num_encoded_chars` (count of `%` — percent
encoding is a common way to hide keywords from naive string scanners).

**Suspicion heuristics (14 features)** — targeted domain knowledge about
known phishing tradecraft, matched against curated lists: `is_shortener`
(bit.ly, tinyurl, etc. — shorteners hide the real destination),
`num_suspicious_keywords` (words like "verify", "secure", "login", "confirm"
that provoke a credential entry), `has_suspicious_tld` (cheap/free TLDs
over-represented in abuse feeds), `brand_domain_mismatch` (a known brand name
appears in the URL but the *registrable domain* isn't that brand's — catches
`paypal.secure-login.xyz` while correctly ignoring `paypal.com/login`),
`is_free_hosting` (free hosting/dynamic-DNS providers, where the registrable
domain identifies the *provider*, not whoever is actually serving content),
`tld_rank` (how common the TLD is on the web at large, rather than a binary
flag — the long tail of rare TLDs shares one bucket), plus the
entropy/composition features already covered above, and `has_punycode`.

### Punycode and homograph attacks

**Punycode** is an encoding (the Bootstring algorithm) that represents
Unicode characters using only ASCII, prefixed with `xn--`. It exists so
internationalized domain names (e.g. Cyrillic, Chinese characters) can be
represented in the ASCII-only DNS system. `xn--e1afmkfd.xn--p1ai` decodes to
a Cyrillic domain.

Attackers exploit this via **homograph attacks**: registering a domain using
Unicode characters that *look* identical to Latin letters in most fonts (e.g.
Cyrillic "а" looks exactly like Latin "a"), so `аpple.com` (with a Cyrillic
"а") visually reads as `apple.com` but is a completely different domain under
the hood, and would appear in DNS/browser address bars as an `xn--...` string.
`has_punycode` just checks for the `xn--` prefix — a cheap, reliable proxy for
"this hostname isn't what it visually appears to be."

### Why extraction has to be vectorized/parallelizable, and why it wasn't parallelized here

`extract_features_frame()` supports chunking work across processes
(`ProcessPoolExecutor`) for scale. In practice, benchmarking showed serial
execution (58 seconds for 1M rows) was *faster* than multiprocessing at this
data size on this 8-core machine — process-spawn overhead on macOS dominates
until you have far more rows or much heavier per-row work. This is worth
knowing generally: **parallelism has fixed overhead (spinning up processes,
serializing data across them), and it only pays off once the per-item work is
large enough to amortize that cost.** Measuring before parallelizing (rather
than assuming "more cores = faster") is the right instinct, and it's exactly
what this project's benchmark did.

---

## 5. Decision trees, the building block

Random Forest and XGBoost are both **ensembles of decision trees** — you
can't understand either without understanding a single tree first.

### What a decision tree is

A decision tree is a sequence of yes/no questions about the features, forming
a tree structure. Each internal node asks one question (e.g., "is
`num_suspicious_keywords > 2`?"), and each leaf makes a prediction. To predict
a new URL, walk from the root down, answering each question, until you reach
a leaf; output that leaf's prediction.

Trees are built greedily, top-down: at each node, the algorithm searches over
all features and all possible thresholds, and picks the single split that
best separates the two classes in that node's data. This is repeated
recursively on each resulting child node.

### How "best split" is measured: Gini impurity and entropy

Two common ways to measure how "mixed" a node's labels are:

**Gini impurity**: `Gini = 1 - Σ p_i²`, summed over each class `i`, where
`p_i` is the fraction of that class in the node. If a node is 100% one class,
Gini = 0 (perfectly pure). If it's 50/50 between two classes, Gini = 0.5 (as
mixed as it gets for binary classification).

**Entropy** (same formula as Shannon entropy from Section 4, applied to class
labels instead of characters): `H = -Σ p_i log2(p_i)`. Same intuition — pure
node has entropy 0, 50/50 split has entropy 1.

A split is chosen to maximize **information gain**: the reduction in
impurity from parent to (weighted average of) children:

```
gain = impurity(parent) - [ (n_left/n_total)·impurity(left) + (n_right/n_total)·impurity(right) ]
```

Gini and entropy usually pick similar splits in practice; Gini is slightly
cheaper to compute (no logarithm) and is scikit-learn's `RandomForestClassifier`
default criterion.

### Why a single tree overfits

A tree grown without limits will keep splitting until every leaf is
pure — potentially one training example per leaf. That's a perfect
memorization of the training data (zero training error) and typically poor
generalization: it has learned the noise, not just the pattern. This is why
single deep decision trees are described as **high variance, low bias**
models — small changes in the training data produce very different trees.

The fix used across this whole field: **don't use one tree — use many, and
combine them so their individual mistakes average out.** That's exactly what
both Random Forest and XGBoost do, via two different combination strategies:
**bagging** and **boosting**.

---

## 6. Random Forest

### Bagging (Bootstrap AGGregatING)

Random Forest builds many decision trees **independently and in parallel**,
each on a different random resample of the training data, then combines their
predictions.

Step by step:
1. For each of `n_estimators` trees, draw a **bootstrap sample**: sample
   `n` rows *with replacement* from the training set (same size as the
   original, but some rows appear multiple times, others not at all — on
   average about 63.2% of unique rows appear in any given bootstrap sample).
2. Grow a decision tree on that sample. Critically, **at each split**, only
   consider a random subset of features (controlled by `max_features`,
   e.g. `"sqrt"` = √(total features) considered at each split) rather than
   all of them.
3. Repeat for all `n_estimators` trees.
4. To predict a new example: run it through every tree, and take a **majority
   vote** across their individual predictions (or average the predicted
   probabilities).

### Why this reduces variance

Each individual tree, grown deep, still overfits its own bootstrap sample.
But the trees overfit *differently* — since they saw different data and
different feature subsets, their mistakes aren't correlated. Averaging many
uncorrelated (or weakly correlated) high-variance estimators cancels out much
of the noise while preserving the genuine signal every tree agrees on. This
is a direct consequence of a statistical fact: the variance of an average of
`k` independent, identically distributed random variables is `1/k` times a
single variable's variance (with some correlation, the reduction is smaller
but still real).

The random feature subsampling at each split (not just bootstrapping rows) is
what prevents all trees from converging to the same few "obviously strongest"
splits and becoming correlated with each other — it's the "Random" in Random
Forest, distinct from plain bagging of full-feature trees.

### This project's Random Forest hyperparameters (tuned)

From `reports/best_params.json`: `n_estimators=364`, `max_depth=None`
(trees grow until pure or hit `min_samples_leaf`), `min_samples_split=4`,
`min_samples_leaf=2`, `max_features=0.4` (40% of features considered per
split, chosen by search over `["sqrt", "log2", 0.4]`).

- `n_estimators` — more trees almost always helps (up to diminishing
  returns) since it's just "average more independent estimators"; the main
  cost is compute time, not overfitting risk (bagging doesn't overfit more
  with more trees, unlike boosting).
- `max_depth` / `min_samples_split` / `min_samples_leaf` — control how deep
  each individual tree can grow, i.e. its capacity to memorize. Left
  unbounded, RF was still tuned to prefer higher `min_samples_split`/`leaf`
  than defaults, effectively lightly regularizing every tree.
- `max_features` — the key anti-correlation knob explained above.

---

## 7. Gradient boosting and XGBoost

### Boosting (fundamentally different from bagging)

Where bagging builds trees **independently in parallel** and averages them,
boosting builds trees **sequentially**, where each new tree is trained to fix
the mistakes of everything built so far.

### Gradient boosting, conceptually

1. Start with a simple initial prediction (e.g., the log-odds of the average
   label).
2. Compute the **residual** — how wrong the current ensemble's predictions
   are for each training example. (For squared-error regression this really
   is `actual - predicted`; more generally it's the negative gradient of the
   loss function with respect to the current predictions — hence "gradient"
   boosting.)
3. Train a new, small tree to predict *that residual* — i.e., a tree whose
   job is specifically to correct the current ensemble's errors.
4. Add this new tree's predictions to the ensemble, scaled down by a
   **learning rate** (so no single tree can swing the ensemble too far).
5. Repeat for `n_estimators` rounds, each new tree targeting whatever error
   remains after all previous trees.

This is why boosted trees are typically shallower individually (they don't
need to be powerful alone — they're specialists correcting a specific
residual pattern) but the sequence as a whole can fit very complex functions.
It's also why boosting is inherently **sequential** — tree 50 depends on
knowing the combined output of trees 1–49 — whereas Random Forest's trees can
all be built in parallel from the start.

### What "XG" (eXtreme Gradient) boosting adds on top of plain gradient boosting

XGBoost is a specific, highly optimized implementation of gradient boosting
with several refinements:

- **Second-order optimization (Newton boosting)**: instead of only using the
  gradient (first derivative) of the loss to decide what each new tree should
  fit, XGBoost also uses the **Hessian** (second derivative), giving a better
  local approximation of the loss and typically faster, more accurate
  convergence.
- **Built-in regularization**: XGBoost's objective directly penalizes tree
  complexity (number of leaves, and the magnitude of leaf output values) via
  L1/L2 terms, controlled by `gamma` (minimum loss reduction required to make
  a further split — effectively a complexity threshold) and implicit
  leaf-weight shrinkage. This is regularization *baked into the split-finding
  criterion itself*, not just a post-hoc adjustment.
- **Histogram-based split finding** (`tree_method="hist"`, used in this
  project): instead of considering every possible threshold for every
  feature at every split (exact greedy search, expensive), continuous
  features are bucketed into histograms first, and splits are only
  considered at bucket boundaries. Much faster with a small, controlled loss
  of split precision — a good tradeoff at the scale of ~800k rows here.
- **Column and row subsampling** (`colsample_bytree`, `subsample`): like
  Random Forest's feature/row randomness, but layered on top of boosting to
  reduce overfitting and correlation between successive trees (this is
  sometimes called *stochastic gradient boosting*).
- **Native handling of missing values**: XGBoost learns a default split
  direction for missing values during training (not used here, since feature
  extraction never produces missing values, but worth knowing).

### This project's XGBoost hyperparameters (tuned)

From `reports/best_params.json`: `n_estimators=544`, `max_depth=12`,
`learning_rate≈0.043`, `min_child_weight=1`, `colsample_bytree≈0.712`,
`subsample≈0.995`, `gamma≈0.217`.

- `learning_rate` (a.k.a. shrinkage) — how much each new tree's correction
  is scaled down before being added to the ensemble. Lower values need more
  trees (`n_estimators`) to reach the same fit, but generalize better because
  no single tree can overreact to noise in its residual target. The tuned
  value here (~0.043) is fairly low, paired with a fairly high tree count
  (544) — a classic "slow and steady" boosting configuration.
- `max_depth=12` — much deeper than boosting theory's usual advice ("many
  shallow trees"), but reasonable here given 47 features and heavy feature
  interactions the suspicion heuristics create (e.g. keyword count only
  matters combined with TLD and brand-mismatch signals).
- `min_child_weight` — minimum sum of instance weight (roughly, minimum
  number of samples, for equal-weighted classification) needed in a leaf
  before a split is allowed; higher values prevent trees from creating
  leaves fit to just a couple of noisy examples.
- `gamma` — the minimum information gain a split must provide to be made at
  all; higher values prune more aggressively.
- `subsample` / `colsample_bytree` — fraction of rows / columns randomly
  used to grow each tree, for the stochastic-boosting variance reduction
  described above.

### Why `RandomizedSearchCV`'s search spaces use particular distributions

`src/models.py::SEARCH_SPACES` uses `randint` for counts (`n_estimators`,
`max_depth`), `uniform` for ratios naturally bounded in a range
(`subsample`, `colsample_bytree`, `gamma`), and `loguniform` specifically for
`learning_rate`. Why log-uniform there specifically? Learning rate spans
orders of magnitude in typical useful ranges (0.03 to 0.4 here), and the
*effect* of moving from 0.03 to 0.06 is roughly as large as moving from 0.15
to 0.3 — it's a multiplicative, not additive, quantity. Sampling uniformly on
a log scale spends equal search effort across each order of magnitude instead
of oversampling the large end.

---

## 8. Random Forest vs XGBoost

| | Random Forest | XGBoost |
|---|---|---|
| tree construction | parallel, independent | sequential, each tree corrects prior errors |
| primary lever against overfitting | averaging many high-variance trees | shrinkage (learning rate) + explicit regularization |
| typical tree depth | can be deep/unbounded per tree | often shallower per tree, sometimes deep with regularization (as tuned here) |
| training speed at this scale | slower (287s to fit on 811,595 rows) | faster (33s to fit on the same rows — ~8.7x) |
| sensitivity to hyperparameters | fairly robust to default settings | more sensitive — learning rate and depth interact strongly |
| typical accuracy ceiling on structured/tabular data | strong | usually slightly stronger, as observed here (XGB: F1 0.8936 vs RF: 0.8909) |
| can overfit given enough boosting rounds without shrinkage/regularization? | not really (more trees just reduces variance further) | yes — unlike bagging, more rounds *can* eventually overfit if learning rate/regularization aren't controlled |
| interpretability of individual estimator | one tree is somewhat readable | one tree less meaningful alone (it only encodes a residual correction) |
| feature importance | available (impurity-based / mean decrease in impurity) | available (gain-based by default in this project) |

This project reports both because a real-world evaluation should never rely
on a single model family's quirks — if RF and XGBoost roughly agree (they do:
0.891 vs 0.894 F1, similar top features), that's evidence the signal is real
rather than an artifact one particular algorithm happens to exploit.

---

## 9. Evaluation metrics

### The confusion matrix

For binary classification with a "positive" class (here: phishing = 1), every
prediction falls into exactly one of four buckets:

|  | predicted legitimate | predicted phishing |
|---|---|---|
| **actually legitimate** | True Negative (TN) | False Positive (FP) |
| **actually phishing** | False Negative (FN) | True Positive (TP) |

In this project's XGBoost test-set results: TN=46,999, FP=3,001, FN=7,195,
TP=42,805 (out of 50,000 legitimate + 50,000 phishing).

**False Positive** = flagged a legitimate URL as phishing (annoying — blocks
a real link, e.g. in a mail gateway). **False Negative** = let a phishing URL
through as legitimate (dangerous — the actual harm the system exists to
prevent). Which one is worse entirely depends on deployment context; this
project is explicit that its threshold (0.5) favors precision, and exposes
`predict_proba` so a deployer can move the threshold toward more recall if
missed phishing is the costlier failure mode for them.

### Accuracy

```
accuracy = (TP + TN) / (TP + TN + FP + FN)
```
Fraction of all predictions that were correct. Misleading on imbalanced data
(a model that always predicts the majority class gets high "accuracy" while
being useless) — this is exactly why the test split here is forced to 50/50.

### Precision

```
precision = TP / (TP + FP)
```
"Of everything I flagged as phishing, what fraction actually was?" High
precision = few false alarms. In this project: XGBoost precision = 0.9345,
i.e. when it says "phishing," it's right 93.45% of the time.

### Recall (a.k.a. sensitivity, true positive rate)

```
recall = TP / (TP + FN)
```
"Of everything that actually was phishing, what fraction did I catch?" High
recall = few missed threats. In this project: XGBoost recall = 0.8561, i.e.
it catches 85.61% of actual phishing URLs, missing the remaining ~14.4%.

**Precision and recall trade off against each other** as you move the
decision threshold (the probability cutoff above which you call something
"phishing"). Raise the threshold (be more conservative about calling
something phishing) → precision goes up, recall goes down. Lower it → the
reverse. There is no single "right" threshold; it depends on the relative
cost of a false positive vs a false negative in your specific deployment.

### F1 score

```
F1 = 2 · (precision · recall) / (precision + recall)
```
The **harmonic mean** of precision and recall — not the arithmetic mean.
The harmonic mean punishes imbalance between the two much more than the
arithmetic mean would: if precision is 0.99 and recall is 0.10, the
arithmetic mean is 0.545 (looks okay), but the harmonic mean (F1) is ~0.182
(correctly reflects that a model missing 90% of phishing is bad, regardless
of how precise its rare positive calls are). F1 is the standard single-number
summary when you care about both false positives and false negatives and
don't have a specific cost ratio between them.

### ROC curve and AUC

The **ROC curve** (Receiver Operating Characteristic) plots **True Positive
Rate** (= recall) against **False Positive Rate** (`FPR = FP / (FP + TN)`) as
you sweep the decision threshold from 1 down to 0. At threshold = 1 (call
nothing phishing), both TPR and FPR are 0. At threshold = 0 (call everything
phishing), both are 1. The curve traces everything in between.

**AUC (Area Under the Curve)** condenses this whole curve into one number
between 0.5 (no better than random guessing) and 1.0 (perfect separation).
AUC has a clean probabilistic interpretation: it's **the probability that a
randomly chosen positive example is scored higher by the model than a
randomly chosen negative example.** This project's XGBoost AUC = 0.9724,
meaning: pick one random phishing URL and one random legitimate URL: 97.24%
of the time, the model assigns the phishing one a higher phishing-probability
score.

**Why report AUC in addition to F1**: F1 (and precision/recall) depend on a
*specific* chosen threshold (0.5 by default). AUC evaluates the model's
ranking ability across **all possible thresholds at once**, so it tells you
about the quality of the underlying probability estimates independent of
where you decide to draw the line. It's also **insensitive to class
balance** in a way accuracy is not, making it a good sanity-check metric even
outside a balanced test split.

### Why this project checks metrics on a class-balanced split specifically

All the ratios above are still mathematically well-defined on an imbalanced
test set. But interpreting them is easier when the test set is 50/50: 90%
accuracy clearly means "way better than guessing" only when guessing gets you
50%, not when guessing already gets you 95% for free on a skewed set. The
balanced test set here removes that ambiguity entirely from every metric at
once.

---

## 10. Hyperparameters and tuning

### Parameters vs hyperparameters

A **parameter** is something the *training algorithm* learns from data (e.g.
which feature and threshold each tree node splits on, what value each leaf
outputs). A **hyperparameter** is something *you* set before training starts,
which controls *how* the algorithm learns (e.g. `n_estimators`, `max_depth`,
`learning_rate`). You can't learn hyperparameters the same way you learn
parameters — instead, you try several values and see which produces the best
*validation* performance (never test — see [Section 2](#2-train--validation--test-splits)).

### Grid search vs randomized search

**Grid search** exhaustively tries every combination in a specified grid
(e.g. every combination of 5 values of `max_depth` × 4 values of
`learning_rate` × 3 values of `n_estimators` = 60 full training runs). Cost
grows multiplicatively with the number of hyperparameters you're tuning —
quickly becomes impractical.

**`RandomizedSearchCV`** (used in this project, `src/train.py`) instead
samples a fixed number of random combinations (`n_iter=15` here) from
specified *distributions* rather than exhaustive lists. This is more
efficient in high-dimensional hyperparameter spaces because — as shown
empirically in the research this technique is based on — a few
hyperparameters usually matter far more than the others, and random sampling
explores the *important* dimensions more thoroughly per unit of compute than
a grid does (a grid wastes evaluations varying unimportant parameters
independently across every setting of the important ones).

### Why the tuning search ran on a subsample, then refit on everything

`src/train.py::tune()` runs the randomized search on a 150,000-row stratified
subsample of the 811,595-row training set, not the whole thing. Rationale:
ranking ~15 candidate configurations against each other doesn't require the
full dataset — the relative *ordering* of "this config is better than that
one" stabilizes well before you've used all the data, and using a subsample
keeps each of the 45 fits (15 configs × 3 CV folds) fast enough that the whole
search finishes in minutes rather than hours. Once the best configuration is
selected, it's refit once on the *complete* training set (`X_train` — all
811,595 rows) so the actual persisted model has seen everything available to
it.

---

## 11. Cross-validation

### Why not just use the validation set directly for tuning?

You could. But with only one validation split, your hyperparameter choice is
sensitive to whatever happened to land in that particular split — a form of
overfitting to the validation set itself, especially with many candidate
configurations. **Cross-validation** reduces this by evaluating each
candidate on *multiple* different splits and averaging.

### K-fold cross-validation, concretely

Split the training data into `k` equal-sized chunks ("folds"). For each of
the `k` rounds: train on `k-1` folds, evaluate on the remaining one. Average
the `k` evaluation scores. Every row gets used for both training and
evaluation across the `k` rounds, but never both *at the same time* within a
single round.

This project uses `StratifiedKFold(3, shuffle=True, ...)` — **3-fold**,
**stratified** (each fold preserves the overall class ratio, important
because the subsample used for tuning is itself imbalanced), and shuffled
before splitting (so folds aren't accidentally correlated with the original
row order).

Why only 3 folds, not the more commonly cited 5 or 10? More folds means more
training runs per candidate (each fold requires a fresh model fit), so it's a
compute/reliability tradeoff. With 15 candidates × 3 folds = 45 fits per
model, 3-fold keeps the search fast; the tradeoff is a slightly noisier
per-candidate score estimate, which is acceptable here because the *goal* of
this search stage is finding a good-enough configuration to refit on full
data, not certifying the final number (that's what the untouched test set is
for).

---

## 12. Data leakage, generalization, and the domain-disjoint experiment

This is the single most important applied-ML lesson this project
demonstrates, and it's exactly the kind of thing that separates a junior
answer from a senior one in an interview.

### What data leakage is, generally

**Data leakage** is any situation where information that wouldn't be
available at real prediction time somehow influences the reported evaluation
metric, making the model look better than it actually is at the task it's
meant to do. Common forms:

- **Target leakage**: a feature is computed using information that itself
  depends on the label (or is only available *after* the label is known).
- **Train/test contamination**: the same or near-identical examples appear in
  both training and test sets, so "predicting" the test set partly reduces to
  recognizing something already memorized.
- **Temporal leakage**: training on data from *after* the point in time the
  model is meant to predict from (e.g. using next month's data to predict
  this month, in a deployed system that will never have next month's data
  available).
- **Group leakage** (the one this project investigates directly): examples
  that share a group identity (same user, same document, same *domain*) are
  split across train and test, so the model can partly succeed by recognizing
  the *group*, not by generalizing to new instances of that group.

### The group leakage in this specific dataset

Look at the raw label statistics per registrable domain:
`duckdns.org` is 99.98% phishing in this corpus, `000webhostapp.com` is
100% phishing, `wikipedia.org` is ~0%. If you compute, for every domain, "what
fraction of its rows share the majority label," and weight by how many rows
each domain has, you get a **ceiling of about 98.14% accuracy achievable
purely by memorizing which domain a URL belongs to** — no URL-structure
reasoning required at all.

The project's default random train/test split doesn't hold domains apart —
a URL from `duckdns.org` can land in training while a *different* URL, also
from `duckdns.org`, lands in the test set. When that happens, the model
doesn't need to understand anything about *why* `duckdns.org` URLs tend to be
phishing — it can partly just learn "I've seen this exact registrable domain
before, tagged phishing" and get credit for it on the test row, even though
that test row is nominally "unseen."

### How this project measured the size of that effect

`src/ablation.py::domain_disjoint_split()` builds an alternative split where
**every registrable domain appears entirely on one side or the other** — no
domain is shared between train and test. It:
1. Computes the registrable domain for every URL.
2. Randomly holds out 15% of the *distinct domains* (not rows) for testing.
3. Balances the resulting test set to 50/50 by label, same as the main split.

Comparing the same tuned model on both splits:

| split | XGBoost F1 | XGBoost AUC |
|---|---|---|
| random (domains shared across train/test) | 0.8936 | 0.9724 |
| domain-disjoint (every test domain unseen) | 0.8500 | 0.9298 |

The gap — about **4.4 F1 points** — is the portion of the headline score
attributable to recognizing already-seen domains rather than reasoning about
URL structure generalizably. That's not nothing (recognizing known-bad
infrastructure is a legitimately useful capability in a real system with a
feedback loop), but it's a different claim than "this model understands what
makes a URL look like phishing," and reporting only the random-split number
would silently conflate the two.

### Why this is worth knowing cold for an interview

"How do you know your model isn't just memorizing?" is a extremely common
senior-level ML interview question, and the textbook answer is: **construct a
held-out split along whatever axis your deployment scenario will actually
see novel values on**, and compare. Here, in deployment, the model will
constantly see brand-new domains that never appeared in training (that's the
entire point of a phishing detector) — so domain-disjoint evaluation is the
*more honest* measurement of real-world performance, and the random split's
higher number is, to a real degree, optimistic.

This generalizes far beyond URLs: the same idea applies to splitting by
patient (not by visit) in medical ML, by user (not by session) in
recommendation systems, by time (not randomly) in any system where the future
must be predicted from the past. The general principle: **identify what
"new" means in your deployment context, and make sure your evaluation split
reflects genuine novelty along that axis** — not just "rows the model hasn't
seen," but "the kind of *new thing* the model will actually face."

---

## 13. Software engineering choices in this codebase

### The `Pipeline` object, and why it's used even without preprocessing

`src/models.py` wraps both classifiers in `sklearn.pipeline.Pipeline([("clf", ...)])`
even though there's no scaling or transformation step here. Two reasons:

1. **Interface stability.** The persisted artifact (`models/xgboost.joblib`)
   is always "one object that goes from a feature DataFrame to a
   prediction," regardless of what preprocessing might be added later. Code
   that calls `model.predict(X)` never needs to change if a scaler gets
   inserted in the future.
2. **Leakage prevention, by construction.** If a scaler *were* added, putting
   it inside the `Pipeline` (rather than scaling the whole dataset once,
   upfront) means that during cross-validation, the scaler's mean/std are
   refit on *only* each fold's training portion — it can never see the
   validation fold's statistics. Scaling the whole dataset before splitting
   would leak information about the validation/test distribution into
   training, even though it feels harmless ("it's just rescaling").

### Why there's no scaler here at all

Both Random Forest and XGBoost split on **thresholds of individual raw
feature values** (`is feature X > threshold?`). Any monotonic rescaling
(e.g., min-max normalization, standardization) doesn't change which side of a
threshold any given value falls on — it just changes the numeric value of
the threshold itself. So scaling literally cannot change what these models
learn; it's pure wasted computation for tree-based models. (It absolutely
would matter for models like logistic regression, k-nearest-neighbors, or
neural networks, where the *magnitude* of feature values directly affects the
learned function — a good interview distinction to know.)

### joblib vs pickle for model persistence

`joblib.dump(model, path, compress=3)` is used rather than Python's built-in
`pickle`. `joblib` is built on top of `pickle` but is specifically optimized
for objects containing large numpy arrays (which is exactly what a fitted
tree ensemble is full of — leaf values, split thresholds, etc. stored as
arrays) — it can be meaningfully faster and produce smaller files for such
objects, especially with compression enabled.

### Why feature extraction is cached to Parquet rather than recomputed every run

`src/data_loader.py::load_xy()` checks whether `data/processed/features.parquet`
already exists and matches the expected shape/columns before recomputing.
Extracting 47 features from ~1M URLs takes about a minute — cheap once, but
every one of `train.py`, `evaluate.py`, and `ablation.py` needs the same
matrix, and re-extracting it in each would triple the total wasted time for
no benefit (the extraction is a pure, deterministic function of the URL
strings). This is a general engineering pattern: **cache the result of any
expensive, deterministic computation that multiple downstream steps need
unchanged.**

---

## 14. Testing strategy

### Why a project like this needs tests, specifically

Two of this project's own bugs were caught by its test suite, not by manual
review:
1. `canonicalize()` originally stripped only *one* trailing slash via
   `re.sub(r"/$", "", url)`; a row ending in `//` survived as ending in `/`,
   leaving a sliver of the collection artifact in place. Caught by
   `test_canonicalize_is_idempotent` in `tests/test_pipeline.py` — applying
   canonicalization twice should be a no-op, and it wasn't.
2. `make_splits()` crashed with a `ValueError` on any corpus smaller than
   `2 × TEST_PER_CLASS` rows (it hard-coded 50,000 rows *per class* for the
   test split regardless of how much data was available), because the
   integration tests intentionally exercise the pipeline on small synthetic
   inputs to keep the suite fast — that's exactly the scenario that revealed
   the bug.

This is the practical case for automated tests over "I read the code and it
looks right": some bugs only manifest at boundary conditions (a corpus
smaller than expected, an idempotency property that isn't obviously encoded
anywhere) that a manual read-through won't naturally probe.

### `pytest` concepts used here, explained

- **Fixtures** (`@pytest.fixture`): reusable setup code injected into test
  functions by argument name. `tests/test_models.py::model` is a
  module-scoped, parametrized fixture — `scope="module"` means the (slow)
  model-loading happens once and is shared across every test in the file
  that requests it, rather than reloading the model fresh for each
  individual test.
- **Parametrization** (`@pytest.mark.parametrize`): run the same test logic
  against many inputs without copy-pasting the test. `test_features.py` runs
  its "never raises, always returns full schema" check across two dozen
  pathological URL strings this way — one assertion, many inputs, individual
  pass/fail reporting per input.
- **Custom markers** (`@pytest.mark.artifacts`, registered in
  `pyproject.toml`): tags a test as requiring the trained model files to
  exist. Tests so marked call `pytest.skip(...)` (not `fail`) when the
  artifacts are missing, so a fresh clone that hasn't run `python -m
  src.train` yet still gets a clean, informative pytest run instead of a wall
  of confusing failures.
- **`pytest.raises`** isn't used here on purpose — the design goal of
  `extract_features` is that it *never* raises, so the tests instead assert
  the *absence* of exceptions across adversarial input, which is a stronger
  and more relevant guarantee for this function's contract than testing that
  it raises on bad input.

### The three test files and what each layer is responsible for

- **`test_features.py`** — unit tests. Tests `extract_features()` in
  complete isolation: given this exact string, is this exact feature value
  correct? Covers edge cases (empty string, IPv6 literals, punycode,
  megabyte-long strings, `None` input) precisely because a pure function like
  this is cheap to test exhaustively and any bug here silently corrupts every
  downstream number.
- **`test_models.py`** — tests the *persisted artifact*, not the code that
  produced it: does it load as the expected object type, does it accept the
  right input shape, are its probability outputs well-formed (rows sum to
  1), does it get obviously-easy examples right, is it deterministic (same
  input twice → same output — trivially true for these models since neither
  has randomness at inference time, but worth asserting explicitly since it's
  a real risk for some model types).
- **`test_pipeline.py`** — integration tests: does the *whole path* from raw
  URL string through canonicalization, feature extraction, and prediction
  work end-to-end without NaNs/infs leaking through, and does an
  already-canonical URL survive the inference-time canonicalization step
  unchanged (a regression guard specifically for the trailing-slash bug
  above).

---

## 15. How this project's numbers were actually produced

A concrete walkthrough, since this ties every abstract concept above to
actual commands and actual output:

1. **`python -m src.data_loader`** — downloads the two parquet shards
   (1,132,306 raw rows), canonicalizes every URL, drops 130,479 exact
   duplicates and 54 label conflicts, writes 1,001,773 clean rows to
   `data/processed/dataset.csv`.
2. **Feature extraction** (triggered automatically by anything calling
   `load_xy()`) — computes all 47 features for all 1,001,773 URLs (~58s),
   caches to `data/processed/features.parquet`.
3. **`make_splits()`** carves out exactly 50,000 legitimate + 50,000
   phishing rows as the untouched test set, then stratified-splits the
   remainder into 811,595 train / 90,178 validation rows.
4. **`python -m src.train`** — for each model, `RandomizedSearchCV` tries 15
   hyperparameter configurations, each scored by 3-fold stratified
   cross-validated F1 on a 150,000-row training subsample; the
   best-scoring configuration is refit on all 811,595 training rows; the
   fitted pipeline is checked once against the validation set and saved to
   `models/{name}.joblib`.
5. **`python -m src.evaluate`** — loads both saved models, predicts on the
   (never-before-touched) 100,000-row balanced test set, computes accuracy /
   precision / recall / F1 / ROC-AUC / confusion matrix for each, and writes
   `reports/metrics.json`, `model_comparison.csv`, and three figures
   (confusion matrices, ROC curves, feature importances).
6. **`python -m src.ablation`** — re-trains multiple restricted variants
   (each feature family alone / removed; 10/25/50/100% of training data; a
   domain-disjoint split) using the *same* tuned hyperparameters found in
   step 4, so that differences in the resulting F1 are attributable to the
   thing being ablated and not to a second, independent hyperparameter
   search muddying the comparison.
7. **`pytest`** — 82 tests across the three files described in
   [Section 14](#14-testing-strategy), all passing against the real trained
   artifacts and the real processed dataset.

Every number in the README and in this guide was produced by actually running
these commands, not estimated or assumed — including the ones that came out
lower than initially hoped (the 0.98 F1 target discussed at length in the
README's Results section was not met, and the README says so explicitly
rather than adjusting the framing to obscure it).

---

## 16. Rapid-fire interview Q&A

**Q: What's the difference between a parameter and a hyperparameter?**
A parameter is learned from data during training (tree split thresholds,
leaf values). A hyperparameter is set before training and controls how
learning happens (number of trees, learning rate, tree depth). You tune
hyperparameters using validation performance, never test performance.

**Q: Why do you need a separate validation set if you already have a test
set?**
Repeatedly checking hyperparameter choices against the test set overfits your
*choices* to that specific test set, even though no model parameter was
directly fit to it. The validation set absorbs that repeated-checking risk;
the test set is touched exactly once, at the end, for the number you report.

**Q: What is overfitting, concretely?**
The model has fit patterns specific to the training sample — including
noise — rather than the general relationship between features and label.
Symptom: training performance is much better than validation/test
performance. Fixed by reducing model capacity/flexibility (fewer trees,
shallower trees, more regularization) or adding more/better training data.

**Q: Explain Random Forest to someone who's never heard of it.**
Train many decision trees independently, each on a random resample of the
training rows (bootstrap sampling) and each considering only a random subset
of features at every split. Predict by majority vote across all trees. Each
individual tree overfits its own sample, but since the trees' errors aren't
correlated, averaging them cancels out most of that overfitting while
keeping the genuine signal every tree picked up on.

**Q: Explain gradient boosting to someone who's never heard of it.**
Build trees one at a time, sequentially. Each new tree's job is specifically
to predict the error (residual) left over after all previous trees'
combined predictions. Add each new tree's output to the running total,
scaled down by a learning rate so no single tree can dominate. Repeat for
many rounds. Unlike bagging, the trees are *dependent* on each other in
sequence and can't be built in parallel.

**Q: Bagging vs boosting — what's the fundamental difference?**
Bagging trains estimators independently/in parallel on resampled data and
averages them to reduce variance. Boosting trains estimators sequentially,
each correcting the previous ensemble's errors, primarily reducing bias
(while using shrinkage/regularization to keep variance in check as more
rounds are added).

**Q: What does XGBoost add on top of standard gradient boosting?**
Second-order (Newton) optimization using both gradient and Hessian for a
better local loss approximation; regularization built directly into the
split-gain formula (via `gamma` and leaf-weight penalties); histogram-based
approximate split-finding for speed at scale; stochastic row/column
subsampling per tree; native missing-value handling.

**Q: Why doesn't a tree-based model need feature scaling?**
Trees split on thresholds of individual raw feature values. Any monotonic
rescaling changes the numeric threshold but not which side of it any given
value falls — so scaling can't change what the model learns. It matters for
distance-based or gradient-based models (KNN, SVM, linear/logistic
regression, neural nets) where feature magnitude directly affects the
learned function.

**Q: Precision vs recall — define both and give an example of when you'd
prioritize one over the other.**
Precision = of everything flagged positive, what fraction truly is (`TP/(TP+FP)`).
Recall = of everything truly positive, what fraction was caught
(`TP/(TP+FN)`). Prioritize recall when missing a positive is costly (e.g.
missing actual cancer in a screening test, missing actual phishing in a
security-critical gateway). Prioritize precision when false alarms are
costly (e.g. blocking legitimate email, unnecessary invasive follow-up
testing).

**Q: Why is F1 the harmonic mean rather than the arithmetic mean of
precision and recall?**
The harmonic mean is dominated by the smaller of the two values, so a model
can't get a good F1 by being extreme in one and terrible in the other (e.g.
precision 0.99, recall 0.10 → arithmetic mean 0.545 looks fine, harmonic
mean ~0.18 correctly flags the model as bad). It forces both to be
reasonably good simultaneously.

**Q: What does an ROC-AUC of 0.5 mean? What about 1.0?**
0.5 means the model's ranking of positive vs negative examples is no better
than random chance. 1.0 means perfect separation — every positive example is
scored higher than every negative example. AUC is the probability a randomly
chosen positive is scored above a randomly chosen negative.

**Q: Why use a class-balanced test split even though the real-world class
distribution is imbalanced?**
So metrics like accuracy and F1 have a stable, comparable baseline (a coin
flip = 50%) rather than being inflated by whichever class happens to
dominate. It isolates "is the model actually good" from "is the model
exploiting an easy majority class."

**Q: What is data leakage? Give an example from this project.**
Any way information unavailable at real prediction time influences the
reported evaluation, inflating it. Example: a random train/test split let the
same registrable domain appear on both sides, so the model partly succeeded
on test rows by having already seen that exact domain in training — a form
of group leakage. Measured directly by comparing a random split (F1 0.894)
against a domain-disjoint split holding whole domains apart (F1 0.850): a
~4.4 point gap attributable to memorized domain identity rather than
generalizable URL-structure reasoning.

**Q: How would you detect a spurious correlation / collection artifact in a
dataset?**
Check whether a single crude rule based on a suspicious feature scores far
above the majority-class baseline on its own. In this project: "does the URL
contain `://`" alone scored 61.3% accuracy against a 55.1% baseline — because
the two source datasets had inconsistently kept the scheme prefix, not
because scheme presence has anything to do with phishing.

**Q: Why use RandomizedSearchCV instead of GridSearchCV?**
Grid search cost grows multiplicatively with the number of hyperparameters
being tuned, quickly becoming infeasible. Randomized search samples a fixed
budget of combinations from specified distributions, and empirically
explores the (usually few) truly important hyperparameters more thoroughly
per unit of compute, since it doesn't waste evaluations exhaustively varying
unimportant ones.

**Q: What is k-fold cross-validation, and why stratify it?**
Split training data into k folds; for each round, train on k-1 folds and
validate on the remaining fold; average the k scores. Reduces sensitivity to
any single lucky/unlucky split. Stratifying ensures every fold preserves the
overall class ratio, avoiding a fold that's accidentally starved of one
class — especially important with any class imbalance.

**Q: What's Shannon entropy, and why use it as a phishing-detection
feature?**
`H = -Σ p(c)·log2(p(c))` over character frequencies in a string. Measures
unpredictability: a repeated character has entropy 0, a string with many
distinct characters at even frequency has high entropy. Algorithmically
generated hostnames tend toward higher entropy than domain names a person
chose, so it's a cheap proxy for "does this look machine-generated."

**Q: What's a Public Suffix List and why does it matter here?**
A maintained list of suffixes (like `.com`, but also multi-part ones like
`.co.uk`) under which anyone can register a name. Needed to correctly compute
a URL's *registrable domain* — without it, `www.bbc.co.uk` would naively
resolve to `co.uk` (wrong) instead of `bbc.co.uk`, corrupting every
domain-level feature (brand-mismatch detection, entropy, hyphen counts) built
on top of it.

**Q: Why put a scikit-learn Pipeline around a model that has no
preprocessing steps?**
Two reasons: it fixes a stable "raw features in, prediction out" interface
for the persisted artifact regardless of future changes, and — more
importantly in general — any preprocessing that *is* added later (e.g. a
scaler) sits inside the pipeline, so cross-validation refits it per-fold on
training data only, preventing it from leaking information about the
held-out fold.

**Q: Why is the Random Forest artifact ~40x larger on disk than the
XGBoost one?**
`max_depth=None` lets RF's 364 trees grow completely unbounded (down to
`min_samples_leaf=2`), producing very large, deep trees with many nodes.
XGBoost's trees, while individually deep (`max_depth=12`), are far more
constrained by regularization (`gamma`, shrinkage) that prunes weak splits,
and its 544 trees are far smaller structures each. Capping RF's `max_depth`
would shrink the artifact substantially at a small cost to F1.

**Q: What would you check first if this model's real-world performance
dropped after deployment?**
Whether the input distribution has drifted from what the model was trained
on — phishing tradecraft evolves (new TLD abuse patterns, new free-hosting
providers, new brands being targeted), and this project's own limitations
section flags that its blocklists (`SUSPICIOUS_TLDS`, `FREE_HOSTING`,
`TARGETED_BRANDS`) are static snapshots that need periodic refreshing against
current abuse patterns, and that the whole corpus itself is a point-in-time
snapshot needing periodic retraining.
