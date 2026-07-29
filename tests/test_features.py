"""Unit tests for URL feature extraction."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.data_loader import canonicalize
from src.features import (
    COMMON_TLDS,
    FEATURE_GROUPS,
    FEATURE_NAMES,
    extract_features,
    extract_features_frame,
    _registered_domain,
)

# Input that has broken naive URL parsers: no scheme, bare punctuation, stray
# delimiters, credentials, non-ASCII, and a payload far past any sane length.
PATHOLOGICAL = [
    "",
    " ",
    "/",
    "///",
    ":",
    "://",
    "http://",
    "https://///",
    "?",
    "#",
    "@",
    "http://@@@",
    "not a url at all",
    "http://[::1]/x",
    "http://[not-closed/x",
    "example.com:notaport/x",
    "http://user:pw@host.com:8080/p?q=1#f",
    "пример.рф/путь",
    "xn--e1afmkfd.xn--p1ai/path",
    "a" * 5000,
    "http://" + "sub." * 200 + "example.com",
    "%%%%",
    "javascript:alert(1)",
    "ftp://1.2.3.4:21/f",
]


def test_feature_names_are_unique_and_partitioned():
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))
    assert len(FEATURE_NAMES) >= 18, "the project requires at least 18 features"
    flattened = [name for group in FEATURE_GROUPS.values() for name in group]
    assert flattened == list(FEATURE_NAMES)
    groups = [set(g) for g in FEATURE_GROUPS.values()]
    for i, a in enumerate(groups):
        for b in groups[i + 1 :]:
            assert not (a & b), "feature groups must not overlap, ablation depends on it"


@pytest.mark.parametrize("url", PATHOLOGICAL)
def test_never_raises_and_always_returns_full_schema(url):
    features = extract_features(url)
    assert list(features) == list(FEATURE_NAMES)
    for name, value in features.items():
        assert isinstance(value, (int, float)), name
        assert not math.isnan(value), name
        assert math.isfinite(value), name


def test_none_is_tolerated():
    assert extract_features(None) == extract_features("")


def test_empty_string_is_all_zeros_apart_from_the_tld_bucket():
    features = extract_features("")
    # An empty URL has no TLD, which lands in the "not a common TLD" bucket
    # rather than at rank 0 (rank 0 is ".com"). Everything else is genuinely 0.
    assert features["tld_rank"] == len(COMMON_TLDS)
    assert set(features.values()) - {features["tld_rank"]} == {0, 0.0}


def test_plain_hostname_without_scheme():
    f = extract_features("example.com")
    assert f["hostname_length"] == 11
    assert f["path_length"] == 0
    assert f["num_path_segments"] == 0
    assert f["tld_length"] == 3
    assert f["num_subdomains"] == 0
    assert f["uses_https"] == 0
    assert f["has_ip_host"] == 0


def test_ipv4_host_is_detected():
    f = extract_features("http://192.168.1.1:8080/login")
    assert f["has_ip_host"] == 1
    assert f["has_port"] == 1
    assert f["num_subdomains"] == 0, "an address has no subdomains to count"
    assert f["tld_length"] == 0, "an address has no TLD"


def test_values_that_look_like_ipv4_but_are_not():
    assert extract_features("999.999.999.999/x")["has_ip_host"] == 0
    assert extract_features("1.2.3/x")["has_ip_host"] == 0
    assert extract_features("http://[::1]/x")["has_ip_host"] == 1


def test_https_and_port_and_userinfo():
    f = extract_features("https://user:pw@shop.example.com:8443/a/b?x=1&y=2#top")
    assert f["uses_https"] == 1
    assert f["has_port"] == 1
    assert f["has_userinfo"] == 1
    assert f["has_at_symbol"] == 1
    assert f["has_fragment"] == 1
    assert f["num_path_segments"] == 2
    assert f["num_query_params"] == 2
    assert f["num_subdomains"] == 1


def test_scheme_is_not_confused_with_host_when_port_present():
    # urllib.parse reads "example.com" as the scheme here; the hand-rolled
    # splitter must not make that mistake.
    f = extract_features("example.com:8080/path")
    assert f["hostname_length"] == len("example.com")
    assert f["has_port"] == 1
    assert f["uses_https"] == 0


def test_punycode_detected():
    assert extract_features("xn--e1afmkfd.xn--p1ai/path")["has_punycode"] == 1
    assert extract_features("example.com")["has_punycode"] == 0


def test_unicode_domain_is_handled_without_error():
    f = extract_features("пример.рф/путь")
    assert f["url_length"] == len("пример.рф/путь")
    assert f["has_punycode"] == 0


def test_very_long_url():
    url = "http://evil.com/" + "a" * 5000
    f = extract_features(url)
    assert f["url_length"] == len(url)
    assert f["longest_consecutive_run"] == 5000
    assert 0 <= f["path_to_url_ratio"] <= 1


def test_shortener_detection():
    assert extract_features("bit.ly/3xK9a")["is_shortener"] == 1
    assert extract_features("http://tinyurl.com/abcd")["is_shortener"] == 1
    assert extract_features("example.com/bit.ly")["is_shortener"] == 0


def test_brand_mismatch_fires_only_on_impersonation():
    assert extract_features("paypal.secure-login.xyz/webscr")["brand_domain_mismatch"] == 1
    assert extract_features("paypal.com/signin")["brand_domain_mismatch"] == 0
    assert extract_features("www.paypal.com/us/signin")["brand_domain_mismatch"] == 0
    assert extract_features("example.com/about")["brand_domain_mismatch"] == 0


def test_suspicious_tld_and_keywords():
    f = extract_features("account-verify.login.top/secure/confirm")
    assert f["has_suspicious_tld"] == 1
    assert f["num_suspicious_keywords"] >= 4
    assert extract_features("example.com/index.html")["has_suspicious_tld"] == 0


def test_double_slash_redirect_only_counts_the_path():
    assert extract_features("https://site.com/redirect//evil.com")["has_double_slash_redirect"] == 1
    assert extract_features("https://site.com/a/b")["has_double_slash_redirect"] == 0


def test_entropy_ordering_matches_intuition():
    random_domain = extract_features("x7fk2q9zvb.top")["domain_entropy"]
    repetitive = extract_features("aaaaaaaaaa.top")["domain_entropy"]
    assert random_domain > repetitive
    assert extract_features("")["domain_entropy"] == 0


def test_digit_letter_ratio_handles_zero_letters():
    f = extract_features("12345")
    assert math.isfinite(f["digit_letter_ratio"])
    assert f["digit_letter_ratio"] > 0


def test_counts_are_consistent_with_the_string():
    url = "a1-b2_c3.example.com/p?x=1&y=2"
    f = extract_features(url)
    assert f["num_digits"] == sum(c.isdigit() for c in url)
    assert f["num_letters"] == sum(c.isalpha() for c in url)
    assert f["num_dots"] == url.count(".")
    assert f["num_hyphens"] == url.count("-")
    assert f["num_slashes"] == url.count("/")


def test_batch_extraction_matches_single_extraction():
    frame = extract_features_frame(pd.Series(PATHOLOGICAL), n_jobs=1)
    assert frame.shape == (len(PATHOLOGICAL), len(FEATURE_NAMES))
    for i, url in enumerate(PATHOLOGICAL):
        expected = np.array(list(extract_features(url).values()), dtype=np.float32)
        np.testing.assert_allclose(frame.iloc[i].to_numpy(), expected, rtol=1e-6)


def test_batch_extraction_is_finite_and_preserves_index():
    series = pd.Series(PATHOLOGICAL, index=range(100, 100 + len(PATHOLOGICAL)))
    frame = extract_features_frame(series, n_jobs=1)
    assert list(frame.index) == list(series.index)
    assert not frame.isna().to_numpy().any()
    assert np.isfinite(frame.to_numpy()).all()


def test_empty_batch_returns_empty_frame_with_schema():
    frame = extract_features_frame([], n_jobs=1)
    assert frame.empty
    assert list(frame.columns) == list(FEATURE_NAMES)


def test_registered_domain_respects_multi_part_suffixes():
    assert _registered_domain("www.bbc.co.uk".split(".")) == "bbc.co.uk"
    assert _registered_domain("bbc.co.uk".split(".")) == "bbc.co.uk"
    assert _registered_domain("a.b.example.com".split(".")) == "example.com"
    assert _registered_domain("example.com".split(".")) == "example.com"
    assert _registered_domain(["localhost"]) == "localhost"
    assert _registered_domain([]) == ""


def test_brand_check_uses_the_registrable_domain_not_the_suffix():
    # Without multi-part suffix handling this reads as domain "co.uk" and
    # wrongly reports impersonation.
    assert extract_features("paypal.co.uk/signin")["brand_domain_mismatch"] == 0
    assert extract_features("paypal.login.co.uk/signin")["brand_domain_mismatch"] == 1


def test_free_hosting_detection():
    assert extract_features("login.duckdns.org/verify")["is_free_hosting"] == 1
    assert extract_features("000webhostapp.com/x")["is_free_hosting"] == 1
    assert extract_features("a.b.herokuapp.com/x")["is_free_hosting"] == 1
    assert extract_features("example.com/duckdns.org")["is_free_hosting"] == 0


def test_tld_rank_buckets_the_long_tail():
    assert extract_features("example.com")["tld_rank"] == 0
    assert extract_features("example.top")["tld_rank"] == len(COMMON_TLDS)
    assert extract_features("example.org")["tld_rank"] < len(COMMON_TLDS)


def test_subdomain_length_excludes_the_registrable_domain():
    assert extract_features("example.com")["subdomain_length"] == 0
    assert extract_features("www.example.com")["subdomain_length"] == 3
    assert extract_features("secure-login-verify.example.com")["subdomain_length"] == 19
    assert extract_features("www.bbc.co.uk")["subdomain_length"] == 3


def test_executable_extension_and_encoding():
    assert extract_features("site.com/a/setup.exe")["has_executable_extension"] == 1
    assert extract_features("site.com/a/index.html")["has_executable_extension"] == 0
    assert extract_features("site.com/a/index.html")["has_file_extension"] == 1
    assert extract_features("site.com/a/page")["has_file_extension"] == 0
    assert extract_features("site.com/%2Fa%2Fb")["num_encoded_chars"] == 2


def test_hex_token_and_consonant_run():
    assert extract_features("site.com/a1b2c3d4e5f6")["has_hex_token"] == 1
    assert extract_features("site.com/hello")["has_hex_token"] == 0
    assert extract_features("bcdfghjklm.top")["longest_consonant_run"] > 5
    assert extract_features("banana.com")["longest_consonant_run"] <= 2


def test_canonicalize_removes_collection_artifacts():
    raw = pd.Series(
        ["https://Example.COM/Path/", "http://example.com", "example.com/", "HTTP://A.B.C/x?q=1"]
    )
    out = canonicalize(raw)
    assert out.tolist() == ["example.com/Path", "example.com", "example.com", "a.b.c/x?q=1"]
    assert not out.str.contains("://").any()


def test_canonicalize_preserves_path_case_but_lowers_host():
    assert canonicalize(pd.Series(["WWW.Example.com/CamelCase"]))[0] == "www.example.com/CamelCase"
