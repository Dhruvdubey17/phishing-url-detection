"""Feature extraction from URL strings.

Every feature here is computed from the URL string alone: no HTTP requests, no
DNS, no WHOIS, no page rendering. That constraint is deliberate. A detector that
only needs the string can run inline on a mail gateway or a proxy before anyone
clicks, and it cannot be starved by an attacker who takes the landing page down.

Features are grouped into three families (``FEATURE_GROUPS``) so the ablation
study can measure what each family actually contributes:

``lexical``     — size and character-composition of the string and its parts.
``structural``  — the shape of the URL as a parsed object.
``suspicion``   — targeted heuristics for known phishing tradecraft.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

# --- reference data for the suspicion heuristics -----------------------------

# URL shorteners hide the destination, so phishing campaigns lean on them. This
# is a pattern match against known hosts, not a network lookup, so it is cheap
# but only covers services we have listed.
SHORTENERS = frozenset(
    """bit.ly tinyurl.com goo.gl t.co ow.ly is.gd buff.ly adf.ly bit.do cutt.ly
    rb.gy shorturl.at tiny.cc rebrand.ly bl.ink lnkd.in db.tt qr.ae cli.gs
    u.to v.gd tr.im soo.gd s2r.co clicky.me budurl.com shorte.st""".split()
)

# Words that appear far more often in URLs trying to provoke a credential entry
# than in ordinary browsing.
SUSPICIOUS_KEYWORDS = frozenset(
    """secure account verify login update banking confirm signin sign-in
    password credential webscr ebayisapi alert suspend suspended billing
    invoice wallet recover unlock validate authenticate security limited
    unusual verification submit customer client service support""".split()
)

# Brands whose names get borrowed into hostnames the brand does not own.
TARGETED_BRANDS = frozenset(
    """paypal apple google amazon microsoft facebook netflix ebay chase
    wellsfargo bankofamerica citibank hsbc santander dhl fedex ups usps
    whatsapp instagram linkedin dropbox steam roblox outlook office365
    icloud yahoo gmail alibaba coinbase binance blockchain metamask""".split()
)

# TLDs that are cheap or free to register and are consistently over-represented
# in abuse feeds relative to their share of the web.
SUSPICIOUS_TLDS = frozenset(
    """top xyz club online site tk ml ga cf gq work click link buzz icu cam
    rest fit surf website space store tech men loan download racing win
    review party stream bid trade date faith accountant science""".split()
)

# Providers that hand out a subdomain or a dynamic-DNS name to anyone, free and
# instantly. The registrable domain then belongs to the provider and says
# nothing about who is serving the content, which is exactly why abuse
# concentrates here. Treating the provider as the "domain" would also make the
# brand and entropy checks meaningless, so these are flagged explicitly.
FREE_HOSTING = frozenset(
    """000webhostapp.com duckdns.org firebaseapp.com herokuapp.com appspot.com
    web.app netlify.app vercel.app glitch.me repl.co replit.dev ngrok.io
    ngrok-free.app serveo.net localtunnel.me weebly.com wixsite.com
    squarespace.com blogspot.com wordpress.com github.io gitlab.io
    pages.dev workers.dev r2.dev sites.google.com googleusercontent.com
    azurewebsites.net cloudapp.net amazonaws.com s3.amazonaws.com
    servebeer.com serveblog.net servegame.com hopto.org no-ip.org no-ip.biz
    zapto.org myftp.biz myftp.org bounceme.net redirectme.net ddns.net
    freedynamicdns.net linkpc.net onedrive.live.com dropboxusercontent.com
    altervista.org 000space.com byethost.com epizy.com rf.gd infinityfree.net
    freehostia.com hostinger.com neocities.org tripod.com angelfire.com
    yolasite.com jimdo.com webnode.com strikingly.com carrd.co
    surge.sh render.com fly.dev onrender.com""".split()
)

# The public suffix list proper has thousands of entries and needs periodic
# refreshing. These are the multi-label suffixes that actually show up in this
# corpus; without them "bbc.co.uk" registers as "co.uk" and every domain-level
# feature (entropy, hyphens, brand match) is computed on the wrong string.
MULTI_PART_SUFFIXES = frozenset(
    """co.uk org.uk ac.uk gov.uk net.uk sch.uk ltd.uk plc.uk me.uk
    com.au net.au org.au edu.au gov.au id.au asn.au
    com.br net.br org.br gov.br edu.br
    co.jp ne.jp or.jp ac.jp go.jp
    com.cn net.cn org.cn gov.cn edu.cn ac.cn
    co.in net.in org.in gen.in firm.in ind.in
    com.mx org.mx net.mx gob.mx
    co.za org.za net.za gov.za ac.za
    com.tr net.tr org.tr gov.tr edu.tr
    com.sg net.sg org.sg edu.sg gov.sg
    com.hk net.hk org.hk idv.hk gov.hk
    com.tw net.tw org.tw idv.tw gov.tw
    co.kr or.kr ne.kr go.kr re.kr pe.kr
    com.ar net.ar org.ar gob.ar
    com.co net.co org.co gov.co
    com.pe net.pe org.pe gob.pe
    com.ph net.ph org.ph gov.ph
    com.my net.my org.my gov.my edu.my
    com.vn net.vn org.vn gov.vn edu.vn
    co.nz net.nz org.nz govt.nz ac.nz
    com.pl net.pl org.pl gov.pl edu.pl waw.pl
    co.il net.il org.il ac.il gov.il
    com.ua net.ua org.ua gov.ua kiev.ua
    com.ru net.ru org.ru msk.ru spb.ru
    co.id or.id ac.id go.id web.id
    com.pk net.pk org.pk gov.pk
    com.eg net.eg org.eg gov.eg
    com.ng net.ng org.ng gov.ng
    com.sa net.sa org.sa gov.sa
    com.es org.es gob.es edu.es
    com.pt net.pt org.pt gov.pt
    com.gr net.gr org.gr gov.gr
    com.uy net.uy org.uy gub.uy
    com.ve net.ve org.ve gob.ve
    com.ec net.ec org.ec gob.ec
    com.bd net.bd org.bd gov.bd
    com.np net.np org.np gov.np
    com.lk net.lk org.lk gov.lk
    com.kw net.kw org.kw gov.kw
    com.bo net.bo org.bo gob.bo
    com.do net.do org.do gob.do
    com.gt net.gt org.gt gob.gt
    com.py net.py org.py gov.py
    com.cy net.cy org.cy gov.cy
    com.mt net.mt org.mt gov.mt
    co.th in.th ac.th go.th or.th
    co.ke or.ke ne.ke go.ke ac.ke
    co.tz or.tz ne.tz go.tz ac.tz
    co.ug or.ug ne.ug go.ug ac.ug
    co.zw org.zw gov.zw ac.zw
    com.gh org.gh gov.gh edu.gh
    com.na org.na net.na
    com.bh net.bh org.bh gov.bh
    com.om net.om org.om gov.om
    com.qa net.qa org.qa gov.qa
    com.jo net.jo org.jo gov.jo
    com.lb net.lb org.lb gov.lb
    com.af net.af org.af gov.af
    com.al net.al org.al gov.al
    com.hr com.mk com.ba com.me
    org.il co.ma net.ma org.ma
    com.tn net.tn org.tn gov.tn
    com.dz net.dz org.dz gov.dz
    com.ly net.ly org.ly gov.ly
    com.sd net.sd org.sd gov.sd
    com.et net.et org.et gov.et
    com.mm net.mm org.mm gov.mm
    com.kh net.kh org.kh gov.kh
    com.la net.la org.la gov.la
    com.bn net.bn org.bn gov.bn
    com.mo net.mo org.mo gov.mo
    com.fj net.fj org.fj gov.fj
    com.pg net.pg org.pg gov.pg
    com.sb net.sb org.sb gov.sb
    com.vu net.vu org.vu gov.vu
    com.to net.to org.to gov.to
    com.ws net.ws org.ws gov.ws
    com.ki net.ki org.ki gov.ki
    com.nf net.nf org.nf
    com.cu net.cu org.cu gob.cu
    com.jm net.jm org.jm gov.jm
    com.tt net.tt org.tt gov.tt
    com.bb net.bb org.bb gov.bb
    com.bs net.bs org.bs gov.bs
    com.bz net.bz org.bz gov.bz
    com.sv net.sv org.sv gob.sv
    com.hn net.hn org.hn gob.hn
    com.ni net.ni org.ni gob.ni
    com.pa net.pa org.pa gob.pa
    com.cr net.cr org.cr go.cr
    com.pr net.pr org.pr gov.pr""".split()
)

# TLDs given their own indicator, ranked by how often they appear on the web at
# large. Encoding the TLD as a rank rather than one-hot keeps it to a single
# column while still separating the common web from the long tail.
COMMON_TLDS = (
    "com org net ru de jp uk br it pl fr au info nl in cn es cz kr ca eu ch"
    " se no be dk fi at gr pt hu ro il tr mx ar za tw hk sg id th vn ph my"
    " io co me tv cc biz us gov edu mil"
).split()
TLD_RANK = {tld: i for i, tld in enumerate(COMMON_TLDS)}
_UNRANKED_TLD = len(COMMON_TLDS)

_CONSONANT_RUN_RE = re.compile(r"[bcdfghjklmnpqrstvwxz]+")
_HEX_TOKEN_RE = re.compile(r"\b[0-9a-f]{8,}\b")
_EXECUTABLE_EXTS = (".exe", ".scr", ".zip", ".rar", ".apk", ".jar", ".msi", ".dmg", ".bat", ".ps1")

_SCHEME_RE = re.compile(r"^([A-Za-z][A-Za-z0-9+.\-]*)://")
_IPV4_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")
_HEX_IP_RE = re.compile(r"^0[xX][0-9a-fA-F]+$")
_RUN_RE = re.compile(r"(.)\1*")
_SPECIALS = "-_?=&%@"
_DIGITS = "0123456789"

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "lexical": (
        "url_length",
        "hostname_length",
        "path_length",
        "query_length",
        "num_digits",
        "num_letters",
        "num_special_chars",
        "num_dots",
        "num_hyphens",
        "num_slashes",
        "num_subdomains",
        "num_dots_in_domain",
        "has_at_symbol",
        "has_hyphen_in_domain",
        "tld_length",
        "host_digit_count",
        "host_hyphen_count",
        "num_host_tokens",
        "longest_host_token",
        "subdomain_length",
    ),
    "structural": (
        "uses_https",
        "has_port",
        "has_ip_host",
        "num_path_segments",
        "longest_path_segment",
        "num_query_params",
        "has_fragment",
        "has_userinfo",
        "has_double_slash_redirect",
        "path_to_url_ratio",
        "has_file_extension",
        "has_executable_extension",
        "num_encoded_chars",
    ),
    "suspicion": (
        "is_shortener",
        "num_suspicious_keywords",
        "has_suspicious_tld",
        "brand_domain_mismatch",
        "digit_letter_ratio",
        "domain_entropy",
        "url_entropy",
        "longest_consecutive_run",
        "has_punycode",
        "vowel_ratio_domain",
        "is_free_hosting",
        "tld_rank",
        "has_hex_token",
        "longest_consonant_run",
    ),
}

FEATURE_NAMES: tuple[str, ...] = tuple(n for g in FEATURE_GROUPS.values() for n in g)


def _parse(url: str) -> tuple[str, str, str, str, str, str, str]:
    """Split a URL into parts without raising on malformed input.

    ``urllib.parse`` is avoided here on purpose: it treats a scheme-less
    ``example.com:8080/x`` as having scheme ``example.com``, and it raises
    ``ValueError`` from ``.port``/``.hostname`` on input that this corpus
    contains plenty of. Feature extraction must never raise, so we split by hand.

    Returns ``(scheme, host, port, path, query, fragment, userinfo)``.
    """
    match = _SCHEME_RE.match(url)
    scheme, rest = (match.group(1).lower(), url[match.end() :]) if match else ("", url)

    rest, _, fragment = rest.partition("#")
    rest, _, query = rest.partition("?")
    netloc, slash, path = rest.partition("/")
    path = slash + path

    userinfo, _, netloc = netloc.rpartition("@")

    if netloc.startswith("["):  # bracketed IPv6 literal
        end = netloc.find("]")
        host, port = (netloc[: end + 1], netloc[end + 1 :].lstrip(":")) if end != -1 else (netloc, "")
    else:
        host, _, port = netloc.partition(":")

    return scheme, host.lower(), port, path, query, fragment, userinfo


def _entropy(text: str) -> float:
    """Shannon entropy over the characters of ``text``, in bits.

    Algorithmically generated hostnames (domain-generation algorithms, throwaway
    subdomains like ``x7fk2q9.example.top``) spread their characters much more
    evenly than words a human chose, so they sit at higher entropy.
    """
    if not text:
        return 0.0
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in Counter(text).values())


def _registered_domain(host_parts: list[str]) -> str:
    """The registrable domain: the public suffix plus one label.

    ``["www", "bbc", "co", "uk"]`` gives ``bbc.co.uk``, not ``co.uk``. Getting
    this wrong shifts every domain-level feature onto a string the registrant
    never chose, which quietly breaks the brand, entropy and hyphen checks.
    """
    if len(host_parts) < 2:
        return host_parts[0] if host_parts else ""
    if ".".join(host_parts[-2:]) in MULTI_PART_SUFFIXES and len(host_parts) >= 3:
        return ".".join(host_parts[-3:])
    return ".".join(host_parts[-2:])


def _is_ip_host(host: str) -> bool:
    """True when the host is a raw address rather than a name.

    A literal address means no domain was registered and no certificate name has
    to match, which is common in phishing and rare in legitimate links a user
    would be sent.
    """
    if host.startswith("[") or _HEX_IP_RE.match(host):
        return True
    if _IPV4_RE.match(host):
        return all(int(p) < 256 for p in host.split("."))
    return False


def extract_features(url: str) -> dict[str, float]:
    """Compute every feature for a single URL.

    Never raises: any input, including ``""``, ``None``-ish junk, or a megabyte
    of garbage, yields a complete dict with all ``FEATURE_NAMES`` keys.
    """
    url = "" if url is None else str(url)
    scheme, host, port, path, query, fragment, userinfo = _parse(url)

    host_parts = host.split(".") if host else []
    is_ip = _is_ip_host(host)
    tld = host_parts[-1] if len(host_parts) > 1 and not is_ip else ""
    registered = "" if is_ip else _registered_domain(host_parts)
    subdomain = host[: -len(registered) - 1] if registered and host.endswith(registered) else ""

    num_digits = sum(map(url.count, _DIGITS))
    num_letters = sum(c.isalpha() for c in url)
    segments = [s for s in path.split("/") if s]
    last_segment = segments[-1] if segments else ""
    extension = last_segment[last_segment.rfind(".") :].lower() if "." in last_segment else ""

    # A '//' inside the path is the classic open-redirect / path-confusion trick
    # ("site.com/redirect//evil.com"); the scheme's own '//' is already stripped.
    double_slash = "//" in path

    domain_letters = [c for c in registered if c.isalpha()]

    features = {
        # --- lexical ---
        "url_length": len(url),
        "hostname_length": len(host),
        "path_length": len(path),
        "query_length": len(query),
        "num_digits": num_digits,
        "num_letters": num_letters,
        "num_special_chars": sum(map(url.count, _SPECIALS)),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_slashes": url.count("/"),
        # Deep subdomain nesting lets an attacker put a trusted-looking word in
        # front of a domain they actually control.
        "num_subdomains": max(0, len(host_parts) - 2) if not is_ip else 0,
        "num_dots_in_domain": host.count("."),
        # '@' makes everything to its left userinfo, so the real host hides at
        # the end of a string that opens with a familiar brand.
        "has_at_symbol": float("@" in url),
        "has_hyphen_in_domain": float("-" in registered),
        "tld_length": len(tld),
        "host_digit_count": sum(map(host.count, _DIGITS)),
        "host_hyphen_count": host.count("-"),
        "num_host_tokens": len(host_parts),
        "longest_host_token": max((len(p) for p in host_parts), default=0),
        # A long subdomain is where "secure-login-yourbank-verify" gets parked in
        # front of a domain the attacker owns.
        "subdomain_length": len(subdomain),
        # --- structural ---
        "uses_https": float(scheme == "https"),
        "has_port": float(bool(port)),
        "has_ip_host": float(is_ip),
        "num_path_segments": len(segments),
        "longest_path_segment": max((len(s) for s in segments), default=0),
        "num_query_params": query.count("=") if query else 0,
        "has_fragment": float(bool(fragment)),
        "has_userinfo": float(bool(userinfo)),
        "has_double_slash_redirect": float(double_slash),
        "path_to_url_ratio": len(path) / len(url) if url else 0.0,
        "has_file_extension": float(bool(extension)),
        # A link that lands directly on an installer or archive is a malware
        # delivery pattern rather than a credential-harvest page, but both sit
        # on the malicious side of this label.
        "has_executable_extension": float(extension in _EXECUTABLE_EXTS),
        # Percent-encoding in the path or query is a routine way to obscure
        # keywords from a naive string scanner.
        "num_encoded_chars": url.count("%"),
        # --- suspicion ---
        "is_shortener": float(registered in SHORTENERS or host in SHORTENERS),
        "num_suspicious_keywords": sum(k in url.lower() for k in SUSPICIOUS_KEYWORDS),
        "has_suspicious_tld": float(tld in SUSPICIOUS_TLDS),
        # The brand name shows up somewhere in the URL, but the registered
        # domain is not the brand's: "paypal.secure-login.xyz" trips this,
        # "paypal.com/login" does not.
        "brand_domain_mismatch": float(
            any(b in url.lower() for b in TARGETED_BRANDS)
            and not any(registered.startswith(b + ".") or registered == b for b in TARGETED_BRANDS)
        ),
        "digit_letter_ratio": num_digits / num_letters if num_letters else float(num_digits > 0),
        "domain_entropy": _entropy(registered),
        "url_entropy": _entropy(url),
        # Long runs of one character ("aaaaaa", "------") show up in generated
        # or padded hostnames far more than in names someone typed on purpose.
        "longest_consecutive_run": max((len(m.group(0)) for m in _RUN_RE.finditer(url)), default=0),
        # Punycode encodes non-ASCII; it is how homograph lookalikes such as
        # "аpple.com" (Cyrillic 'а') reach the wire.
        "has_punycode": float("xn--" in host),
        "vowel_ratio_domain": (
            sum(c in "aeiou" for c in domain_letters) / len(domain_letters) if domain_letters else 0.0
        ),
        # Free hosting and dynamic DNS: the registrable domain belongs to the
        # provider, so nothing about it identifies who is actually serving.
        "is_free_hosting": float(
            registered in FREE_HOSTING or any(host.endswith("." + p) for p in FREE_HOSTING)
        ),
        # Rank of the TLD among the common web; everything off the list shares
        # one bucket at the end.
        "tld_rank": TLD_RANK.get(tld, _UNRANKED_TLD),
        # Long hex strings are session tokens and generated identifiers, common
        # in throwaway phishing paths and rare in URLs meant to be readable.
        "has_hex_token": float(bool(_HEX_TOKEN_RE.search(url.lower()))),
        # Unpronounceable consonant clusters are the clearest cheap signal that
        # a hostname was generated rather than chosen.
        "longest_consonant_run": max(
            (len(m.group(0)) for m in _CONSONANT_RUN_RE.finditer(registered)), default=0
        ),
    }
    return features


def _extract_chunk(urls: list[str]) -> list[tuple[float, ...]]:
    """Worker entry point: extract a chunk of URLs into positional tuples."""
    return [tuple(extract_features(u).values()) for u in urls]


def extract_features_frame(
    urls: pd.Series | list[str], n_jobs: int = -1, chunk_size: int = 20_000
) -> pd.DataFrame:
    """Extract features for many URLs, in parallel across processes.

    A single ``extract_features`` implementation serves both this batch path and
    single-URL callers, so there is no second copy of the logic that could drift
    out of agreement. ``tests/test_features.py`` asserts the two paths match.
    """
    values = list(urls)
    if not values:
        return pd.DataFrame(columns=list(FEATURE_NAMES), dtype=np.float32)

    chunks = [values[i : i + chunk_size] for i in range(0, len(values), chunk_size)]
    if n_jobs == 1 or len(chunks) == 1:
        rows = [row for chunk in chunks for row in _extract_chunk(chunk)]
    else:
        workers = None if n_jobs == -1 else n_jobs
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = [row for chunk in pool.map(_extract_chunk, chunks) for row in chunk]

    frame = pd.DataFrame(rows, columns=list(FEATURE_NAMES), dtype=np.float32)
    index = urls.index if isinstance(urls, pd.Series) else None
    if index is not None:
        frame.index = index
    return frame
