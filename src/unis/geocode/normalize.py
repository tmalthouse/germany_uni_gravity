"""The normalization ladder.

Each rung produces a progressively more aggressive key. The same ladder is
applied to student strings and to gazetteer terms, which is the point: matching
happens between two keys built the same way, rather than between a hand-cleaned
student string and whatever form the gazetteer happens to store.

Rungs, coarsening downward:

    raw        the string as given
    fold       casefold, umlaut/eszett expansion, punctuation and space collapse
    modernize  19th-century orthography mapped to modern (c/k, th/t, y/i, ...)
    strip      administrative qualifiers and locational suffixes removed
    phonetic   Koelner Phonetik code

`strip` also returns what it removed, because a stripped qualifier is a
disambiguation hint rather than noise.

All functions here are pure and operate on Polars expressions where possible,
falling back to Python only for the phonetic code (which is inherently
sequential). Phonetic keys are computed once and cached, so the Python cost is
paid on the gazetteer only when the index is rebuilt.
"""

from __future__ import annotations

import re
import unicodedata

import polars as pl

# --------------------------------------------------------------------------
# Rung 1: fold
# --------------------------------------------------------------------------

_UMLAUT_MAP = {
    "ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss",
    "Ä": "ae", "Ö": "oe", "Ü": "ue",
    "å": "aa", "æ": "ae", "ø": "oe", "œ": "oe",
    "Å": "aa", "Æ": "ae", "Ø": "oe", "Œ": "oe",
}


def fold_expr(col: pl.Expr) -> pl.Expr:
    """Casefold, expand umlauts, drop punctuation, collapse whitespace.

    Umlaut expansion runs before Unicode decomposition so that 'ü' becomes
    'ue' rather than 'u'. That matters: the gazetteer may store either, and
    expansion is the form that also matches archaic 'ue' digraph spellings.
    """
    out = col.str.strip_chars()
    for src, dst in _UMLAUT_MAP.items():
        out = out.str.replace_all(src, dst, literal=True)
    # Strip remaining diacritics (French/Polish/Scandinavian place names).
    out = out.str.normalize("NFKD").str.replace_all(r"[\u0300-\u036f]", "")
    out = out.str.to_lowercase()
    # Hyphens and slashes become spaces; other punctuation vanishes.
    out = out.str.replace_all(r"[-/_]", " ")
    out = out.str.replace_all(r"[^\w\s]", "")
    out = out.str.replace_all(r"\s+", " ").str.strip_chars()
    return out


# --------------------------------------------------------------------------
# Rung 1b: expand abbreviations
# --------------------------------------------------------------------------
# Applied to the folded form, where punctuation is already gone, so "a. M."
# and "a/M." have both become "a m".
#
# Expansion rather than stripping. "Frankfurt a. M." stripped of its qualifier
# is just "Frankfurt", which is ambiguous with Frankfurt an der Oder; expanded
# it is "Frankfurt am Main", which resolves exactly. The qualifier is the
# disambiguating information, so discarding it is the wrong move.

_ABBREV_RULES: list[tuple[str, str]] = [
    (r"\ba\s+d\s+o\b", "an der oder"),
    (r"\ba\s+d\s+s\b", "an der saale"),
    (r"\ba\s+d\s+h\b", "an der havel"),
    (r"\ba\s+d\s+l\b", "an der lahn"),
    (r"\ba\s+m\b", "am main"),
    (r"\ba\s+rh\b", "am rhein"),
    (r"\ba\s+n\b", "am neckar"),
    (r"\bi\s+pr\b", "in preussen"),
    (r"\bi\s+b\b", "im breisgau"),
    (r"\bi\s+schles\b", "in schlesien"),
    (r"\bi\s+schl\b", "in schlesien"),
    (r"\bi\s+w\b", "in westfalen"),
    (r"\bi\s+bay\b", "in bayern"),
    (r"\bi\s+th\b", "in thueringen"),
    (r"\bi\s+hann\b", "in hannover"),
    (r"\bi\s+sa\b", "in sachsen"),
    (r"\bst\b", "sankt"),
    (r"\bvorm\s+wald\b", "vorm wald"),
]


def expand_abbrev_expr(col: pl.Expr) -> pl.Expr:
    out = col
    for pattern, replacement in _ABBREV_RULES:
        out = out.str.replace_all(pattern, replacement)
    return out.str.replace_all(r"\s+", " ").str.strip_chars()


# --------------------------------------------------------------------------
# Rung 2: modernize
# --------------------------------------------------------------------------
# Ordered rewrite rules over the *folded* form. Order matters: Latin endings
# are removed before consonant rewrites so that 'wirceburgensis' loses its
# suffix before 'c' handling runs.
#
# Each rule is (pattern, replacement). Patterns are anchored where the rule is
# only valid at a word boundary.

#
# Written for the Rust regex engine Polars uses: no lookaround, no
# backreferences. Capture groups are referenced as $1, converted to \1 for the
# Python fallback below so both implementations stay identical.
#
# Doubled consonants are enumerated rather than matched with a backreference,
# which is why that block is long.

_DOUBLES = "bdfgklmnprst"

_MODERNIZE_RULES: list[tuple[str, str]] = [
    # --- Latin adjectival / genitive endings on place names ---------------
    (r"\b(\w{4,}?)iensis\b", "$1"),
    (r"\b(\w{4,}?)ensis\b", "$1"),       # heidelbergensis -> heidelberg
    (r"\b(\w{4,}?)ensium\b", "$1"),
    (r"\b(\w{4,}?)anus\b", "$1"),        # cassellanus -> cassell
    (r"\b(\w{4,}?)inus\b", "$1"),
    # --- 19th-century orthography ----------------------------------------
    (r"th", "t"),                        # herzogthum, grossherzogthum
    (r"\bc([aouhkqrl])", "k$1"),         # cassel -> kassel, coeln -> koeln
    (r"c([aouk])", "k$1"),               # berncastel -> bernkastel
    (r"\bzw?ey", "zwei"),                # zweybruecken -> zweibruecken
    (r"ey([a-z])", "ei$1"),              # heydingsfeld -> heidingsfeld
    (r"\bmayn", "main"),                 # maynz -> mainz
    (r"y", "i"),                         # residual y -> i (speier/speyer)
    (r"ii+", "i"),
    (r"tz", "z"),
    (r"ck", "k"),
    (r"dt\b", "t"),                      # stuttgardt -> stuttgart
    (r"ph", "f"),
    # "im Badischen" -> baden, "in den Hessischen" -> hessen. Only the dative
    # plural -ischen form: bare -isch is left alone because "Schwedisch
    # Pommern" is a compound name, not an adjectival construction.
    # Braces are required: in the Rust regex crate "$1en" parses as a capture
    # group *named* "1en", not group 1 followed by "en", and silently expands
    # to the empty string.
    (r"(\w{3,}?)ischen\b", "${1}en"),
] + [
    (c + c, c) for c in _DOUBLES        # collapse doubled consonants
] + [
    (r"ss", "s"),
    (r"\s+", " "),
]


def _to_python_pattern(replacement: str) -> str:
    """Convert $1 and ${1} group references to Python's \\1 style."""
    replacement = re.sub(r"\$\{(\d+)\}", r"\\\1", replacement)
    return re.sub(r"\$(\d+)", r"\\\1", replacement)


_MODERNIZE_COMPILED = [
    (re.compile(p), _to_python_pattern(r)) for p, r in _MODERNIZE_RULES
]


def modernize_expr(col: pl.Expr) -> pl.Expr:
    """Apply orthographic modernization to a folded column."""
    out = col
    for pattern, replacement in _MODERNIZE_RULES:
        out = out.str.replace_all(pattern, replacement)
    return out.str.replace_all(r"\s+", " ").str.strip_chars()


def modernize_str(s: str) -> str:
    for pattern, replacement in _MODERNIZE_COMPILED:
        s = pattern.sub(replacement, s)
    return re.sub(r"\s+", " ", s).strip()


# --------------------------------------------------------------------------
# Rung 3: strip qualifiers
# --------------------------------------------------------------------------
# Applied to the *modernized* form. Leading administrative words and trailing
# locational qualifiers are removed; both are returned as hints.

# Leading forms: "herzogthum bremen", "fuerstenthum waldeck", "grafschaft mark".
# Full forms and the truncated variants that actually appear in the registers
# ("Herz. Magdeburg", "Koenigr. Hannover", "Anh. Bernburg", "Cant. St Gallen").
# The abbreviations matter more than the full forms: the full forms were
# already in the correction file, the abbreviations were the ones failing.
_LEAD_QUALIFIERS = [
    "herzogtum", "herzogt", "herz", "hzgt",
    "grosherzogtum", "grossherzogtum", "grossherzogt", "grossherz", "grossh", "ghzt",
    "erzherzogtum", "kurfuerstentum", "kurfuerst",
    "fuerstentum", "furstentum", "fuerstent", "fuerstl", "fstm",
    "koenigreich", "konigreich", "koenigr", "kgr",
    "grafschaft", "landgrafschaft", "markgrafschaft", "gft",
    "reichsstadt", "freie reichsstadt", "freie stadt",
    "provinz", "prov", "kreis", "kr", "amt", "bistum", "erzbistum",
    "republik", "freistaat", "departement", "dept", "dep",
    "canton", "kanton", "cant", "kant",
    "anhalt", "anh", "mecklenburg", "mecklb", "meckl",
    "schwarzburg", "schwarzb", "sachsen", "saechs",
]

# Trailing/embedded locational qualifiers: "greifenberg in schlesien",
# "neustadt bei coburg", "frankfurt an der oder".
_SPLIT_PATTERNS = [
    r"\s+(?:an der|a\s*d|am|a\s*m|ob dem|unter dem|vor dem|im|in|bei|b)\s+(.+)$",
]

# Qualifiers are matched against the *modernized* string, so they have to be
# spelled the way modernization leaves them -- "meckl" has become "mekl" by
# that point and would never match. Running the list through the same
# transform keeps the two in step automatically instead of by hand.
_LEAD_QUALIFIERS_MODERN = sorted(
    {modernize_str(q) for q in _LEAD_QUALIFIERS}, key=len, reverse=True
)
_LEAD_RE = re.compile(r"^(?:" + "|".join(_LEAD_QUALIFIERS_MODERN) + r")\s+")
# "in Pommern", "im Badischen", "aus dem Nassauischen". The old splitter only
# handled prepositions with a place name before them, so a bare prepositional
# phrase never matched anything.
_LEAD_PREP_RE = re.compile(r"^(?:in der|in dem|aus dem|aus der|im|in|aus|bei|zu|von)\s+")
_SPLIT_RE = [re.compile(p) for p in _SPLIT_PATTERNS]
_COMMA_RE = re.compile(r"^([^,]+),\s*(.+)$")


def strip_qualifiers(s: str) -> tuple[str, str | None]:
    """Return (core, hint). Hint is the removed qualifier, or None.

    The hint is not thrown away because it constrains disambiguation: the
    'schlesien' in 'greifenberg in schlesien' tells you which Greifenberg.
    """
    hint_parts: list[str] = []

    m = _COMMA_RE.match(s)
    if m:
        s, tail = m.group(1).strip(), m.group(2).strip()
        hint_parts.append(tail)

    new = _LEAD_PREP_RE.sub("", s)
    if new != s:
        s = new

    new = _LEAD_RE.sub("", s)
    if new != s:
        hint_parts.append(s[: len(s) - len(new)].strip())
        s = new

    for pattern in _SPLIT_RE:
        m = pattern.search(s)
        if m:
            hint_parts.append(m.group(1).strip())
            s = s[: m.start()].strip()
            break

    hint = " ".join(p for p in hint_parts if p) or None
    return s.strip(), hint


# --------------------------------------------------------------------------
# Rung 4: Koelner Phonetik
# --------------------------------------------------------------------------
# Designed for German, and it natively absorbs exactly the variation this
# corpus has: c/k, y/i, th/t, doubled consonants, f/v/w.

_KP_PREP = str.maketrans({"ä": "a", "ö": "o", "ü": "u", "é": "e", "è": "e"})


def koelner_phonetik(word: str) -> str:
    """Return the Koelner Phonetik code for a single word.

    Returns "" for input with no codable letters.
    """
    if not word:
        return ""
    w = word.lower().translate(_KP_PREP).replace("ß", "ss")
    w = unicodedata.normalize("NFKD", w)
    w = "".join(c for c in w if c.isalpha() and c.isascii())
    if not w:
        return ""

    codes: list[str] = []
    n = len(w)
    for i, ch in enumerate(w):
        prev = w[i - 1] if i > 0 else ""
        nxt = w[i + 1] if i + 1 < n else ""

        if ch in "aeijouy":
            code = "0"
        elif ch == "h":
            continue
        elif ch == "b":
            code = "1"
        elif ch == "p":
            code = "3" if nxt == "h" else "1"
        elif ch in "dt":
            code = "8" if nxt in "csz" else "2"
        elif ch in "fvw":
            code = "3"
        elif ch in "gkq":
            code = "4"
        elif ch == "c":
            if i == 0:
                code = "4" if nxt in "ahkloqrux" else "8"
            elif prev in "sz":
                code = "8"
            else:
                code = "4" if nxt in "ahkoqux" else "8"
        elif ch == "x":
            if prev in "ckq":
                code = "8"
            else:
                codes.append("4")
                code = "8"
        elif ch == "l":
            code = "5"
        elif ch in "mn":
            code = "6"
        elif ch == "r":
            code = "7"
        elif ch in "sz":
            code = "8"
        else:
            continue
        codes.append(code)

    # Collapse runs of identical codes.
    deduped: list[str] = []
    for c in codes:
        if not deduped or deduped[-1] != c:
            deduped.append(c)
    # Drop all zeros except a leading one.
    if not deduped:
        return ""
    head, tail = deduped[0], [c for c in deduped[1:] if c != "0"]
    return head + "".join(tail)


def phonetic_key(s: str) -> str:
    """Phonetic key for a possibly multi-word string, words joined by '-'."""
    if not s:
        return ""
    parts = [koelner_phonetik(w) for w in s.split()]
    return "-".join(p for p in parts if p)


# --------------------------------------------------------------------------
# Assembling the ladder
# --------------------------------------------------------------------------

LADDER_COLUMNS = ["k_fold", "k_expand", "k_modern", "k_strip", "k_phon", "hint"]


def add_ladder(df: pl.DataFrame, source: str) -> pl.DataFrame:
    """Add all ladder keys for `source` to an eager DataFrame.

    Operates on a DataFrame of *distinct strings*, not on the full student
    table. Doing it on distinct strings is what keeps the Python-level phonetic
    step affordable.
    """
    df = df.with_columns(
        fold_expr(pl.col(source)).alias("k_fold"),
    ).with_columns(
        expand_abbrev_expr(pl.col("k_fold")).alias("k_expand"),
    ).with_columns(
        modernize_expr(pl.col("k_expand")).alias("k_modern"),
    )

    stripped = [strip_qualifiers(s) for s in df["k_modern"].to_list()]
    df = df.with_columns(
        pl.Series("k_strip", [c for c, _ in stripped], dtype=pl.Utf8),
        pl.Series("hint", [h for _, h in stripped], dtype=pl.Utf8),
    )
    df = df.with_columns(
        pl.Series(
            "k_phon",
            [phonetic_key(s) for s in df["k_strip"].to_list()],
            dtype=pl.Utf8,
        )
    )
    # Empty keys are useless and would match everything. Null them.
    return df.with_columns(
        [
            pl.when(pl.col(c).str.len_chars() == 0).then(None).otherwise(pl.col(c)).alias(c)
            for c in ["k_fold", "k_expand", "k_modern", "k_strip", "k_phon"]
        ]
    )
