#!/usr/bin/env python3
"""
Name normalisation for 19th-century German university registers.

Built around the finding that given names are abbreviated in ~49% of records,
and that the abbreviation rate is school-specific (Berlin 99.8%, Giessen 0.4%).
The consequence is that given names must be compared as INITIAL SEQUENCES,
with full-token agreement available only as a bonus level.

Provides:
    koelner(s)            Koelner Phonetik code (German-tuned, unlike Soundex)
    canon_surname(s)      orthographic surname normalisation + particle split
    canon_given(s)        Latin/variant -> vernacular canonical given tokens
    initials(s)           normalised initial sequence (C/K and I/J folded)
    add_columns(con, tbl) materialise all derived columns in DuckDB

    GIVEN_COMPARISON_SQL  CASE expression implementing the comparison levels
"""

import re
import unicodedata

# ---------------------------------------------------------------------------
# orthography
# ---------------------------------------------------------------------------

_UML = {"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss",
        "Ä": "ae", "Ö": "oe", "Ü": "ue", "é": "e", "è": "e", "ç": "c"}

PARTICLES = ("von der", "von dem", "vom", "von", "van der", "van den", "van",
             "zu", "zur", "zum", "de la", "de", "di", "du", "af", "auf dem")


def _fold(s):
    if not s:
        return ""
    s = s.lower()
    s = "".join(_UML.get(ch, ch) for ch in s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s


def canon_surname(s):
    """Return (particle, normalised_surname)."""
    s = _fold(s)
    s = re.sub(r"[^a-z' -]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    particle = ""
    for p in PARTICLES:
        if s.startswith(p + " "):
            particle, s = p, s[len(p) + 1:]
            break
    # Note: do NOT rewrite ph/th/dt/ck here. Koelner already folds them, and
    # pre-rewriting corrupts its context rules. Only doubled letters are safe.
    s = re.sub(r"(.)\1+", r"\1", s)
    return particle, s.strip()


# ---------------------------------------------------------------------------
# Koelner Phonetik
# ---------------------------------------------------------------------------

def koelner(s):
    """Koelner Phonetik. Tuned to German; use instead of Soundex/NYSIIS."""
    s = _fold(s)
    s = re.sub(r"[^a-z]", "", s)
    if not s:
        return ""
    codes = []
    n = len(s)
    for i, ch in enumerate(s):
        prev = s[i - 1] if i > 0 else ""
        nxt = s[i + 1] if i + 1 < n else ""
        if ch in "aeijouy":
            c = "0"
        elif ch == "h":
            continue
        elif ch == "b":
            c = "1"
        elif ch == "p":
            c = "3" if nxt == "h" else "1"
        elif ch in "dt":
            c = "8" if (nxt and nxt in "csz") else "2"
        elif ch in "fvw":
            c = "3"
        elif ch in "gkq":
            c = "4"
        elif ch == "c":
            if i == 0:
                c = "4" if (nxt and nxt in "ahkloqrux") else "8"
            elif prev and prev in "sz":
                c = "8"
            else:
                c = "4" if (nxt and nxt in "ahkoqux") else "8"
        elif ch == "x":
            c = "8" if (prev and prev in "ckq") else "48"
        elif ch == "l":
            c = "5"
        elif ch in "mn":
            c = "6"
        elif ch == "r":
            c = "7"
        elif ch in "sz":
            c = "8"
        else:
            continue
        codes.append(c)
    out = "".join(codes)
    out = re.sub(r"(.)\1+", r"\1", out)              # collapse repeats
    return out[0] + out[1:].replace("0", "") if out else ""


# ---------------------------------------------------------------------------
# given names
# ---------------------------------------------------------------------------

# Latinised forms -> vernacular. Marburg is 70.8% latinate, Freiburg 16.9%,
# Kiel 10.2%; the rest are low. Extend from your own top-token frequencies.
LATIN = {
    "carolus": "karl", "caroli": "karl",
    "joannes": "johann", "ioannes": "johann", "johannes": "johann",
    "joannis": "johann", "ioannis": "johann",
    "guilielmus": "wilhelm", "wilhelmus": "wilhelm", "gulielmus": "wilhelm",
    "franciscus": "franz", "josephus": "joseph", "iosephus": "joseph",
    "henricus": "heinrich", "heinricus": "heinrich",
    "fridericus": "friedrich", "friedericus": "friedrich",
    "georgius": "georg", "ludovicus": "ludwig", "augustus": "august",
    "augustinus": "augustin", "christianus": "christian",
    "antonius": "anton", "jacobus": "jakob", "iacobus": "jakob",
    "petrus": "peter", "paulus": "paul", "martinus": "martin",
    "nicolaus": "nikolaus", "philippus": "philipp",
    "theodorus": "theodor", "ernestus": "ernst", "adolphus": "adolf",
    "gustavus": "gustav", "mauritius": "moritz",
    "sebastianus": "sebastian", "bernardus": "bernhard",
    "conradus": "konrad", "eduardus": "eduard",
    "ferdinandus": "ferdinand", "gothofredus": "gottfried",
    "hermannus": "hermann", "leopoldus": "leopold",
    "rudolphus": "rudolf", "stephanus": "stephan",
    "valentinus": "valentin", "vincentius": "vinzenz",
    "xaverius": "xaver", "aloysius": "alois", "ignatius": "ignaz",
    "benedictus": "benedikt", "gregorius": "gregor",
    "laurentius": "lorenz", "matthaeus": "matthias", "mathias": "matthias",
    "michaelis": "michael", "emanuel": "immanuel",
}

VARIANT = {
    "carl": "karl", "karl": "karl", "carolus": "karl",
    "joh": "johann", "hans": "johann", "johan": "johann",
    "friederich": "friedrich", "fridrich": "friedrich",
    "josef": "joseph", "wilh": "wilhelm", "willhelm": "wilhelm",
    "christoff": "christoph", "kristoph": "christoph",
    "theodor": "theodor", "gottlob": "gottlob",
    "ludewig": "ludwig", "louis": "ludwig",
    "andreas": "andreas", "andres": "andreas",
    "ernest": "ernst", "guenther": "gunther",
    "maximilian": "maximilian", "max": "maximilian",
}

_ABBREV = re.compile(r"\.$|^[a-z]{1,3}$")


def _tokens(s):
    s = _fold(s)
    s = re.sub(r"[^a-z. -]", " ", s)
    return [t for t in re.split(r"[\s-]+", s) if t]


def canon_given(s):
    """Canonical full given tokens. Abbreviated tokens are dropped."""
    out = []
    for t in _tokens(s):
        if t.endswith(".") or len(t) <= 3:
            continue                      # abbreviation: no reliable expansion
        t = LATIN.get(t, t)
        t = VARIANT.get(t, t)
        out.append(t)
    return out


def is_abbreviated(s):
    return any(t.endswith(".") or len(t) <= 3 for t in _tokens(s))


def initials(s, loose=True):
    """Initial sequence. loose folds C->K and I/Y->J (Carl/Karl, Ioannes/Johann).

    The cost of loose folding is that Christian collides with Karl; keep both
    a strict and a loose version and let the model weight them separately.
    """
    out = []
    for t in _tokens(s):
        if not (t.endswith(".") or len(t) <= 3):
            t = VARIANT.get(LATIN.get(t, t), LATIN.get(t, t))
        c = t[0]
        if loose:
            if c == "c":
                c = "k"
            elif c in "iy":
                c = "j"
        out.append(c)
    return "".join(out)


# ---------------------------------------------------------------------------
# DuckDB materialisation
# ---------------------------------------------------------------------------

def _nullsafe(fn, empty_is_null=True):
    def w(s):
        if s is None:
            return None
        v = fn(s)
        if empty_is_null and (v == "" or v == []):
            return None
        return v
    return w


def register(con):
    f = con.create_function
    f("kphon", _nullsafe(koelner), ["VARCHAR"], "VARCHAR", null_handling="special")
    f("sn_particle", _nullsafe(lambda s: canon_surname(s)[0]), ["VARCHAR"], "VARCHAR", null_handling="special")
    f("sn_norm", _nullsafe(lambda s: canon_surname(s)[1]), ["VARCHAR"], "VARCHAR", null_handling="special")
    f("gn_canon", _nullsafe(canon_given), ["VARCHAR"], "VARCHAR[]", null_handling="special")
    f("gn_init", _nullsafe(lambda s: initials(s, True)), ["VARCHAR"], "VARCHAR", null_handling="special")
    f("gn_init_strict", _nullsafe(lambda s: initials(s, False)), ["VARCHAR"], "VARCHAR", null_handling="special")
    f("gn_init_sorted", _nullsafe(lambda s: "".join(sorted(initials(s, True)))), ["VARCHAR"], "VARCHAR", null_handling="special")
    f("gn_abbrev", _nullsafe(is_abbreviated, False), ["VARCHAR"], "BOOLEAN", null_handling="special")


def add_columns(con, src="d", dst="rec"):
    """Materialise derived columns into table `dst`."""
    register(con)
    con.execute(f"""
        CREATE OR REPLACE TABLE {dst} AS
        SELECT
            *,
            sn_norm(last_name)                     AS ln_norm,
            sn_particle(last_name)                 AS ln_particle,
            kphon(sn_norm(last_name))              AS ln_phon,
            gn_canon(first_names)                  AS gn_tokens,
            list_sort(gn_canon(first_names))       AS gn_set,
            gn_init(first_names)                   AS gn_init,
            gn_init_sorted(first_names)            AS gn_init_set,
            substr(gn_init(first_names), 1, 1)     AS gn_i1,
            gn_abbrev(first_names)                 AS gn_is_abbrev,
            coalesce(len(gn_canon(first_names)), 0) > 0  AS gn_has_full
        FROM {src}
    """)
    return dst


# ---------------------------------------------------------------------------
# comparison levels
# ---------------------------------------------------------------------------

# Ordered most- to least-informative. Level 0 is the "both sides usable and
# they agree" case; levels 2-3 are the only ones available for pairs involving
# Berlin, Jena or Erlangen, so they must carry their own weights.
GIVEN_COMPARISON_SQL = """
CASE
  WHEN l.gn_has_full AND r.gn_has_full AND l.gn_set = r.gn_set          THEN 0
  WHEN l.gn_has_full AND r.gn_has_full
       AND len(list_intersect(l.gn_tokens, r.gn_tokens)) > 0
       AND l.gn_i1 = r.gn_i1                                            THEN 1
  WHEN l.gn_init = r.gn_init AND length(l.gn_init) >= 2                 THEN 2
  WHEN l.gn_init_set = r.gn_init_set AND length(l.gn_init) >= 2         THEN 3
  WHEN l.gn_i1 = r.gn_i1                                                THEN 4
  WHEN l.gn_i1 IS NULL OR r.gn_i1 IS NULL OR l.gn_i1 = '' OR r.gn_i1 = '' THEN 5
  ELSE 6
END
"""

# Non-comparability guard: a pair where both sides are abbreviated can never
# reach levels 0-1, so the observed level distribution differs by school pair.
# Estimate m/u separately for abbreviated-abbreviated pairs, or the EM step
# will attribute the school-composition effect to the name agreement itself.
ABBREV_STRATUM_SQL = """
CASE
  WHEN l.gn_has_full AND r.gn_has_full THEN 'both_full'
  WHEN l.gn_has_full OR  r.gn_has_full THEN 'mixed'
  ELSE 'both_abbrev'
END
"""


if __name__ == "__main__":
    tests = [
        ("Mueller", "657"), ("Müller", "657"), ("Miller", "657"),
        ("Schmidt", "862"), ("Schmitt", "862"), ("Schmid", "862"),
        ("Meyer", "67"), ("Meier", "67"), ("Maier", "67"), ("Mayr", "67"),
        ("Wikipedia", "3412"), ("Breschnew", "17863"),
    ]
    print("Koelner Phonetik")
    for s, want in tests:
        got = koelner(s)
        print(f"  {s:12s} -> {got:8s} {'ok' if got == want else 'EXPECTED ' + want}")

    print("\nsurname normalisation")
    for s in ["von Müller", "Schmidt", "Schmitt", "Zur Mühlen", "Baur", "Bauer"]:
        print(f"  {s:14s} -> {canon_surname(s)}  phon={koelner(canon_surname(s)[1])}")

    print("\ngiven names (the abbreviation problem)")
    for s in ["Johann Friedrich", "Joh. Frdr.", "J. F.", "Carolus Fridericus",
              "Carl Friedrich", "K. F.", "F.", "Guilielmus", "Joannes Baptista"]:
        print(f"  {s:20s} full={canon_given(s)!s:34s} init={initials(s)!r:8s} abbrev={is_abbreviated(s)}")
