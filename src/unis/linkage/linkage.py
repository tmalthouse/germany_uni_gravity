#!/usr/bin/env python3
"""
Record linkage for 19th-century German university registers.

Pure Python + DuckDB. Stages:

  1 prepare   normalised name columns (names.py)
  2 spells    within-school collapse of annual registers -> person-school spells
  3 block     union of blocking rules over spells, cross-school only
  4 compare   discrete comparison levels + term frequencies
  5 em        Fellegi-Sunter m/u by EM, estimated SEPARATELY per abbreviation
              stratum, on the aggregated pattern table (fast and exact)
  6 score     log2 Bayes factor + term-frequency adjustment
  7 cluster   constrained agglomeration with must-not-link rules

Usage:
    python -m unis.linkage.linkage students.parquet --out linked.parquet --threshold 0.9
    python -m unis.linkage.linkage synth.parquet --eval    # ground-truth evaluation

KNOWN ISSUE (not yet fixed): stage_u draws its random pairs with an unseeded
`USING SAMPLE`, so u -- and with it the student clustering -- changes from run
to run. Spells (stage 2) are deterministic.
"""

import argparse
import math
from collections import defaultdict

import duckdb

from unis.linkage import names

# ---------------------------------------------------------------------------
# comparison field definitions. Each is (name, n_levels).
# Level 0 is always the most-agreeing level.
# ---------------------------------------------------------------------------

FIELDS = [("sn", 4), ("gn", 7), ("geo", 4), ("yr", 5), ("pt", 3)]

ANNUAL_THRESHOLD = 1.25      # rows per name-key above which a source is annual
MAX_OVERLAP = 1              # years two spells at different schools may overlap
MAX_SPAN = 9                 # years from first matriculation to last record
MAX_SCHOOLS = 3


# ---------------------------------------------------------------------------
# 1. prepare
# ---------------------------------------------------------------------------

def stage_prepare(con, path, keep_truth=False):
    reader = (f"read_parquet('{path}')" if path.endswith((".parquet", ".pq"))
              else f"read_csv_auto('{path}')")
    con.execute(f"CREATE OR REPLACE VIEW d0 AS SELECT * FROM {reader}")
    cols = {c.lower() for c in con.sql("SELECT * FROM d0 LIMIT 0").columns}

    have = lambda c: c in cols                                    # noqa: E731
    extra = []
    for c in ("field", "prov_id", "standard_polity_id", "father",
              "religion", "prev_university"):
        extra.append(f"{c}" if have(c) else f"CAST(NULL AS VARCHAR) AS {c}")
    if keep_truth and have("truth_person_id"):
        extra.append("truth_person_id")

    con.execute(f"""
        CREATE OR REPLACE VIEW d AS
        SELECT last_name, first_names, first_year, school, location_id,
               {', '.join(extra)}
        FROM d0
        WHERE last_name IS NOT NULL AND first_year IS NOT NULL
    """)
    names.add_columns(con, src="d", dst="rec")
    con.execute("ALTER TABLE rec ADD COLUMN rec_id BIGINT")
    con.execute("UPDATE rec SET rec_id = rowid")
    n = con.sql("SELECT count(*) FROM rec").fetchone()[0]
    print(f"[1] prepared {n:,} records")
    return n


def detect_annual(con, override=None):
    rows = con.sql(f"""
        SELECT school, sum(n) * 1.0 / count(*) AS rpk FROM (
            SELECT school, count(*) AS n FROM rec
            GROUP BY school, ln_norm, gn_init, coalesce(location_id, -999))
        GROUP BY school
    """).fetchall()
    annual = ({s for s, r in rows if r and r > ANNUAL_THRESHOLD}
              if override is None else set(override))
    for s, r in sorted(rows, key=lambda x: -(x[1] or 0)):
        print(f"      {s:12s} rows/key {r:.2f}")
    print(f"[1] annual registers: {sorted(annual)}")
    print(f"[1] matriculation-only: {sorted({s for s, _ in rows} - annual)}")
    return annual


# ---------------------------------------------------------------------------
# 2. within-school collapse
# ---------------------------------------------------------------------------

def _compatible(ia, ib, ta, tb):
    """Initial sequences prefix-compatible and full tokens not contradictory."""
    ia, ib = ia or "", ib or ""
    if not (ia.startswith(ib) or ib.startswith(ia)):
        return False
    sa, sb = set(ta or []), set(tb or [])
    if sa and sb and not (sa & sb):
        return False
    return True


def stage_spells(con, annual):
    """Chain within-school records into spells.

    Only annual registers are chained. At a matriculation-only source every row
    is its own spell, because two same-named students in adjacent years are two
    people, not one person seen twice.
    """
    rows = con.sql("""
        SELECT rec_id, school, ln_phon, gn_i1, coalesce(location_id, -999) AS loc,
               first_year, gn_init, gn_tokens
        FROM rec ORDER BY school, ln_phon, gn_i1, loc, first_year
    """).fetchall()

    assign = {}
    groups = defaultdict(list)
    for r in rows:
        groups[(r[1], r[2], r[3], r[4])].append(r)

    sid = 0
    for (school, _, _, _), recs in groups.items():
        if school not in annual:
            for r in recs:
                assign[r[0]] = sid
                sid += 1
            continue
        chains = []                       # (spell_id, last_year, init, tokens)
        for rec_id, _, _, _, _, yr, init, toks in recs:
            hit = None
            for ci, (cs, cy, cinit, ctoks) in enumerate(chains):
                if cy < yr <= cy + 2 and _compatible(init, cinit, toks, ctoks):
                    hit = ci
                    break
            if hit is None:
                chains.append([sid, yr, init, toks])
                assign[rec_id] = sid
                sid += 1
            else:
                cs, _, cinit, ctoks = chains[hit]
                # keep the longer initial string and the union of full tokens
                chains[hit] = [cs, yr,
                               init if len(init or "") > len(cinit or "") else cinit,
                               list({*(ctoks or []), *(toks or [])})]
                assign[rec_id] = cs

    con.execute("CREATE OR REPLACE TABLE map(rec_id BIGINT, spell_id BIGINT)")
    con.executemany("INSERT INTO map VALUES (?,?)", list(assign.items()))

    con.execute("""
        CREATE OR REPLACE TABLE spell AS
        SELECT
            m.spell_id,
            any_value(r.school)                          AS school,
            min(r.first_year)                            AS first_seen,
            max(r.first_year)                            AS last_seen,
            count(*)                                     AS n_rec,
            -- consensus name: longest initial string, union of full tokens
            arg_max(r.ln_norm, length(r.ln_norm))        AS ln_norm,
            any_value(r.ln_phon)                         AS ln_phon,
            coalesce(any_value(r.ln_particle), '')       AS ln_particle,
            arg_max(r.gn_init, length(r.gn_init))        AS gn_init,
            arg_max(r.gn_init_set, length(r.gn_init_set)) AS gn_init_set,
            any_value(r.gn_i1)                           AS gn_i1,
            list_distinct(flatten(list(r.gn_tokens)))    AS gn_tokens,
            bool_or(r.gn_has_full)                       AS gn_has_full,
            mode(r.location_id)                          AS location_id,
            mode(r.field)                                AS field
        FROM map m JOIN rec r USING (rec_id)
        GROUP BY m.spell_id
    """)
    con.execute("""
        ALTER TABLE spell ADD COLUMN gn_set VARCHAR;
        UPDATE spell SET gn_set = list_aggregate(list_sort(gn_tokens), 'string_agg', '|');
    """)
    n = con.sql("SELECT count(*) FROM spell").fetchone()[0]
    print(f"[2] {n:,} spells")
    return n


# ---------------------------------------------------------------------------
# 3. blocking
# ---------------------------------------------------------------------------

# `loc_ok` excludes region-sized geocoding buckets (Wuerttemberg is 7.6% of
# records) from location-based blocking. Those records are still reachable
# through the surname rules; letting them block on location would generate an
# enormous number of pairs that carry no information.
BLOCK_RULES = [
    "l.ln_phon = r.ln_phon AND l.gn_i1 = r.gn_i1",
    "l.ln_phon = r.ln_phon AND l.location_id = r.location_id",
    "l.loc_ok AND r.loc_ok AND l.location_id = r.location_id AND l.gn_init = r.gn_init",
    "l.ln_norm = r.ln_norm",
    "substr(l.ln_phon,1,3) = substr(r.ln_phon,1,3) AND l.gn_init = r.gn_init AND l.gn_i1 = r.gn_i1",
]
LOC_FREQ_CAP = 0.002        # location_ids above this share are region buckets


def stage_block(con, max_gap=9):
    con.execute(f"""
        ALTER TABLE spell ADD COLUMN IF NOT EXISTS loc_ok BOOLEAN;
        UPDATE spell SET loc_ok = location_id IS NOT NULL AND location_id IN (
            SELECT location_id FROM spell WHERE location_id IS NOT NULL
            GROUP BY 1 HAVING count(*) * 1.0 / (SELECT count(*) FROM spell) < {LOC_FREQ_CAP});
    """)
    big = con.sql("SELECT count(*) FROM spell WHERE NOT coalesce(loc_ok, false)").fetchone()[0]
    print(f"[3] {big:,} spells in region-sized or missing location buckets "
          f"(excluded from location blocking)")
    parts = [f"""
        SELECT l.spell_id AS lid, r.spell_id AS rid
        FROM spell l JOIN spell r
          ON {rule}
         AND l.spell_id < r.spell_id
         AND l.school <> r.school
         AND least(r.first_seen - l.last_seen, l.first_seen - r.last_seen) <= {max_gap}
         AND greatest(l.first_seen, r.first_seen)
             - least(l.last_seen, r.last_seen) >= -{MAX_OVERLAP + 1}
    """ for rule in BLOCK_RULES]
    con.execute("CREATE OR REPLACE TABLE pairs AS SELECT lid, rid FROM ("
                + " UNION ALL ".join(parts) + ") GROUP BY lid, rid")
    n = con.sql("SELECT count(*) FROM pairs").fetchone()[0]
    print(f"[3] {n:,} candidate pairs")
    return n


# ---------------------------------------------------------------------------
# 4. comparison vectors
# ---------------------------------------------------------------------------

def _compare_sql(pair_tbl, annual, with_tf=True):
    annual_sql = "(" + ",".join(f"'{s}'" for s in annual) + ")" if annual else "('')"
    F = ["ln_norm", "ln_phon", "ln_particle", "gn_init", "gn_init_set", "gn_i1",
         "gn_tokens", "gn_has_full", "gn_set", "location_id"]
    lsel = ", ".join(f"l.{c} AS l_{c}" for c in F)
    rsel = ", ".join(f"r.{c} AS r_{c}" for c in F)
    gap = ("greatest(0, t_first - e_last - CASE WHEN e_school IN "
           f"{annual_sql} THEN 0 ELSE 2 END)")
    tf_cols = """,
            coalesce(a.f, 1.0) AS f_sn_l, coalesce(b.f, 1.0) AS f_sn_r,
            coalesce(g.f, 1.0) AS f_geo_l, coalesce(h.f, 1.0) AS f_geo_r"""
    tf_join = """
        LEFT JOIN tf_sn  a ON a.ln_phon     = p.l_ln_phon
        LEFT JOIN tf_sn  b ON b.ln_phon     = p.r_ln_phon
        LEFT JOIN tf_geo g ON g.location_id = p.l_location_id
        LEFT JOIN tf_geo h ON h.location_id = p.r_location_id"""
    return f"""
        WITH p AS (
            SELECT q.lid, q.rid, {lsel}, {rsel},
                   CASE WHEN l.first_seen <= r.first_seen THEN l.last_seen ELSE r.last_seen END AS e_last,
                   CASE WHEN l.first_seen <= r.first_seen THEN r.first_seen ELSE l.first_seen END AS t_first,
                   CASE WHEN l.first_seen <= r.first_seen THEN l.school ELSE r.school END AS e_school
            FROM {pair_tbl} q
            JOIN spell l ON l.spell_id = q.lid
            JOIN spell r ON r.spell_id = q.rid
        )
        SELECT
            p.lid, p.rid,
            CASE WHEN p.l_ln_norm = p.r_ln_norm THEN 0
                 WHEN p.l_ln_phon = p.r_ln_phon THEN 1
                 WHEN jaro_winkler_similarity(p.l_ln_norm, p.r_ln_norm) >= 0.92 THEN 2
                 ELSE 3 END AS sn,
            CASE
              WHEN p.l_gn_has_full AND p.r_gn_has_full AND p.l_gn_set = p.r_gn_set   THEN 0
              WHEN p.l_gn_has_full AND p.r_gn_has_full
                   AND len(list_intersect(p.l_gn_tokens, p.r_gn_tokens)) > 0
                   AND p.l_gn_i1 = p.r_gn_i1                                          THEN 1
              WHEN p.l_gn_init = p.r_gn_init AND length(p.l_gn_init) >= 2             THEN 2
              WHEN p.l_gn_init_set = p.r_gn_init_set AND length(p.l_gn_init) >= 2     THEN 3
              WHEN p.l_gn_i1 = p.r_gn_i1                                              THEN 4
              WHEN p.l_gn_i1 IS NULL OR p.r_gn_i1 IS NULL                             THEN 5
              ELSE 6 END AS gn,
            CASE WHEN p.l_location_id IS NULL OR p.r_location_id IS NULL THEN 3
                 WHEN p.l_location_id = p.r_location_id THEN 0
                 ELSE 2 END AS geo,
            CASE WHEN {gap} <= 1 THEN 0
                 WHEN {gap} =  2 THEN 1
                 WHEN {gap} <= 4 THEN 2
                 WHEN {gap} <= 8 THEN 3
                 ELSE 4 END AS yr,
            CASE WHEN p.l_ln_particle = p.r_ln_particle AND p.l_ln_particle <> '' THEN 0
                 WHEN p.l_ln_particle = '' AND p.r_ln_particle = '' THEN 1
                 ELSE 2 END AS pt,
            CASE WHEN p.l_gn_has_full AND p.r_gn_has_full THEN 'both_full'
                 WHEN p.l_gn_has_full OR  p.r_gn_has_full THEN 'mixed'
                 ELSE 'both_abbrev' END AS stratum
            {tf_cols if with_tf else ""}
        FROM p
        {tf_join if with_tf else ""}
    """


def stage_compare(con, annual):
    con.execute("""
        CREATE OR REPLACE TABLE tf_sn AS
        SELECT ln_phon, count(*) * 1.0 /
               (SELECT count(*) FROM spell WHERE ln_phon IS NOT NULL) AS f
        FROM spell WHERE ln_phon IS NOT NULL GROUP BY 1;
        CREATE OR REPLACE TABLE tf_geo AS
        SELECT location_id, count(*) * 1.0 /
               (SELECT count(*) FROM spell WHERE location_id IS NOT NULL) AS f
        FROM spell WHERE location_id IS NOT NULL GROUP BY 1;
    """)
    con.execute("CREATE OR REPLACE TABLE cmp AS " + _compare_sql("pairs", annual))
    print("[4] comparison vectors built")


def stage_u(con, annual, n=3000):
    """Estimate u from RANDOM cross-school pairs.

    Estimating u from blocked pairs is the classic way to get a degenerate EM
    fit: blocking selects for agreement, so u is biased up, the Bayes factors
    collapse, and lambda runs away. Random pairs are non-matches with
    probability ~1, so their level frequencies are u directly.
    """
    con.execute(f"""
        CREATE OR REPLACE TABLE rnd AS
        SELECT a.spell_id AS lid, b.spell_id AS rid
        FROM (SELECT spell_id, school FROM spell USING SAMPLE {n} ROWS) a
        CROSS JOIN (SELECT spell_id, school FROM spell USING SAMPLE {n} ROWS) b
        WHERE a.spell_id < b.spell_id AND a.school <> b.school
    """)
    con.execute("CREATE OR REPLACE TABLE cmp_u AS "
                + _compare_sql("rnd", annual, with_tf=False))
    rows = con.sql(f"""
        SELECT stratum, {', '.join(f for f, _ in FIELDS)}, count(*) AS n
        FROM cmp_u GROUP BY ALL
    """).fetchall()
    tot = con.sql("SELECT count(*) FROM cmp_u").fetchone()[0]
    print(f"[5a] u estimated from {tot:,} random cross-school pairs")

    u = {}
    for r in rows:
        st, lv, c = r[0], r[1:1 + len(FIELDS)], r[-1]
        d = u.setdefault(st, {f: [0.0] * k for f, k in FIELDS})
        for (f, _), l in zip(FIELDS, lv):
            d[f][l] += c
    for st, d in u.items():
        for f, k in FIELDS:
            s = sum(d[f]) or 1.0
            # Laplace smoothing: an unobserved level must not get u = 0
            d[f] = [(x + 0.5) / (s + 0.5 * k) for x in d[f]]
    # strata with too few random pairs fall back to the pooled estimate
    pooled = {f: [sum(d[f][i] for d in u.values()) / len(u) for i in range(k)]
              for f, k in FIELDS}
    for st in ("both_full", "mixed", "both_abbrev"):
        u.setdefault(st, pooled)
    return u


# ---------------------------------------------------------------------------
# 5. EM, per stratum, on aggregated patterns
# ---------------------------------------------------------------------------

def em_stratum(patterns, u, lam, m0=None, n_iter=8):
    """EM for m only, with u fixed at its random-pair estimate and lambda pinned.

    Letting EM estimate lambda as well is not identified here: with m free, the
    blocked-pair distribution is explained perfectly by "everything is a match",
    and lambda runs to its cap. Pinning lambda to an externally known transfer
    rate is both stable and more honest about where the information comes from.
    """
    m = ({f: list(m0[f]) for f, _ in FIELDS} if m0 else
         {f: [0.6 if i == 0 else 0.4 / (k - 1) for i in range(k)] for f, k in FIELDS})
    total = sum(c for _, c in patterns)
    if total == 0:
        return lam, m
    for _ in range(n_iter):
        num = {f: [1e-9] * k for f, k in FIELDS}
        sm = 0.0
        for lv, c in patterns:
            lm, lu = math.log(lam), math.log(1 - lam)
            for (f, _), l in zip(FIELDS, lv):
                lm += math.log(max(m[f][l], 1e-12))
                lu += math.log(max(u[f][l], 1e-12))
            g = 1.0 / (1.0 + math.exp(min(700, max(-700, lu - lm))))
            sm += g * c
            for (f, _), l in zip(FIELDS, lv):
                num[f][l] += g * c
        for f, k in FIELDS:
            t = sum(num[f])
            m[f] = [x / t for x in num[f]]
    return lam, m


def expected_links(con, mean_schools):
    """Expected number of true cross-school spell pairs.

    With S spells and mean_schools = mu, there are S/mu students and roughly
    S - S/mu excess spells, each contributing about one cross-school pair.
    Calibrate mu from prev_university: 36% reporting a prior school implies
    mu of roughly 1.45 once second transfers are allowed for.
    """
    S = con.sql("SELECT count(*) FROM spell").fetchone()[0]
    return S * (1 - 1 / mean_schools)


def seed_m(con, u_all, rare=2e-4):
    """Estimate m from seed sets of near-certain matches.

    Cold-start EM is not identified here: it drags m toward u and reports that
    matches disagree on given names, which is backwards. Instead each field's m
    is estimated on a seed set made near-certain by the OTHER fields, so no
    field is used to validate itself. This is the same idea as Splink's
    "train on a blocking rule" step.
    """
    seeds = {
        # rare exact surname + exact hometown -> given name, year, particle
        ("gn", "yr", "pt"):
            f"sn = 0 AND geo = 0 AND f_sn_l < {rare}",
        # exact full given-name agreement + exact hometown -> surname
        ("sn",):
            "gn <= 1 AND geo = 0",
        # rare exact surname + given-name agreement -> hometown
        ("geo",):
            f"sn = 0 AND gn <= 2 AND f_sn_l < {rare}",
    }
    m = {}
    for fields, where in seeds.items():
        cols = ", ".join(fields)
        rows = con.sql(f"""
            SELECT stratum, {cols}, count(*) AS n FROM cmp WHERE {where} GROUP BY ALL
        """).fetchall()
        n_tot = sum(r[-1] for r in rows)
        print(f"[5b] seed [{where[:46]}...]  {n_tot:,} pairs -> m for {fields}")
        acc = {}
        for r in rows:
            st, lv, c = r[0], r[1:1 + len(fields)], r[-1]
            d = acc.setdefault(st, {f: [0.0] * dict(FIELDS)[f] for f in fields})
            for f, l in zip(fields, lv):
                d[f][l] += c
        for st, d in acc.items():
            for f in fields:
                k = dict(FIELDS)[f]
                s = sum(d[f]) or 1.0
                m.setdefault(st, {})[f] = [(x + 0.5) / (s + 0.5 * k) for x in d[f]]

    # any stratum/field the seeds could not reach falls back to the pooled value
    for st in u_all:
        for f, k in FIELDS:
            if f not in m.get(st, {}):
                vals = [m[s][f] for s in m if f in m[s]]
                m.setdefault(st, {})[f] = (
                    [sum(v[i] for v in vals) / len(vals) for i in range(k)]
                    if vals else [1.0 / k] * k)
    return m


def stage_em(con, u_all, n_links):
    rows = con.sql(f"""
        SELECT stratum, {', '.join(f for f, _ in FIELDS)}, count(*) AS n
        FROM cmp GROUP BY ALL
    """).fetchall()
    by = defaultdict(list)
    for r in rows:
        by[r[0]].append((tuple(r[1:1 + len(FIELDS)]), r[-1]))

    total_pairs = sum(c for pats in by.values() for _, c in pats)
    lam0 = min(0.5, n_links / max(total_pairs, 1))
    print(f"[5b] expecting ~{n_links:,.0f} true links among {total_pairs:,} "
          f"candidate pairs -> lambda = {lam0:.2e}")

    m_seed = seed_m(con, u_all)
    params = {}
    for stratum, pats in by.items():
        u = u_all[stratum]
        lam, m = em_stratum(pats, u, lam0, m0=m_seed[stratum])
        params[stratum] = (lam, m, u)
        n = sum(c for _, c in pats)
        print(f"[5b] {stratum:12s} {n:>10,} pairs   "
              f"gn m={['%.3f' % x for x in m['gn']]}")
    return params


# ---------------------------------------------------------------------------
# 6. scoring
# ---------------------------------------------------------------------------

def stage_score(con, params, tf_cap=4.0):
    con.execute("""CREATE OR REPLACE TABLE w(
        stratum VARCHAR, field VARCHAR, level INT, bf DOUBLE, u DOUBLE)""")
    rows, prior = [], {}
    for stratum, (lam, m, u) in params.items():
        prior[stratum] = math.log2(lam / (1 - lam))
        for f, k in FIELDS:
            for l in range(k):
                bf = math.log2(max(m[f][l], 1e-12) / max(u[f][l], 1e-12))
                rows.append((stratum, f, l, bf, u[f][l]))
    con.executemany("INSERT INTO w VALUES (?,?,?,?,?)", rows)
    con.execute("CREATE OR REPLACE TABLE prior(stratum VARCHAR, lp DOUBLE)")
    con.executemany("INSERT INTO prior VALUES (?,?)", list(prior.items()))

    joins = "\n".join(
        f"LEFT JOIN w w_{f} ON w_{f}.stratum = c.stratum "
        f"AND w_{f}.field = '{f}' AND w_{f}.level = c.{f}"
        for f, _ in FIELDS)
    bfs = " + ".join(f"coalesce(w_{f}.bf, 0)" for f, _ in FIELDS)

    # Term-frequency adjustment: replace the average u for the exact-agreement
    # level with the product of the two observed value frequencies. This is what
    # separates agreement on Wuerttemberg (7.6% of records) from agreement on a
    # village. Capped to keep rare-value pairs from dominating.
    con.execute(f"""
        CREATE OR REPLACE TABLE edge AS
        SELECT c.lid, c.rid, c.stratum,
               ({bfs}) AS bf_base,
               CASE WHEN c.sn = 0 THEN greatest(-{tf_cap}, least({tf_cap},
                    log2(coalesce(w_sn.u, 0.1)) - log2(c.f_sn_l * c.f_sn_r)))
                    ELSE 0 END AS tf_sn,
               CASE WHEN c.geo = 0 THEN greatest(-{tf_cap}, least({tf_cap},
                    log2(coalesce(w_geo.u, 0.1)) - log2(c.f_geo_l * c.f_geo_r)))
                    ELSE 0 END AS tf_geo,
               p.lp
        FROM cmp c
        {joins}
        LEFT JOIN prior p ON p.stratum = c.stratum
    """)
    con.execute("""
        CREATE OR REPLACE TABLE edge AS
        SELECT lid, rid, stratum, w, 1.0 / (1.0 + pow(2.0, -w)) AS p
        FROM (SELECT lid, rid, stratum, lp + bf_base + tf_sn + tf_geo AS w FROM edge)
        WHERE w > -2
    """)
    n = con.sql("SELECT count(*) FROM edge").fetchone()[0]
    print(f"[6] scored; {n:,} edges retained above w > -2")


# ---------------------------------------------------------------------------
# 7. constrained clustering
# ---------------------------------------------------------------------------

class Clusters:
    """Union-find with must-not-link constraints checked at merge time."""

    def __init__(self, spells):
        self.parent = {}
        self.info = {}
        for sid, school, first, last in spells:
            self.parent[sid] = sid
            self.info[sid] = ({school}, [(first, last)])

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def _ok(self, a, b):
        sa, ia = self.info[a]
        sb, ib = self.info[b]
        if sa & sb:                                  # same school twice
            return False
        if len(sa | sb) > MAX_SCHOOLS:
            return False
        for f1, l1 in ia:
            for f2, l2 in ib:
                if min(l1, l2) - max(f1, f2) + 1 > MAX_OVERLAP:
                    return False
        lo = min(min(f for f, _ in ia), min(f for f, _ in ib))
        hi = max(max(l for _, l in ia), max(l for _, l in ib))
        if hi - lo + 1 > MAX_SPAN:
            return False
        return True

    def merge(self, x, y):
        a, b = self.find(x), self.find(y)
        if a == b or not self._ok(a, b):
            return False
        self.parent[b] = a
        self.info[a] = (self.info[a][0] | self.info[b][0],
                        self.info[a][1] + self.info[b][1])
        del self.info[b]
        return True


def stage_cluster(con, threshold):
    spells = con.sql("SELECT spell_id, school, first_seen, last_seen FROM spell").fetchall()
    edges = con.sql(f"""
        SELECT lid, rid, w FROM edge WHERE p >= {threshold} ORDER BY w DESC
    """).fetchall()
    cl = Clusters(spells)
    kept = sum(cl.merge(a, b) for a, b, _ in edges)
    print(f"[7] {len(edges):,} edges over threshold, {kept:,} merges accepted "
          f"({len(edges) - kept:,} blocked by constraints)")

    con.execute("CREATE OR REPLACE TABLE cluster(spell_id BIGINT, student_id BIGINT)")
    con.executemany("INSERT INTO cluster VALUES (?,?)",
                    [(s, cl.find(s)) for s in cl.parent])
    n = con.sql("SELECT count(DISTINCT student_id) FROM cluster").fetchone()[0]
    print(f"[7] {n:,} distinct students")
    return n


# ---------------------------------------------------------------------------
# prev_university validation (works on the real data)
# ---------------------------------------------------------------------------

import re as _re


def resolve_prev(con):
    """Map free-text prev_university strings onto schools present in the data.

    Handles the patterns visible in the diagnostics: 'ex ac. ' prefixes,
    abbreviations ('Heidelb.', 'Marb.', 'Berl.'), umlauts ('Tuebingen' vs
    'tuebingen'), and compound entries ('Berlin u. Heidelberg'). Anything that
    does not resolve is an unreachable prior school and caps recall.
    """
    schools = [s for (s,) in con.sql(
        "SELECT DISTINCT school FROM rec WHERE school IS NOT NULL").fetchall()]
    folded = {names._fold(s).replace(" ", ""): s for s in schools}

    def resolve(v):
        if not v:
            return []
        t = names._fold(v)
        t = _re.sub(r"^ex\s*ac\.?\s*", "", t)
        out = []
        for part in _re.split(r"\su\.\s|\sund\s|,|/", t):
            key = _re.sub(r"[^a-z]", "", part)
            if len(key) < 4:
                continue
            hits = sorted({orig for f, orig in folded.items()
                           if f.startswith(key) or key.startswith(f)})
            if len(hits) == 1:
                out.append(hits[0])
        return sorted(set(out))

    rows = con.sql("""
        SELECT rec_id, school, prev_university FROM rec
        WHERE prev_university IS NOT NULL
    """).fetchall()
    gold, unreachable, reporting = [], 0, {}
    for rec_id, school, prev in rows:
        reporting[school] = reporting.get(school, 0) + 1
        tgt = [t for t in resolve(prev) if t != school]
        if tgt:
            gold += [(rec_id, t) for t in tgt]
        else:
            unreachable += 1
    con.execute("CREATE OR REPLACE TABLE gold(rec_id BIGINT, target VARCHAR)")
    if gold:
        con.executemany("INSERT INTO gold VALUES (?,?)", gold)
    n = len(rows)
    print(f"\n[gold] {n:,} records report a prior university "
          f"(reporting schools: {sorted(reporting)})")
    print(f"[gold] {n - unreachable:,} resolve to a school in the data "
          f"({100 * (n - unreachable) / max(n, 1):.1f}%); {unreachable:,} unreachable")
    print(f"[gold] recall ceiling is therefore "
          f"{100 * (n - unreachable) / max(n, 1):.1f}%")
    return n - unreachable


def validate_gold(con):
    """Score the current clustering against prev_university."""
    r = con.sql("""
        WITH cl AS (
            SELECT c.student_id, list(DISTINCT sp.school) AS schools
            FROM cluster c JOIN spell sp USING (spell_id) GROUP BY 1),
        g AS (
            SELECT gold.rec_id, gold.target, cl.schools
            FROM gold JOIN map USING (rec_id) JOIN cluster USING (spell_id)
                      JOIN cl USING (student_id))
        SELECT count(*) AS reachable,
               sum(CASE WHEN list_contains(schools, target) THEN 1 ELSE 0 END) AS hit,
               sum(CASE WHEN NOT list_contains(schools, target)
                         AND len(schools) > 1 THEN 1 ELSE 0 END) AS wrong_school,
               sum(CASE WHEN len(schools) = 1 THEN 1 ELSE 0 END) AS unlinked
        FROM g
    """).fetchone()
    reachable, hit, wrong, unlinked = r
    if not reachable:
        print("[gold] no resolvable gold pairs")
        return
    print(f"[gold] recall {100 * hit / reachable:5.1f}%  "
          f"({hit:,}/{reachable:,})   linked-to-wrong-school "
          f"{100 * wrong / reachable:4.1f}%   left unlinked "
          f"{100 * unlinked / reachable:4.1f}%")


# ---------------------------------------------------------------------------
# evaluation (synthetic only)
# ---------------------------------------------------------------------------

def evaluate(con, brief=False):
    con.execute("""
        CREATE OR REPLACE TABLE spell_truth AS
        SELECT m.spell_id, mode(r.truth_person_id) AS truth,
               count(DISTINCT r.truth_person_id) AS n_truth
        FROM map m JOIN rec r USING (rec_id) GROUP BY 1
    """)
    pure = con.sql("SELECT avg(CASE WHEN n_truth = 1 THEN 1.0 ELSE 0 END) FROM spell_truth").fetchone()[0]
    print(f"\n[eval] within-school collapse purity: {100 * pure:.2f}% of spells single-person")

    res = con.sql("""
        WITH s AS (
            SELECT c.spell_id, c.student_id, t.truth, sp.school
            FROM cluster c JOIN spell_truth t USING (spell_id)
                           JOIN spell sp USING (spell_id)
        ),
        pred AS (
            SELECT a.spell_id l, b.spell_id r, a.truth = b.truth AS correct
            FROM s a JOIN s b ON a.student_id = b.student_id
                 AND a.spell_id < b.spell_id AND a.school <> b.school
        ),
        truth AS (
            SELECT a.spell_id l, b.spell_id r
            FROM s a JOIN s b ON a.truth = b.truth
                 AND a.spell_id < b.spell_id AND a.school <> b.school
        )
        SELECT (SELECT count(*) FROM pred) AS predicted,
               (SELECT count(*) FROM pred WHERE correct) AS tp,
               (SELECT count(*) FROM truth) AS actual
    """).fetchone()
    predicted, tp, actual = res
    prec = tp / predicted if predicted else 0
    rec = tp / actual if actual else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    print(f"[eval] cross-school pairs: predicted {predicted:,}  true {actual:,}  correct {tp:,}")
    print(f"[eval] precision {prec:.3f}   recall {rec:.3f}   F1 {f1:.3f}")
    if brief:
        return

    print("\n[eval] blocking recall (ceiling for everything downstream):")
    con.sql("""
        WITH s AS (SELECT sp.spell_id, t.truth, sp.school
                   FROM spell sp JOIN spell_truth t USING (spell_id)),
        truth AS (SELECT a.spell_id l, b.spell_id r FROM s a JOIN s b
                  ON a.truth = b.truth AND a.spell_id < b.spell_id AND a.school <> b.school)
        SELECT count(*) AS true_pairs,
               sum(CASE WHEN p.lid IS NOT NULL THEN 1 ELSE 0 END) AS in_blocks,
               round(100.0 * sum(CASE WHEN p.lid IS NOT NULL THEN 1 ELSE 0 END) / count(*), 1) AS pct
        FROM truth LEFT JOIN pairs p ON p.lid = truth.l AND p.rid = truth.r
    """).show()

    print("[eval] threshold sweep:")
    con.sql("""
        WITH s AS (SELECT sp.spell_id, t.truth, sp.school
                   FROM spell sp JOIN spell_truth t USING (spell_id)),
        e AS (SELECT edge.*, (a.truth = b.truth) AS correct
              FROM edge JOIN s a ON a.spell_id = edge.lid JOIN s b ON b.spell_id = edge.rid),
        n AS (SELECT count(*) AS total FROM s a JOIN s b
              ON a.truth = b.truth AND a.spell_id < b.spell_id AND a.school <> b.school)
        SELECT th,
               count(*) FILTER (p >= th) AS edges,
               round(avg(CASE WHEN correct THEN 1.0 ELSE 0 END) FILTER (p >= th), 3) AS edge_precision,
               round(count(*) FILTER (p >= th AND correct) * 1.0 / any_value(n.total), 3) AS edge_recall
        FROM e, n, (SELECT unnest([0.5,0.7,0.8,0.9,0.95,0.99]) AS th)
        GROUP BY th ORDER BY th
    """).show()

    print("[eval] recall by abbreviation stratum (the Berlin problem, measured):")
    con.sql("""
        WITH s AS (SELECT sp.spell_id, t.truth, sp.school, sp.gn_has_full, c.student_id
                   FROM spell sp JOIN spell_truth t USING (spell_id)
                                 JOIN cluster c USING (spell_id))
        SELECT CASE WHEN a.gn_has_full AND b.gn_has_full THEN 'both_full'
                    WHEN a.gn_has_full OR  b.gn_has_full THEN 'mixed'
                    ELSE 'both_abbrev' END AS stratum,
               count(*) AS true_pairs,
               round(100.0 * avg(CASE WHEN a.student_id = b.student_id THEN 1.0 ELSE 0 END), 1) AS recall_pct
        FROM s a JOIN s b ON a.truth = b.truth AND a.spell_id < b.spell_id AND a.school <> b.school
        GROUP BY 1 ORDER BY 2 DESC
    """).show()

    print("[eval] recall by school pair (where the linkage is hardest):")
    con.sql("""
        WITH s AS (
            SELECT c.spell_id, c.student_id, t.truth, sp.school
            FROM cluster c JOIN spell_truth t USING (spell_id) JOIN spell sp USING (spell_id))
        SELECT least(a.school, b.school) AS s1, greatest(a.school, b.school) AS s2,
               count(*) AS true_pairs,
               round(100.0 * avg(CASE WHEN a.student_id = b.student_id THEN 1.0 ELSE 0 END), 1) AS recall_pct
        FROM s a JOIN s b ON a.truth = b.truth AND a.spell_id < b.spell_id AND a.school <> b.school
        GROUP BY 1, 2 HAVING count(*) >= 60 ORDER BY recall_pct LIMIT 12
    """).show()


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--out")
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--annual", help="comma-separated schools with annual registers")
    ap.add_argument("--mean-schools", type=float, default=1.45,
                    help="mean universities per student; calibrate from prev_university")
    args = ap.parse_args()

    con = duckdb.connect()
    stage_prepare(con, args.path, keep_truth=args.eval)
    annual = detect_annual(con, args.annual.split(',') if args.annual else None)
    stage_spells(con, annual)
    stage_block(con)
    stage_compare(con, annual)
    u_all = stage_u(con, annual)
    n_links = expected_links(con, args.mean_schools)
    params = stage_em(con, u_all, n_links)
    stage_score(con, params)
    has_gold = con.sql(
        "SELECT count(prev_university) FROM rec").fetchone()[0] > 0
    if has_gold:
        resolve_prev(con)

    if args.sweep:
        for th in (0.7, 0.8, 0.9, 0.95, 0.99):
            print(f"\n===== threshold {th} =====")
            stage_cluster(con, th)
            if has_gold:
                validate_gold(con)
            if args.eval:
                evaluate(con, brief=True)
        return
    stage_cluster(con, args.threshold)
    if has_gold:
        validate_gold(con)

    if args.eval:
        evaluate(con)
    if args.out:
        con.execute(f"""COPY (
            SELECT r.*, m.spell_id, c.student_id
            FROM rec r JOIN map m USING (rec_id) JOIN cluster c USING (spell_id)
        ) TO '{args.out}' (FORMAT PARQUET)""")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
