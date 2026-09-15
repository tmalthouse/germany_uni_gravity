"""Candidate generation.

Work happens at the level of *distinct source strings*, not students. A corpus
of tens of thousands of students typically holds only a few thousand distinct
region/hometown values, so everything expensive -- normalization, phonetic
keying, fuzzy matching -- is paid once per string and joined back.

The output is deliberately not a decision. It is one row per
(string, candidate place) with the evidence attached, and the ranking stage
decides. Separating these two is what lets the scorer weigh a nearby weak match
against a distant strong one instead of having the join silently pick.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import math

import duckdb
import polars as pl

from .config import Config
from .distance import EARTH_RADIUS_KM, haversine_expr
from .normalize import add_ladder

# Ladder rungs tried in order, most to least reliable. Once a string has
# candidates from a rung, coarser rungs are not consulted for it.
def _pick_best_method(df: pl.DataFrame) -> pl.DataFrame:
    """One row per (string, place), keeping the highest-scoring match method.

    The retry pass can rediscover a place the first pass already found, so the
    two runs have to be reconciled rather than concatenated blindly.
    """
    return (
        df.sort(
            ["src_raw", "SUBJECT_ID", "string_score", "TERM_ID"],
            descending=[False, False, True, False],
        )
        .group_by(["src_raw", "SUBJECT_ID"], maintain_order=True)
        .agg(
            pl.col("TERM").first(), pl.col("TERM_ID").first(),
            pl.col("match_method").first(), pl.col("string_score").first(),
            *([pl.col("term_quality").first()] if "term_quality" in df.columns else []),
        )
    )


def _pick_representative_term(df: pl.DataFrame) -> pl.DataFrame:
    """Collapse to one row per (string, place) with a deterministic term.

    A coarse rung matches many of a place's terms at once -- 'Koeln', 'Köln'
    and 'Cologne' all share a phonetic key -- so something has to choose which
    one is reported as location_name_tgn. Leaving that to unique(keep="first")
    makes the output non-reproducible across runs even though the resolved
    location is identical, which is exactly the failure the contract's
    determinism guarantee is about.

    Rule: lowest TERM_ID, tie-broken on the term text.
    """
    by = ["src_raw", "SUBJECT_ID"]
    has_quality = "term_quality" in df.columns
    sort_cols = by + (["term_quality"] if has_quality else []) + ["TERM_ID", "TERM"]
    desc = [False, False] + ([True] if has_quality else []) + [False, False]
    aggs = [pl.col("TERM").first(), pl.col("TERM_ID").first()]
    if has_quality:
        aggs.append(pl.col("term_quality").first())
    return df.sort(sort_cols, descending=desc).group_by(by, maintain_order=True).agg(aggs)


LADDER_STAGES = [
    ("exact", "TERM", "src_raw"),
    ("fold", "k_fold", "k_fold"),
    ("modernize", "k_modern", "k_modern"),
    ("strip", "k_strip", "k_strip"),
    ("phonetic", "k_phon", "k_phon"),
]


def string_inventory(students: pl.LazyFrame) -> pl.DataFrame:
    """Distinct non-null values of region and hometown, with frequencies."""
    frames = []
    for field, code in (("region", "region"), ("hometown", "hometown")):
        frames.append(
            students.select(
                pl.col(field).alias("src_raw"),
                pl.lit(code).alias("field"),
            )
        )
    return (
        pl.concat(frames)
        .filter(pl.col("src_raw").is_not_null() & (pl.col("src_raw").str.strip_chars() != ""))
        .group_by("src_raw")
        .agg(pl.len().alias("freq"))
        .sort("freq", descending=True)
        .collect()
    )


def apply_corrections(inventory: pl.DataFrame, cfg: Config) -> pl.DataFrame:
    """Apply curated string rewrites and explicit non-place exclusions.

    corrections.csv columns:
        raw        source string, matched case-insensitively after trimming
        canonical  replacement string, or empty to mark 'not a place'
        context    optional ancestor constraint: the canonical must resolve to
                   a place whose gazetteer parent chain contains this one
        category   territory_seat entries are skipped when territory_mode="self"
        reason     free text, for auditability

    The context column exists because rewriting a territory to the bare name of
    its seat throws away the only thing that disambiguated it.
    "Mecklenburg-Schwerin" is unambiguous; "Schwerin" is not, and will happily
    resolve to a suburb of Bochum. Context restores the constraint.
    """
    empty = inventory.with_columns(
        pl.col("src_raw").alias("src_corrected"),
        pl.lit(False).alias("corrected"),
        pl.lit(False).alias("not_a_place"),
        pl.lit(None, dtype=pl.Utf8).alias("context"),
    )
    if not cfg.paths.corrections.exists():
        return empty

    raw_corr = pl.read_csv(cfg.paths.corrections)
    for col in ("context", "category"):
        if col not in raw_corr.columns:
            raw_corr = raw_corr.with_columns(pl.lit(None, dtype=pl.Utf8).alias(col))

    if cfg.tuning.territory_mode == "self":
        before = raw_corr.height
        raw_corr = raw_corr.filter(pl.col("category") != "territory_seat")
        if cfg.verbose:
            print(f"[candidates] territory_mode='self': skipping "
                  f"{before - raw_corr.height} territory_seat corrections")

    corr = (
        raw_corr
        .with_columns(pl.col("raw").str.strip_chars().str.to_lowercase().alias("_key"))
        .with_row_index("_row")
        .sort("_row")
        .group_by("_key", maintain_order=True)
        .agg(pl.col("canonical").first(), pl.col("context").first())
        .select("_key", "canonical", "context")
        # A matched row with an empty canonical marks 'not a place'. The
        # marker is what separates that case from an inventory string no
        # correction row touches: after read_csv an empty canonical is null,
        # exactly like the nulls a left join produces for no match at all.
        .with_columns(_in_corr=pl.lit(True))
    )

    out = (
        inventory.with_columns(
            pl.col("src_raw").str.strip_chars().str.to_lowercase().alias("_key")
        )
        .join(corr, on="_key", how="left")
        .with_columns(
            not_a_place=pl.col("_in_corr").fill_null(False)
            & pl.col("canonical").is_null(),
            corrected=pl.col("_in_corr").fill_null(False)
            & pl.col("canonical").is_not_null(),
        )
        .with_columns(
            src_corrected=pl.when(pl.col("corrected"))
            .then(pl.col("canonical"))
            .otherwise(pl.col("src_raw")),
            context=pl.when(pl.col("corrected")).then(pl.col("context")).otherwise(None),
        )
        .drop("_key", "canonical", "_in_corr")
    )
    return out


def _jaro_filter(
    df: pl.DataFrame, left: str, right: str, threshold: float
) -> pl.DataFrame:
    """Keep rows whose two string columns are at least `threshold` similar."""
    if df.height == 0:
        return df.drop(left, right)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        df.write_parquet(tmp / "in.parquet")
        con = duckdb.connect()
        con.execute(
            f"""
            COPY (
                SELECT * FROM read_parquet('{tmp / "in.parquet"}')
                WHERE jaro_winkler_similarity({left}, {right}) >= {threshold}
            ) TO '{tmp / "out.parquet"}' (FORMAT PARQUET)
            """
        )
        con.close()
        out = pl.read_parquet(tmp / "out.parquet")
    return out.drop(left, right)


def _fuzzy_stage(
    residual: pl.DataFrame, index: pl.DataFrame, cfg: Config
) -> pl.DataFrame:
    """Blocked Jaro-Winkler fallback for strings with no ladder match.

    Runs in DuckDB, which has jaro_winkler_similarity built in. Blocking on a
    shared prefix and a length window keeps this from being a cross join; it is
    only ever applied to the residual, which is small by construction.
    """
    t = cfg.tuning
    residual = residual.filter(
        pl.col("k_strip").is_not_null()
        & (pl.col("k_strip").str.len_chars() >= t.fuzzy_min_len)
    )
    if residual.height == 0:
        return residual.select(
            pl.col("src_raw"),
            pl.lit(None, dtype=pl.Int64).alias("SUBJECT_ID"),
            pl.lit(None, dtype=pl.Utf8).alias("TERM"),
            pl.lit(None, dtype=pl.Int64).alias("TERM_ID"),
            pl.lit(None, dtype=pl.Utf8).alias("match_method"),
            pl.lit(None, dtype=pl.Float64).alias("string_score"),
        )

    left = residual.select("src_raw", "k_strip").with_columns(
        blk=pl.col("k_strip").str.slice(0, t.fuzzy_prefix),
        # Signed. str.len_chars() is UInt32, and the length-window predicate
        # below is a subtraction that goes negative half the time; in an
        # unsigned type that underflows before abs() ever sees it.
        ln=pl.col("k_strip").str.len_chars().cast(pl.Int64),
    )
    right = (
        index.filter(pl.col("k_strip").is_not_null())
        .select("k_strip", "TERM", "TERM_ID", "SUBJECT_ID")
        .with_columns(
            blk=pl.col("k_strip").str.slice(0, t.fuzzy_prefix),
            ln=pl.col("k_strip").str.len_chars().cast(pl.Int64),
        )
    )

    # Data is handed to DuckDB as parquet rather than in-memory Arrow. Both
    # libraries read and write parquet natively, so this keeps pyarrow out of
    # the dependency set.
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        left.write_parquet(tmp / "left.parquet")
        right.write_parquet(tmp / "right.parquet")
        con = duckdb.connect()
        con.execute(
            f"""
            COPY (
                SELECT
                    l.src_raw,
                    r.SUBJECT_ID,
                    r.TERM,
                    r.TERM_ID,
                    jaro_winkler_similarity(l.k_strip, r.k_strip) AS sim
                FROM read_parquet('{tmp / "left.parquet"}') l
                JOIN read_parquet('{tmp / "right.parquet"}') r
                  ON l.blk = r.blk
                 AND abs(CAST(l.ln AS BIGINT) - CAST(r.ln AS BIGINT))
                     <= {t.fuzzy_len_tolerance}
                WHERE jaro_winkler_similarity(l.k_strip, r.k_strip)
                      >= {t.fuzzy_threshold}
            ) TO '{tmp / "out.parquet"}' (FORMAT PARQUET)
            """
        )
        con.close()
        result = pl.read_parquet(tmp / "out.parquet")

    base = cfg.tuning.method_scores["fuzzy"]
    # Best similarity per (string, place), then a deterministic representative.
    result = (
        result.sort(
            ["src_raw", "SUBJECT_ID", "sim", "TERM_ID", "TERM"],
            descending=[False, False, True, False, False],
        )
        .group_by(["src_raw", "SUBJECT_ID"], maintain_order=True)
        .agg(pl.col("TERM").first(), pl.col("TERM_ID").first(), pl.col("sim").first())
    )
    return result.with_columns(
        match_method=pl.lit("fuzzy"),
        # Scale the fuzzy score by how good the match actually was, so a 0.99
        # similarity outranks a 0.92 one within the same method.
        string_score=pl.lit(base) * pl.col("sim"),
    ).drop("sim")


def _run_ladder(
    live: pl.DataFrame, index: pl.DataFrame, cfg: Config, source: str
) -> tuple[list[pl.DataFrame], pl.DataFrame]:
    """Walk the ladder over `source`, returning (hits, strings still unmatched)."""
    live = add_ladder(live, source)
    index_cols = ["TERM", "TERM_ID", "SUBJECT_ID"]
    if "term_quality" in index.columns:
        index_cols.append("term_quality")
    collected: list[pl.DataFrame] = []
    remaining = live

    for method, index_key, source_key in LADDER_STAGES:
        if remaining.height == 0:
            break
        left_key = source if method == "exact" else source_key
        needs_check = method == "phonetic"
        idx = index.filter(pl.col(index_key).is_not_null()).select(
            [pl.col(index_key).alias("_jk")]
            + index_cols
            + ([pl.col("k_strip").alias("_rs")] if needs_check else [])
        )
        left_sel = ["src_raw", pl.col(left_key).alias("_jk")]
        if needs_check:
            left_sel.append(pl.col("k_strip").alias("_ls"))
        hit = (
            remaining.filter(pl.col(left_key).is_not_null())
            .select(left_sel)
            .join(idx, on="_jk", how="inner")
            .drop("_jk")
        )
        if needs_check:
            before = hit.height
            hit = _jaro_filter(
                hit.filter(pl.col("_ls").is_not_null() & pl.col("_rs").is_not_null()),
                "_ls", "_rs", cfg.tuning.phonetic_min_similarity,
            )
            if cfg.verbose and before - hit.height:
                print(
                    f"[candidates] {"":>10}  dropped {before - hit.height:,} phonetic "
                    f"collisions below similarity {cfg.tuning.phonetic_min_similarity}"
                )
        hit = _pick_representative_term(hit)
        if hit.height:
            collected.append(
                hit.with_columns(
                    match_method=pl.lit(method),
                    string_score=pl.lit(
                        cfg.tuning.method_scores[method], dtype=pl.Float64
                    ),
                )
            )
            remaining = remaining.filter(
                ~pl.col("src_raw").is_in(hit["src_raw"].unique())
            )
        if cfg.verbose:
            print(
                f"[candidates] {method:>10}: {hit.height:,} pairs, "
                f"{remaining.height:,} strings still unmatched"
            )

    if remaining.height:
        fz = _fuzzy_stage(remaining, index, cfg)
        if fz.height:
            collected.append(fz)
            remaining = remaining.filter(
                ~pl.col("src_raw").is_in(fz["src_raw"].unique())
            )
        if cfg.verbose:
            print(f"[candidates] {"fuzzy":>10}: {fz.height:,} pairs")

    return collected, remaining


def _gate_distant_coarse_matches(
    cand: pl.DataFrame, unis: pl.DataFrame, cfg: Config
) -> pl.DataFrame:
    """Drop coarse-rung candidates implausibly far from every university.

    Only the rungs in `gated_methods` are affected. An exact match to a distant
    place is evidence; a phonetic collision with one is noise.
    """
    t = cfg.tuning
    gated = list(t.gated_methods)
    if not gated:
        return cand

    places = cand.select("SUBJECT_ID", "LAT", "LON").unique(subset=["SUBJECT_ID"])
    reach = (
        places.filter(pl.col("LAT").is_not_null())
        .rename({"LAT": "LAT1", "LON": "LON1"})
        .join(unis.select("LAT0", "LON0"), how="cross")
        .with_columns(d=haversine_expr())
        .group_by("SUBJECT_ID")
        .agg(pl.col("d").min().alias("uni_dist_min"))
    )

    out = cand.join(reach, on="SUBJECT_ID", how="left")
    drop = pl.col("match_method").is_in(gated) & (
        pl.col("uni_dist_min").is_null() | (pl.col("uni_dist_min") > t.coarse_gate_km)
    )
    n_drop = out.filter(drop).height
    if cfg.verbose and n_drop:
        print(
            f"[candidates] gated {n_drop:,} coarse-rung candidates beyond "
            f"{t.coarse_gate_km:,.0f} km from any university"
        )
    return out.filter(~drop).drop("uni_dist_min")


def _collapse_coincident(cand: pl.DataFrame, cfg: Config) -> pl.DataFrame:
    """Collapse candidates that are the same place under different subjects.

    Greedy, deterministic, and school-independent by construction: candidates
    are ordered by match quality, then prominence, then id, and each is kept
    only if it is not already within `dedupe_radius_km` of something kept. The
    ordering deliberately excludes distance to the university, because the
    whole point is that the representative must not depend on which university
    the student attended.
    """
    t = cfg.tuning
    radius = t.dedupe_radius_km
    if radius <= 0:
        return cand.with_columns(n_merged=pl.lit(1, dtype=pl.UInt32))

    # Settlements rank first, so when a city and its administrative container
    # collapse together the city is the survivor. Depth breaks remaining ties
    # toward the finer place.
    has_settle = "is_settlement" in cand.columns
    has_anc = "ancestors" in cand.columns
    sort_cols, desc = ["src_raw"], [False]
    if has_settle:
        sort_cols.append("is_settlement"); desc.append(True)
    sort_cols += ["string_score"]; desc += [True]
    if "depth" in cand.columns:
        sort_cols.append("depth"); desc.append(True)
    tail = "prominence" if "prominence" in cand.columns else "n_terms"
    sort_cols += [tail, "SUBJECT_ID"]; desc += [True, False]

    ordered = (
        cand.sort(sort_cols, descending=desc)
        .with_columns(_r=pl.int_range(pl.len()).over("src_raw"))
        .filter(pl.col("_r") < t.dedupe_pool)
        .drop("_r")
    )

    lat = ordered["LAT"].to_list()
    lon = ordered["LON"].to_list()
    src = ordered["src_raw"].to_list()
    sid = ordered["SUBJECT_ID"].to_list()
    anc = (
        [set(a) if a is not None else set() for a in ordered["ancestors"].to_list()]
        if has_anc
        else [set()] * len(src)
    )

    keep = [False] * len(src)
    merged = [1] * len(src)
    kept_idx: list[int] = []
    current: str | None = None
    lat_r = math.radians

    for i in range(len(src)):
        if src[i] != current:
            current, kept_idx = src[i], []
        if lat[i] is None or lon[i] is None:
            keep[i] = True
            kept_idx.append(i)
            continue
        hit = None
        for j in kept_idx:
            # A gazetteer ancestor relation is proof that two candidates are
            # the same place at different administrative levels -- Hannover the
            # city inside Hannover the province -- rather than an inference
            # from proximity. Checked first, and independent of distance.
            if sid[j] in anc[i] or sid[i] in anc[j]:
                hit = j
                break
            if lat[j] is None:
                continue
            # Equirectangular approximation: exact enough at 25 km, and far
            # cheaper than haversine inside an O(k^2) loop.
            dlat = lat_r(lat[i] - lat[j])
            dlon = lat_r(lon[i] - lon[j]) * math.cos(lat_r((lat[i] + lat[j]) / 2))
            if EARTH_RADIUS_KM * math.sqrt(dlat * dlat + dlon * dlon) <= radius:
                hit = j
                break
        if hit is None:
            keep[i] = True
            kept_idx.append(i)
        else:
            merged[hit] += 1

    out = ordered.with_columns(
        pl.Series("_keep", keep),
        pl.Series("n_merged", merged, dtype=pl.UInt32),
    )
    n_dropped = out.filter(~pl.col("_keep")).height
    if cfg.verbose and n_dropped:
        print(
            f"[candidates] collapsed {n_dropped:,} candidates coincident within "
            f"{radius:.0f} km of a better-ranked one"
        )
    return out.filter(pl.col("_keep")).drop("_keep")


def _resolve_context(inventory: pl.DataFrame, index: pl.DataFrame) -> pl.DataFrame:
    """Map each context label to the gazetteer subjects it could name."""
    ctx = (
        inventory.filter(pl.col("context").is_not_null())
        .select("src_raw", pl.col("context").alias("ctx_raw"))
        .unique()
    )
    if ctx.height == 0:
        return pl.DataFrame(schema={"src_raw": pl.Utf8, "ctx_id": pl.Int64})
    keys = add_ladder(ctx, "ctx_raw").select("src_raw", pl.col("k_modern").alias("_ck"))
    coarse = index.filter(
        pl.col("geo_precision").is_in(["district", "territory", "country"])
    ).select(pl.col("k_modern").alias("_ck"), pl.col("SUBJECT_ID").alias("ctx_id"))
    return (
        keys.drop_nulls("_ck").join(coarse, on="_ck", how="inner")
        .select("src_raw", "ctx_id").unique()
    )


def _apply_context_filter(
    cand: pl.DataFrame, ctx: pl.DataFrame, cfg: Config
) -> pl.DataFrame:
    """Keep only candidates whose ancestors satisfy the curated context.

    A hard constraint, not a bonus: the point is to force the correct Schwerin,
    not to nudge toward it. If nothing satisfies the constraint the candidates
    are left untouched and the string is reported, because silently emptying a
    candidate set would turn a curated fix into an unresolved row.
    """
    if ctx.height == 0 or "ancestors" not in cand.columns:
        return cand

    constrained = set(ctx["src_raw"].unique().to_list())
    subject = cand.filter(pl.col("src_raw").is_in(constrained))
    if subject.height == 0:
        return cand

    marked = (
        subject.join(ctx, on="src_raw", how="left")
        .with_columns(
            _ok=pl.col("ancestors").fill_null([]).list.contains(pl.col("ctx_id"))
            | (pl.col("SUBJECT_ID") == pl.col("ctx_id"))
        )
        .group_by(["src_raw", "SUBJECT_ID"])
        .agg(pl.col("_ok").any())
    )
    satisfied = marked.filter(pl.col("_ok")).select("src_raw").unique()
    keep_pairs = marked.filter(pl.col("_ok")).select("src_raw", "SUBJECT_ID")

    unsatisfiable = sorted(constrained - set(satisfied["src_raw"].to_list()))
    if cfg.verbose and unsatisfiable:
        print(f"[candidates] {len(unsatisfiable)} contexts matched no candidate and "
              f"were ignored: {unsatisfiable[:8]}")

    kept = cand.join(keep_pairs, on=["src_raw", "SUBJECT_ID"], how="semi")
    untouched = cand.filter(
        ~pl.col("src_raw").is_in(satisfied["src_raw"].implode().first() or [])
    )
    out = pl.concat([kept, untouched]).unique(subset=["src_raw", "SUBJECT_ID"])
    if cfg.verbose:
        print(f"[candidates] context constraints removed "
              f"{cand.height - out.height:,} candidates")
    return out


def hint_subjects(
    students: pl.LazyFrame, index: pl.DataFrame, cfg: Config
) -> pl.DataFrame:
    """Resolve parsed qualifiers ("...in Schlesien") to candidate ancestor ids.

    The strip rung has always returned a hint and nothing ever consumed it.
    With a hierarchy available the hint becomes a containment constraint: the
    Greifenberg meant is the one whose parent chain contains Silesia.
    """
    if "ancestors" not in index.columns:
        return pl.DataFrame(schema={"src_raw": pl.Utf8, "hint_id": pl.Int64})

    inv = apply_corrections(string_inventory(students), cfg).filter(~pl.col("not_a_place"))
    laddered = add_ladder(inv, "src_corrected").filter(pl.col("hint").is_not_null())
    if laddered.height == 0:
        return pl.DataFrame(schema={"src_raw": pl.Utf8, "hint_id": pl.Int64})

    hints = add_ladder(
        laddered.select("src_raw", pl.col("hint").alias("hint_raw")), "hint_raw"
    ).select("src_raw", pl.col("k_modern").alias("_hk"))

    # Hints name regions, not settlements, so only coarse places qualify.
    coarse = index.filter(
        pl.col("geo_precision").is_in(["district", "territory", "country"])
    ).select(pl.col("k_modern").alias("_hk"), pl.col("SUBJECT_ID").alias("hint_id"))

    return (
        hints.drop_nulls("_hk")
        .join(coarse, on="_hk", how="inner")
        .select("src_raw", "hint_id")
        .unique()
    )


def build(
    students: pl.LazyFrame, index: pl.DataFrame, unis: pl.DataFrame, cfg: Config
) -> pl.DataFrame:
    """Produce the (string -> candidate place) table."""
    if cfg.use_cache and cfg.paths.candidates.exists():
        if cfg.verbose:
            print(f"[candidates] loading cache {cfg.paths.candidates}")
        return pl.read_parquet(cfg.paths.candidates)

    inventory = apply_corrections(string_inventory(students), cfg)
    if cfg.verbose:
        print(
            f"[candidates] {inventory.height:,} distinct strings "
            f"({inventory['freq'].sum():,} student-field values)"
        )

    live = inventory.filter(~pl.col("not_a_place"))
    collected, remaining = _run_ladder(live, index, cfg, "src_corrected")

    # A correction that rewrites to a name the gazetteer does not contain is
    # strictly worse than no correction, because the ladder only ever runs on
    # the corrected form. Retry those on the original string.
    retry = remaining.filter(pl.col("corrected"))
    if retry.height:
        if cfg.verbose:
            print(
                f"[candidates] {retry.height:,} corrected strings matched nothing; "
                "retrying on the raw string"
            )
            for row in retry.select("src_raw", "src_corrected", "freq").sort(
                "freq", descending=True
            ).head(20).iter_rows():
                print(f"             dead correction: {row[0]!r} -> {row[1]!r} ({row[2]:,})")
        extra, _ = _run_ladder(
            retry.select("src_raw", "src_corrected", "corrected", "freq"),
            index, cfg, "src_raw",
        )
        collected.extend(extra)

    if collected:
        cand = pl.concat(collected, how="diagonal")
    else:
        cand = pl.DataFrame(
            schema={
                "src_raw": pl.Utf8, "TERM": pl.Utf8, "TERM_ID": pl.Int64,
                "SUBJECT_ID": pl.Int64, "match_method": pl.Utf8,
                "string_score": pl.Float64,
            }
        )
    cand = cand.pipe(_pick_best_method)

    corrected_keys = inventory.filter(pl.col("corrected"))["src_raw"]
    cand = cand.with_columns(
        string_score=pl.when(
            pl.col("src_raw").is_in(corrected_keys) & (pl.col("match_method") == "exact")
        )
        .then(pl.lit(cfg.tuning.method_scores["correction"]))
        .otherwise(pl.col("string_score"))
    )

    attr_cols = ["n_terms", "geo_precision", "LAT", "LON"]
    for extra in ("is_settlement", "depth", "placetype", "prominence",
                  "is_sub_settlement"):
        if extra in index.columns:
            attr_cols.append(extra)
    attrs = (
        index.select(["SUBJECT_ID"] + attr_cols)
        .group_by("SUBJECT_ID")
        .agg([pl.col(c).min() for c in attr_cols])
    )
    if "ancestors" in index.columns:
        anc = (
            index.select("SUBJECT_ID", "ancestors")
            .sort("SUBJECT_ID")
            .group_by("SUBJECT_ID", maintain_order=True)
            .agg(pl.col("ancestors").first())
        )
        attrs = attrs.join(anc, on="SUBJECT_ID", how="left")
    cand = (
        cand.join(attrs, on="SUBJECT_ID", how="left")
        .join(inventory.select("src_raw", "freq"), on="src_raw", how="left")
        .pipe(_gate_distant_coarse_matches, unis, cfg)
        .pipe(_apply_context_filter, _resolve_context(inventory, index), cfg)
        .pipe(_collapse_coincident, cfg)
    )

    cand = (
        cand.sort(
            ["src_raw", "string_score", "n_terms", "SUBJECT_ID"],
            descending=[False, True, True, False],
        )
        .with_columns(
            _rank=pl.int_range(pl.len()).over("src_raw"),
            n_candidates=pl.len().over("src_raw"),
        )
        .filter(pl.col("_rank") < cfg.tuning.max_candidates_per_string)
        .with_columns(cap_hit=pl.col("n_candidates") > cfg.tuning.max_candidates_per_string)
        .drop("_rank")
    )

    cfg.paths.cache_dir.mkdir(parents=True, exist_ok=True)
    cand.write_parquet(cfg.paths.candidates)
    if cfg.verbose:
        print(
            f"[candidates] {cand.height:,} candidate pairs for "
            f"{cand['src_raw'].n_unique():,} strings -> {cfg.paths.candidates}"
        )
    return cand
