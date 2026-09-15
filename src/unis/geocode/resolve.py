"""Scoring, decision, and output assembly.

Scoring is a sum of log-scale terms, one per signal:

    score = w_string     * log(string_score)
          + w_prominence * log1p(n_terms)
          + w_distance   * log P(dist | school)
          + w_coherence  * log f(cross-field distance)

None of this is calibrated probability; it is a ranking function whose weights
live in config.Weights. What matters is that no single signal can dominate by
construction -- in particular that proximity to the university is one term
among several rather than the sole criterion, so a prominent distant city can
outscore a nearby hamlet with the same name.
"""

from __future__ import annotations

import polars as pl

from .config import PRECISION_RANKS, Config
from .distance import (
    bin_expr,
    coherence_logdensity,
    fit_prior,
    haversine_expr,
    interpolated_log_density,
)

FIELDS = [("region", 1), ("hometown", 2)]


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------

def school_period_expr() -> pl.Expr:
    """Period-adjusted school identifier.

    Ludwig-Maximilians moved from Landshut to Munich in 1826, so enrollments
    before then belong to the Landshut seat (muenchen_old, TGN 7077307) and
    those from 1826 on to Munich (muenchen, TGN 7004333).

    This threshold read `< 1926` -- a transposed digit. Because the corpus ends
    in 1848 the predicate was true for every Munich row, so all 6,293 of them
    were placed at Landshut, 62 km from Munich, and school_fixed == 'muenchen'
    never occurred in the output at all. checks.py did not catch it: it only
    asserts that each variant produced here exists in uni_tgn_mapping.csv, and
    muenchen_old does.
    """
    return (
        pl.when((pl.col("school") == "muenchen") & (pl.col("first_year") < 1826))
        .then(pl.lit("muenchen_old"))
        .otherwise(pl.col("school"))
        .alias("school_fixed")
    )


def load_students(cfg: Config) -> pl.DataFrame:
    return (
        pl.scan_parquet(cfg.paths.students)
        .with_row_index()
        .with_columns(school_period_expr())
        .collect()
    )


RESOLUTION_KEY = ["region", "hometown", "school_fixed"]


def key_table(students: pl.DataFrame) -> pl.DataFrame:
    """Distinct resolution keys.

    A student's resolution depends only on their two source strings and their
    period-adjusted school -- nothing else in the row enters the score. So all
    scoring happens over distinct triples and is joined back, which collapses
    the candidate expansion from (students x candidates) to
    (distinct triples x candidates). On a real corpus that is two or three
    orders of magnitude smaller, and it is the difference between the scoring
    stage fitting in memory and not.

    If a future signal is genuinely per-student -- period validity against
    first_year is the obvious candidate -- it has to be added to this key.
    """
    return (
        students.group_by(RESOLUTION_KEY)
        .agg(pl.len().alias("n_students"))
        .sort(RESOLUTION_KEY, nulls_last=True)
        .with_row_index("key_id")
    )


def load_university_coords(cfg: Config) -> pl.DataFrame:
    """school_fixed -> university seat coordinates."""
    mapping = pl.read_csv(cfg.paths.uni_mapping)
    coords = pl.read_parquet(cfg.paths.tgn_coords).select("SUBJECT_ID", "LAT", "LON")
    return (
        mapping.rename({"school": "school_fixed"})
        .join(coords, on="SUBJECT_ID", how="left")
        .select(
            "school_fixed",
            pl.col("LAT").alias("LAT0"),
            pl.col("LON").alias("LON0"),
        )
        .group_by("school_fixed")
        .agg(pl.col("LAT0").min(), pl.col("LON0").min())
    )


# --------------------------------------------------------------------------
# Cross-field coherence
# --------------------------------------------------------------------------

def coherence_table(keys: pl.DataFrame, cand: pl.DataFrame, cfg: Config) -> pl.DataFrame:
    """Minimum distance from each candidate to the *other* field's candidates.

    Computed over distinct (region, hometown) string pairs rather than per
    student, which keeps the cross join small. Standing in for gazetteer
    hierarchy, which TGN's terms table does not expose: if the region names a
    territory and the hometown names a village, the correct village is the one
    near that territory.
    """
    k = cfg.tuning.coherence_topk

    has_anc = "ancestors" in cand.columns
    keep = ["src_raw", "SUBJECT_ID", "LAT", "LON"] + (["ancestors"] if has_anc else [])
    top = (
        cand.filter(pl.col("LAT").is_not_null())
        .sort(
            ["src_raw", "string_score", "n_terms", "SUBJECT_ID"],
            descending=[False, True, True, False],
        )
        .with_columns(_r=pl.int_range(pl.len()).over("src_raw"))
        .filter(pl.col("_r") < k)
        .select(keep)
    )

    pairs = (
        keys.select(
            pl.col("region").alias("r_src"), pl.col("hometown").alias("h_src")
        )
        .filter(pl.col("r_src").is_not_null() & pl.col("h_src").is_not_null())
        .unique()
    )
    if pairs.height == 0:
        return pl.DataFrame(
            schema={
                "src_raw": pl.Utf8,
                "other_src": pl.Utf8,
                "SUBJECT_ID": pl.Int64,
                "cross_dist": pl.Float64,
            }
        )

    left = top.rename({"src_raw": "r_src", "SUBJECT_ID": "r_sid",
                       "LAT": "LAT0", "LON": "LON0"})
    right = top.rename({"src_raw": "h_src", "SUBJECT_ID": "h_sid",
                        "LAT": "LAT1", "LON": "LON1"})
    if has_anc:
        left = left.rename({"ancestors": "r_anc"})
        right = right.rename({"ancestors": "h_anc"})

    joined = (
        pairs.join(left, on="r_src", how="inner")
        .join(right, on="h_src", how="inner")
        .with_columns(cross=haversine_expr())
    )
    if has_anc:
        # Containment is proof, not proximity: the Neustadt inside Mecklenburg
        # is the one whose parent chain actually names Mecklenburg.
        joined = joined.with_columns(
            nested=pl.col("r_anc").fill_null([]).list.contains(pl.col("h_sid"))
            | pl.col("h_anc").fill_null([]).list.contains(pl.col("r_sid"))
        )
    else:
        joined = joined.with_columns(nested=pl.lit(False))

    region_side = (
        joined.group_by(["r_src", "h_src", "r_sid"])
        .agg(pl.col("cross").min().alias("cross_dist"), pl.col("nested").any())
        .rename({"r_src": "src_raw", "h_src": "other_src", "r_sid": "SUBJECT_ID"})
    )
    hometown_side = (
        joined.group_by(["h_src", "r_src", "h_sid"])
        .agg(pl.col("cross").min().alias("cross_dist"), pl.col("nested").any())
        .rename({"h_src": "src_raw", "r_src": "other_src", "h_sid": "SUBJECT_ID"})
    )
    return pl.concat([region_side, hometown_side])


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def expand(keys: pl.DataFrame, cand: pl.DataFrame, unis: pl.DataFrame) -> pl.DataFrame:
    """One row per (key, field, candidate place), with distances attached."""
    frames = []
    for field, code in FIELDS:
        other = "hometown" if field == "region" else "region"
        frames.append(
            keys.select(
                "key_id",
                "school_fixed",
                pl.col(field).alias("src_raw"),
                pl.col(other).alias("other_src"),
                pl.lit(code, dtype=pl.Int32).alias("field_code"),
            ).filter(pl.col("src_raw").is_not_null())
        )
    long = pl.concat(frames)

    return (
        long.join(cand, on="src_raw", how="inner")
        .join(unis, on="school_fixed", how="left")
        .rename({"LAT": "LAT1", "LON": "LON1"})
        .with_columns(dist=haversine_expr())
    )


def fit_distance_prior(
    keys: pl.DataFrame, cand: pl.DataFrame, unis: pl.DataFrame, cfg: Config
) -> pl.DataFrame:
    """Fit log P(distance bin | school) on unambiguous strings only.

    Weighted by the number of students behind each key, so a string used by
    four hundred students counts four hundred times. The fitting set is built
    directly from the single-candidate strings rather than from the full
    expansion, which keeps it small enough to compute in one pass before
    chunked scoring begins.
    """
    unambiguous = cand.filter(pl.col("n_candidates") == 1).select(
        "src_raw", "SUBJECT_ID", "LAT", "LON"
    )
    frames = []
    for field, _ in FIELDS:
        frames.append(
            keys.select(
                "school_fixed", "n_students", pl.col(field).alias("src_raw")
            ).filter(pl.col("src_raw").is_not_null())
        )
    rows = (
        pl.concat(frames)
        .join(unambiguous, on="src_raw", how="inner")
        .join(unis, on="school_fixed", how="left")
        .rename({"LAT": "LAT1", "LON": "LON1"})
        .with_columns(dist=haversine_expr())
        .filter(pl.col("dist").is_not_null())
        .select(
            pl.col("school_fixed").repeat_by("n_students").explode(),
            pl.col("dist").repeat_by("n_students").explode(),
        )
    )
    prior = fit_prior(rows, cfg)
    if cfg.verbose:
        print(
            f"[score] distance prior fitted on {rows.height:,} unambiguous "
            f"student-field values across {prior['school_fixed'].n_unique()} schools"
        )
    return prior


def score(
    expanded: pl.DataFrame,
    coherence: pl.DataFrame,
    prior: pl.DataFrame,
    cfg: Config,
    hints: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Attach the distance prior, coherence, and the combined score."""
    w = cfg.weights

    out = (
        expanded.with_columns(dist_bin=bin_expr("dist", cfg))
        .pipe(interpolated_log_density, prior, cfg)
        # A null here means the school never appeared in the fitting set at all.
        # Fall back to the pooled expectation rather than to zero.
        .with_columns(
            pl.col("dist_ll").fill_null(pl.col("dist_ll").mean()).fill_null(0.0)
        )
        .join(coherence, on=["src_raw", "other_src", "SUBJECT_ID"], how="left")
        .with_columns(has_other=pl.col("other_src").is_not_null())
        .with_columns(coh_ll=coherence_logdensity("cross_dist", "has_other", cfg))
    )
    if "nested" not in out.columns:
        out = out.with_columns(nested=pl.lit(False))
    out = out.with_columns(pl.col("nested").fill_null(False))

    # Hint containment: does a qualifier parsed off the string name one of this
    # candidate's ancestors?
    if hints is not None and hints.height and "ancestors" in out.columns:
        out = (
            out.join(hints, on="src_raw", how="left")
            .with_columns(
                _hit=pl.col("hint_id").is_not_null()
                & pl.col("ancestors").fill_null([]).list.contains(pl.col("hint_id"))
            )
            .group_by(
                [c for c in out.columns if c != "ancestors"] + ["ancestors"],
                maintain_order=True,
            )
            .agg(pl.col("_hit").any().alias("hint_match"))
        )
    else:
        out = out.with_columns(hint_match=pl.lit(False))

    if "term_quality" not in out.columns:
        out = out.with_columns(term_quality=pl.lit(1.0))

    out = out.with_columns(
        score=(
            w.string * pl.col("string_score").log()
            + w.term_quality * pl.col("term_quality").fill_null(1.0).log()
            + w.prominence
            * (
                pl.col("prominence")
                if "prominence" in out.columns
                else pl.col("n_terms").log1p()
            )
            - (
                pl.when(pl.col("is_sub_settlement").fill_null(False))
                .then(pl.lit(cfg.tuning.sub_settlement_penalty))
                .otherwise(pl.lit(0.0))
                if "is_sub_settlement" in out.columns
                else pl.lit(0.0)
            )
            + w.distance * pl.col("dist_ll")
            + w.coherence * pl.col("coh_ll")
            + w.nesting * pl.col("nested").cast(pl.Float64)
            + w.hint_containment * pl.col("hint_match").fill_null(False).cast(pl.Float64)
            - pl.when(pl.col("LAT1").is_null())
            .then(pl.lit(cfg.tuning.no_coords_penalty))
            .otherwise(pl.lit(0.0))
        )
    )
    return out


def resolve_keys(
    keys: pl.DataFrame,
    cand: pl.DataFrame,
    unis: pl.DataFrame,
    coherence: pl.DataFrame,
    prior: pl.DataFrame,
    cfg: Config,
    hints: pl.DataFrame | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Score and decide in chunks of resolution keys.

    Returns (best, review_candidates). The chunked loop keeps only the
    per-key decision and a small per-string candidate summary; the full
    expansion is never materialized all at once.
    """
    best_parts: list[pl.DataFrame] = []
    review_parts: list[pl.DataFrame] = []
    n = keys.height
    size = max(1, cfg.tuning.chunk_size)

    for start in range(0, n, size):
        chunk = keys.slice(start, size)
        expanded = expand(chunk, cand, unis)
        if expanded.height == 0:
            continue
        scored = score(expanded, coherence, prior, cfg, hints)
        best_parts.append(decide_per_field(scored, cfg))
        review_parts.append(
            scored.sort(
                ["src_raw", "SUBJECT_ID", "score"], descending=[False, False, True]
            )
            .group_by(["src_raw", "SUBJECT_ID"], maintain_order=True)
            .agg(
                pl.col("TERM").first(),
                pl.col("score").first(),
                pl.col("dist").first(),
                pl.col("geo_precision").first(),
            )
        )
        if cfg.verbose and n > size:
            print(f"[score] keys {min(start + size, n):,}/{n:,}")

    if not best_parts:
        empty = expand(keys.head(0), cand, unis)
        return (
            decide_per_field(score(empty, coherence, prior, cfg, hints), cfg),
            empty,
        )

    best = pl.concat(best_parts)
    review_cand = (
        pl.concat(review_parts)
        .sort(["src_raw", "SUBJECT_ID", "score"], descending=[False, False, True])
        .group_by(["src_raw", "SUBJECT_ID"], maintain_order=True)
        .agg(
            pl.col("TERM").first(),
            pl.col("score").first(),
            pl.col("dist").first(),
            pl.col("geo_precision").first(),
        )
    )
    return best, review_cand


def decide_per_field(scored: pl.DataFrame, cfg: Config) -> pl.DataFrame:
    """Pick the best candidate per (student, field), with an explicit tiebreak."""
    ordered = scored.sort(
        ["key_id", "field_code", "score", "n_terms", "SUBJECT_ID"],
        descending=[False, False, True, True, False],
    )
    best = (
        ordered.group_by(["key_id", "field_code"], maintain_order=True)
        .agg(
            pl.col("SUBJECT_ID").first(),
            pl.col("TERM").first(),
            pl.col("TERM_ID").first(),
            pl.col("LAT1").first().alias("lat"),
            pl.col("LON1").first().alias("lon"),
            pl.col("dist").first(),
            pl.col("match_method").first(),
            pl.col("geo_precision").first(),
            pl.col("school_fixed").first(),
            *([pl.col("is_settlement").first()] if "is_settlement" in scored.columns else []),
            *([pl.col("placetype").first()] if "placetype" in scored.columns else []),
            *([pl.col("prominence").first()] if "prominence" in scored.columns else []),
            pl.col("string_score").first(),
            pl.col("n_candidates").first(),
            pl.col("cap_hit").first(),
            pl.col("score").first().alias("best_score"),
            pl.col("score").get(1, null_on_oob=True).alias("runner_up_score"),
        )
        .with_columns(
            score_margin=(pl.col("best_score") - pl.col("runner_up_score"))
            .fill_null(cfg.tuning.margin_cap)
            .clip(upper_bound=cfg.tuning.margin_cap)
        )
        .drop("runner_up_score")
    )
    return best


def choose_field(
    best: pl.DataFrame, cfg: Config, unis: pl.DataFrame | None = None
) -> pl.DataFrame:
    """Collapse the two fields to one resolution per student.

    Hometown is preferred, unless the hometown's best candidate sits at the
    university's own city: there the hometown string is usually just the
    institution's address repeated, and the region field carries the actual
    origin (the Berlin case -- 'Berlin' as hometown means 'at the university',
    while the region names where the student is from). Otherwise the hometown
    is the more specific declaration and wins even when it is no finer than
    the region in gazetteer precision: a region string that shares its TGN
    term with a city ('Baden' the territory vs Baden-Baden the city) is the
    classic false match this rule prevents.

    The prior behaviour is retained as an override: when the region resolves
    materially coarser than the hometown -- a territory against a settlement
    -- the finer place wins regardless, and `matched` reports which field was
    used.
    """
    rank = pl.col("geo_precision").replace_strict(
        PRECISION_RANKS, default=PRECISION_RANKS["unknown"], return_dtype=pl.Int32
    )
    best = best.with_columns(prec_rank=rank)

    region = best.filter(pl.col("field_code") == 1)
    home = best.filter(pl.col("field_code") == 2)

    has_settle = "is_settlement" in best.columns

    # University seat coordinates, for the at-university exception. Joined on
    # school_fixed, which the per-field best table now carries.
    unis_local = (
        unis.rename({"LAT0": "U_LAT0", "LON0": "U_LON0"})
        if unis is not None
        else None
    )

    region_sel = ["key_id", pl.col("prec_rank").alias("r_prec")]
    if unis is not None and "school_fixed" in best.columns:
        region_sel.append(pl.col("school_fixed"))
    joined = home.join(
        region.select(region_sel),
        on="key_id",
        how="left",
    )
    if unis is not None and "school_fixed" in joined.columns:
        joined = joined.join(unis_local, on="school_fixed", how="left")

    # The Berlin exception: the hometown candidate sits at the university's
    # own city, so the hometown string is 'studying here', not an origin.
    # Equirectangular, exact enough at 15 km. Region-only keys (no hometown
    # candidate) are unaffected: home rows only exist when a hometown was
    # resolved.
    at_uni = pl.lit(False)
    if "U_LAT0" in joined.columns:
        cos_lat = ((pl.col("lat") + pl.col("U_LAT0")) / 2).radians().cos()
        at_uni = (
            pl.col("lat").is_not_null()
            & pl.col("U_LAT0").is_not_null()
            & (
                (
                    (pl.col("lat") - pl.col("U_LAT0")).pow(2)
                    + ((pl.col("lon") - pl.col("U_LON0")) * cos_lat).pow(2)
                ).sqrt()
                * 111.0
                <= 15.0
            )
        )

    joined = joined.with_columns(
        prefer_hometown=(
            # Default: the hometown is the more specific declaration and
            # wins. The overrides below only ever hand the resolution BACK
            # to the region.
            pl.col("r_prec").is_not_null()
            # Region materially finer than the hometown (a settlement
            # against a hometown that only names a territory): the finer
            # place wins. The old precision-gap rule read the same way.
            & ~(
                (pl.col("prec_rank") - pl.col("r_prec"))
                >= cfg.tuning.precision_override_gap
            )
        )
    )
    if "U_LAT0" in joined.columns:
        # ...and never when the hometown is merely 'at the university'.
        joined = joined.with_columns(prefer_hometown=pl.col("prefer_hometown") & ~at_uni)

    drop_cols = ["r_prec", "prefer_hometown"]
    if "U_LAT0" in joined.columns:
        drop_cols += ["U_LAT0", "U_LON0"]
    home_keep = joined.filter(pl.col("prefer_hometown") | pl.col("r_prec").is_null()).drop(drop_cols)
    # Hometown won on these keys; the region resolution is discarded.
    # The Berlin exception is just prefer_hometown=False with both fields
    # present: region stays.
    region_stays = joined.filter(
        ~pl.col("prefer_hometown") & pl.col("r_prec").is_not_null()
    ).select("key_id")

    # Keys where the hometown field never resolved stay with the region field.
    region_only = region.join(home.select("key_id"), on="key_id", how="anti")
    region_keep = region.join(
        pl.concat([region_stays, region_only.select("key_id")]).unique(),
        on="key_id",
        how="semi",
    )

    return pl.concat([home_keep, region_keep], how="diagonal").drop("prec_rank")


def apply_overrides(
    resolved: pl.DataFrame, keys: pl.DataFrame, index: pl.DataFrame, cfg: Config
) -> pl.DataFrame:
    """Force curated resolutions, applied after ranking so nothing undoes them.

    overrides.csv columns:
        raw         source string to override, case-insensitive
        subject_id  forced SUBJECT_ID, or empty to force 'unresolved'
        reason      free text
    """
    if not cfg.paths.overrides.exists():
        return resolved

    ov = pl.read_csv(cfg.paths.overrides, schema_overrides={"subject_id": pl.Int64})
    if ov.height == 0:
        return resolved
    ov = (
        ov.with_columns(pl.col("raw").str.strip_chars().str.to_lowercase().alias("_key"))
        .unique(subset=["_key"], keep="first")
        .select("_key", "subject_id")
    )

    attrs = (
        index.select("SUBJECT_ID", "TERM", "TERM_ID", "geo_precision", "LAT", "LON")
        .sort(["SUBJECT_ID", "TERM_ID", "TERM"])
        .group_by("SUBJECT_ID", maintain_order=True)
        .agg(
            pl.col("TERM").first(), pl.col("TERM_ID").first(),
            pl.col("geo_precision").first(), pl.col("LAT").first(), pl.col("LON").first(),
        )
    )

    # An override fires on whichever field carries the overridden string,
    # region first.
    targets = []
    for field, code in FIELDS:
        targets.append(
            keys.select(
                "key_id",
                "school_fixed",
                pl.col(field).str.strip_chars().str.to_lowercase().alias("_key"),
                pl.lit(code, dtype=pl.Int32).alias("field_code"),
            )
        )
    hits = (
        pl.concat(targets)
        .join(ov, on="_key", how="inner")
        .sort(["key_id", "field_code"])
        .group_by("key_id", maintain_order=True)
        .agg(
            pl.col("subject_id").first(),
            pl.col("field_code").first(),
            pl.col("school_fixed").first(),
        )
    )
    if hits.height == 0:
        return resolved

    unis = load_university_coords(cfg)
    forced = (
        hits.join(attrs, left_on="subject_id", right_on="SUBJECT_ID", how="left")
        .join(unis, on="school_fixed", how="left")
        .rename({"LAT": "LAT1", "LON": "LON1"})
        .with_columns(dist=haversine_expr())
        .select(
            "key_id",
            pl.when(pl.col("subject_id").is_null())
            .then(None)
            .otherwise(pl.col("field_code"))
            .alias("field_code"),
            pl.col("subject_id").alias("SUBJECT_ID"),
            "TERM",
            "TERM_ID",
            pl.col("LAT1").alias("lat"),
            pl.col("LON1").alias("lon"),
            "dist",
            pl.lit("override").alias("match_method"),
            pl.col("geo_precision").fill_null("unknown"),
            pl.lit(1.0).alias("string_score"),
            pl.lit(1, dtype=pl.UInt32).alias("n_candidates"),
            pl.lit(False).alias("cap_hit"),
            pl.lit(None, dtype=pl.Float64).alias("best_score"),
            pl.lit(cfg.tuning.margin_cap).alias("score_margin"),
        )
    )

    if cfg.verbose:
        print(f"[overrides] forced {forced.height:,} student resolutions")

    return pl.concat(
        [resolved.join(forced.select("key_id"), on="key_id", how="anti"), forced],
        how="diagonal",
    )


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------

OUTPUT_RENAMES = {"SUBJECT_ID": "location_id", "TERM": "location_name_tgn"}


def assemble(
    students: pl.DataFrame, keys: pl.DataFrame, resolved: pl.DataFrame, cfg: Config
) -> pl.DataFrame:
    """Left join resolutions back onto the full student table."""
    keep = resolved.select(
        "key_id",
        "SUBJECT_ID",
        "TERM",
        "TERM_ID",
        "lat",
        "lon",
        "dist",
        pl.col("field_code").alias("matched"),
        "match_method",
        "geo_precision",
        *(["is_settlement"] if "is_settlement" in resolved.columns else []),
        *(["placetype"] if "placetype" in resolved.columns else []),
        *(["prominence"] if "prominence" in resolved.columns else []),
        "string_score",
        "n_candidates",
        "cap_hit",
        "score_margin",
    )

    out = (
        students.join(
            keys.select("key_id", *RESOLUTION_KEY),
            on=RESOLUTION_KEY, how="left", nulls_equal=True,
        )
        .join(keep, on="key_id", how="left")
        .drop("key_id")
        .rename(OUTPUT_RENAMES)
        .with_columns(
            # Null coherence guarantee: the identity columns stand or fall
            # together. A forced-null override, or a place with no coordinates,
            # must not leave a half-populated row.
            low_confidence=pl.col("score_margin") < cfg.tuning.low_confidence_margin,
        )
        .with_columns(
            [
                pl.when(pl.col("location_id").is_null())
                .then(None)
                .otherwise(pl.col(c))
                .alias(c)
                for c in ["location_name_tgn", "matched", "lat", "lon", "dist"]
            ]
        )
        .sort("index")
    )
    return out
