"""The review queue.

The highest-leverage artifact in this pipeline is not the scorer, it is a work
queue ordered by how much a human decision would buy. That is frequency times
uncertainty: a string used by 400 students whose top two candidates are nearly
tied is worth far more attention than a singleton that failed to match.

Each row is a *string*, not a student, and carries the top candidates with
their scores so the decision can be made without going back to the data. The
resolution column is left blank for a human to fill; the file is shaped to be
pasted straight into overrides.csv or corrections.csv.
"""

from __future__ import annotations

import polars as pl

from .config import Config


def build(
    students: pl.DataFrame,
    keys: pl.DataFrame,
    review_cand: pl.DataFrame,
    best: pl.DataFrame,
    cfg: Config,
) -> pl.DataFrame:
    """Assemble the ranked review queue.

    Stats come from `best` -- the per-(student, field) decision -- rather than
    from the final output. A region string that lost the field-preference
    contest to its hometown still has a resolution and a margin of its own, and
    that is what a reviewer needs to see. Reading stats off the final output
    instead would leave exactly those strings blank.
    """
    # Every (student, field) instance of every string, carrying the resolution
    # key so per-key decisions can be counted at student weight -- frequency is
    # what makes a review decision worth making.
    keyed = students.join(
        keys, on=["region", "hometown", "school_fixed"], how="left", nulls_equal=True
    )
    instances = pl.concat(
        [
            keyed.select(
                "key_id", "school_fixed",
                pl.col("region").alias("src_raw"),
                pl.lit(1, dtype=pl.Int32).alias("field_code"),
            ),
            keyed.select(
                "key_id", "school_fixed",
                pl.col("hometown").alias("src_raw"),
                pl.lit(2, dtype=pl.Int32).alias("field_code"),
            ),
        ]
    ).filter(pl.col("src_raw").is_not_null())

    usage = instances.group_by("src_raw").agg(
        pl.len().alias("freq"),
        pl.col("school_fixed").unique().sort().str.join(", ").alias("schools"),
    )

    stats = (
        instances.join(
            best.select("key_id", "field_code", "SUBJECT_ID", "score_margin", "match_method"),
            on=["key_id", "field_code"],
            how="left",
        )
        .group_by("src_raw")
        .agg(
            pl.col("SUBJECT_ID").is_null().mean().alias("unresolved_rate"),
            pl.col("score_margin").mean().alias("mean_margin"),
            pl.col("match_method").drop_nulls().min().alias("match_method"),
        )
    )

    # Distinct candidate places per string, best first.
    top = (
        review_cand.sort(
            ["src_raw", "score", "SUBJECT_ID"], descending=[False, True, False]
        )
        .group_by("src_raw", maintain_order=True)
        .agg(
            pl.col("TERM").head(4).alias("cand_names"),
            pl.col("SUBJECT_ID").head(4).alias("cand_ids"),
            pl.col("score").head(4).round(2).alias("cand_scores"),
            pl.col("dist").head(4).round(0).alias("cand_dist_km"),
            pl.col("geo_precision").head(4).alias("cand_precision"),
            pl.len().alias("n_distinct_candidates"),
        )
    )

    queue = (
        usage.join(stats, on="src_raw", how="left")
        .join(top, on="src_raw", how="left")
        .with_columns(
            pl.col("unresolved_rate").fill_null(1.0),
            pl.col("n_distinct_candidates").fill_null(0),
        )
        .with_columns(
            # Uncertainty blends "we found nothing" with "we found several and
            # could barely tell them apart".
            uncertainty=pl.col("unresolved_rate")
            + (1 - pl.col("unresolved_rate"))
            / (1.0 + pl.col("mean_margin").fill_null(0.0))
        )
        .with_columns(priority=pl.col("freq") * pl.col("uncertainty"))
        .filter(pl.col("uncertainty") > 0.01)
        .sort(["priority", "src_raw"], descending=[True, False])
        .head(cfg.tuning.review_queue_size)
        .with_columns(
            cand_names=pl.col("cand_names").list.join(" | "),
            cand_ids=pl.col("cand_ids").cast(pl.List(pl.Utf8)).list.join(" | "),
            cand_scores=pl.col("cand_scores").cast(pl.List(pl.Utf8)).list.join(" | "),
            cand_dist_km=pl.col("cand_dist_km").cast(pl.List(pl.Utf8)).list.join(" | "),
            cand_precision=pl.col("cand_precision").list.join(" | "),
            unresolved_rate=pl.col("unresolved_rate").round(3),
            mean_margin=pl.col("mean_margin").round(2),
            priority=pl.col("priority").round(1),
            # Blank columns for the reviewer to fill in.
            verdict=pl.lit(""),
            canonical_or_subject_id=pl.lit(""),
            reviewer_note=pl.lit(""),
        )
        .select(
            "src_raw", "freq", "unresolved_rate", "mean_margin", "priority",
            "match_method", "n_distinct_candidates", "cand_names", "cand_ids",
            "cand_scores", "cand_dist_km", "cand_precision", "schools",
            "verdict", "canonical_or_subject_id", "reviewer_note",
        )
    )

    cfg.paths.cache_dir.mkdir(parents=True, exist_ok=True)
    queue.write_csv(cfg.paths.review_queue)
    if cfg.verbose:
        covered = queue["freq"].sum()
        print(
            f"[review] {queue.height} strings queued, covering {covered:,} "
            f"student-field values -> {cfg.paths.review_queue}"
        )
    return queue


def diff_against(previous_path, out: pl.DataFrame, cfg: Config) -> pl.DataFrame | None:
    """Compare resolutions against a previous run's output.

    Aggregate match rate can improve while specific important cases regress, so
    migration should be judged on the diff, not the headline number.
    """
    from pathlib import Path

    previous_path = Path(previous_path)
    if not previous_path.exists():
        return None

    prev = pl.read_parquet(previous_path)
    id_col = "location_id" if "location_id" in prev.columns else "SUBJECT_ID"
    if "index" not in prev.columns:
        return None

    joined = (
        prev.select("index", pl.col(id_col).alias("old_id"))
        .join(out.select("index", "region", "hometown", "school_fixed",
                         pl.col("location_id").alias("new_id"),
                         pl.col("location_name_tgn").alias("new_name")),
              on="index", how="inner")
        .filter(pl.col("old_id") != pl.col("new_id"))
    )

    summary = (
        joined.group_by(["region", "hometown", "old_id", "new_id", "new_name"])
        .agg(pl.len().alias("n_students"))
        .sort("n_students", descending=True)
    )
    summary.write_csv(cfg.paths.diff_report)
    if cfg.verbose:
        print(
            f"[diff] {joined.height:,} students changed resolution across "
            f"{summary.height:,} distinct string pairs -> {cfg.paths.diff_report}"
        )
    return summary
