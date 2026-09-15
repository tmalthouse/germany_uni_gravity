"""Contract assertions.

These are hard failures, not printed diagnostics. The original script printed a
duplication rate and a match rate and carried on regardless; a broken
cardinality guarantee is the kind of thing that silently corrupts every
downstream group-by, so it should stop the run.
"""

from __future__ import annotations

import polars as pl

from .config import Config


class ContractViolation(AssertionError):
    pass


def _fail(msg: str) -> None:
    raise ContractViolation(msg)


def check_preflight(students: pl.DataFrame, cfg: Config) -> None:
    """Checks that can be run before any work happens."""
    required = {"region", "hometown", "school", "first_year"}
    missing = required - set(students.columns)
    if missing:
        _fail(f"students table is missing required columns: {sorted(missing)}")

    mapping = pl.read_csv(cfg.paths.uni_mapping)
    if "school" not in mapping.columns or "SUBJECT_ID" not in mapping.columns:
        _fail("uni_tgn_mapping.csv must have columns 'school' and 'SUBJECT_ID'")

    dupe_schools = (
        mapping.group_by("school").agg(pl.len().alias("n")).filter(pl.col("n") > 1)
    )
    if dupe_schools.height:
        _fail(
            "uni_tgn_mapping.csv has duplicate school rows: "
            f"{dupe_schools['school'].to_list()}"
        )

    needed = set(students["school_fixed"].unique().drop_nulls().to_list())
    have = set(mapping["school"].to_list())
    unmapped = needed - have
    if unmapped:
        _fail(
            "these school identifiers have no row in uni_tgn_mapping.csv, so "
            f"distance disambiguation is impossible for them: {sorted(unmapped)}"
        )

    # Remap targets must be real places, not just syntactically valid ids. A
    # remap pointing at a SUBJECT_ID absent from the gazetteer produces a
    # dangling identifier that referential checks cannot catch, because the
    # remap itself is what introduced it.
    if cfg.paths.subject_remap.exists():
        remap = pl.read_csv(
            cfg.paths.subject_remap,
            schema_overrides={"from_subject_id": pl.Int64, "to_subject_id": pl.Int64},
        ).drop_nulls(["from_subject_id", "to_subject_id"])
        if remap.height:
            known = (
                pl.scan_parquet(cfg.paths.tgn_terms)
                .select("SUBJECT_ID")
                .unique()
                .collect()["SUBJECT_ID"]
            )
            bad = remap.filter(~pl.col("to_subject_id").is_in(known))
            if bad.height:
                _fail(
                    "subject_remap.csv points at SUBJECT_IDs that do not exist "
                    f"in tgn_terms: {bad['to_subject_id'].to_list()}"
                )
            coords_ids = (
                pl.scan_parquet(cfg.paths.tgn_coords)
                .filter(pl.col("LAT").is_not_null() & pl.col("LON").is_not_null())
                .select("SUBJECT_ID")
                .collect()["SUBJECT_ID"]
            )
            no_xy = remap.filter(~pl.col("to_subject_id").is_in(coords_ids))
            if no_xy.height:
                _fail(
                    "subject_remap.csv points at SUBJECT_IDs with no coordinates, "
                    "which would silently produce resolved-but-unlocatable rows: "
                    f"{no_xy['to_subject_id'].to_list()}"
                )

    coords = pl.read_parquet(cfg.paths.tgn_coords).select("SUBJECT_ID", "LAT", "LON")
    uni_coords = mapping.join(coords, on="SUBJECT_ID", how="left")
    no_coords = uni_coords.filter(pl.col("LAT").is_null() | pl.col("LON").is_null())
    if no_coords.height:
        _fail(
            "these universities map to a SUBJECT_ID with no coordinates, which "
            "would silently disable the distance prior for all their students: "
            f"{no_coords['school'].to_list()}"
        )


def check_output(
    students: pl.DataFrame, out: pl.DataFrame, index: pl.DataFrame, cfg: Config
) -> dict[str, float]:
    """Full post-run contract verification. Returns summary statistics."""
    # 3.1 cardinality
    if out.height != students.height:
        _fail(
            f"row count changed: {students.height:,} in, {out.height:,} out. "
            "Some join multiplied rows."
        )
    if out["index"].n_unique() != out.height:
        _fail("index is not unique in the output")

    # 3.2 column preservation
    lost = set(students.columns) - set(out.columns)
    if lost:
        _fail(f"input columns dropped from the output: {sorted(lost)}")
    for col in students.columns:
        if col in ("index",):
            continue
        if not students.sort("index")[col].equals(out.sort("index")[col]):
            _fail(f"input column '{col}' was modified")

    # 3.4 null coherence
    bad = out.filter(
        pl.col("location_id").is_null() != pl.col("location_name_tgn").is_null()
    )
    if bad.height:
        _fail(f"{bad.height} rows have location_id/location_name_tgn null mismatch")
    bad = out.filter(pl.col("location_id").is_null() != pl.col("matched").is_null())
    if bad.height:
        _fail(f"{bad.height} rows have location_id/matched null mismatch")
    bad = out.filter(pl.col("location_id").is_null() & pl.col("lat").is_not_null())
    if bad.height:
        _fail(f"{bad.height} rows have coordinates without a location_id")

    # 3.5 referential validity
    valid = set(index["SUBJECT_ID"].unique().to_list())
    unknown = (
        out.filter(pl.col("location_id").is_not_null())
        .filter(~pl.col("location_id").is_in(list(valid)))
    )
    if unknown.height:
        _fail(
            f"{unknown.height} rows carry a location_id absent from the gazetteer: "
            f"{unknown['location_id'].unique().head(5).to_list()}"
        )

    coord_check = (
        out.filter(pl.col("lat").is_not_null())
        .join(
            index.select("SUBJECT_ID", "LAT", "LON")
            .group_by("SUBJECT_ID")
            .agg(pl.col("LAT").min(), pl.col("LON").min()),
            left_on="location_id",
            right_on="SUBJECT_ID",
            how="left",
        )
        .filter(
            ((pl.col("lat") - pl.col("LAT")).abs() > 1e-9)
            | ((pl.col("lon") - pl.col("LON")).abs() > 1e-9)
        )
    )
    if coord_check.height:
        _fail(
            f"{coord_check.height} rows have coordinates that do not match their "
            "location_id in the gazetteer"
        )

    # 2.3 matched domain
    bad_matched = out.filter(
        pl.col("matched").is_not_null() & ~pl.col("matched").is_in([1, 2])
    )
    if bad_matched.height:
        _fail("matched contains values outside {1, 2, null}")

    return {
        "n_students": out.height,
        "resolution_rate": out.select(
            pl.col("location_id").is_not_null().mean()
        ).item(),
        "coordinate_rate": out.select(pl.col("lat").is_not_null().mean()).item(),
        "low_confidence_rate": out.select(
            (pl.col("low_confidence") & pl.col("location_id").is_not_null()).mean()
        ).item(),
        "region_share": out.select((pl.col("matched") == 1).mean()).item(),
    }


def check_determinism(run_fn, cfg: Config) -> None:
    """Run the pipeline twice and require identical output. Opt-in; it is slow."""
    a = run_fn(cfg)
    b = run_fn(cfg)
    if not a.equals(b):
        _fail("two runs produced different output; the tiebreak is not stable")
