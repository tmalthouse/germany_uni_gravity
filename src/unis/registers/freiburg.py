"""Freiburg: parsed from the printed Matrikel (raw/universities/freiburg.csv)."""

import polars as pl

from unis import paths


def build() -> pl.LazyFrame:
    return (
        pl.scan_csv(paths.FREIBURG_RAW)
        .with_columns(
            law_admin=pl.col.field.is_in(["jur"]),
            theology=pl.col.field.is_in(["theol"]),
            medicine=pl.col.field.is_in(["med", "pharm", "chir"]),
            sciences=pl.col.field == "math",
            humanities=pl.col.field == "phil",
        )
        .rename({"matrikel_year": "first_year"})
        .drop("freiburg_id")
    )
