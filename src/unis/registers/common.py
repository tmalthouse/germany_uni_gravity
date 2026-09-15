"""Pieces shared by the per-school register cleaners."""

from collections.abc import Callable
from pathlib import Path

import polars as pl

from unis import paths

# What the extractor wrote where the register had a ditto mark ("as above").
DITTO_MARKS = ['"', "", "=", "-", "„", "—", "."]


def volume_files(*needles: str) -> list[Path]:
    """Extraction CSVs whose path contains any of `needles`, in pipeline order.

    The order is pinned in data/manual/out_final_order.txt: it is the unsorted
    directory order the original pipeline happened to read the files in.
    Several cleaners forward-fill ditto marks across file boundaries, so the
    order is part of the result, and a fresh directory listing would not
    reproduce it.
    """
    names = paths.OUT_FINAL_ORDER.read_text().splitlines()
    return [paths.OUT_FINAL / n for n in names if n and any(s in n for s in needles)]


def scan_volumes(
    *needles: str, per_file: Callable[[pl.LazyFrame], pl.LazyFrame] | None = None
) -> pl.LazyFrame:
    """All extraction volumes for a school, every column read as text."""
    frames = [pl.scan_csv(f, infer_schema_length=0) for f in volume_files(*needles)]
    if per_file is not None:
        frames = [per_file(lf) for lf in frames]
    return pl.concat(frames, how="diagonal_relaxed")


def location_full() -> pl.Expr:
    """'hometown, region', or whichever of the two is present."""
    return (
        pl.when(pl.col.hometown.is_not_null() & pl.col.region.is_not_null())
        .then(pl.col.hometown + ", " + pl.col.region)
        .when(pl.col.hometown.is_not_null())
        .then(pl.col.hometown)
        .otherwise(pl.col.region)
    )


def volume_year() -> pl.Expr:
    """Register year from a volume id such as 'berlin_21' (-> 1821)."""
    return 1800 + pl.col.volume_id.str.extract(r"(\d{2})").cast(pl.Int64)
