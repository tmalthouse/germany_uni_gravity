"""Jena: annual registers, one row per student per register year."""

import polars as pl

from unis.registers.common import location_full, scan_volumes, volume_year


def build() -> pl.LazyFrame:
    return (
        # 'jona' catches volumes whose id the extractor misread.
        scan_volumes("jena", "jona")
        .sort("volume_id", maintain_order=True)
        .with_columns(
            pl.col.hometown.str.replace(r"\.$", ""),
            pl.col.region.str.replace(r"\.$", ""),
        )
        .with_columns(
            location_full=location_full()
            .str.strip_chars()
            .replace(r"Daher", None)
            .forward_fill(limit=10),
            register_year=volume_year(),
            address=pl.col.lodging,
        )
        .select(
            "last_name",
            "first_names",
            pl.col.register_year.alias("first_year"),
            "field",
            "location_full",
            "hometown",
            "region",
            "address",
        )
    )
