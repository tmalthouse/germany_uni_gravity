"""Berlin: annual registers, one row per student per register year."""

import polars as pl

from unis.registers.common import DITTO_MARKS, location_full, scan_volumes, volume_year


def build() -> pl.LazyFrame:
    return (
        scan_volumes("berlin")
        .with_columns(
            pl.col.field.replace(old=DITTO_MARKS, new=None).forward_fill(limit=12),
            pl.col.hometown.replace(old=DITTO_MARKS, new=None).forward_fill(limit=12),
            pl.col.region.replace(old=DITTO_MARKS, new=None).forward_fill(limit=12),
            pl.col.street.replace(old=DITTO_MARKS, new=None).forward_fill(limit=12),
        )
        .with_columns(pl.col.first_names.fill_null(pl.col.first_name))
        .with_columns(
            register_year=volume_year(),
            location_full=location_full(),
            address=pl.col.street + " " + pl.col.house_number,
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
