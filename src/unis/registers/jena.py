"""Jena: annual registers, one row per student per register year."""

import polars as pl

from unis.registers.common import location_full, scan_volumes, volume_year


def field_flags() -> dict[str, pl.Expr]:
    """Field groups from the register's abbreviations ('J.', 'Th.', 'M.', ...)."""
    return dict(
        law_admin=pl.col.field.is_in(["J.", "Cam.", "J.u.C.", "J. u. C."]),
        theology=pl.col.field.is_in(["T.", "Th.", "T. u. P.", "T.u.P."]),
        medicine=pl.col.field.is_in(["M.", "Pm.", "Ph.", "Chir."]),
        sciences=pl.col.field.is_in(["Math."]),
        humanities=pl.col.field.is_in(["P.", "Oec.", "Phil.", "Phll."]),
    )


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
        .with_columns(**field_flags())
    )
