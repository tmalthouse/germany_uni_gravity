"""Tübingen: annual registers, one row per student per register year."""

import polars as pl

from unis.registers.common import scan_volumes, volume_year


def field_flags() -> dict[str, pl.Expr]:
    """Field groups from the register's field names, punctuation stripped."""
    field = pl.col.field.str.replace_all(r"\W", "")
    return dict(
        law_admin=field.str.contains(r"(Rechts|Cameral|Staats)"),
        theology=field.str.contains(r"(Theol|theol|Théol)"),
        medicine=field.str.contains(r"(Med|Medicin|Chirur|Pharm)"),
        sciences=field.str.contains("Forstwissenschaft"),
        humanities=field.str.contains(r"(Phil)"),
    )


def build() -> pl.LazyFrame:
    return (
        # Field is forward-filled within each volume, never across volumes.
        scan_volumes("tuebingen", per_file=lambda lf: lf.with_columns(pl.col.field.forward_fill()))
        .with_columns(
            first_year=pl.col.years.str.extract(r"(\d{4})").cast(pl.Int64).fill_null(volume_year()),
            location_full=pl.col.hometown_full,
        )
        .sort("first_year", "last_name", "first_names")
        .select(
            "last_name",
            "first_names",
            "first_year",
            "field",
            "location_full",
            "hometown",
            "region",
        )
        .with_columns(**field_flags())
    )
