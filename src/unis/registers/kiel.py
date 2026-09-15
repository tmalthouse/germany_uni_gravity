"""Kiel: matriculation register."""

import polars as pl

from unis.registers.common import scan_volumes


def build() -> pl.LazyFrame:
    return (
        scan_volumes("kiel")
        .with_columns(
            location_full=pl.col.hometown,
            first_year=pl.col.date_year.cast(pl.Int64),
            field=pl.col.field_expanded.fill_null(pl.col.field),
        )
        .select("last_name", "first_names", "field", "hometown", "location_full", "first_year")
        .with_columns(
            law_admin=pl.col.field.is_in(
                ["Jura", "Cameralwissenschaft", "Jura und Cameralwissenschaft"]
            ),
            theology=pl.col.field.is_in([
                "Theologie",
                "Theologie und Philologie",
                "Theologie und Philosophie",
                "Philosophie und Theologie",
                "Philologie und Theologie",
            ]),
            medicine=pl.col.field.is_in(
                ["Medizin", "Pharmazie", "Medizin und Chirurgie", "Pharmacie"]
            ),
            sciences=pl.col.field.is_in(["Mathematik", "Chemie"]),
            humanities=pl.col.field.is_in(["Philosophie", "Philologie"]),
        )
    )
