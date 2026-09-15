"""Marburg: the 1796, 1811 and 1823 matriculation volumes."""

import polars as pl

from unis.registers.common import location_full, scan_volumes


def build() -> pl.LazyFrame:
    return (
        scan_volumes("marburg")
        .with_columns(
            first_year=pl.col.date_iso.str.extract(r"(\d{4})").cast(pl.Int64).forward_fill(),
            location_full=location_full(),
            field=pl.col.field_expanded.fill_null(pl.col.field),
        )
        .select(
            "last_name", "first_names", "field", "hometown", "region", "location_full",
            "first_year",
        )
        .with_columns(
            law_admin=pl.col.field.is_in([
                "Rechtswissenschaft (Jura)",
                "Rechtswissenschaft",
                "Oeconomia politica (Staatswirtschaft)",
                "Kameralwissenschaft",
                "Staatswirtschaft",
                "Rechtswissenschaft und Staatswirtschaft",
            ]),
            theology=pl.col.field.is_in(
                ["Theologie", "Theologie und Philologie", "Theologie & Philologie"]
            ),
            medicine=pl.col.field.is_in([
                "Medizin", "Chirurgie", "Pharmazie", "Veterinärmedizin", "Tierheilkunde",
                "Medizin und Chirurgie",
            ]),
            sciences=pl.col.field.is_in([
                "Forstwissenschaft", "Res saltuaria (Forstwissenschaft)", "Mathematik",
                "Bergbauwissenschaft", "Chemie", "Architektur",
            ]),
            humanities=pl.col.field.is_in(["Philosophie", "Philologie"]),
        )
    )
