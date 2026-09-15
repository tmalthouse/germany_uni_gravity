"""Würzburg: matriculation register."""

import polars as pl

from unis.registers.common import scan_volumes


def build() -> pl.LazyFrame:
    return (
        scan_volumes("wuerzburg")
        .with_columns(
            location_full=pl.col.hometown,
            first_year=pl.col.date_iso.str.extract(r"(\d{4})").cast(pl.Int64),
            field=pl.col.field_expanded.fill_null(pl.col.field),
        )
        .select("last_name", "first_names", "field", "hometown", "location_full", "first_year")
        .with_columns(
            law_admin=pl.col.field.is_in([
                "Jura", "Kameralwissenschaft", "Cameralia", "Oeconomie", "Staatswirtschaft",
                "Jura und Kameralwissenschaft", "Kameralistik",
            ]),
            theology=pl.col.field.is_in(["Theologie"]),
            medicine=pl.col.field.is_in([
                "Medizin", "Chirurgie", "Pharmazie", "Veterinärmedizin", "Tierheilkunde",
                "Medizin und Chirurgie",
            ]),
            sciences=pl.col.field.is_in([
                "Forstwissenschaft", "Res saltuaria (Forstwissenschaft)", "Mathematik",
                "Bergbauwissenschaft", "Chemie", "Logik", "Architektur", "Log.", "Archit.",
                "Phys.", "Physik", "Bergw.", "Bauw.", "Bauwesen", "Technologie",
            ]),
            humanities=pl.col.field.is_in(["Philosophie", "Philologie", "Philol."]),
        )
    )
