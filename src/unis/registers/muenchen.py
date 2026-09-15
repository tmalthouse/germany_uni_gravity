"""München (Landshut until 1826): matriculation register, with page-year dates."""

import polars as pl

from unis import paths
from unis.registers.common import location_full, scan_volumes


def build() -> pl.LazyFrame:
    return (
        scan_volumes("muenchen")
        .with_columns(pl.col.academic_year.replace("ibid", None))
        .with_columns(
            academic_year=pl.when(pl.col.extraction_notes.str.contains(r"Year\w* inferred"))
            .then(None)
            .otherwise(pl.col.academic_year),
            first_names=pl.when(
                pl.col.extraction_notes.str.contains("'v.'", literal=True)
                | pl.col.extraction_notes.str.contains("'von'", literal=True)
            )
            .then(pl.col.first_names + "von")
            .otherwise(pl.col.first_names),
        )
        .with_columns(
            location_full=location_full(),
            field=pl.col.field_expanded.fill_null(pl.col.field),
            first_year=pl.col.academic_year.str.extract(r"(\d{4})").cast(pl.Int64),
            source_page=pl.col.source_page.cast(pl.Int64),
        )
        .filter(pl.col.first_year >= 1799)
        .filter(pl.col.source_page != 261)
        .join(
            pl.scan_csv(paths.MUNICH_DATE_HELPER),
            how="left",
            maintain_order="left",
            on=["source_page"],
        )
        .with_columns(
            first_year=pl.when(
                pl.col.year_helper.is_null()
                | ((pl.col.first_year - pl.col.year_helper).abs() > 2)
                | (pl.col.source_page == 262)
            )
            .then(None)
            .otherwise(pl.col.first_year)
            .fill_null(pl.col.year_helper)
            .forward_fill()
        )
        .select(
            "last_name", "first_names", "field", "hometown", "region", "location_full",
            "first_year", "source_page",
        )
        .with_columns(
            law_admin=pl.col.field.is_in([
                "Jura", "Cameralwissenschaft", "Jura Cameralwissenschaft",
                "Jura, Cameralwissenschaft",
            ]),
            theology=pl.col.field.is_in(["Theologie", "Theologie Philosophie"]),
            medicine=pl.col.field.is_in([
                "Medizin", "Chirurgie", "Pharmazie", "Veterinärmedizin", "Tierheilkunde",
                "Medizin und Chirurgie",
            ]),
            sciences=pl.col.field.is_in([
                "Forstwissenschaft", "Res saltuaria (Forstwissenschaft)", "Mathematik",
                "Bergbauwissenschaft", "Chemie", "Logik", "Architektur", "Log.", "Archit.",
                "Phys.", "Physik", "Bergw.", "Bauw.", "Bauwesen",
            ]),
            humanities=pl.col.field.is_in(["Philosophie", "Philologie", "Philol."]),
        )
    )
