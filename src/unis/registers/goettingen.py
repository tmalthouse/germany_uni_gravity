"""Göttingen: the 1734 and 1837 matriculation volumes."""

import polars as pl

from unis import paths
from unis.mappings.register_fields import GOETTINGEN_YEAR_ANCHORS
from unis.registers.common import location_full

VOLUMES = [
    "goettingen_1734/goettingen_1734_pp409-943.csv",
    "goettingen_1837/goettingen_1837_pp9-145.csv",
]


def build() -> pl.LazyFrame:
    lf = (
        pl.concat(
            [pl.scan_csv(paths.OUT_FINAL / v, infer_schema_length=0) for v in VOLUMES],
            how="diagonal_relaxed",
        )
        .with_columns(
            pl.col.source_page.cast(pl.Int64),
            location_full=location_full(),
        )
        .with_columns(
            field=pl.col.field_expanded.fill_null(pl.col.field),
            year=pl.col.date.str.extract(r"(\d{4})").cast(pl.Int64),
        )
        .join(
            pl.DataFrame(GOETTINGEN_YEAR_ANCHORS).lazy(),
            on=["last_name", "first_names", "source_page"],
            how="left",
            maintain_order="left",
        )
        .with_columns(pl.col.year.fill_null(pl.col.year_helper).forward_fill())
        .select(
            "last_name",
            "first_names",
            "hometown",
            "region",
            "location_full",
            "field",
            pl.col.year.alias("first_year"),
            "father",
            "father_occupation",
            "father_location",
            "prev_university",
        )
        .with_columns(f_help=pl.col.field.str.to_lowercase().str.replace_all(r"\W", ""))
        .with_columns(
            law_admin=pl.col.f_help.str.contains(r"(jura|jur|kam|cam|staatsw)"),
            theology=pl.col.f_help.str.contains(r"(theol)"),
            medicine=pl.col.f_help.str.contains(r"(med|chir|pharm)"),
            sciences=pl.col.f_help.str.contains(r"(math|physik|berge|naturw|forstw)"),
            humanities=pl.col.f_help.str.contains(r"(phil|philol|geschichte|lit|hist)"),
        )
        .drop("f_help")
    )
    n_1840 = lf.select((pl.col.first_year == 1840).sum()).collect().item()
    print(f"Number of entries reporting year 1840: {n_1840}")
    return lf
