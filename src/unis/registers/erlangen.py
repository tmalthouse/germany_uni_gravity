"""Erlangen: matriculation register."""

import polars as pl

from unis.mappings.register_fields import FIELD_MAPPING_ERLANGEN
from unis.registers.common import location_full, scan_volumes


def build() -> pl.LazyFrame:
    return (
        scan_volumes("erlangen")
        .with_columns(
            pl.col.field.fill_null("")
            .replace([r"desgl.", "="], None)
            .forward_fill(limit=10)
            .replace("", None)
        )
        .with_columns(pl.col.field.replace(FIELD_MAPPING_ERLANGEN))
        .with_columns(location_full=location_full())
        .select(
            "last_name",
            "first_names",
            "field",
            "hometown",
            "region",
            "location_full",
            first_year=pl.col.year.cast(pl.Int64),
        )
        .with_columns(
            law_admin=pl.col.field.str.contains(r"(Jura|Kam|Staatsw|Kriegsw|Rechte)"),
            theology=pl.col.field.str.contains(r"(Theol)"),
            medicine=pl.col.field.str.contains(r"(Med|Chir|Pharm)"),
            sciences=pl.col.field.str.contains(r"(Math|Physik|Berge|Naturw|Forstw|Eng|Chem)"),
            humanities=pl.col.field.str.contains(r"(Phil|Philol|Geschichte|Lit|Aesth)"),
        )
    )
