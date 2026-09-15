"""Berlin: annual registers, one row per student per register year."""

import polars as pl

from unis.mappings.register_fields import FIELD_MAPPING_BERLIN
from unis.registers.common import DITTO_MARKS, location_full, scan_volumes, volume_year


def field_flags() -> dict[str, pl.Expr]:
    """Field groups from the register's abbreviations ('Theol.', 'Kam.', ...).

    Abbreviations missing from FIELD_MAPPING_BERLIN get no group.
    """
    field = pl.col.field.str.replace_all(r"\W", "").replace_strict(
        FIELD_MAPPING_BERLIN, default=None
    )
    return dict(
        law_admin=field.is_in(["Kam", "Recht", "Rechte"]),
        theology=field == "Theol",
        medicine=field == "Med",
        sciences=field.is_in(["Mineral", "Math", "Naturw"]),
        humanities=field.is_in(["Phil"]),
    )


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
        .with_columns(**field_flags())
    )
