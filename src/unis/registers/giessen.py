"""Giessen: parsed from BA-25.pdf (raw/universities/giessen.csv)."""

import polars as pl

from unis import paths
from unis.mappings.giessen_fathers import canonical_expr as giessen_father_fix
from unis.mappings.giessen_fields import FIELD_OF_STUDY_MAP


def build() -> pl.LazyFrame:
    return (
        pl.scan_csv(paths.GIESSEN_RAW)
        .with_columns(
            field=pl.col.field_of_study.replace(FIELD_OF_STUDY_MAP),
            father_occupation=giessen_father_fix("father_profession"),
            first_year=pl.col.enrolment_date.str.to_date(strict=False).dt.year(),
        )
        .with_columns(
            law_admin=pl.col.field.str.contains("(Rechtswissenschaft|Kameralwissenschaft)"),
            theology=pl.col.field.str.contains("Theologie"),
            medicine=pl.col.field.str.contains("(Medizin|Chirurgie|Pharmazie|Tierheilkunde)"),
            sciences=pl.col.field.str.contains("(Forstwissenschaft|Chemie|Mathematik|Architektur)"),
            humanities=pl.col.field.str.contains("(Philosophie|Philologie)"),
        )
        .select(
            "last_name",
            "first_names",
            "first_year",
            "hometown",
            "field",
            "law_admin",
            "theology",
            "medicine",
            "sciences",
            "humanities",
            "father_occupation",
        )
    )
