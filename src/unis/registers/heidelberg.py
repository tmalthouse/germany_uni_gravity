"""Heidelberg: the 1704, 1807 and 1846 volumes, with page-year date correction."""

from pathlib import Path

import polars as pl

from unis import paths
from unis.registers.common import scan_volumes

# Volumes whose page years are authoritative for the "no entry predates its
# page" rule (page boundaries confirmed against the printed register).
STRICT_PAGE_YEAR_VOLUMES = ["heidelberg_1846"]


def correct_years(lf: pl.LazyFrame, fixer: Path = paths.HEIDELBERG_DATE_FIXER) -> pl.LazyFrame:
    """Correct `first_year` using the page start years in the date fixer.

    `lf` needs volume_id, source_page and first_year. Page years (year_helper)
    are forward-filled within each volume, then:

    - STRICT_PAGE_YEAR_VOLUMES: the register is chronological, so no entry can
      predate its page's year. An entry dated earlier than its page year takes
      the page year, as does one dated more than 2 years later. Entries up to 2
      years later are kept: a page spanning a year change legitimately holds
      next-year entries (heidelberg_1846 pp. 42-43 run Nov 1846 - Feb 1847).
    - Other volumes: an entry carrying the volume's opening year on a page of a
      later year takes the page year; otherwise an entry more than 2 years from
      its page year takes the page year.

    Why (2026-09-14): where an entry's date was a ditto mark, the extractor took
    the year from the scan footer, which names the volume. In heidelberg_1846
    that dated 372 entries on 1847-48 pages as 1846 (1846: 955 entries vs 347 in
    1847 and 179 in 1848), and the old rule only corrected gaps of more than 2
    years. The strict rule is NOT applied to every volume: in heidelberg_1807 it
    would move 1,263 entries by 1-3 years, which is unverified against the
    register.
    """
    fy, page_year = pl.col.first_year, pl.col.year_helper
    opening_year = pl.col.volume_id.str.extract(r"(\d{4})").cast(pl.Int64)
    strict = (
        pl.when(page_year.is_null()).then(fy)
        .when(fy.is_null()).then(page_year)
        .when(fy < page_year).then(page_year)
        .when(fy > page_year + 2).then(page_year)
        .otherwise(fy)
    )
    default = (
        pl.when((fy == opening_year) & page_year.is_not_null() & (page_year > fy))
        .then(page_year)
        .when((fy - page_year.fill_null(fy)).abs() > 2)
        .then(page_year)
        .otherwise(fy)
    )
    return (
        lf.sort("volume_id", "source_page", maintain_order=True)
        .join(
            pl.scan_parquet(fixer),
            on=["volume_id", "source_page"],
            how="left",
            maintain_order="left",
        )
        # Forward-fill within each volume. A plain forward_fill() carried the
        # last page year of one volume into the start of the next (the 1846
        # volume's opening pages inherited 1845 from the 1807 volume).
        .with_columns(pl.col.year_helper.forward_fill().over("volume_id"))
        .with_columns(
            first_year=pl.when(pl.col.volume_id.is_in(STRICT_PAGE_YEAR_VOLUMES))
            .then(strict)
            .otherwise(default)
        )
        .drop("year_helper")
    )


def build() -> pl.LazyFrame:
    entries = (
        scan_volumes("heidelberg")
        .with_columns(
            year=pl.col.date_iso.str.extract(r"(\d{4})").cast(pl.Int64),
            field=pl.col.field_expanded.fill_null(pl.col.field),
            location_full=pl.col.hometown,
            source_page=pl.col.source_page.cast(pl.Int64),
        )
        .select(
            "last_name",
            "first_names",
            "hometown",
            "region",
            "location_full",
            "field",
            "religion",
            "volume_id",
            "source_page",
            pl.col.year.alias("first_year"),
            "father",
            "father_occupation",
            "father_location",
        )
    )
    return (
        correct_years(entries)
        .drop("volume_id", "source_page")
        .with_columns(
            law_admin=pl.col.field.is_in([
                "Cameralia", "Jura Cameralia", "Jura", "Criminalia", "Jura, Cameralia",
                "Notariat", "Jura und Cameralia", "juris candidatus", "Staatswissenschaften",
            ]),
            theology=pl.col.field.is_in(
                ["Theologie", "Theologie Philosophie", "Philosophie Theologie"]
            ),
            medicine=pl.col.field.is_in([
                "Medizin", "Cr.", "Chirurgie", "Pharmacie", "Pharmazie", "Medizin Chirurgie",
            ]),
            sciences=pl.col.field.is_in(
                ["Forstwissenschaft", "Mathematik", "Naturwissenschaften"]
            ),
            humanities=pl.col.field.is_in(["Philosophie", "Pl.", "Philologie", "Geschichte"]),
        )
    )
