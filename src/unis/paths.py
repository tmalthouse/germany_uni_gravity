"""Every file location the pipeline reads or writes.

Stages refer to these names rather than to paths relative to the working
directory, so any module can be run from anywhere.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
RAW = DATA / "raw"  # frozen sources, imported once with `make import-raw` (gitignored)
MANUAL = DATA / "manual"  # small hand-curated inputs (tracked)
INTERIM = DATA / "interim"  # rebuilt intermediates
PROCESSED = DATA / "processed"  # rebuilt analysis datasets

OUTPUT = ROOT / "output"
TABLES = OUTPUT / "tables"
FIGURES = OUTPUT / "figures"

# --- raw sources -------------------------------------------------------------
OUT_FINAL = RAW / "universities" / "out_final"  # LLM register extractions, one dir per volume
FREIBURG_RAW = RAW / "universities" / "freiburg.csv"
GIESSEN_RAW = RAW / "universities" / "giessen.csv"
TGN_REL = RAW / "tgn_rel_0126"  # Getty TGN relational release (.out files)
GHGIS = RAW / "ghgis"  # historical boundary shapefiles


def ghgis_layer(name: str) -> Path:
    """Shapefile for a GHGIS layer such as '1820CORE' or '1848GERMANCONFED'."""
    return GHGIS / f"GHGIS{name}" / f"GHGIS{name}.shp"


# --- hand-curated inputs -----------------------------------------------------
OUT_FINAL_ORDER = MANUAL / "out_final_order.txt"
MUNICH_DATE_HELPER = MANUAL / "munich_date_helper.csv"
HEIDELBERG_HEADS = MANUAL / "heidelberg_heads"
UNI_TGN_MAPPING = MANUAL / "uni_tgn_mapping.csv"
GUILD_DATES = MANUAL / "guild_dates_ogilvie.csv"
GEOCODE_CURATED = MANUAL / "geocode"

# --- intermediates -----------------------------------------------------------
TGN = INTERIM / "tgn"
HEIDELBERG_DATE_FIXER = INTERIM / "heidelberg_date_fixer.parquet"
REGISTERS = INTERIM / "registers"
STUDENTS_UNLINKED = INTERIM / "all_students_unlinked.parquet"
GEOCODE_CACHE = INTERIM / "geocode_cache"
STUDENTS_WITH_TGN = INTERIM / "students_with_tgn.parquet"
STUDENTS_GEOCODED = INTERIM / "students_geocoded.parquet"
STUDENTS_LINKED = INTERIM / "students_linked.parquet"
STUDENTS_CLEAN = INTERIM / "students_clean.parquet"

# --- analysis datasets -------------------------------------------------------
STUDENTS_FINAL = PROCESSED / "students_final.parquet"
OD = PROCESSED / "od.parquet"  # origin x destination x era
OD_YEAR = PROCESSED / "od_year.parquet"  # origin x destination x enrollment year
