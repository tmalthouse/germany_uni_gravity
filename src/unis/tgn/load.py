"""Load the TGN relational release into parquet.

The .out files have no header, so every column name here is positional and
comes from the field order in the TGN REL data dictionary. That order has
already proved not to be perfectly reliable once -- COORDINATES carries an
undocumented column, which is why the existing loader has a "NUL" in its list
-- so every categorical field with a documented value domain is checked after
loading. A domain violation almost always means the columns are off by one,
and it is far better to fail here than to silently geocode against a misread
table.

Run with --check to validate and print samples without writing parquet.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from unis import paths

# --------------------------------------------------------------------------
# Column orders, from the data dictionary
# --------------------------------------------------------------------------

TERM_COLS = [
    "AACR2_FLAG", "DISPLAY_DATE", "DISPLAY_NAME", "DISPLAY_ORDER", "END_DATE",
    "HISTORIC_FLAG", "OTHER_FLAGS", "PREFERRED", "START_DATE", "SUBJECT_ID",
    "TERM", "TERM_ID", "VERNACULAR",
]

SUBJECT_COLS = [
    "LEGACY_ID", "MERGED_STAT", "PARENT_KEY", "RECORD_TYPE", "SORT_ORDER",
    "SPECIAL_PROJ", "SUBJECT_ID",
]

SUBJECT_RELS_COLS = [
    "DISPLAY_DATE", "END_DATE", "HISTORIC_FLAG", "PREFERRED", "REL_TYPE",
    "START_DATE", "SUBJECTA_ID", "SUBJECTB_ID", "HIER_REL_TYPE",
]

PTYPE_ROLE_COLS = ["PTYPE_ROLE", "PTYPE_ROLE_ID"]

PTYPE_ROLE_RELS_COLS = [
    "DISPLAY_DATE", "DISPLAY_ORDER", "END_DATE", "HISTORIC_FLAG", "PREFERRED",
    "PTYPE_ROLE_ID", "START_DATE", "SUBJECT_ID",
]

LANGUAGE_RELS_COLS = [
    "LANGUAGE_CODE", "PREFERRED", "SUBJECT_ID", "TERM_ID", "QUALIFIER",
    "TERM_TYPE", "PART_OF_SPEECH", "LANG_STAT",
]

SUBJECT_MERGE_COLS = ["NEW_ID", "DOMINANT_ID", "MERGE_ID"]

SOURCE_RELS_SUBJECT_COLS = ["HOST_TYPE", "PAGE", "SOURCE_ID", "SUBJECT_ID"]

SCOPE_NOTES_COLS = ["SCOPE_NOTE_ID", "SUBJECT_ID", "LANGUAGE_CODE", "NOTE_TEXT"]

# COORDINATES keeps the empirically verified list from the original loader,
# including the undocumented column.
COORDINATES_COLS = [
    "ELEVATION_FEET", "ELEVATION_METERS",
    "LAT_DECIMAL", "LAT_DEGREE", "LAT_DIRECTION", "LAT_MIN", "LAT_SEC",
    "LATLEAST_DECIMAL", "LATLEAST_DEGREE", "LATLEAST_DIR", "LATLEAST_MIN",
    "LATLEAST_SEC",
    "LATMOST_DECIMAL", "LATMOST_DEGREE", "LATMOST_DIR", "LATMOST_MIN",
    "LATMOST_SEC",
    "LONG_DECIMAL", "LONG_DEGREE", "LONG_DIRECTION", "LONG_MIN", "LONG_SEC",
    "LONGLEAST_DECIMAL", "LONGLEAST_DEGREE", "LONGLEAST_DIR", "LONGLEAST_MIN",
    "LONGLEAST_SEC",
    "LONGMOST_DECIMAL", "LONGMOST_DEGREE", "LONGMOST_DIR", "LONGMOST_MIN",
    "LONGMOST_SEC",
    "NUL", "SUBJECT_ID",
]

# Columns that must parse as integers, and columns that must look like text.
# These are the reliable misalignment detectors: if the columns were off by
# one, an ID column would contain names and a text column would contain digits.
# The flag domains below are a weaker signal, because the data dictionary turns
# out not to document every value that actually occurs.
STRUCTURE = {
    "TERM.out": {"ids": ["SUBJECT_ID", "TERM_ID"], "text": ["TERM"]},
    "SUBJECT.out": {"ids": ["SUBJECT_ID"], "text": []},
    "SUBJECT_RELS.out": {"ids": ["SUBJECTA_ID", "SUBJECTB_ID"], "text": []},
    "PTYPE_ROLE.out": {"ids": ["PTYPE_ROLE_ID"], "text": ["PTYPE_ROLE"]},
    "PTYPE_ROLE_RELS.out": {"ids": ["SUBJECT_ID", "PTYPE_ROLE_ID"], "text": []},
    "COORDINATES.out": {"ids": ["SUBJECT_ID"], "text": []},
    "SOURCE_RELS_SUBJECT.out": {"ids": ["SOURCE_ID", "SUBJECT_ID"], "text": []},
    "SCOPE_NOTES.out": {"ids": ["SCOPE_NOTE_ID", "SUBJECT_ID"], "text": []},
    "LANGUAGE_RELS.out": {"ids": ["SUBJECT_ID", "TERM_ID"], "text": []},
    "SUBJECT_MERGE.out": {"ids": [], "text": []},
}

# Minimum share of non-null values that must fall inside the documented domain
# before a column is considered plausibly misaligned rather than merely
# under-documented.
DOMAIN_COVERAGE_FLOOR = 0.80

# Documented value domains. Advisory: the dictionary is incomplete in places
# (TERM.PREFERRED also takes N, HISTORIC_FLAG also takes LU, DISPLAY_NAME also
# takes I), so an unexpected value is reported, not fatal, unless coverage
# collapses or --strict is passed.
DOMAINS = {
    "TERM.out": {
        # The dictionary documents P and V only; real releases also carry N
        # (non-preferred), which the pipeline treats the same as V.
        "PREFERRED": {"P", "V", "N"},
        # LU is undocumented but occurs; it is not used for anything here.
        "HISTORIC_FLAG": {"B", "C", "H", "NA", "U", "LU"},
        "VERNACULAR": {"V", "O", "U"},
        # I is undocumented but occurs.
        "DISPLAY_NAME": {"Y", "N", "NA", "I"},
    },
    "SUBJECT.out": {
        "MERGED_STAT": {"M", "N"},
        "RECORD_TYPE": {"A", "P", "B", "G", "F"},
    },
    "SUBJECT_RELS.out": {
        "PREFERRED": {"P", "N"},
        "REL_TYPE": {"P"},
        "HIER_REL_TYPE": {"G", "I", "P"},
        "HISTORIC_FLAG": {"C", "H", "B", "NA", "U"},
    },
    "PTYPE_ROLE_RELS.out": {
        "PREFERRED": {"P", "N"},
        "HISTORIC_FLAG": {"C", "H", "B", "NA", "U"},
    },
}


def load(
    tgn_dir: Path, name: str, columns: list[str], *, required: bool = True, strict: bool = False
) -> pl.DataFrame | None:
    """Read one .out file, checking width and documented value domains."""
    path = tgn_dir / name
    if not path.exists():
        if required:
            raise FileNotFoundError(f"{path} not found")
        print(f"  {name}: not present, skipping")
        return None

    # Every column is read as text and cast deliberately afterwards. Letting
    # Polars infer types here is actively harmful: it samples the first rows,
    # and several columns are numeric for millions of rows before turning out
    # not to be -- SUBJECT.LEGACY_ID is varchar2(30) and holds values like
    # "T2084". Inference also silently strips leading zeros from codes and can
    # turn a flag column into a boolean. Text in, explicit casts out.
    df = pl.read_csv(
        path,
        has_header=False,
        separator="\t",
        quote_char=None,
        truncate_ragged_lines=True,
        infer_schema=False,
    )

    if df.width != len(columns):
        raise ValueError(
            f"{name}: expected {len(columns)} columns, found {df.width}. "
            "The data dictionary order may not match this release. First row:\n"
            f"{df.head(1).to_dicts()}"
        )
    df = df.rename({old: new for old, new in zip(df.columns, columns)})

    # The .out files are fixed-width in places: HISTORIC_FLAG arrives as "C "
    # rather than "C". Left unstripped this silently breaks every downstream
    # equality test against a flag value, so strip before anything else looks
    # at the data. Empty strings become nulls.
    df = df.with_columns([pl.col(c).str.strip_chars() for c in df.columns])
    df = df.with_columns(
        [
            pl.when(pl.col(c).str.len_chars() == 0).then(None).otherwise(pl.col(c)).alias(c)
            for c in df.columns
        ]
    )

    # --- structural checks: these really do catch misalignment -------------
    spec = STRUCTURE.get(name, {"ids": [], "text": []})
    for col in spec["ids"]:
        not_int = pl.col(col).cast(pl.Int64, strict=False).is_null() & pl.col(col).is_not_null()
        bad = df.select(not_int.mean()).item()
        if bad:
            n_bad = df.select(not_int.sum()).item()
            print(f"    note: {col} has {n_bad:,} values that are not integers "
                  f"({bad:.2%}); these become null")
        if bad and bad > 0.01:
            raise ValueError(
                f"{name}: column '{col}' should hold integers but {bad:.1%} of "
                "values do not parse. The columns are misaligned.\n"
                f"Sample: {df.select(col).head(3).to_series().to_list()}"
            )
    for col in spec["text"]:
        numeric_share = df.select(
            pl.col(col).cast(pl.Float64, strict=False).is_not_null().mean()
        ).item()
        if numeric_share and numeric_share > 0.5:
            raise ValueError(
                f"{name}: column '{col}' should hold text but {numeric_share:.1%} "
                "of values are numeric. The columns are misaligned.\n"
                f"Sample: {df.select(col).head(3).to_series().to_list()}"
            )

    # --- flag domains: advisory ------------------------------------------
    fatal, notes = [], []
    for col, allowed in DOMAINS.get(name, {}).items():
        vals = df[col].drop_nulls().cast(pl.Utf8)
        if vals.len() == 0:
            continue
        coverage = vals.is_in(list(allowed)).mean()
        counts = (
            df.group_by(col).agg(pl.len().alias("n")).drop_nulls(col)
            .filter(~pl.col(col).cast(pl.Utf8).is_in(list(allowed)))
            .sort("n", descending=True).head(6)
        )
        if counts.height == 0:
            continue
        detail = ", ".join(f"{v!r} x{n:,}" for v, n in counts.iter_rows())
        line = f"    {col}: {coverage:.1%} documented; also saw {detail}"
        if coverage < DOMAIN_COVERAGE_FLOOR or strict:
            fatal.append(line)
        else:
            notes.append(line)

    if fatal:
        raise ValueError(
            f"{name}: flag columns do not match the data dictionary.\n"
            + "\n".join(fatal)
            + "\n  Structural checks passed, so this may just be an "
            "under-documented domain; rerun without --strict to continue."
        )

    print(f"  {name}: {df.height:,} rows x {df.width} cols, structure OK")
    for line in notes:
        print(f"    note: {line.strip()}")
    return df


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tgn-dir", default=paths.TGN_REL, type=Path)
    parser.add_argument("--out-dir", default=paths.TGN, type=Path)
    parser.add_argument("--check", action="store_true",
                        help="validate and sample, but write nothing")
    parser.add_argument("--strict", action="store_true",
                        help="treat every undocumented flag value as a fatal error")
    args = parser.parse_args(argv)

    def read(name, columns, required=True):
        return load(args.tgn_dir, name, columns, required=required, strict=args.strict)

    def write(df: pl.DataFrame, filename: str) -> None:
        if args.check:
            print(f"    (check mode, not writing {filename})")
            return
        args.out_dir.mkdir(parents=True, exist_ok=True)
        df.write_parquet(args.out_dir / filename)

    print(f"Loading TGN from {args.tgn_dir}")

    # --- terms -------------------------------------------------------------
    terms = read("TERM.out", TERM_COLS)
    write(
        terms.with_columns(
            pl.col("SUBJECT_ID").cast(pl.Int64, strict=False),
            pl.col("TERM_ID").cast(pl.Int64, strict=False),
            pl.col("TERM").cast(pl.Utf8),
            pl.col("START_DATE").cast(pl.Int64, strict=False),
            pl.col("END_DATE").cast(pl.Int64, strict=False),
        ),
        "tgn_terms.parquet",
    )

    # --- subjects: record type, merge status, preferred parent -------------
    # RECORD_TYPE is the single most useful new filter. A student's origin is
    # an administrative place; 'P' records are rivers and mountains, and
    # 'G'/'F' are structural nodes that are not places at all.
    subjects = read("SUBJECT.out", SUBJECT_COLS)
    write(
        subjects.select(
            pl.col("SUBJECT_ID").cast(pl.Int64, strict=False),
            pl.col("PARENT_KEY").cast(pl.Int64, strict=False),
            pl.col("RECORD_TYPE").cast(pl.Utf8),
            pl.col("MERGED_STAT").cast(pl.Utf8),
        ),
        "tgn_subjects.parquet",
    )

    # --- place types -------------------------------------------------------
    ptype = read("PTYPE_ROLE.out", PTYPE_ROLE_COLS)
    ptype_rels = read("PTYPE_ROLE_RELS.out", PTYPE_ROLE_RELS_COLS)
    write(
        ptype_rels.select(
            pl.col("SUBJECT_ID").cast(pl.Int64, strict=False),
            pl.col("PTYPE_ROLE_ID").cast(pl.Int64, strict=False),
            pl.col("PREFERRED").cast(pl.Utf8),
            pl.col("HISTORIC_FLAG").cast(pl.Utf8),
            pl.col("START_DATE").cast(pl.Int64, strict=False),
            pl.col("END_DATE").cast(pl.Int64, strict=False),
            pl.col("DISPLAY_ORDER").cast(pl.Int64, strict=False),
        ).join(
            ptype.select(
                pl.col("PTYPE_ROLE_ID").cast(pl.Int64, strict=False),
                pl.col("PTYPE_ROLE").cast(pl.Utf8).alias("PLACETYPE"),
            ),
            on="PTYPE_ROLE_ID",
            how="left",
        ),
        "tgn_placetypes.parquet",
    )

    # --- hierarchy ---------------------------------------------------------
    rels = read("SUBJECT_RELS.out", SUBJECT_RELS_COLS)
    write(
        rels.select(
            pl.col("SUBJECTA_ID").cast(pl.Int64, strict=False).alias("PARENT_ID"),
            pl.col("SUBJECTB_ID").cast(pl.Int64, strict=False).alias("CHILD_ID"),
            pl.col("PREFERRED").cast(pl.Utf8),
            pl.col("HIER_REL_TYPE").cast(pl.Utf8),
            pl.col("HISTORIC_FLAG").cast(pl.Utf8),
            pl.col("START_DATE").cast(pl.Int64, strict=False),
            pl.col("END_DATE").cast(pl.Int64, strict=False),
        ),
        "tgn_rels.parquet",
    )

    # --- coordinates -------------------------------------------------------
    # The bounding box is kept as well as the point. Box extent is the closest
    # thing to a size measure available without leaving TGN: a city has a real
    # box, a hamlet has a point or nothing, and merely *having* a box is itself
    # a prominence signal.
    coords = read("COORDINATES.out", COORDINATES_COLS)
    write(
        coords.select(
            pl.col("SUBJECT_ID").cast(pl.Int64, strict=False),
            pl.col("LAT_DECIMAL").cast(pl.Float64, strict=False).alias("LAT"),
            pl.col("LONG_DECIMAL").cast(pl.Float64, strict=False).alias("LON"),
            pl.col("LATLEAST_DECIMAL").cast(pl.Float64, strict=False).alias("LAT_MIN_BOX"),
            pl.col("LATMOST_DECIMAL").cast(pl.Float64, strict=False).alias("LAT_MAX_BOX"),
            pl.col("LONGLEAST_DECIMAL").cast(pl.Float64, strict=False).alias("LON_MIN_BOX"),
            pl.col("LONGMOST_DECIMAL").cast(pl.Float64, strict=False).alias("LON_MAX_BOX"),
        ),
        "tgn_coords.parquet",
    )

    # --- optional extras ---------------------------------------------------
    langs = read("LANGUAGE_RELS.out", LANGUAGE_RELS_COLS, required=False)
    if langs is not None:
        write(
            langs.select(
                pl.col("SUBJECT_ID").cast(pl.Int64, strict=False),
                pl.col("TERM_ID").cast(pl.Int64, strict=False),
                pl.col("LANGUAGE_CODE").cast(pl.Utf8),
                pl.col("PREFERRED").cast(pl.Utf8).alias("LANG_PREFERRED"),
                pl.col("TERM_TYPE").cast(pl.Utf8),
            ),
            "tgn_languages.parquet",
        )

    # --- prominence proxies ------------------------------------------------
    # Aggregated at load time: only the per-subject counts matter downstream,
    # and the raw tables are large.
    sources = read("SOURCE_RELS_SUBJECT.out", SOURCE_RELS_SUBJECT_COLS, required=False)
    if sources is not None:
        write(
            sources.select(pl.col("SUBJECT_ID").cast(pl.Int64, strict=False))
            .drop_nulls()
            .group_by("SUBJECT_ID")
            .agg(pl.len().alias("n_sources")),
            "tgn_sources.parquet",
        )

    notes = read("SCOPE_NOTES.out", SCOPE_NOTES_COLS, required=False)
    if notes is not None:
        write(
            notes.select(pl.col("SUBJECT_ID").cast(pl.Int64, strict=False))
            .drop_nulls()
            .group_by("SUBJECT_ID")
            .agg(pl.len().alias("n_notes")),
            "tgn_scope_notes.parquet",
        )

    merges = read("SUBJECT_MERGE.out", SUBJECT_MERGE_COLS, required=False)
    if merges is not None:
        write(
            merges.select(
                pl.col("MERGE_ID").cast(pl.Int64, strict=False),
                pl.col("DOMINANT_ID").cast(pl.Int64, strict=False),
                pl.col("NEW_ID").cast(pl.Int64, strict=False),
            ),
            "tgn_merges.parquet",
        )

    print("done" if not args.check else "check complete, nothing written")


if __name__ == "__main__":
    main()
