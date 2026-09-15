"""Build heidelberg_date_fixer.parquet from the register's running heads.

For each Heidelberg register volume, the first scan page of each enrollment
year (`year_helper`), read from the running-head dates in
data/manual/heidelberg_heads/heidelberg{1704,1807,1846}.json. The Heidelberg
cleaner joins this on (volume_id, source_page), forward-fills it within each
volume, and uses it to correct entry years.

A page's head year is the first year on its running head ("29. November 1842 -
1. Mai 1843" -> 1842). A page starts year Y when its head year is Y and the
previous head's year is Y - 1. Year changes fall on a spanning spread: e.g. in
heidelberg_1846, pp. 42-43 are headed "12. November 1846 - 13. Februar 1847",
so 1847 starts at p44 but pp. 42-43 also hold early-1847 entries.

Fixes (2026-09-14):
- Each volume's first head used to be dropped (its `shift(1)` is null), so no
  volume had an entry for its opening year: heidelberg_1807 lacked 1807 and
  heidelberg_1846 lacked 1846. The first head is now kept.
- A single misread head broke the year sequence: heidelberg_1807 p712 is headed
  "1. Mai - 14. Mai 1841)" for May 1843, so p714's 1843 failed the previous + 1
  test and 1843 was never recorded. A head year outside [previous head, next
  head] is now replaced by the next head's year, which repairs isolated
  misreads in either direction and leaves monotone runs unchanged.
- MANUAL_START_PAGES holds start pages confirmed against the printed register;
  they replace the detected page for that (volume, year).
"""

import json
from pathlib import Path

import polars as pl

from unis import paths

VOLUMES = {  # volume_id: (page-head json, first scan page, last scan page)
    "heidelberg_1704": ("heidelberg1704.json", 380, 425),
    "heidelberg_1807": ("heidelberg1807.json", 8, 789),
    "heidelberg_1846": ("heidelberg1846.json", 10, 112),
}

# (volume_id, year) -> first scan page, confirmed against the printed register.
MANUAL_START_PAGES = {
    ("heidelberg_1846", 1846): 10,  # pp. 10-43 cover 1846
}


def start_pages(volume_id: str, json_file: Path, first_page: int, last_page: int) -> pl.DataFrame:
    blobs = json.loads(json_file.read_text())
    heads = (
        pl.DataFrame({"source_page": list(blobs.keys()), "blob": list(blobs.values())})
        .with_columns(
            pl.col.source_page.cast(pl.Int64),
            year=pl.col.blob.str.extract(r"(\d{4})").cast(pl.Int64),
        )
        .filter(pl.col.source_page.is_between(first_page, last_page))
        .filter(pl.col.year.is_not_null())
        .sort("source_page")
    )
    # Repair isolated misreads: a head year outside [previous, next] takes the
    # next head's year (only where previous <= next, i.e. the neighbours agree
    # on direction).
    prev = pl.col.year.shift(1).fill_null(pl.col.year)
    nxt = pl.col.year.shift(-1).fill_null(pl.col.year)
    outside = (prev <= nxt) & ((pl.col.year < prev) | (pl.col.year > nxt))
    heads = heads.with_columns(year=pl.when(outside).then(nxt).otherwise(pl.col.year))
    return heads.filter(
        (pl.col.year == pl.col.year.shift(1) + 1)
        | (pl.int_range(pl.len()) == 0)  # keep the volume's first head
    ).select(
        pl.lit(volume_id).alias("volume_id"),
        "source_page",
        pl.col.year.alias("year_helper"),
    )


def build_table(heads_dir: Path = paths.HEIDELBERG_HEADS) -> pl.DataFrame:
    full_tab = pl.concat(
        [start_pages(v, heads_dir / f, first, last) for v, (f, first, last) in VOLUMES.items()]
    )
    for (volume_id, year), page in MANUAL_START_PAGES.items():
        full_tab = pl.concat([
            full_tab.filter(~((pl.col.volume_id == volume_id) & (pl.col.year_helper == year))),
            pl.DataFrame(
                {"volume_id": [volume_id], "source_page": [page], "year_helper": [year]},
                schema=full_tab.schema,
            ),
        ])
    return full_tab.sort("volume_id", "source_page")


def main() -> None:
    table = build_table()
    print(table)
    paths.HEIDELBERG_DATE_FIXER.parent.mkdir(parents=True, exist_ok=True)
    table.write_parquet(paths.HEIDELBERG_DATE_FIXER)
    print(f"wrote {paths.HEIDELBERG_DATE_FIXER}")


if __name__ == "__main__":
    main()
