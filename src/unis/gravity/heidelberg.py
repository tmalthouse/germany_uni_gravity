"""Heidelberg religion sample, shared by the two religion analyses.

`religion` is only recorded for Heidelberg enrollees; all other universities
have zero religion observations. Both analyses read students_final directly
and do not deduplicate spells: at Heidelberg one row is one spell.
"""

import polars as pl

from unis import paths

RELIGION_COLORS = {'catholic': 'tab:orange', 'protestant': 'tab:blue', 'jewish': 'tab:green'}


def religion_sample() -> pl.DataFrame:
    """Heidelberg enrollees in the broad-Europe sample with a classifiable religion.

    Catholic = code starts with k; Jewish = j; Protestant = e, l, p, r, g, v, c
    or h (ev., lu., pr., re., gr., e.p., ...). Unparseable codes are dropped.
    """
    df = pl.read_parquet(paths.STUDENTS_FINAL).filter(
        pl.col.in_germany_broad & pl.col.religion.is_not_null()
    ).filter(pl.col.school_fixed == 'heidelberg')
    return df.with_columns(
        first=pl.col.religion.str.strip_chars().str.to_lowercase().str.slice(0, 1)
    ).with_columns(
        rel=pl.when(pl.col.first == 'k').then(pl.lit('catholic'))
              .when(pl.col.first == 'j').then(pl.lit('jewish'))
              .when(pl.col.first.is_in(['e', 'l', 'p', 'r', 'g', 'v', 'c', 'h']))
              .then(pl.lit('protestant'))
              .otherwise(None)
    ).drop('first').filter(pl.col.rel.is_not_null())


def seat(df: pl.DataFrame) -> tuple[float, float]:
    """Heidelberg's coordinates, taken from the data (lat_uni/lon_uni).

    Never a literal: a previously hardcoded pair was Tübingen's, 101 km off.
    """
    return df['lat_uni'][0], df['lon_uni'][0]
