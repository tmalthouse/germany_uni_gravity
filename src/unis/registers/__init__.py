"""Per-school register cleaning.

Each school module exposes ``build() -> pl.LazyFrame`` returning one row per
register entry in a common column set; ``combine`` writes each school out and
stacks them into ``all_students_unlinked.parquet``.
"""
