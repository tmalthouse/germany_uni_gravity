"""Clean every school's register and stack them into all_students_unlinked.parquet."""

import polars as pl

from unis import paths
from unis.registers import (
    berlin,
    bonn,
    erlangen,
    freiburg,
    giessen,
    goettingen,
    heidelberg,
    jena,
    kiel,
    marburg,
    muenchen,
    tuebingen,
    wuerzburg,
)

SCHOOLS = {
    "berlin": berlin.build,
    "bonn": bonn.build,
    "erlangen": erlangen.build,
    "freiburg": freiburg.build,
    "giessen": giessen.build,
    "goettingen": goettingen.build,
    "heidelberg": heidelberg.build,
    "jena": jena.build,
    "kiel": kiel.build,
    "marburg": marburg.build,
    "muenchen": muenchen.build,
    "tuebingen": tuebingen.build,
    "wuerzburg": wuerzburg.build,
}


def main() -> None:
    paths.REGISTERS.mkdir(parents=True, exist_ok=True)
    for school, build in SCHOOLS.items():
        print(f"Cleaning {school}")
        lf = build()
        print(f"Columns: {list(lf.collect_schema().keys())}")
        lf.sink_parquet(paths.REGISTERS / f"{school}.parquet")

    all_students = (
        pl.concat(
            [
                pl.scan_parquet(paths.REGISTERS / f"{school}.parquet").with_columns(
                    school=pl.lit(school)
                )
                for school in sorted(SCHOOLS)
            ],
            how="diagonal_relaxed",
        )
        .filter(pl.col.first_year.is_in(range(1800, 1851)))
        .with_columns(pl.col.last_name.fill_null(""), pl.col.first_names.fill_null(""))
    )
    all_students.sink_parquet(paths.STUDENTS_UNLINKED)
    print(f"wrote {paths.STUDENTS_UNLINKED}")


if __name__ == "__main__":
    main()
