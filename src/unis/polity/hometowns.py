"""Assign each hometown its historical state/province and guild-abolition status.

Point-in-polygon of the geocoded hometown against the GHGIS map in force at
first enrollment (gis_year), plus membership in the concave hull of the 1848
German Confederation, plus guild abolition/reinstatement dates by polity.

students_with_tgn.parquet -> students_geocoded.parquet
"""

import polars as pl

from unis import paths
from unis.polity import ghgis


def main() -> None:
    con = ghgis.connect()

    con.execute(f"""
        CREATE TABLE students
        AS SELECT *,
        CASE
            WHEN first_year < 1826 THEN 1820
            WHEN first_year < 1830 THEN 1826
            WHEN first_year < 1834 THEN 1830
            WHEN first_year < 1848 THEN 1834
            WHEN first_year <= 1850 THEN 1848
        END as gis_year,
        ST_Transform(
            ST_Point(lat, lon),
            'EPSG:4326',
            'EPSG:32633'
        ) as geom
        FROM '{paths.STUDENTS_WITH_TGN}';
    """)

    ghgis.create_boundary_tables(con)

    con.execute(f"""
        CREATE TABLE confederation_borders AS
        SELECT
            ST_ConcaveHull(ST_Collect(LIST(geom)), 0.85, FALSE) AS geom,
            1 as in_germany
        FROM
        ST_Read('{paths.ghgis_layer("1848GERMANCONFED")}');
    """)

    con.execute("""
        CREATE TABLE students_matched AS
        SELECT
           students.* EXCLUDE (students.geom),
           states.STAAT_NAME, states.G1 as staat_id,
           provinces.PROV_NAME, provinces.F as prov_id,
           concat_ws('/', states.STAAT_NAME, provinces.PROV_NAME) AS standard_location_name,
           coalesce(states.G1, 0)*100 + coalesce(provinces.F, 0) as standard_polity_id,
           (districts.STAAT_NAME = 'Bayern' AND districts.REG_NAME = 'Rheinkreis') as in_rheinkreis,
           coalesce(confederation_borders.in_germany, 0)::Boolean as in_germany
        FROM
            students
        LEFT JOIN states
            ON students.gis_year = states.gis_year AND
            ST_Intersects(students.geom, states.geom)
        LEFT JOIN provinces
            ON students.gis_year = provinces.gis_year AND
            ST_Intersects(students.geom, provinces.geom)
        LEFT JOIN districts
            ON ST_Intersects(students.geom, districts.geom)
        LEFT JOIN confederation_borders
            ON ST_Intersects(students.geom, confederation_borders.geom)
        ORDER BY index;
    """)

    # The Bavarian Palatinate (Rheinkreis) is its own polity id.
    con.execute("""
        UPDATE students_matched
        SET
            standard_location_name = 'Bayern/Rheinland',
            standard_polity_id = 801
        WHERE in_rheinkreis;
    """)

    out_tab = con.execute("SELECT * FROM students_matched;").pl()

    guild_years = pl.read_csv(paths.GUILD_DATES).drop("standard_location_name").with_columns(
        pl.col.abolition_year.replace(0, None),
        pl.col.reinstatement_year.replace(0, None),
    )

    out_tab = out_tab.join(
        guild_years,
        on="standard_polity_id",
        how="left",
        coalesce=True,
    ).with_columns(
        guild_abolished=(pl.col.first_year >= pl.col.abolition_year)
        & (pl.col.first_year < pl.col.reinstatement_year).fill_null(True),
        guild_abolished_ever=pl.col.first_year >= pl.col.abolition_year,
        guild_reinstated=(pl.col.first_year >= pl.col.reinstatement_year).fill_null(False),
    ).with_columns(
        guild_abolished=pl.col.guild_abolished_ever & (~pl.col.guild_reinstated),
        guild_status=pl.when(pl.col.guild_reinstated)
        .then(pl.lit("reinstated"))
        .when(pl.col.guild_abolished_ever)
        .then(pl.lit("abolished"))
        .when(pl.col.in_germany)
        .then(pl.lit("present")),
    )

    print(out_tab)
    print(out_tab["guild_status"].value_counts())

    out_tab.write_parquet(paths.STUDENTS_GEOCODED)
    print(f"wrote {paths.STUDENTS_GEOCODED}")


if __name__ == "__main__":
    main()
