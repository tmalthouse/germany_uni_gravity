"""Assign each university a polity and compute same_polity / same_state.

Point-in-polygon of each university seat against the same GHGIS layers as the
hometown stage (14 school labels x 5 map years = 70 tests). Every
(school, gis_year) combination with zero or multiple polygon matches is logged.

students_clean.parquet -> students_final.parquet
"""

import polars as pl

from unis import paths
from unis.polity import ghgis

GIS_YEAR_POLARS = (
    pl.when(pl.col.first_year < 1826).then(pl.lit(1820))
    .when(pl.col.first_year < 1830).then(pl.lit(1826))
    .when(pl.col.first_year < 1834).then(pl.lit(1830))
    .when(pl.col.first_year < 1848).then(pl.lit(1834))
    .otherwise(pl.lit(1848))
)


def main() -> None:
    con = ghgis.connect()

    # University points: one row per school (TGN coords are the university city).
    # Projected to EPSG:32633 to match the GHGIS shapefiles.
    con.execute(f"""
        CREATE TABLE unis AS
        SELECT
            m.school,
            c.LAT,
            c.LON,
            ST_Transform(ST_Point(c.LAT, c.LON), 'EPSG:4326', 'EPSG:32633') AS geom
        FROM '{paths.UNI_TGN_MAPPING}' m
        JOIN '{paths.TGN / "tgn_coords.parquet"}' c USING (SUBJECT_ID);
    """)

    ghgis.create_boundary_tables(con)

    # Point-in-polygon per (school, gis_year), same join logic as the hometown
    # stage. Cross join gives every school every map year; the LEFT JOINs on
    # gis_year keep one output row per (school, gis_year) combination.
    years_sql = ", ".join(str(y) for y in ghgis.GIS_YEARS)
    con.execute(f"""
        CREATE TABLE uni_matched AS
        SELECT
            g.school,
            g.gis_year,
            g.LAT,
            g.LON,
            states.STAAT_NAME,
            states.G1 AS staat_id,
            provinces.PROV_NAME,
            provinces.F AS prov_id,
            concat_ws('/', states.STAAT_NAME, provinces.PROV_NAME) AS uni_standard_location_name,
            coalesce(states.G1, 0) * 100 + coalesce(provinces.F, 0) AS standard_polity_id
        FROM (
            SELECT u.school, y.gis_year, u.LAT, u.LON, u.geom
            FROM unis u
            CROSS JOIN (SELECT unnest([{years_sql}]) AS gis_year) y
        ) g
        LEFT JOIN states
            ON g.gis_year = states.gis_year AND
            ST_Intersects(g.geom, states.geom)
        LEFT JOIN provinces
            ON g.gis_year = provinces.gis_year AND
            ST_Intersects(g.geom, provinces.geom)
        ORDER BY school, gis_year;
    """)

    # Rheinkreis override (parity with the hometown stage; no university city
    # is in the Palatinate, but keep the rule for consistency).
    con.execute("""
        CREATE OR REPLACE TABLE uni_rheinkreis AS
        SELECT u.school,
               (d.STAAT_NAME = 'Bayern' AND d.REG_NAME = 'Rheinkreis') AS in_rheinkreis
        FROM unis u
        LEFT JOIN districts d ON ST_Intersects(u.geom, d.geom);
    """)
    con.execute("""
        UPDATE uni_matched
        SET
            uni_standard_location_name = 'Bayern/Rheinland',
            standard_polity_id = 801
        WHERE school IN (SELECT school FROM uni_rheinkreis WHERE in_rheinkreis);
    """)

    # Validation: every (school, gis_year) combination must resolve. Log anything odd.
    report = con.execute("""
        SELECT school, gis_year, LAT, LON, STAAT_NAME, PROV_NAME,
               uni_standard_location_name, standard_polity_id
        FROM uni_matched ORDER BY school, gis_year;
    """).pl()

    for school, grp in report.group_by("school"):
        if grp.height != len(ghgis.GIS_YEARS):
            print(f"WARNING {school}: {grp.height} rows, expected {len(ghgis.GIS_YEARS)}")
        nulls = grp.filter(pl.col.STAAT_NAME.is_null())
        if nulls.height:
            print(f"WARNING {school}: no state match for gis_years {nulls['gis_year'].to_list()}")
        nulls_p = grp.filter(pl.col.PROV_NAME.is_null())
        if nulls_p.height:
            print(f"NOTE {school}: no province match for gis_years {nulls_p['gis_year'].to_list()}")
        states_hit = grp.group_by("gis_year").agg(pl.col.STAAT_NAME.n_unique().alias("n"))
        multi = states_hit.filter(pl.col.n > 1)
        if multi.height:
            print(f"WARNING {school}: matched multiple states for gis_years "
                  f"{multi['gis_year'].to_list()}")

    print(report)
    uni_polity = report.select(
        "school",
        "gis_year",
        pl.col.LAT.alias("lat_uni"),
        pl.col.LON.alias("lon_uni"),
        "uni_standard_location_name",
        pl.col.standard_polity_id.alias("uni_standard_polity_id"),
    )

    # Join onto students, keyed on the same per-record gis_year as the hometown side.
    students = pl.read_parquet(paths.STUDENTS_CLEAN)
    # Sanity check: recompute the hometown rule and confirm it matches what the
    # hometown stage already stored.
    stored = students["gis_year"]
    students = students.with_columns(gis_year=GIS_YEAR_POLARS)
    assert (students["gis_year"] == stored).all(), "gis_year rule disagrees with stored column"

    students = students.join(
        uni_polity,
        left_on=["school_fixed", "gis_year"],
        right_on=["school", "gis_year"],
        how="left",
    )

    students = students.with_columns(
        same_polity=(pl.col.uni_standard_polity_id == pl.col.standard_polity_id),
        same_state=(pl.col.uni_standard_polity_id // 100 == pl.col.standard_polity_id // 100),
        # Broad Europe bounding box (lat 34-72, lon -25..45): true also for students
        # whose hometown geocoded outside the confederation hull but still in Europe
        # (e.g. East Prussia, Schleswig, Switzerland, Habsburg lands).
        in_germany_broad=pl.col.in_germany
        | (pl.col.lat.is_between(34, 72) & pl.col.lon.is_between(-25, 45)),
    )

    print(students["same_polity"].value_counts())
    print(students["same_state"].value_counts())
    print(students["in_germany_broad"].value_counts())
    print(students.group_by("school_fixed").agg(
        pl.col.same_polity.mean().alias("share_same_polity"),
        pl.col.same_state.mean().alias("share_same_state"),
        pl.len().alias("n"),
    ).sort("n", descending=True))

    paths.STUDENTS_FINAL.parent.mkdir(parents=True, exist_ok=True)
    students.write_parquet(paths.STUDENTS_FINAL)
    print(f"wrote {paths.STUDENTS_FINAL}: {students.shape}")


if __name__ == "__main__":
    main()
