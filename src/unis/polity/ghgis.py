"""GHGIS boundary layers in DuckDB, shared by the hometown and university stages.

Both stages use the same five map years, the same EPSG:32633 projection, the
same standard_polity_id formula (states.G1 * 100 + provinces.F) and the same
Rheinkreis override, so a hometown and a university in the same territory get
the same id.
"""

import duckdb

from unis import paths

GIS_YEARS = [1820, 1826, 1830, 1834, 1848]


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("INSTALL spatial;")
    con.execute("LOAD spatial;")
    return con


def create_boundary_tables(con: duckdb.DuckDBPyConnection) -> None:
    """Create `states` and `provinces` (all map years, tagged gis_year) and `districts` (1820)."""
    for table, layer in (("states", "CORE"), ("provinces", "PROVINCES")):
        subqueries = [
            f"SELECT *, {y} AS gis_year FROM ST_Read('{paths.ghgis_layer(f'{y}{layer}')}')"
            for y in GIS_YEARS
        ]
        con.execute(f"CREATE TABLE {table} AS {' UNION ALL '.join(subqueries)}")
    con.execute(
        f"CREATE TABLE districts AS SELECT * FROM ST_Read('{paths.ghgis_layer('1820DISTRICTS')}');"
    )
