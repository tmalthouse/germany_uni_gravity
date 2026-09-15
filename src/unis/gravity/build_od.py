"""Build the origin-destination datasets for the student-university gravity model.

Origin: hometown place, identified by its TGN location_id (geocoded lat/lon).
Destination: the 13 universities (school_fixed), with uni lat/lon from tgn_coords.
Note school_fixed carries 14 labels: Munich appears as `muenchen_old` (Landshut
seat, enrollments before 1826) and `muenchen` (from the 1826 move onward).
Flows: number of student-spells from origin o enrolled at university u.
Sample: students whose hometown geocode falls in broad Europe (in_germany_broad).

Output: od.parquet (era grid: one row per origin x dest x era, era = decade of
first enrollment) and od_year.parquet (year grid: one row per origin x dest x
first_year, true annual flows -- use this for event studies).
Columns: origin_id, dest, flow, lat/lon, dist_km (great-circle), same_state,
same_polity, confession vars. Both grids are zero-padded: every (origin, dest,
period) pair appears, including flows = 0.
"""

import polars as pl

from unis import paths
from unis.geo import haversine_km

# Broad Europe bounding box (matches in_germany_broad in polity.universities)
EU_LAT = (34.0, 72.0)
EU_LON = (-25.0, 45.0)

# Dominant confession by origin polity (standard_location_name), classified by
# ruling house / territorial church affiliation, c. 1800-1850.
# protestant: Lutheran/Reformed or unionist ruling house with Protestant majority
# catholic: Catholic ruling house with Catholic majority or Catholic-ruled mixed
# mixed: confessionally divided territory (Rheinprovinz, Baden, Hessen-Darmstadt)
CONFESSION_MAP = {
    # Protestant north/east/center
    'Hannover': 'protestant', 'Mecklenburg-Schwerin': 'protestant',
    'Mecklenburg-Strelitz': 'protestant', 'Sachsen': 'protestant',
    'Sachsen-Weimar-Eisenach': 'protestant', 'Sachsen-Coburg und Gotha': 'protestant',
    'Sachsen-Meiningen': 'protestant', 'Sachsen-Altenburg': 'protestant',
    'Sachsen-Gotha-Altenburg': 'protestant', 'Sachsen-Hildburghausen': 'protestant',
    'Sachsen-Coburg-Saalfeld': 'protestant',
    'Braunschweig': 'protestant', 'Oldenburg': 'protestant',
    'Anhalt-Bernburg': 'protestant', 'Anhalt-Dessau': 'protestant',
    'Anhalt-Köthen': 'protestant', 'Schwarzburg-Sondershausen': 'protestant',
    'Schwarzburg-Rudolstadt': 'protestant', 'Reuß-Gera': 'protestant',
    'Reuß-Schleiz': 'protestant', 'Reuß-Ebersdorf': 'protestant',
    'Reuß-Lobenstein und Ebersdorf': 'protestant',
    'Reuß ältere Linie (Reuß-Greiz)': 'protestant',
    'Lippe-Detmold': 'protestant', 'Schaumburg-Lippe': 'protestant',
    'Waldeck': 'protestant', 'Hessen-Kassel': 'protestant',
    'Hessen-Homburg': 'protestant', 'Frankfurt': 'protestant',
    'Hamburg': 'protestant', 'Bremen': 'protestant', 'Lübeck': 'protestant',
    'Schleswig': 'protestant', 'Holstein': 'protestant', 'Lauenburg': 'protestant',
    # Prussia: Protestant-ruled union state; eastern provinces firmly Protestant,
    # western provinces (Rheinprovinz, Jülich-Kleve-Berg, Niederrhein) majority
    # Catholic -> mixed; Brandenburg/Westfalen also mixed (Westfalen strongly
    # Catholic in the south). Posen mixed (Catholic Poles). Preußen (East/Hinter-
    # Pommern province) protestant; Westpreußen mixed.
    'Preußen/Brandenburg': 'protestant', 'Preußen/Preußen': 'protestant',
    'Preußen/Pommern': 'protestant', 'Preußen/Sachsen': 'protestant',
    'Preußen/Ostpreußen': 'protestant',
    'Preußen/Schlesien (Preußen)': 'mixed',
    'Preußen/Rheinprovinz': 'mixed', 'Preußen/Jülich-Kleve-Berg': 'mixed',
    'Preußen/Niederrhein': 'mixed', 'Preußen/Westfalen': 'mixed',
    'Preußen/Posen': 'mixed', 'Preußen/Westpreußen': 'mixed',
    # Catholic south and west
    'Bayern': 'catholic', 'Bayern/Rheinland': 'catholic',  # Rheinkreis: Catholic-ruled
    # Württemberg: Lutheran state church and overwhelmingly Lutheran population.
    # Only the ruling house was Catholic (from 1733), so the ruling-house rule
    # used elsewhere in this map misclassifies it. Coding it 'catholic' put
    # Württemberg->Tübingen (7,898 flows, ~21% of the whole ss_dc cell) in
    # "same state, different confession". Recoded 2026-09-14.
    'Württemberg': 'protestant',
    'Baden': 'mixed',  # Catholic-ruled, confessional mix
    'Hessen-Darmstadt': 'mixed',
    'Hessen-Darmstadt/Starkenburg': 'mixed',
    'Hessen-Darmstadt/Oberhessen (H.-Darmstadt)': 'protestant',
    'Hessen-Darmstadt/Rheinhessen': 'mixed',
    'Nassau': 'mixed',  # Catholic-ruled, mixed population
    'Hohenzollern-Sigmaringen': 'catholic', 'Hohenzollern-Hechingen': 'catholic',
}

# University confession from the pipeline's own flags (catholic_uni/protestant_uni);
# berlin/bonn are flagged neither (newer, state-controlled) -> protestant by
# Prussian establishment.
UNI_CONFESSION = {
    'freiburg': 'catholic', 'wuerzburg': 'catholic', 'muenchen': 'catholic',
    'muenchen_old': 'catholic',
    'berlin': 'protestant', 'bonn': 'protestant', 'erlangen': 'protestant',
    'giessen': 'protestant', 'goettingen': 'protestant', 'heidelberg': 'protestant',
    'jena': 'protestant', 'kiel': 'protestant', 'marburg': 'protestant',
    'tuebingen': 'protestant',
}


def attach_covariates(od: pl.DataFrame, orig: pl.DataFrame) -> pl.DataFrame:
    """Add distance, border, and confession covariates to a grid with
    columns: lat, lon, lat_uni, lon_uni, polity_o, polity_d, dest,
    location_id (origin key), plus optional era/first_year."""
    od = od.with_columns(
        dist_km=haversine_km(pl.col.lat, pl.col.lon, pl.col.lat_uni, pl.col.lon_uni),
        same_state=(pl.col.polity_o // 100 == pl.col.polity_d // 100),
        same_polity=(pl.col.polity_o == pl.col.polity_d),
    ).with_columns(
        # Zero-distance cells (hometown = university city): keep them (2.5% of
        # flows, and dropping would bias destination FEs). Same-city gets its
        # own dummy; log_dist is floored at log(0.5 km) for identification.
        same_city=(pl.col.dist_km < 1),
    ).with_columns(
        log_dist=pl.when(pl.col.dist_km >= 1).then(pl.col.dist_km.log())
                  .otherwise(pl.lit(0.5).log()),
    )

    od = od.rename({'location_id': 'origin_id'})

    # Confession: origin polity via CONFESSION_MAP on the polity name; destination
    # via UNI_CONFESSION on the school label. same_confession compares the two.
    name_map = orig.select(pl.col.location_id.alias('origin_id'), 'location_name_o')
    od = od.join(name_map, on='origin_id', how='left')
    od = od.with_columns(
        # KNOWN ISSUE (docs/findings.md, Data): Series.replace leaves unmatched
        # values unchanged, so origins with an empty polity name keep '' rather
        # than becoming 'unknown', and the warning below can never fire.
        confession_o=pl.col.location_name_o.replace(CONFESSION_MAP).fill_null('unknown'),
    ).drop('location_name_o')
    od = od.with_columns(
        confession_d=pl.col.dest.replace(UNI_CONFESSION),
    )
    od = od.with_columns(
        same_confession=(pl.col.confession_o == pl.col.confession_d)
        & pl.col.confession_o.is_in(['protestant', 'catholic']),
    )
    # Four-way border x confession dummies for the confession_split spec
    # (formulaic's parser rejects '~', so precompute them here).
    od = od.with_columns(
        ss_sc=(pl.col.same_state & pl.col.same_confession),   # same state, same confession
        ss_dc=(pl.col.same_state & ~pl.col.same_confession),  # same state, diff confession
        ds_sc=(~pl.col.same_state & pl.col.same_confession),  # diff state, same confession
    )
    unmapped = od.filter(pl.col.confession_o == 'unknown')['origin_id'].n_unique()
    if unmapped:
        print(f'WARNING: {unmapped} origins with unmapped confession')
    return od


def spell_sample(students: pl.DataFrame) -> pl.DataFrame:
    """One row per enrollment spell, in the broad-Europe sample.

    Some schools (e.g. Tübingen, Jena, Bonn, Berlin) record every semester of a
    spell, others only the first. Keep the first row per spell_id so flows are
    counted per enrollment spell, not per recorded semester (otherwise
    record-dense schools are overrepresented ~2x).
    """
    return students.filter(pl.col.in_germany_broad).sort('first_year').unique(
        subset='spell_id', keep='first', maintain_order=True
    )


def origin_table(df: pl.DataFrame) -> pl.DataFrame:
    """Origins: hometown places, one row per TGN id, with coordinates and polity."""
    return (df
            .group_by('location_id')
            .agg(
                pl.col.hometown.first(),
                pl.col.lat.first(),
                pl.col.lon.first(),
                pl.col.standard_polity_id.first().alias('polity_o'),
                pl.col.standard_location_name.first().alias('location_name_o'),
                # modal gis_year (the year whose map the hometown was assigned under)
                pl.col.gis_year.mode().first().alias('gis_year_o'),
                pl.len().alias('n_students_o'),
            ))


def destination_table(df: pl.DataFrame) -> pl.DataFrame:
    """Destinations: the 14 school_fixed labels (13 institutions; Munich splits into
    its Landshut and Munich seats at the 1826 move) with coordinates and polity."""
    return (df
            .group_by('school_fixed')
            .agg(
                pl.col.lat_uni.first(),
                pl.col.lon_uni.first(),
                pl.col.uni_standard_polity_id.first().alias('polity_d'),
                pl.col.uni_standard_location_name.first().alias('location_name_d'),
            )
            .rename({'school_fixed': 'dest'})
            .sort('dest'))


def era_grid(df: pl.DataFrame, orig: pl.DataFrame, dest: pl.DataFrame) -> pl.DataFrame:
    """Zero-padded origin x destination x era grid of spell flows, with covariates.

    Era = decade of first enrollment (1 = 1800s ... 5 = 1840s), as defined in
    unis.finalize.
    """
    eras = df.select('era').unique().sort('era')
    grid = (orig.select('location_id', 'lat', 'lon', 'polity_o')
            .join(dest.select('dest', 'lat_uni', 'lon_uni', 'polity_d'), how='cross')
            .join(eras, how='cross'))

    flows = (df
             .group_by('location_id', 'school_fixed', 'era')
             .agg(pl.len().alias('flow'))
             .rename({'school_fixed': 'dest'}))

    # Enrollment year of an (origin, dest, era) cell: the modal first_year within
    # the cell. Never use it for annual analysis -- that is what od_year is for.
    years = (df.group_by('location_id', 'school_fixed', 'era')
             .agg(pl.col.first_year.mode().first().alias('first_year'))
             .rename({'school_fixed': 'dest'}))

    od = grid.join(flows, on=['location_id', 'dest', 'era'], how='left')
    od = od.join(years, on=['location_id', 'dest', 'era'], how='left')
    od = od.with_columns(pl.col.flow.fill_null(0))
    return attach_covariates(od, orig)


def main() -> None:
    df = spell_sample(pl.read_parquet(paths.STUDENTS_FINAL))
    orig = origin_table(df)
    print(f'origins: {orig.height} unique hometown places')
    dest = destination_table(df)
    print(f'destinations: {dest.height} universities')

    od = era_grid(df, orig, dest)
    paths.OD.parent.mkdir(parents=True, exist_ok=True)
    od.write_parquet(paths.OD)
    print(f'wrote {paths.OD}: {od.shape}')
    print('by era:', od.group_by('era').agg(
        pl.col.flow.sum().alias('flow'), pl.len().alias('cells')).sort('era'))
    print('flow distribution:', od['flow'].describe())
    print('total students covered:', od['flow'].sum())
    print('nonzero cells:', (od['flow'] > 0).sum())

    # ---- Year-level companion grid ----
    # One row per (origin, dest, first_year) with true annual flows.
    years_all = sorted(
        y for y in df['first_year'].unique().to_list() if y is not None
    )
    grid_y = (orig.select('location_id', 'lat', 'lon', 'polity_o')
              .join(dest.select('dest', 'lat_uni', 'lon_uni', 'polity_d'), how='cross')
              .join(pl.DataFrame({'first_year': years_all}), how='cross'))
    flows_y = (df.drop_nulls('first_year')
               .group_by('location_id', 'school_fixed', 'first_year')
               .agg(pl.len().alias('flow'))
               .rename({'school_fixed': 'dest'}))
    od_y = grid_y.join(flows_y, on=['location_id', 'dest', 'first_year'], how='left')
    od_y = od_y.with_columns(pl.col.flow.fill_null(0))
    od_y = attach_covariates(od_y, orig)
    od_y.write_parquet(paths.OD_YEAR)
    print(f'wrote {paths.OD_YEAR}: {od_y.shape}')


if __name__ == '__main__':
    main()
