"""Where Berlin's students come from, and how precisely they are geocoded (findings §4-5).

Descriptive companions to design_b_berlin_checks:

1. Geocode precision by university: the share of enrollment spells placed only
   at a territory, district or country point, placed via the register's region
   field, from 300 km or more, and both coarse and that far.
2. Berlin's enrollments by origin province.
3. Berlin's East and West Prussian origins, with how each was geocoded. Most of
   that flow is two province-level points for students whose recorded hometown
   is "Berlin"; the rest comes from towns.

All counts are enrollment spells in the gravity sample (unis.gravity.build_od).

students_final.parquet, od.parquet
    -> output/tables/design_b_berlin_geocodes.{csv,tex},
       output/tables/design_b_berlin_provinces.{csv,tex},
       output/tables/design_b_berlin_east_prussia.{csv,tex}
"""

import polars as pl

from unis import paths
from unis.analysis.design_b_berlin_checks import EAST_PRUSSIA
from unis.geo import haversine_km
from unis.gravity.build_od import spell_sample
from unis.gravity.constants import UNIVERSITY_NAMES
from unis.latex import Tabular, escape, integer, multicolumn, num

f = pl.col
COARSE = ['territory', 'district', 'country', 'unknown']
FAR_KM = 300
TOP_PROVINCES = 12
MIN_EAST_PRUSSIA_ENROLLMENTS = 5

# Prussian provinces by polity id; East Prussia appears under two map vintages,
# and the Rhine Province under its earlier and later divisions.
PRUSSIAN_PROVINCES = {
    202: 'Brandenburg', 203: 'Rhine Province', 204: 'Rhine Province', 205: 'Pomerania',
    206: 'Westphalia', 207: 'Posen', 208: 'West Prussia', 209: 'East Prussia',
    210: 'Silesia', 211: 'Saxony (province)', 230: 'Rhine Province', 231: 'East Prussia',
}
STATE_NAMES = {
    'Hannover': 'Hanover', 'Sachsen': 'Saxony', 'Bayern': 'Bavaria',
    'Hessen-Kassel': 'Hesse-Kassel', 'Hessen-Darmstadt': 'Hesse-Darmstadt',
    'Braunschweig': 'Brunswick', 'Holstein': 'Holstein', 'Schleswig': 'Schleswig',
}


def province_label(polity_id: int | None, name: str | None) -> str:
    """English label for an origin polity (GHGIS standard_polity_id and name)."""
    if polity_id in PRUSSIAN_PROVINCES:
        return f'Prussia: {PRUSSIAN_PROVINCES[polity_id]}'
    if not name:
        return 'No mapped polity'
    state = name.split('/')[0]
    return STATE_NAMES.get(state, state)


def geocode_precision(spells: pl.DataFrame) -> pl.DataFrame:
    coarse = f.geo_precision.is_in(COARSE)
    far = haversine_km(f.lat, f.lon, f.lat_uni, f.lon_uni) >= FAR_KM
    return (spells
            .group_by('school_fixed')
            .agg(pl.len().alias('spells'),
                 coarse.mean().alias('coarse'),
                 (f.matched == 1).fill_null(False).mean().alias('region_field'),
                 far.mean().alias('far'),
                 (coarse & far).mean().alias('coarse_and_far'))
            .with_columns(university=f.school_fixed.replace_strict(UNIVERSITY_NAMES))
            .sort('university'))


def berlin_origins(od: pl.DataFrame, spells: pl.DataFrame) -> pl.DataFrame:
    """One row per origin sending students to Berlin, with province and geocode details."""
    places = (spells.group_by(f.location_id.alias('origin_id'))
              .agg(f.location_name_tgn.first().alias('place'),
                   f.geo_precision.first().alias('level'),
                   f.hometown.mode().first().alias('hometown'),
                   f.standard_location_name.first().alias('polity_name')))
    return (od.filter((f.dest == 'berlin') & (f.flow > 0))
            .group_by('origin_id', 'polity_o')
            .agg(f.flow.sum().alias('enrollments'), f.dist_km.first())
            .join(places, on='origin_id', how='left')
            .with_columns(province=pl.struct('polity_o', 'polity_name').map_elements(
                lambda r: province_label(r['polity_o'], r['polity_name']), return_dtype=pl.String)))


def province_table(origins: pl.DataFrame) -> pl.DataFrame:
    by_province = (origins.group_by('province')
                   .agg(f.enrollments.sum(), pl.len().alias('origins'),
                        ((f.enrollments * f.dist_km).sum() / f.enrollments.sum()).alias('mean_km'))
                   .sort('enrollments', 'province', descending=[True, False]))
    top = by_province.head(TOP_PROVINCES)
    rest = by_province.slice(TOP_PROVINCES)
    other = rest.select(
        pl.lit('Other').alias('province'), f.enrollments.sum(), f.origins.sum(),
        ((f.enrollments * f.mean_km).sum() / f.enrollments.sum()).alias('mean_km'))
    table = pl.concat([top, other], how='vertical_relaxed')
    total = table['enrollments'].sum()
    return table.with_columns(share=f.enrollments / total)


def write_geocodes_tex(table: pl.DataFrame, path) -> None:
    t = Tabular('lrcccc')
    t.row('', '', multicolumn(4, 'c', escape('Share of spells (%)'))).cmidrule(3, 6)
    t.row('', '', 'Coarse', 'Via region', '', 'Coarse and')
    t.row('University', 'Spells', 'geocode', 'field', f'{FAR_KM}+ km', f'{FAR_KM}+ km').midrule()
    for r in table.iter_rows(named=True):
        t.row(r['university'], integer(r['spells']), num(100 * r['coarse'], 1),
              num(100 * r['region_field'], 1), num(100 * r['far'], 1),
              num(100 * r['coarse_and_far'], 1))
    t.write(path)


def write_provinces_tex(table: pl.DataFrame, path) -> None:
    t = Tabular('lrrrr')
    t.row('Origin', 'Enrollments', escape('Share (%)'), 'Origins', 'Mean distance (km)').midrule()
    for r in table.iter_rows(named=True):
        if r['province'] == 'Other':
            t.midrule()
        t.row(escape(r['province']), integer(r['enrollments']), num(100 * r['share'], 1),
              integer(r['origins']), num(r['mean_km'], 0))
    t.midrule()
    t.row('Total', integer(table['enrollments'].sum()), num(100.0, 1),
          integer(table['origins'].sum()), '')
    t.write(path)


def write_east_prussia_tex(table: pl.DataFrame, path) -> None:
    t = Tabular('llllrr')
    t.row('', 'Geocode', 'Recorded', 'Prussian', '', 'Distance')
    t.row('Geocoded place', 'level', 'hometown', 'province', 'Enrollments', '(km)').midrule()
    for r in table.iter_rows(named=True):
        if r['place'].startswith('Other'):
            t.midrule()
        province = (r['province'] or '').removeprefix('Prussia: ')
        t.row(escape(r['place']), escape(r['level'] or ''), escape(r['hometown'] or ''),
              escape(province), integer(r['enrollments']), num(r['dist_km'], 0))
    t.midrule()
    t.row('Total', '', '', '', integer(table['enrollments'].sum()), '')
    t.write(path)


def main() -> None:
    spells = spell_sample(pl.read_parquet(paths.STUDENTS_FINAL))
    od = pl.read_parquet(paths.OD)
    paths.TABLES.mkdir(parents=True, exist_ok=True)

    precision = geocode_precision(spells)
    out = paths.TABLES / 'design_b_berlin_geocodes.csv'
    precision.write_csv(out)
    print(f'wrote {out}')
    print(precision)
    write_geocodes_tex(precision, out.with_suffix('.tex'))

    origins = berlin_origins(od, spells)
    provinces = province_table(origins)
    out = paths.TABLES / 'design_b_berlin_provinces.csv'
    provinces.write_csv(out)
    print(f'wrote {out}')
    print(provinces)
    write_provinces_tex(provinces, out.with_suffix('.tex'))

    east = (origins.filter(f.polity_o.is_in(EAST_PRUSSIA))
            .select('place', 'level', 'hometown', 'province', 'enrollments', 'dist_km')
            .sort('enrollments', 'place', descending=[True, False]))
    shown = east.filter(f.enrollments >= MIN_EAST_PRUSSIA_ENROLLMENTS)
    rest = east.filter(f.enrollments < MIN_EAST_PRUSSIA_ENROLLMENTS)
    if rest.height:
        shown = pl.concat([shown, pl.DataFrame({
            'place': [f'Other ({rest.height} origins)'], 'level': [None], 'hometown': [None],
            'province': [None], 'enrollments': [rest['enrollments'].sum()],
            'dist_km': [None],
        }, schema=shown.schema)])
    out = paths.TABLES / 'design_b_berlin_east_prussia.csv'
    shown.write_csv(out)
    print(f'wrote {out}')
    print(shown)
    write_east_prussia_tex(shown, out.with_suffix('.tex'))


if __name__ == '__main__':
    main()
