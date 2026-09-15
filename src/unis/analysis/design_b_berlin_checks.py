"""Is Berlin's flatter distance gradient about Berlin, or about where its students come from?

Re-estimates the static Design B specification (findings §4-5)

    flow ~ log_dist + same_city + berlin x log_dist + bonn x log_dist
           + berlin x C(era) + bonn x C(era) | origin_id + dest + era

on restricted samples (without East/West Prussia, the eastern provinces, or
origins outside every mapped polity; from settlement-level geocodes only) and
with added controls and interactions (Prussian border, Berlin's gradient by
origin region, small origins). The Berlin x log distance coefficient is the
object throughout: positive means Berlin's distance decay is flatter than the
old universities'.

od.parquet, students_final.parquet -> output/tables/design_b_berlin_checks.{csv,tex}
"""

from dataclasses import dataclass

import polars as pl
import pyfixest as pf

from unis import paths
from unis.gravity.build_od import destination_table, era_grid, origin_table, spell_sample
from unis.latex import Tabular, escape, integer, multicolumn, num, se, stars

OUT_CSV = paths.TABLES / 'design_b_berlin_checks.csv'

EAST_PRUSSIA = [208, 209, 231]  # West Prussia; East Prussia under two map vintages
EASTERN = EAST_PRUSSIA + [207, 210]  # plus Posen and Silesia
SMALL_ORIGIN = 10  # origins sending fewer students than this, all years and universities

f = pl.col
CONTROLS = 'same_city + is_berlin:C(era) + is_bonn:C(era)'
FE = ' | origin_id + dest + era'
BASE = f'flow ~ log_dist + bl + bn + {CONTROLS}'


@dataclass
class Check:
    key: str
    label: str
    formula: str = BASE
    grid: str = 'all'  # 'all': od.parquet; 'settlement': rebuilt from settlement geocodes
    sample: pl.Expr | None = None
    terms: tuple[str, ...] = ('bl',)


SAMPLE_CHECKS = [
    Check('published', 'All origins (published)'),
    Check('no_east_prussia', 'Without East and West Prussia', sample=~f.east_prussia),
    Check('no_eastern', 'Without the eastern provinces', sample=~f.eastern),
    Check('no_unmapped', 'Without unmapped origins', sample=~f.unmapped),
    Check('no_eastern_unmapped', 'Without eastern or unmapped origins', sample=f.rest),
    Check('settlement', 'Settlement-level geocodes only', grid='settlement'),
    Check('settlement_no_eastern_unmapped', 'Settlement-level, no eastern/unmapped',
          grid='settlement', sample=f.rest),
]
CONTROL_CHECKS = [
    Check('state_controls', 'Same-state and same-polity controls',
          formula=BASE + ' + same_state_f + same_polity_f'),
    Check('berlin_state', r'Adding Berlin $\times$ same state',
          formula=BASE + ' + same_state_f + same_polity_f + berlin_state',
          terms=('bl', 'berlin_state')),
    Check('by_region', 'By origin region',
          formula=(f'flow ~ log_dist + bl_eastern + bl_unmapped + bl_rest + berlin_eastern'
                   f' + berlin_unmapped + bn + {CONTROLS}'),
          terms=('bl_eastern', 'bl_unmapped', 'bl_rest')),
    Check('small_origins', rf'Small origins ($<${SMALL_ORIGIN} students)',
          formula=BASE + ' + berlin_small + bl_small', terms=('bl', 'bl_small')),
]
TERM_LABELS = {
    'bl': r'Berlin $\times$ log distance',
    'berlin_state': r'Berlin $\times$ same state',
    'bl_eastern': 'Eastern provinces',
    'bl_unmapped': 'Unmapped origins',
    'bl_rest': 'Other origins',
    'bl_small': r'$\times$ small origin',
}


def add_terms(od: pl.DataFrame) -> pl.DataFrame:
    """Origin groups and the Berlin/Bonn interaction columns used by the checks."""
    od = od.join(od.group_by('origin_id').agg(f.flow.sum().alias('sent')), on='origin_id')
    return od.with_columns(
        is_berlin=(f.dest == 'berlin').cast(pl.Float64),
        is_bonn=(f.dest == 'bonn').cast(pl.Float64),
        east_prussia=f.polity_o.is_in(EAST_PRUSSIA),
        eastern=f.polity_o.is_in(EASTERN),
        # no polygon of any mapped polity: mostly foreign or border-gap origins
        unmapped=(f.polity_o == 0) | f.polity_o.is_null(),
        small=f.sent < SMALL_ORIGIN,
    ).with_columns(
        rest=~f.eastern & ~f.unmapped,
        bl=f.is_berlin * f.log_dist,
        bn=f.is_bonn * f.log_dist,
        same_state_f=f.same_state.cast(pl.Float64),
        same_polity_f=f.same_polity.cast(pl.Float64),
    ).with_columns(
        berlin_state=f.is_berlin * f.same_state_f,
        berlin_eastern=f.is_berlin * f.eastern.cast(pl.Float64),
        berlin_unmapped=f.is_berlin * f.unmapped.cast(pl.Float64),
        bl_eastern=f.bl * f.eastern.cast(pl.Float64),
        bl_unmapped=f.bl * f.unmapped.cast(pl.Float64),
        bl_rest=f.bl * f.rest.cast(pl.Float64),
        berlin_small=f.is_berlin * f.small.cast(pl.Float64),
        bl_small=f.bl * f.small.cast(pl.Float64),
    )


def settlement_grid(students: pl.DataFrame) -> pl.DataFrame:
    """The era grid rebuilt from spells whose hometown geocoded to a settlement.

    Excludes students placed only at a territory, district or country point.
    These are far more common at Berlin (27% of its spells, against 3-5% at most
    universities), and the point can sit far from where the student lived.
    """
    settled = spell_sample(students).filter(f.geo_precision == 'settlement')
    return era_grid(settled, origin_table(settled), destination_table(settled))


def write_tex(results: dict, path) -> None:
    t = Tabular('lcccr')
    t.row('', 'Estimate', 'SE', escape('Berlin flow (%)'), 'Observations').midrule()

    t.row(multicolumn(5, 'l', r'\textit{A. Berlin $\times$ log distance, by sample}'))
    for check in SAMPLE_CHECKS:
        r = results[check.key]
        term = r['terms']['bl']
        t.row(check.label, num(term['est']) + stars(term['p']), se(term['se']),
              num(100 * r['berlin_share'], 1), integer(r['n']))

    t.space()
    t.row(multicolumn(5, 'l', r'\textit{B. Controls and interactions, all origins}'))
    for check in CONTROL_CHECKS:
        r = results[check.key]
        if check.terms == ('bl',):
            term = r['terms']['bl']
            t.row(check.label, num(term['est']) + stars(term['p']), se(term['se']), '',
                  integer(r['n']))
            continue
        t.row(check.label, '', '', '', integer(r['n']))
        for name in check.terms:
            term = r['terms'][name]
            t.row(rf'\quad {TERM_LABELS[name]}', num(term['est']) + stars(term['p']),
                  se(term['se']), '', '')
    t.write(path)


def main() -> None:
    grids = {
        'all': add_terms(pl.read_parquet(paths.OD)),
        'settlement': add_terms(settlement_grid(pl.read_parquet(paths.STUDENTS_FINAL))),
    }
    berlin_total = grids['all'].filter(f.dest == 'berlin')['flow'].sum()

    rows, results = [], {}
    for check in SAMPLE_CHECKS + CONTROL_CHECKS:
        data = grids[check.grid]
        if check.sample is not None:
            data = data.filter(check.sample)
        fit = pf.fepois(check.formula + FE, data=data.to_pandas(), vcov={'CRV1': 'origin_id'},
                        solver='np.linalg.lstsq')
        tidy = fit.tidy()
        share = data.filter(f.dest == 'berlin')['flow'].sum() / berlin_total
        results[check.key] = {'n': fit._N, 'berlin_share': share, 'terms': {}}
        for name in check.terms:
            r = tidy.loc[name]
            term = {'est': r['Estimate'], 'se': r['Std. Error'], 'p': r['Pr(>|t|)']}
            results[check.key]['terms'][name] = term
            rows.append({'check': check.key, 'term': name, **term,
                         'n': fit._N, 'berlin_share': share})
        print(f'===== {check.key} =====')
        print(tidy.loc[list(check.terms)].round(4))

    out = pl.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(OUT_CSV)
    print(f'wrote {OUT_CSV}')
    write_tex(results, OUT_CSV.with_suffix('.tex'))


if __name__ == '__main__':
    main()
