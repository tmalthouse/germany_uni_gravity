"""Event study around 1819 (Carlsbad Decrees): yearly same_state interactions (findings §3).

    flow ~ log_dist + same_city + same_state + same_state:C(first_year)
           | origin_id + dest + first_year^dest

Uses the year grid (true annual flows), not the era grid, whose first_year is
only the modal year per decade cell. The path is baseline + interaction per
year, with conservative CIs that ignore their covariance; read joint
significance from the Wald test instead.

od_year.parquet -> output/tables/event_study_1819.csv, output/tables/event_study_1819.tex
"""

import math

import numpy as np
import polars as pl
import pyfixest as pf

from unis import paths
from unis.latex import Tabular, integer, multicolumn, num, pvalue, se

OUT = paths.TABLES / 'event_study_1819.csv'


def write_tex(rows: list[tuple[int, float, float]], n_obs: int, wald_p: float, path) -> None:
    """Border effect by year in two side-by-side blocks.

    `rows` holds (year, estimate, se); the reference year's estimate is the
    same_state coefficient itself, every other year's is baseline + interaction.
    """
    half = math.ceil(len(rows) / 2)
    blocks = (rows[:half], rows[half:])
    t = Tabular(r'rcc@{\hspace{2em}}rcc')
    t.row('Year', 'Estimate', 'SE', 'Year', 'Estimate', 'SE').midrule()
    for i in range(half):
        cells = []
        for block in blocks:
            if i < len(block):
                year, est, err = block[i]
                cells += [str(int(year)), num(est), se(err)]
            else:
                cells += ['', '', '']
        t.row(*cells)
    t.midrule()
    t.row(multicolumn(3, 'l', 'Observations'), multicolumn(3, 'r', integer(n_obs)))
    t.row(multicolumn(3, 'l', r'Joint test, year interactions $= 0$: $p$'),
          multicolumn(3, 'r', pvalue(wald_p)))
    t.write(path)


def main() -> None:
    od = pl.read_parquet(paths.OD_YEAR).to_pandas()
    fml = ('flow ~ log_dist + same_city + same_state '
           '+ same_state:C(first_year) | origin_id + dest + first_year^dest')
    fit = pf.fepois(fml, data=od, vcov={'CRV1': 'origin_id'})
    print('\n===== event_study_1819 =====')
    print(fit.tidy().round(4).to_string())

    t = fit.tidy().reset_index()
    base = t[t['Coefficient'] == 'same_state'][['Estimate', 'Std. Error']].iloc[0]
    rows = []
    for _, r in t.iterrows():
        c = r['Coefficient']
        if ':C(first_year)' in c:
            yr = int(float(c.split('[T.')[1].rstrip(']')))
            rows.append({'year': yr, 'est_int': r['Estimate'], 'se_int': r['Std. Error'],
                         'est_base': base['Estimate'], 'se_base': base['Std. Error']})
    ev = pl.DataFrame(rows).sort('year')
    ev = ev.with_columns(
        est=pl.col.est_base + pl.col.est_int,
        se=(pl.col.se_base**2 + pl.col.se_int**2).sqrt(),
    ).to_pandas()
    ev['lo'] = ev['est'] - 1.96 * ev['se']
    ev['hi'] = ev['est'] + 1.96 * ev['se']
    OUT.parent.mkdir(parents=True, exist_ok=True)
    ev.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(ev)} years, {ev['year'].min()}--{ev['year'].max()})")

    inter = [c for c in fit._coefnames if ':C(first_year)' in c]
    idx = [fit._coefnames.index(c) for c in inter]
    R = np.eye(len(fit._coefnames))[idx]
    wald = fit.wald_test(R=R, q=np.zeros(len(inter)))
    print('joint Wald test (year interactions = 0):')
    print(wald)

    ref_year = int(min(set(od['first_year'].unique()) - set(ev['year'])))
    path_rows = [(ref_year, base['Estimate'], base['Std. Error'])]
    path_rows += list(zip(ev['year'], ev['est'], ev['se']))
    write_tex(path_rows, fit._N, float(wald['pvalue']), OUT.with_suffix('.tex'))


if __name__ == '__main__':
    main()
