"""Event study around 1819 (Carlsbad Decrees): yearly same_state interactions (findings §3).

    flow ~ log_dist + same_city + same_state + same_state:C(first_year)
           | origin_id + dest + first_year^dest

Uses the year grid (true annual flows), not the era grid, whose first_year is
only the modal year per decade cell. The path is baseline + interaction per
year, with conservative CIs that ignore their covariance; read joint
significance from the Wald test instead.

od_year.parquet -> output/tables/event_study_1819.csv
"""

import numpy as np
import polars as pl
import pyfixest as pf

from unis import paths

OUT = paths.TABLES / 'event_study_1819.csv'


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
    if inter:
        idx = [fit._coefnames.index(c) for c in inter]
        R = np.eye(len(fit._coefnames))[idx]
        print('joint Wald test (year interactions = 0):')
        print(fit.wald_test(R=R, q=np.zeros(len(inter))))


if __name__ == '__main__':
    main()
