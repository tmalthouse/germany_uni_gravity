"""Baseline PPML gravity model of student-university choice (findings §1-3).

    flow(o,u) ~ Poisson(exp(origin FE + destination FE
                            + b1 * log_dist + b2 * same_state + b3 * same_polity))

plus era-interaction specifications and a confession split, on the era grid.
Estimated with pyfixest's fepois; SEs clustered by origin (hometown place).

The OD grid is zero-padded: zeros identify the extensive margin, which is what
PPML needs for consistent estimation with FE (Santos Silva & Tenreyro 2006).

od.parquet -> output/tables/gravity_results.csv
"""

import numpy as np
import polars as pl
import pyfixest as pf

from unis import paths

OUT = paths.TABLES / 'gravity_results.csv'

SPECS = {
    'baseline': 'flow ~ log_dist + same_city + same_state + same_polity | origin_id + dest',
    # Time variation: interactions with era (1=1800s ... 5=1840s), era FEs
    # absorb period level effects. C(era) reference is era 1 (1800s).
    'dist_x_era': 'flow ~ log_dist + same_city + same_state + same_polity '
                  '+ log_dist:C(era) | origin_id + dest + era^dest',
    'state_x_era': 'flow ~ log_dist + same_city + same_state + same_polity '
                   '+ same_state:C(era) | origin_id + dest + era^dest',
    'both_x_era': 'flow ~ log_dist + same_city + same_state + same_polity '
                  '+ log_dist:C(era) + same_state:C(era) | origin_id + dest + era^dest',
    # Confessional pairing: four-way split of the border effect.
    # Baseline (omitted): different state AND different confession.
    # Dummies precomputed in build_od since formulaic rejects '~'.
    'confession_split': 'flow ~ log_dist + same_city + same_polity '
                        '+ ss_sc + ss_dc + ds_sc | origin_id + dest',
}


def main() -> None:
    od = pl.read_parquet(paths.OD)
    print(f'OD cells: {od.height}, nonzero: {(od["flow"] > 0).sum()}')

    results = {}
    for name, fml in SPECS.items():
        fit = pf.fepois(fml, data=od.to_pandas(), vcov={'CRV1': 'origin_id'})
        results[name] = fit
        print(f'\n===== {name} =====')
        print(fit.tidy().round(4))

    # Combined coefficient table across specifications
    rows = []
    for name, fit in results.items():
        t = fit.tidy().reset_index()
        for _, r in t.iterrows():
            rows.append({'spec': name, 'coef': r['Coefficient'], 'est': r['Estimate'],
                         'se': r['Std. Error'], 'p': r['Pr(>|t|)']})
    out = pl.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(OUT)
    print(f'wrote {OUT}: {out.shape}')

    # Joint test: are all era interactions zero? (Wald, per specification)
    for name, fit in results.items():
        inter = [c for c in fit._coefnames if ':C(era)' in c]
        if inter:
            R = np.eye(len(fit._coefnames))[[fit._coefnames.index(c) for c in inter]]
            w = fit.wald_test(R=R, q=np.zeros(len(inter)))
            print(f'joint Wald test {name} (era interactions = 0): {w}')

    print('\n--- diagnostics ---')
    for name, fit in results.items():
        print(f'{name}: deviance {fit.deviance:.1f}, n {fit._N}')


if __name__ == '__main__':
    main()
