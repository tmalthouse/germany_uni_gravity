"""Baseline PPML gravity model of student-university choice (findings §1-3).

    flow(o,u) ~ Poisson(exp(origin FE + destination FE
                            + b1 * log_dist + b2 * same_state + b3 * same_polity))

plus era-interaction specifications and a confession split, on the era grid.
Estimated with pyfixest's fepois; SEs clustered by origin (hometown place).

The OD grid is zero-padded: zeros identify the extensive margin, which is what
PPML needs for consistent estimation with FE (Santos Silva & Tenreyro 2006).

od.parquet -> output/tables/gravity_results.csv, output/tables/gravity_results.tex
"""

import re

import numpy as np
import polars as pl
import pyfixest as pf

from unis import paths
from unis.gravity.constants import ERA_LABELS
from unis.latex import Tabular, integer, num, se, stars

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

# --- LaTeX table ---------------------------------------------------------------
SPEC_LABELS = {
    'baseline': 'Baseline',
    'dist_x_era': r'Distance $\times$ era',
    'state_x_era': r'Border $\times$ era',
    'both_x_era': r'Both $\times$ era',
    'confession_split': 'Confession',
}
# Also the row order.
VARIABLE_LABELS = {
    'log_dist': 'Log distance',
    'same_state': 'Same state',
    'same_polity': 'Same polity',
    'same_city': 'Same city',
    'ss_sc': 'Same state, same confession',
    'ss_dc': 'Same state, different confession',
    'ds_sc': 'Different state, same confession',
}
FIXED_EFFECTS = [
    ('Origin FE', 'origin_id'),
    ('Destination FE', 'dest'),
    (r'Destination $\times$ era FE', 'era^dest'),
]


def split_coef(name: str) -> tuple[str, int | None]:
    """'log_dist:C(era)[T.3]' -> ('log_dist', 3); 'same_state' -> ('same_state', None)."""
    m = re.fullmatch(r'(.+):C\(era\)\[T\.(\d+)\]', name)
    return (m.group(1), int(m.group(2))) if m else (name, None)


def coef_label(name: str) -> str:
    var, era = split_coef(name)
    label = VARIABLE_LABELS[var]
    return label if era is None else rf'{label} $\times$ {ERA_LABELS[era]}'


def write_tex(results: dict, path) -> None:
    """One column per specification: estimates with stars, SEs beneath."""
    names = list(results)
    tidy = {n: fit.tidy() for n, fit in results.items()}
    order = list(VARIABLE_LABELS)
    coefs = sorted({c for t in tidy.values() for c in t.index},
                   key=lambda c: (order.index(split_coef(c)[0]), split_coef(c)[1] or 0))

    t = Tabular('l' + 'c' * len(names))
    t.row('', *[f'({i})' for i in range(1, len(names) + 1)])
    t.row('', *[SPEC_LABELS[n] for n in names]).midrule()
    for c in coefs:
        estimates, errors = [], []
        for n in names:
            if c in tidy[n].index:
                r = tidy[n].loc[c]
                estimates.append(num(r['Estimate']) + stars(r['Pr(>|t|)']))
                errors.append(se(r['Std. Error']))
            else:
                estimates.append('')
                errors.append('')
        t.row(coef_label(c), *estimates)
        t.row('', *errors)
    t.midrule()
    t.row('Observations', *[integer(results[n]._N) for n in names])
    for label, fe in FIXED_EFFECTS:
        t.row(label, *['Yes' if fe in fixed_effects(SPECS[n]) else '' for n in names])
    t.write(path)


def fixed_effects(formula: str) -> list[str]:
    return [f.strip() for f in formula.split('|')[1].split('+')]


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
    write_tex(results, OUT.with_suffix('.tex'))

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
