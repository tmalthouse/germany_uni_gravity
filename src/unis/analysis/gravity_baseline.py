"""Baseline PPML gravity model of student-university choice (findings §1-3).

    flow(o,u) ~ Poisson(exp(origin FE + destination FE
                            + b1 * log_dist + b2 * same_state + b3 * same_polity))

plus era-interaction specifications and a confession split, on the era grid.
Estimated with pyfixest's fepois; SEs clustered by origin (hometown place).

The OD grid is zero-padded: zeros identify the extensive margin, which is what
PPML needs for consistent estimation with FE (Santos Silva & Tenreyro 2006).

od.parquet -> output/tables/gravity_results.csv (every specification),
              output/tables/gravity_structure.tex (baseline and confession split),
              output/tables/gravity_by_decade.tex (era interactions, backs Figure 1)
"""

import re

import numpy as np
import polars as pl
import pyfixest as pf

from unis import paths
from unis.gravity.constants import ERA_LABELS
from unis.latex import integer, pvalue, regression_table

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

# --- LaTeX tables --------------------------------------------------------------
VARIABLE_LABELS = {
    'log_dist': 'Log distance',
    'same_state': 'Same state',
    'same_polity': 'Same polity',
    'ss_sc': 'Same state, same confession',
    'ss_dc': 'Same state, different confession',
    'ds_sc': 'Different state, same confession',
}
DECADE_SPECS = {
    'dist_x_era': r'Distance $\times$ era',
    'state_x_era': r'Border $\times$ era',
    'both_x_era': 'Both',
}
LATER_ERAS = (2, 3, 4, 5)  # era 1 (1800s) is the reference


def era_term(var: str, era: int) -> str:
    return f'{var}:C(era)[T.{era}]'


def era_terms(fit, var: str) -> list[str]:
    return [c for c in fit._coefnames if re.fullmatch(rf'{var}:C\(era\)\[T\.\d+\]', c)]


def wald_p(fit, R: np.ndarray) -> float:
    return float(fit.wald_test(R=R, q=np.zeros(R.shape[0]))['pvalue'])


def joint_zero_p(fit, coefs: list[str]) -> float:
    """p-value for all `coefs` jointly zero."""
    names = list(fit._coefnames)
    return wald_p(fit, np.eye(len(names))[[names.index(c) for c in coefs]])


def equal_p(fit, a: str, b: str) -> float:
    """p-value for coefficient a equal to coefficient b."""
    names = list(fit._coefnames)
    R = np.zeros((1, len(names)))
    R[0, names.index(a)], R[0, names.index(b)] = 1, -1
    return wald_p(fit, R)


def write_structure_tex(results: dict, path) -> None:
    """Main-text table: the baseline and the confession split of the border effect."""
    fits = [results['baseline'], results['confession_split']]
    rows = [(VARIABLE_LABELS[v], [v, v])
            for v in ('log_dist', 'same_state', 'same_polity', 'ss_sc', 'ss_dc', 'ds_sc')]
    regression_table(['Baseline', 'Confession'], [f.tidy() for f in fits], rows, footer=[
        ('Same state: same = different confession, $p$',
         ['', pvalue(equal_p(results['confession_split'], 'ss_sc', 'ss_dc'))]),
        ('Observations', [integer(f._N) for f in fits]),
        ('Same-city control', ['Yes', 'Yes']),
        ('Origin FE', ['Yes', 'Yes']),
        ('Destination FE', ['Yes', 'Yes']),
    ]).write(path)


def write_decade_tex(results: dict, path) -> None:
    """Appendix table: distance and border effects interacted with decade."""
    fits = [results[s] for s in DECADE_SPECS]
    rows = []
    for var in ('log_dist', 'same_state'):
        rows.append((VARIABLE_LABELS[var], [var] * len(fits)))
        rows += [(rf'{VARIABLE_LABELS[var]} $\times$ {ERA_LABELS[e]}', [era_term(var, e)] * len(fits))
                 for e in LATER_ERAS]
    rows.append((VARIABLE_LABELS['same_polity'], ['same_polity'] * len(fits)))

    def era_p(fit, var):
        terms = era_terms(fit, var)
        return pvalue(joint_zero_p(fit, terms)) if terms else ''

    regression_table(list(DECADE_SPECS.values()), [f.tidy() for f in fits], rows, footer=[
        (r'Distance $\times$ era terms $= 0$, $p$', [era_p(f, 'log_dist') for f in fits]),
        (r'Border $\times$ era terms $= 0$, $p$', [era_p(f, 'same_state') for f in fits]),
        ('Observations', [integer(f._N) for f in fits]),
        ('Same-city control', ['Yes'] * len(fits)),
        ('Origin FE', ['Yes'] * len(fits)),
        (r'Destination $\times$ era FE', ['Yes'] * len(fits)),
    ]).write(path)


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

    # Joint tests of the era interactions, per specification
    for name, fit in results.items():
        inter = [c for c in fit._coefnames if ':C(era)' in c]
        if inter:
            print(f'joint Wald test {name} (era interactions = 0): '
                  f'p={joint_zero_p(fit, inter):.4g}')
    # Within states, does shared confession add to the border effect?
    print('Wald test ss_sc = ss_dc (confession_split): '
          f'p={equal_p(results["confession_split"], "ss_sc", "ss_dc"):.4f}')

    print('\n--- diagnostics ---')
    for name, fit in results.items():
        print(f'{name}: deviance {fit.deviance:.1f}, n {fit._N}')

    write_structure_tex(results, paths.TABLES / 'gravity_structure.tex')
    write_decade_tex(results, paths.TABLES / 'gravity_by_decade.tex')


if __name__ == '__main__':
    main()
