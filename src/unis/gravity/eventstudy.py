"""Single-destination event studies on the year grid (Göttingen Seven design).

    flow ~ log_dist + same_city + sum_y 1[dest == d & year == y] | origin + dest + year

PPML, SEs clustered by origin. Year dummies are built by hand with the
treatment year omitted, because a complete C(first_year) interaction set would
be silently re-referenced by the estimator.
"""

import numpy as np
import pandas as pd
import pyfixest as pf

from unis.gravity.constants import TREAT_YEAR


def fit_year_path(od: pd.DataFrame, dest: str, *, prefix: str = 'got_y',
                  treat_year: int = TREAT_YEAR):
    """Estimate the event study for `dest`. Adds the dummy columns to `od` in place.

    Returns (fit, years): the pyfixest fit and the sorted years in the sample.
    """
    years = sorted(od['first_year'].unique())
    dummies = []
    for y in years:
        if y == treat_year:
            continue
        col = f'{prefix}{y}'
        od[col] = ((od['dest'] == dest) & (od['first_year'] == y)).astype(float)
        dummies.append(col)
    fml = ('flow ~ log_dist + same_city + ' + ' + '.join(dummies)
           + ' | origin_id + dest + first_year')
    fit = pf.fepois(fml, data=od, vcov={'CRV1': 'origin_id'})
    return fit, years


def year_path(fit, prefix: str = 'got_y', treat_year: int = TREAT_YEAR) -> pd.DataFrame:
    """Coefficient path (year, est, se): the reference year first at 0, then by year."""
    t = fit.tidy().reset_index()
    inter = t[t['Coefficient'].str.startswith(prefix)].copy()
    inter['year'] = inter['Coefficient'].str.replace(prefix, '', regex=False).astype(float)
    rows = [{'year': treat_year, 'est': 0.0, 'se': 0.0}]
    for _, r in inter.sort_values('year').iterrows():
        rows.append({'year': r['year'], 'est': r['Estimate'], 'se': r['Std. Error']})
    return pd.DataFrame(rows)


def joint_wald(fit, coefs: list[str]):
    """Wald test that all `coefs` are jointly zero."""
    names = list(fit._coefnames)
    idx = [names.index(c) for c in coefs]
    R = np.eye(len(names))[idx]
    return fit.wald_test(R=R, q=np.zeros(len(coefs)))
