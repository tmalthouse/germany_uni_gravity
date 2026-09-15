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
from unis.latex import Tabular, multicolumn, num, se


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


def path_table(groups: dict[str, pd.DataFrame], footer: list[tuple[str, list[str]]] = (),
               treat_year: int = TREAT_YEAR) -> Tabular:
    """Year-by-year coefficient paths side by side, one Estimate/SE pair per group.

    Each group is a frame with year, est and se. The treatment year is shown as
    the reference. Footer rows carry one value per group, or a single value
    spanning all groups.
    """
    k = len(groups)
    t = Tabular("l" + "cc" * k)
    if k > 1:
        t.row("", *[multicolumn(2, "c", label) for label in groups])
        for i in range(k):
            t.cmidrule(2 + 2 * i, 3 + 2 * i)
    t.row("Year", *["Estimate", "SE"] * k).midrule()

    years = sorted({int(y) for g in groups.values() for y in g["year"]} | {treat_year})
    for y in years:
        cells = []
        for g in groups.values():
            if y == treat_year:
                cells.append(multicolumn(2, "c", "reference"))
                continue
            r = g[g["year"] == y]
            cells += [num(r["est"].iloc[0]), se(r["se"].iloc[0])] if len(r) else ["", ""]
        t.row(str(y), *cells)

    if footer:
        t.midrule()
        for label, values in footer:
            if len(values) == 1:
                t.row(label, multicolumn(2 * k, "c", values[0]))
            else:
                t.row(label, *[multicolumn(2, "c", v) for v in values])
    return t
