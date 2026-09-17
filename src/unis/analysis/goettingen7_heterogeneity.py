"""Göttingen Seven event study by origin distance to Göttingen (findings §8b, Fig. 4).

Origins split at 200 km from Göttingen. One pooled regression with separate
Göttingen x year dummies for near and far origins (1837 omitted for both), so
the two paths share the likelihood and FE structure:

    flow ~ log_dist + same_city + got_near + got_far
           + got_near x year dummies + got_far x year dummies
           | origin_id + dest + first_year

od_year.parquet -> output/tables/goettingen7_heterogeneity.{csv,tex},
                   output/figures/goettingen7_heterogeneity.png
"""

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import pyfixest as pf
import seaborn as sns

from unis import paths
from unis.gravity.constants import (
    FAR_KM,
    GOETTINGEN_SEVEN_LABEL,
    TREAT_YEAR,
    distance_to_goettingen,
    load_od_year_window,
)
from unis.gravity.eventstudy import path_table
from unis.latex import integer
from unis.plotting import mark_event, save, set_style

OUT_CSV = paths.TABLES / 'goettingen7_heterogeneity.csv'


def main() -> None:
    od = load_od_year_window()
    od = od.with_columns(dist_got=distance_to_goettingen(od))
    od = od.with_columns(
        far_orig=(pl.col.dist_got >= FAR_KM),
    ).with_columns(
        got_far=(pl.col.dest == 'goettingen') & pl.col.far_orig,
        got_near=(pl.col.dest == 'goettingen') & ~pl.col.far_orig,
    ).to_pandas()

    years = sorted(od['first_year'].unique())
    dummy_cols = []
    for pref in ['got_near', 'got_far']:
        for y in years:
            if y == TREAT_YEAR:
                continue
            col = f'{pref}_y{y}'
            mask = (od['dest'] == 'goettingen') & (od['first_year'] == y)
            if pref == 'got_near':
                mask &= ~od['far_orig']
            else:
                mask &= od['far_orig']
            od[col] = mask.astype(float)
            dummy_cols.append(col)
    fml = ('flow ~ log_dist + same_city + got_near + got_far + '
           + ' + '.join(dummy_cols)
           + ' | origin_id + dest + first_year')
    fit = pf.fepois(fml, data=od, vcov={'CRV1': 'origin_id'})
    t = fit.tidy().reset_index()
    parts = {}
    for pref, label in [('got_near', 'near'), ('got_far', 'far')]:
        inter = t[t['Coefficient'].str.startswith(f'{pref}_y')].copy()
        inter['year'] = inter['Coefficient'].str.replace(f'{pref}_y', '', regex=False).astype(float)
        parts[label] = inter[['year', 'Estimate', 'Std. Error']].assign(group=label)
        print(f'--- {label} ---')
        print(inter[['year', 'Estimate', 'Std. Error', 'Pr(>|t|)']].round(3).to_string(index=False))

    ev = pd.concat(parts.values())
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    ev.to_csv(OUT_CSV, index=False)
    print(f'wrote {OUT_CSV}')
    paths_for_table = {
        rf'Near origins ($<${FAR_KM} km)': parts['near'],
        rf'Far origins ($\geq${FAR_KM} km)': parts['far'],
    }
    path_table(
        {label: p.rename(columns={'Estimate': 'est', 'Std. Error': 'se'})
         for label, p in paths_for_table.items()},
        footer=[('Observations', [integer(fit._N)]),
                ('Same-city control', ['Yes'])],
    ).write(OUT_CSV.with_suffix('.tex'))

    ev['lo'] = ev['Estimate'] - 1.96 * ev['Std. Error']
    ev['hi'] = ev['Estimate'] + 1.96 * ev['Std. Error']
    set_style()
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    palette = {'near': sns.color_palette('deep')[1], 'far': sns.color_palette('deep')[0]}
    for label, g in ev.groupby('group'):
        ax.errorbar(g['year'], g['Estimate'],
                    yerr=[g['Estimate'] - g['lo'], g['hi'] - g['Estimate']],
                    fmt='o', color=palette[label], ecolor=palette[label], elinewidth=1.4,
                    capsize=2.5, markersize=5,
                    label=f'{label} (<{FAR_KM} km)' if label == 'near' else f'{label} (>={FAR_KM} km)')
    ax.axhline(0, color='gray', lw=0.8, ls='--')
    mark_event(ax, TREAT_YEAR, GOETTINGEN_SEVEN_LABEL)
    ax.set_title('Göttingen Seven effect, by origin distance to Göttingen')
    ax.set_xlabel('First enrollment year')
    ax.set_ylabel('Göttingen × year coefficient by origin group\n(log points, 1837 = 0)')
    ax.legend(title='Origin')
    save(fig, paths.FIGURES / 'goettingen7_heterogeneity.png')


if __name__ == '__main__':
    main()
