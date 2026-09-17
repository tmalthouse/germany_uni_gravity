"""Göttingen Seven event study against restricted donor pools (findings §8d).

Same specification as goettingen7_event_study, estimated on three pools:

  all                 : every destination label
  old protestant (6)  : kiel, marburg, giessen, erlangen, jena, tuebingen
  west protestant (4) : kiel, marburg, giessen, erlangen

The pools are thinner than their labels in the later years; see findings §8d.

od_year.parquet -> output/tables/goettingen7_matched.{csv,tex},
                   output/figures/goettingen7_matched.png
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from unis import paths
from unis.gravity.constants import GOETTINGEN_SEVEN_LABEL, TREAT_YEAR, load_od_year_window
from unis.gravity.eventstudy import fit_year_path, joint_wald, path_table, year_path
from unis.latex import integer, pvalue
from unis.plotting import mark_event, save, set_style

OUT_CSV = paths.TABLES / 'goettingen7_matched.csv'

POOLS = {
    'all (14 labels)': None,
    'old protestant (6)': ['goettingen', 'kiel', 'marburg', 'giessen',
                           'erlangen', 'jena', 'tuebingen'],
    'west protestant (4)': ['goettingen', 'kiel', 'marburg', 'giessen', 'erlangen'],
}
POOL_LABELS = {
    'all (14 labels)': 'All universities',
    'old protestant (6)': 'Old Protestant',
    'west protestant (4)': 'Western Protestant',
}


def main() -> None:
    od = load_od_year_window().to_pandas()

    parts, footer = {}, {'pre': [], 'post': [], 'n': []}
    for label, keep in POOLS.items():
        sub = (od if keep is None else od[od['dest'].isin(keep)]).copy()
        fit, years = fit_year_path(sub, 'goettingen')
        ev = year_path(fit)
        ev['pool'] = label
        parts[label] = ev
        w_pre = joint_wald(fit, [f'got_y{y}' for y in years if y < TREAT_YEAR])
        w_post = joint_wald(fit, [f'got_y{y}' for y in years if y > TREAT_YEAR])
        footer['pre'].append(pvalue(float(w_pre['pvalue'])))
        footer['post'].append(pvalue(float(w_post['pvalue'])))
        footer['n'].append(integer(fit._N))
        print(f'===== donor pool: {label} =====')
        print('joint Wald pre-trends:', w_pre)
        print('joint Wald post-1837:', w_post)
        print(ev.round(3).to_string(index=False))
        print()

    all_ev = pd.concat(parts.values())
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    all_ev.to_csv(OUT_CSV, index=False)
    print(f'wrote {OUT_CSV}')
    path_table({POOL_LABELS[label]: ev for label, ev in parts.items()}, footer=[
        ('Pre-trend test (1834--36), $p$', footer['pre']),
        ('Post-period test (1838--47), $p$', footer['post']),
        ('Observations', footer['n']),
        ('Same-city control', ['Yes']),
    ]).write(OUT_CSV.with_suffix('.tex'))

    set_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    palette = sns.color_palette('deep')
    for i, (label, g) in enumerate(all_ev.groupby('pool', sort=False)):
        g = g.sort_values('year').reset_index(drop=True)
        yerr = np.vstack([1.96 * g['se'].to_numpy(), 1.96 * g['se'].to_numpy()])
        ax.errorbar(g['year'], g['est'], yerr=yerr,
                    fmt='o-', color=palette[i], ecolor=palette[i],
                    elinewidth=1.1, capsize=2, markersize=4, label=label)
    ax.axhline(0, color='gray', lw=0.8, ls='--')
    mark_event(ax, TREAT_YEAR, GOETTINGEN_SEVEN_LABEL)
    ax.set_title('Göttingen Seven event study, by donor pool')
    ax.set_xlabel('First enrollment year')
    ax.set_ylabel('goettingen × year coefficient\n(log points, 1837 = 0)')
    ax.legend(title='Comparison group')
    save(fig, paths.FIGURES / 'goettingen7_matched.png')


if __name__ == '__main__':
    main()
