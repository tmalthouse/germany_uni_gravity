"""Era-by-era gravity coefficients from the interaction specifications (findings §3, Fig. 1).

Two panels:
1. State-border effect by era: same_state baseline + same_state:C(era) interactions
   (from state_x_era; both_x_era gives the same picture).
2. Distance elasticity by era: log_dist baseline + log_dist:C(era) interactions
   (from dist_x_era; both_x_era again gives the same picture).

Point estimates with 95% CIs (cluster-robust by origin).

output/tables/gravity_results.csv -> output/figures/era_coefficients.png
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns

from unis import paths
from unis.gravity.constants import ERA_LABELS
from unis.plotting import save, set_style


def era_series(df: pl.DataFrame, spec: str, base_coef: str, inter_prefix: str) -> pd.DataFrame:
    """Reconstruct baseline + interaction point estimates and CIs per era."""
    base = df.filter((pl.col.spec == spec) & (pl.col.coef == base_coef))
    b0, se0 = base['est'][0], base['se'][0]

    rows = []
    for era in range(1, 6):
        if era == 1:
            est, se = b0, se0
        else:
            name = f'{inter_prefix}:C(era)[T.{era}]'
            r = df.filter((pl.col.spec == spec) & (pl.col.coef == name))
            est = b0 + r['est'][0]
            se = float(np.sqrt(se0**2 + r['se'][0]**2))  # conservative: ignores cov
        rows.append({'era': era, 'period': ERA_LABELS[era], 'est': est, 'se': se})
    out = pd.DataFrame(rows)
    out['lo'] = out['est'] - 1.96 * out['se']
    out['hi'] = out['est'] + 1.96 * out['se']
    return out


def main() -> None:
    df = pl.read_csv(paths.TABLES / 'gravity_results.csv')

    state = era_series(df, 'state_x_era', 'same_state', 'same_state')
    dist = era_series(df, 'dist_x_era', 'log_dist', 'log_dist')

    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    color = sns.color_palette('deep')[0]
    for ax, d, title, ylabel in [
        (axes[0], state, 'State-border effect by decade',
         'log points (flow multiplier $e^{\\beta}$)'),
        (axes[1], dist, 'Distance elasticity by decade', 'log points'),
    ]:
        ax.errorbar(d['period'], d['est'],
                    yerr=[d['est'] - d['lo'], d['hi'] - d['est']],
                    fmt='o', color=color, ecolor=color,
                    elinewidth=1.6, capsize=4, markersize=6)
        ax.axhline(0, color='gray', lw=0.8, ls='--')
        ax.set_title(title)
        ax.set_xlabel('')
        ax.set_ylabel(ylabel)

    # annotate the flow multiplier on the state panel
    for _, r in state.iterrows():
        axes[0].annotate(f'×{np.exp(r.est):.1f}', (r['period'], r['est']),
                         textcoords='offset points', xytext=(10, 6), fontsize=8)

    save(fig, paths.FIGURES / 'era_coefficients.png')


if __name__ == '__main__':
    main()
