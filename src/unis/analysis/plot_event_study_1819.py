"""State-border effect by enrollment year, with 1819 (Carlsbad Decrees) marked.

output/tables/event_study_1819.csv -> output/figures/event_study_1819.png
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from unis import paths
from unis.plotting import mark_event, save, set_style


def main() -> None:
    ev = pd.read_csv(paths.TABLES / 'event_study_1819.csv')
    set_style()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    color = sns.color_palette('deep')[0]
    ax.errorbar(ev['year'], ev['est'],
                yerr=[ev['est'] - ev['lo'], ev['hi'] - ev['est']],
                fmt='o', color=color, ecolor=color,
                elinewidth=1.4, capsize=2.5, markersize=4.5,
                label='state-border effect (log points)')
    ax.axhline(0, color='gray', lw=0.8, ls='--')
    mark_event(ax, 1819, 'Carlsbad\nDecrees', x_offset=0.4)

    ax.set_title('State-border effect on student flows, by enrollment year')
    ax.set_xlabel('First enrollment year')
    ax.set_ylabel('same_state coefficient (log points)')
    save(fig, paths.FIGURES / 'event_study_1819.png')


if __name__ == '__main__':
    main()
