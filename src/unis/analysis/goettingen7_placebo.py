"""Placebo event studies for the Göttingen Seven (findings §8f).

The same event study, 1837 reference and 1834-1847 window, with each other
university treated in turn. Göttingen's actual path is overlaid. The figure and
summary use the well-identified placebos only; the CSV keeps every path.

od_year.parquet, output/tables/goettingen7_event_study.csv
    -> output/tables/goettingen7_placebo.{csv,tex}, output/figures/goettingen7_placebo.png
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

from unis import paths
from unis.gravity.constants import (
    GOETTINGEN_SEVEN_LABEL,
    TREAT_YEAR,
    UNIVERSITY_NAMES,
    load_od_year_window,
)
from unis.gravity.eventstudy import fit_year_path, year_path
from unis.latex import Tabular, multicolumn, num
from unis.plotting import mark_event, save, set_style

OUT_CSV = paths.TABLES / 'goettingen7_placebo.csv'
POST = (1838, 1847)

# The non-Göttingen destinations with year-level flow in the window. 'muenchen'
# (not 'muenchen_old'): the university sits in Munich from 1826, so muenchen_old
# is structurally zero here. Marburg and Würzburg have no flow in the window.
PLACEBOS = ['berlin', 'bonn', 'erlangen', 'freiburg', 'giessen', 'heidelberg',
            'jena', 'kiel', 'muenchen', 'tuebingen']
# Excluded from the figure and summary (still written to the CSV): their year
# dummies are near-unidentified because source coverage ends or lapses inside
# the window. Erlangen has no records after 1843, Jena lapses in 1845 (17
# students), München in 1842 (1 student).
DEGENERATE = ['erlangen', 'jena', 'muenchen']


def post_mean(ev: pd.DataFrame) -> float:
    return ev[(ev['year'] >= POST[0]) & (ev['year'] <= POST[1])]['est'].mean()


def summary_cells(dest: str, ev: pd.DataFrame) -> list[str]:
    """University, 1838 coefficient, deepest dip and its year, post-period mean."""
    rest = ev[ev['year'] != TREAT_YEAR]
    dip = rest.loc[rest['est'].idxmin()]
    at_1838 = ev.loc[ev['year'] == POST[0], 'est'].iloc[0]
    return [UNIVERSITY_NAMES[dest], num(at_1838), num(dip['est']), str(int(dip['year'])),
            num(post_mean(ev))]


def write_tex(got: pd.DataFrame, paths_by_dest: dict[str, pd.DataFrame], clean: list[str], path) -> None:
    t = Tabular('lcccc')
    t.row('University', str(POST[0]), 'Deepest dip', 'Year of dip',
          f'Mean {POST[0]}--{str(POST[1])[2:]}').midrule()
    panels = [
        ('Treated', {'goettingen': got}),
        ('Placebos', {d: paths_by_dest[d] for d in clean}),
        ('Placebos with gaps in source coverage', {d: paths_by_dest[d] for d in DEGENERATE}),
    ]
    for i, (title, evs) in enumerate(panels):
        if i:
            t.space()
        t.row(multicolumn(5, 'l', rf'\textit{{{title}}}'))
        for dest, ev in evs.items():
            t.row(*summary_cells(dest, ev))
    t.write(path)


def main() -> None:
    od = load_od_year_window().to_pandas()

    paths_by_dest = {}
    for d in PLACEBOS:
        fit, _ = fit_year_path(od.copy(), d, prefix='plc_y')
        ev = year_path(fit, prefix='plc_y')
        ev['dest'] = d
        paths_by_dest[d] = ev
        deepest = ev.loc[ev[ev['year'] != TREAT_YEAR]['est'].idxmin()]
        print(f'{d}: deepest dip {deepest["est"]:+.2f} ({int(deepest["year"])}), '
              f'post-1838-47 mean {post_mean(ev):+.2f}')

    got = pl.read_csv(paths.TABLES / 'goettingen7_event_study.csv').to_pandas()
    got['dest'] = 'goettingen'
    got = got[['year', 'est', 'se', 'dest']]

    all_ev = pd.concat(list(paths_by_dest.values()) + [got], ignore_index=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    all_ev.to_csv(OUT_CSV, index=False)
    print(f'wrote {OUT_CSV}')

    # How unusual is Göttingen? Depth and persistence, well-identified placebos only.
    clean = [d for d in PLACEBOS if d not in DEGENERATE]
    write_tex(got, paths_by_dest, clean, OUT_CSV.with_suffix('.tex'))
    plc_mins = pd.Series({d: paths_by_dest[d]['est'].min() for d in clean})
    plc_post = pd.Series({d: post_mean(paths_by_dest[d]) for d in clean})
    got_min = got['est'].min()
    print(f'\nwell-identified placebos ({len(clean)}): {clean}')
    print(f'excluded as near-unidentified ({len(DEGENERATE)}): {DEGENERATE}')
    print(f'Göttingen: min {got_min:+.2f}, post-1838-47 mean {post_mean(got):+.2f}')
    print(f'placebo mins: {plc_mins.round(2).to_dict()}')
    print(f'placebo post-1838-47 means: {plc_post.round(2).to_dict()}')
    print(f'Göttingen min more extreme than {int((plc_mins > got_min).sum())}'
          f'/{len(clean)} well-identified placebos')

    set_style()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for _, g in all_ev[all_ev['dest'].isin(clean)].groupby('dest'):
        g = g.sort_values('year')
        ax.plot(g['year'], g['est'], color='gray', lw=0.7, alpha=0.55)
    gg = got.sort_values('year')
    gse = gg['se'].to_numpy()
    ax.errorbar(gg['year'], gg['est'], yerr=np.vstack([1.96 * gse, 1.96 * gse]),
                fmt='o-', color='crimson', ecolor='crimson', elinewidth=1.3,
                capsize=2.5, markersize=5, lw=2.0, label='Göttingen (actual)')
    ax.axhline(0, color='black', lw=0.8, ls='--')
    mark_event(ax, TREAT_YEAR, GOETTINGEN_SEVEN_LABEL)
    ax.set_title(f'Placebo event studies: Göttingen vs. {len(clean)} well-identified '
                 'placebo universities\n(gray: one placebo each, same spec, 1837 '
                 'reference; excluded as near-unidentified: '
                 f'{", ".join(d.title() for d in DEGENERATE)})', fontsize=11)
    ax.set_xlabel('First enrollment year')
    ax.set_ylabel('university × year coefficient\n(log points, 1837 = 0)')
    ax.legend()
    save(fig, paths.FIGURES / 'goettingen7_placebo.png')


if __name__ == '__main__':
    main()
