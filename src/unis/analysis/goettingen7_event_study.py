"""Göttingen Seven pooled event study (findings §8a, Fig. 3).

Did first-enrollment flows to Göttingen drop relative to other universities
after the December 1837 dismissals?

    flow(o,d,t) ~ log_dist + same_city + sum_y 1[d == goettingen & t == y]
                  | origin + dest + year

Year FEs absorb economy-wide and cohort shocks, origin FEs origin size, dest
FEs level differences. 1837 is the reference; 1834-1836 are the pre-trend
diagnostic. No Göttingen trend term: with a complete year-dummy set it is
exactly collinear.

od_year.parquet -> output/tables/goettingen7_event_study.{csv,tex},
                   output/figures/goettingen7_event_study.png
"""

import matplotlib.pyplot as plt
import seaborn as sns

from unis import paths
from unis.gravity.constants import GOETTINGEN_SEVEN_LABEL, TREAT_YEAR, load_od_year_window
from unis.gravity.eventstudy import fit_year_path, joint_wald, path_table, year_path
from unis.latex import integer, pvalue
from unis.plotting import mark_event, save, set_style

OUT_CSV = paths.TABLES / 'goettingen7_event_study.csv'


def main() -> None:
    od = load_od_year_window().to_pandas()
    fit, years = fit_year_path(od, 'goettingen')
    ev = year_path(fit).sort_values('year')

    w_pre = joint_wald(fit, [f'got_y{y}' for y in years if y < TREAT_YEAR])
    w_post = joint_wald(fit, [f'got_y{y}' for y in years if y > TREAT_YEAR])
    print('joint Wald pre-trends (1834-36):', w_pre)
    print('joint Wald post-1837 effects:', w_post)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    ev.to_csv(OUT_CSV, index=False)
    print(f'wrote {OUT_CSV}: {len(ev)} years, {ev["year"].min()}--{ev["year"].max()}')
    print(ev.round(3).to_string(index=False))
    path_table({'Göttingen': ev}, footer=[
        ('Pre-trend test (1834--36), $p$', [pvalue(float(w_pre['pvalue']))]),
        ('Post-period test (1838--47), $p$', [pvalue(float(w_post['pvalue']))]),
        ('Observations', [integer(fit._N)]),
        ('Same-city control', ['Yes']),
    ]).write(OUT_CSV.with_suffix('.tex'))

    ev['lo'] = ev['est'] - 1.96 * ev['se']
    ev['hi'] = ev['est'] + 1.96 * ev['se']
    set_style()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    color = sns.color_palette('deep')[0]
    ax.errorbar(ev['year'], ev['est'],
                yerr=[ev['est'] - ev['lo'], ev['hi'] - ev['est']],
                fmt='o', color=color, ecolor=color,
                elinewidth=1.4, capsize=2.5, markersize=5)
    ax.axhline(0, color='gray', lw=0.8, ls='--')
    mark_event(ax, TREAT_YEAR, GOETTINGEN_SEVEN_LABEL, y_frac=0.9)
    ax.set_title('Göttingen enrollment, relative to other universities')
    ax.set_xlabel('First enrollment year')
    ax.set_ylabel('goettingen × year coefficient\n(log points, 1837 = 0)')
    save(fig, paths.FIGURES / 'goettingen7_event_study.png')


if __name__ == '__main__':
    main()
