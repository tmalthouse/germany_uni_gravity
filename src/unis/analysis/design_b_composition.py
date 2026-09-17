"""Design B: composition of new-university (Berlin/Bonn) flows vs old universities (findings §4-5).

Who did the new research universities attract? Gravity interactions on the era
grid, Berlin and Bonn split:

    static:  flow ~ log_dist + same_city + is_berlin:log_dist + is_bonn:log_dist
                    + is_berlin:C(era) + is_bonn:C(era) | origin_id + dest + era
    dynamic: flow ~ log_dist + same_city + is_berlin:log_dist:C(era)
                    + is_bonn:log_dist:C(era) | origin_id + dest^era
             (pre-founding Berlin/Bonn cells dropped)

plus the flow-weighted mean origin distance by year and school (year grid).

od.parquet, od_year.parquet -> output/figures/design_b_composition.png,
                               output/tables/design_b_composition.tex
"""

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pyfixest as pf
import seaborn as sns

from unis import paths
from unis.gravity.constants import ERA_LABELS, NEW_UNIVERSITIES as NEW
from unis.latex import integer, pvalue, regression_table
from unis.plotting import save, set_style

DYNAMIC_ERAS = (3, 4, 5)  # Berlin and Bonn have no students before era 3


def write_tex(static, dynamic, equal_era_p: dict[str, float], path) -> None:
    """Static and by-era distance gradients, relative to the old universities.

    The static model's Berlin/Bonn era dummies are fit to pre-founding zeros
    and numerically degenerate (findings §4-5), so they are indicated, not shown.
    """
    rows = [('Log distance', ['log_dist', 'log_dist'])]
    for school, name in (('berlin', 'Berlin'), ('bonn', 'Bonn')):
        rows.append((rf'{name} $\times$ log distance', [f'is_{school}:log_dist', None]))
        rows += [(rf'{name} $\times$ log distance $\times$ {ERA_LABELS[e]}',
                  [None, f'is_{school}:log_dist:C(era)[{e}]']) for e in DYNAMIC_ERAS]
    # same_city is flagged, not reported: on those cells log_dist carries the
    # arbitrary log(0.5 km) floor from build_od, so the dummy's coefficient
    # shifts one-for-one with that choice and is not a hometown premium.
    regression_table(['Static', 'By era'], [static.tidy(), dynamic.tidy()], rows, footer=[
        (r'Berlin gradients equal across eras, $p$', ['', pvalue(equal_era_p['berlin'])]),
        (r'Bonn gradients equal across eras, $p$', ['', pvalue(equal_era_p['bonn'])]),
        ('Observations', [integer(static._N), integer(dynamic._N)]),
        ('Same-city control', ['Yes', 'Yes']),
        (r'Berlin, Bonn $\times$ era', ['Yes', '']),
        ('Origin FE', ['Yes', 'Yes']),
        ('Destination FE', ['Yes', '']),
        ('Era FE', ['Yes', '']),
        (r'Destination $\times$ era FE', ['', 'Yes']),
        ('Pre-founding Berlin, Bonn cells', ['Included', 'Dropped']),
    ]).write(path)


def main() -> None:
    od = pl.read_parquet(paths.OD).to_pandas()
    od['is_berlin'] = (od['dest'] == 'berlin').astype(float)
    od['is_bonn'] = (od['dest'] == 'bonn').astype(float)

    fml = ('flow ~ log_dist + same_city + is_berlin:log_dist + is_bonn:log_dist '
           '+ is_berlin:C(era) + is_bonn:C(era) | origin_id + dest + era')
    # solver='np.linalg.lstsq': with Munich split into its Landshut and Munich
    # seats, each structurally zero outside its own operating window, this
    # design is numerically borderline -- scipy's default Cholesky path
    # (assume_a='pos') raises "singular matrix" nondeterministically on it.
    # lstsq returns identical estimates where the default succeeds.
    fit = pf.fepois(fml, data=od, vcov={'CRV1': 'origin_id'}, solver='np.linalg.lstsq')
    print('===== design B split: berlin/bonn x distance, x era =====')
    print(fit.tidy().round(4))

    # Dynamic distance gradient: Berlin/Bonn x era x log_dist, RELATIVE to the
    # old universities' gradient (the base log_dist term).
    # - Pre-founding cells dropped: Berlin and Bonn have no data before era 3,
    #   so their era-1/2 cells are structural zeros (58,372 cells, zero flow).
    #   Kept, era-specific coefficients would be fit to pure zeros.
    # - dest^era FEs absorb each school's enrollment level in each era, so
    #   level growth cannot leak into the era gradients (a level shift times
    #   mean log_dist ~6 mimics a change in reach).
    # - same_city absorbs the arbitrary 0.5 km same-town floor from build_od.
    dyn = od[~(od['dest'].isin(NEW) & (od['era'] < DYNAMIC_ERAS[0]))]
    fml2 = ('flow ~ log_dist + same_city + is_berlin:log_dist:C(era) '
            '+ is_bonn:log_dist:C(era) | origin_id + dest^era')
    fit2 = pf.fepois(fml2, data=dyn, vcov={'CRV1': 'origin_id'}, solver='np.linalg.lstsq')
    print('\n===== design B dynamic: berlin/bonn x era x log_dist (relative) =====')
    print(fit2.tidy().round(4))

    # Flatness: are each school's era-3/4/5 gradients equal? Test this rather
    # than eyeballing CIs: the era coefficients are correlated ~0.99, so level
    # SEs are wide while era differences are precisely estimated.
    names = list(fit2._coefnames)
    equal_era_p = {}
    for school in NEW:
        k = [names.index(f'is_{school}:log_dist:C(era)[{e}]') for e in DYNAMIC_ERAS]
        R = np.zeros((2, len(names)))
        R[0, k[0]], R[0, k[1]] = -1, 1
        R[1, k[0]], R[1, k[2]] = -1, 1
        w = fit2.wald_test(R=R, q=np.zeros(2))
        equal_era_p[school] = float(w['pvalue'])
        print(f'joint Wald, {school} eras 3 = 4 = 5: '
              f'stat={float(w["statistic"]):.3f}, p={float(w["pvalue"]):.4f}')

    # Descriptives: flow-weighted mean origin distance by school x year, from
    # the year grid. Filter to flow > 0: zero-flow university-year cells (132 of
    # them) give 0/0 = NaN weighted means, which would poison the
    # old-universities average for any year containing one.
    dfp = (pl.read_parquet(paths.OD_YEAR)
           .with_columns(new=pl.col.dest.is_in(NEW))
           .filter(pl.col.flow > 0))
    desc = (dfp.group_by('first_year', 'dest', 'new')
            .agg(((pl.col.flow * pl.col.dist_km).sum() / pl.col.flow.sum())
                 .alias('mean_dist_km'),
                 pl.col.flow.sum().alias('flows')))
    print('\nflow-weighted mean distance by year x school:')
    print(desc.sort('first_year', 'dest'))

    # Plot: mean distance over years, Berlin vs Bonn vs old universities
    d = desc.to_pandas()
    d['group'] = np.select(
        [d['dest'] == 'berlin', d['dest'] == 'bonn'],
        ['Berlin (new, 1810)', 'Bonn (new, 1818)'],
        default='Old universities')
    # Old universities: flow-weighted average of their per-school means per year
    old_avg = (d[d['group'] == 'Old universities']
               .groupby('first_year')
               .apply(lambda g: np.average(g['mean_dist_km'], weights=g['flows']),
                      include_groups=False)
               .reset_index(name='mean_dist_km'))
    old_avg['group'] = 'Old universities'
    set_style()
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    palette = {'Berlin (new, 1810)': sns.color_palette('deep')[3],
               'Bonn (new, 1818)': sns.color_palette('deep')[2],
               'Old universities': sns.color_palette('deep')[0]}
    for g in ['Berlin (new, 1810)', 'Bonn (new, 1818)', 'Old universities']:
        sub = old_avg if g == 'Old universities' else d[d['group'] == g]
        sub = sub.sort_values('first_year')
        is_old = g == 'Old universities'
        ax.plot(sub['first_year'], sub['mean_dist_km'],
                marker='s' if g == 'Bonn (new, 1818)' else 'o',
                ls='--' if g == 'Bonn (new, 1818)' else '-',
                color=palette[g], lw=1.4 if is_old else 2.2, alpha=0.55 if is_old else 1.0,
                markersize=3 if is_old else 4, label=g)
    ax.set_title('Mean origin distance of enrollment flows, by year')
    ax.set_xlabel('First enrollment year')
    ax.set_ylabel('Flow-weighted mean distance (km)')
    ax.legend(title='')
    save(fig, paths.FIGURES / 'design_b_composition.png')

    # Written after the figure, so the table is the newer of the two outputs.
    write_tex(fit, fit2, equal_era_p, paths.TABLES / 'design_b_composition.tex')


if __name__ == '__main__':
    main()
