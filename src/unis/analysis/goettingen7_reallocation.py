"""Where did Göttingen's far-origin students go after the Göttingen Seven? (findings §8e, Fig. 5)

Treated origins: far origins (>= 200 km from Göttingen) with positive Göttingen
flow in 1834-1837. Destination shares pre (1834-1837) vs post (1838-1841), the
same for near origins as a comparison, and a PPML check on far origins:

    flow ~ got_post + new_post | origin_id + dest + period

od_year.parquet -> output/tables/goettingen7_reallocation.{csv,tex},
                   output/tables/goettingen7_reallocation_ppml.tex,
                   output/figures/goettingen7_reallocation.png
"""

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pyfixest as pf
import seaborn as sns

from unis import paths
from unis.gravity.constants import (
    FAR_KM,
    NEW_UNIVERSITIES as NEW,
    UNIVERSITY_NAMES,
    distance_to_goettingen,
)
from unis.latex import Tabular, integer, multicolumn, num, regression_table
from unis.plotting import save, set_style

OUT_CSV = paths.TABLES / 'goettingen7_reallocation.csv'
PRE = (1834, 1837)
POST = (1838, 1841)


def origins_with_pre_flow(od: pl.DataFrame, far: bool) -> pl.DataFrame:
    """Origins on one side of FAR_KM that sent students to Göttingen in PRE."""
    side = pl.col.far_orig if far else ~pl.col.far_orig
    return (od.filter((pl.col.dest == 'goettingen') & side
                      & (pl.col.first_year >= PRE[0]) & (pl.col.first_year <= PRE[1])
                      & (pl.col.flow > 0))
            .select('origin_id').unique())


def destination_shares(od: pl.DataFrame, origins: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Restrict to `origins`, label pre/post, and tabulate destination shares by period."""
    t = (od.join(origins, on='origin_id', how='inner')
         .with_columns(period=pl.when(pl.col.first_year <= PRE[1]).then(pl.lit('pre'))
                                .otherwise(pl.lit('post'))))
    shares = (t.group_by('period', 'dest').agg(pl.col.flow.sum().alias('flow'))
              .pivot(on='period', index='dest', aggregate_function='sum')
              .fill_null(0)
              .with_columns(
                  share_pre=pl.col('pre') / pl.col('pre').sum(),
                  share_post=pl.col('post') / pl.col('post').sum(),
              ).with_columns(d_share=pl.col.share_post - pl.col.share_pre)
              .sort('d_share', descending=True))
    return t, shares


def period(years: tuple[int, int]) -> str:
    return f'{years[0]}--{str(years[1])[2:]}'


def write_shares_tex(far: pl.DataFrame, near: pl.DataFrame, n_far: int, n_near: int, path) -> None:
    """Destination shares (%) pre and post, and their change (percentage points)."""
    groups = [{r['dest']: r for r in g.iter_rows(named=True)} for g in (far, near)]

    def no_enrollments(dest: str) -> bool:
        return all(g.get(dest) is None or g[dest]['pre'] == g[dest]['post'] == 0 for g in groups)

    # Destinations with no enrollments from either group in either period are
    # all-zero rows; they stay in the CSV but not in the publication table.
    dests = sorted((d for d in set(groups[0]) | set(groups[1]) if not no_enrollments(d)),
                   key=lambda d: (-groups[0][d]['d_share'] if d in groups[0] else 0.0,
                                  UNIVERSITY_NAMES[d]))
    t = Tabular('l' + 'ccc' * 2)
    t.row('', multicolumn(3, 'c', rf'Far origins ($\geq${FAR_KM} km)'),
          multicolumn(3, 'c', rf'Near origins ($<${FAR_KM} km)'))
    t.cmidrule(2, 4).cmidrule(5, 7)
    t.row('Destination', *[period(PRE), period(POST), 'Change'] * 2).midrule()
    for d in dests:
        cells = [UNIVERSITY_NAMES[d]]
        for g in groups:
            r = g.get(d)
            cells += (['', '', ''] if r is None else
                      [num(100 * r['share_pre'], 1), num(100 * r['share_post'], 1),
                       num(100 * r['d_share'], 1)])
        t.row(*cells)
    t.midrule()
    t.row('Origins', multicolumn(3, 'c', integer(n_far)), multicolumn(3, 'c', integer(n_near)))
    t.row('Enrollments', *[x for g in (far, near)
                           for x in (integer(g['pre'].sum()), integer(g['post'].sum()), '')])
    t.write(path)


def write_ppml_tex(fit, path) -> None:
    rows = [
        (r'Göttingen $\times$ post', ['got_post']),
        (r'Berlin or Bonn $\times$ post', ['new_post']),
    ]
    regression_table(['Flow'], [fit.tidy()], rows, numbered=False, footer=[
        ('Observations', [integer(fit._N)]),
        *[(fe, ['Yes']) for fe in ('Origin FE', 'Destination FE', 'Period FE')],
    ]).write(path)


def main() -> None:
    od = pl.read_parquet(paths.OD_YEAR).filter(
        (pl.col.first_year >= PRE[0]) & (pl.col.first_year <= POST[1])
    )
    od = od.with_columns(dist_got=distance_to_goettingen(od)).with_columns(
        far_orig=(pl.col.dist_got >= FAR_KM),
    )

    treated = origins_with_pre_flow(od, far=True)
    print(f'treated far origins (Göttingen flow in {PRE}): {treated.height}')
    t, piv = destination_shares(od, treated)
    print('\nTreated far origins: destination shares, pre vs post:')
    with pl.Config(tbl_rows=16):
        print(piv.select(['dest', 'pre', 'post', 'share_pre', 'share_post', 'd_share'])
              .with_columns(pl.col.share_pre.round(3), pl.col.share_post.round(3),
                            pl.col.d_share.round(3)))

    near_origins = origins_with_pre_flow(od, far=False)
    _, piv_n = destination_shares(od, near_origins)
    print('\nComparison (near origins): destination shares, pre vs post:')
    with pl.Config(tbl_rows=16):
        print(piv_n.select(['dest', 'share_pre', 'share_post', 'd_share'])
              .with_columns(pl.col.share_pre.round(3), pl.col.share_post.round(3),
                            pl.col.d_share.round(3)))

    cmp = (piv.select(['dest', 'd_share'])
           .rename({'d_share': 'd_far'})
           .join(piv_n.select(['dest', 'd_share']).rename({'d_share': 'd_near'}),
                 on='dest', how='full', coalesce=True)
           .fill_null(0).sort('d_far', descending=True))
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    cmp.write_csv(OUT_CSV)
    print(f'\nwrote {OUT_CSV}')
    write_shares_tex(piv, piv_n, treated.height, near_origins.height, OUT_CSV.with_suffix('.tex'))

    # PPML check, far treated origins only: post x destination group, with
    # other old universities as the reference.
    tf = t.filter(pl.col.far_orig).to_pandas()
    tf['post'] = (tf['period'] == 'post').astype(float)
    tf['got'] = (tf['dest'] == 'goettingen').astype(float)
    tf['new'] = tf['dest'].isin(NEW).astype(float)
    tf['got_post'] = tf['got'] * tf['post']
    tf['new_post'] = tf['new'] * tf['post']
    fit = pf.fepois('flow ~ got_post + new_post | origin_id + dest + period',
                    data=tf, vcov={'CRV1': 'origin_id'})
    print('\n===== PPML reallocation (far origins, pre/post x dest group) =====')
    print(fit.tidy().round(4))
    write_ppml_tex(fit, paths.TABLES / 'goettingen7_reallocation_ppml.tex')

    d = cmp.to_pandas()
    set_style()
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    y = np.arange(len(d))
    h = 0.38
    ax.barh(y + h / 2, d['d_far'] * 100, height=h, color=sns.color_palette('deep')[0],
            label='far origins (>=200 km, treated)')
    ax.barh(y - h / 2, d['d_near'] * 100, height=h, color=sns.color_palette('deep')[1],
            label='near origins (<200 km, comparison)')
    ax.set_yticks(y)
    ax.set_yticklabels(d['dest'])
    ax.axvline(0, color='gray', lw=0.8)
    ax.invert_yaxis()
    ax.set_title('Change in destination shares after the Göttingen Seven\n'
                 f'(origins with pre-{PRE[1]} Göttingen flow; {PRE[0]}–{PRE[1]} vs {POST[0]}–{POST[1]})')
    ax.set_xlabel('Change in destination share (percentage points)')
    ax.legend()
    save(fig, paths.FIGURES / 'goettingen7_reallocation.png')


if __name__ == '__main__':
    main()
