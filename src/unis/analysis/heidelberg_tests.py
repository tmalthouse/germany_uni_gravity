"""Discriminating tests for the Heidelberg religion story (findings §4-5 caveats).

Test A (reputation vs proximity over time): does the Protestant long-distance
share (>300 km) peak in the 1810s-20s or stay flat?

Test B (field composition): are long-distance Protestants concentrated in
law/philology (reputation-driven) vs theology (seminary/proximity story)?

students_final.parquet -> output/figures/heidelberg_religion_tests.png
"""

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from unis import paths
from unis.geo import haversine_km
from unis.gravity.constants import ERA_LABELS
from unis.gravity.heidelberg import RELIGION_COLORS, religion_sample, seat
from unis.plotting import save, set_style

FAR_KM = 300


def load() -> pl.DataFrame:
    df = religion_sample()
    hd_lat, hd_lon = seat(df)
    return df.filter(pl.col.lat.is_not_null()).with_columns(
        dist_km=haversine_km(hd_lat, hd_lon, pl.col.lat, pl.col.lon),
    ).with_columns(
        far=pl.col.dist_km > FAR_KM,
    )


def main() -> None:
    df = load()

    # --- Test A: far share by era and religion ---------------------------
    a = (df.group_by(['era', 'rel'])
         .agg(pl.col.far.mean().alias('share_far'), pl.len())
         .sort(['era', 'rel']).to_pandas())
    a['period'] = a['era'].map(ERA_LABELS)
    print('share >300km by era x religion:')
    print(a.pivot(index='era', columns='rel', values='share_far').round(3))

    # --- Test B: field composition, far vs near, within religion ---------
    p = df.filter((pl.col.rel == 'protestant') & pl.col.field.is_not_null())
    # Field shares WITHIN near and WITHIN far Protestants: each near/far column
    # sums to 1 over fields. Columns are selected by name because pivot orders
    # them by first appearance.
    t = (p.group_by(['far', 'field']).agg(pl.len())
         .pivot(on='far', index='field', values='len').fill_null(0)
         .select('field', 'false', 'true'))
    t = t.with_columns(pl.exclude('field') / pl.exclude('field').sum())
    print('\nProtestants: field shares, near vs far:')
    print(t)

    far_protestants = p.filter(pl.col.far & pl.col.field.is_not_null())
    t2 = (far_protestants.group_by(['era', 'field']).agg(pl.len())
          .pivot(on='field', index='era', values='len').fill_null(0))
    t2 = t2.with_columns(pl.exclude('era').truediv(pl.sum_horizontal(pl.exclude('era'))))
    print('\nFar Protestants: field shares by era:')
    print(t2)
    print('\nfar-Protestant counts per era:',
          far_protestants.group_by('era').agg(pl.len()).sort('era').to_dicts())

    print('\nTop origins of far Protestants:',
          df.filter((pl.col.rel == 'protestant') & pl.col.far)
          ['standard_location_name'].value_counts().sort('count', descending=True)
          .head(8).to_dicts())

    # --- Plot -------------------------------------------------------------
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    a_p = a[a['rel'] != 'jewish']
    sns.lineplot(data=a_p, x='period', y='share_far', hue='rel', marker='o', ax=axes[0],
                 palette={k: RELIGION_COLORS[k] for k in ('catholic', 'protestant')})
    axes[0].set_title('Share of Heidelberg enrollees from >300 km, by decade')
    axes[0].set_ylabel('Share from >300 km')
    axes[0].set_xlabel('')

    t_pd = (t.rename({'false': 'near (<300 km)', 'true': 'far (>300 km)'})
             .to_pandas().set_index('field').T)
    t_pd.plot(kind='bar', stacked=True, ax=axes[1], colormap='tab20', width=0.7)
    axes[1].set_title('Field composition, near vs far Protestants')
    axes[1].set_ylabel('Field share')
    axes[1].legend(fontsize=7, ncol=2)
    axes[1].set_xlabel('')

    save(fig, paths.FIGURES / 'heidelberg_religion_tests.png')


if __name__ == '__main__':
    main()
