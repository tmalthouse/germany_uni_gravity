"""Robustness: student-level religion at Heidelberg (findings §4-5 caveats).

Heidelberg is the only destination with religion recorded, so a four-way state
x confession split is not estimable. Instead: among Heidelberg enrollees, does
origin composition differ by student confession?

1. Descriptive: distance distribution and Baden-origin share by religion.
2. PPML on (origin polity x religion) counts, Catholic and Protestant only:
       flow ~ log_dist + baden + catholic x baden | religion
   HC-robust SEs, with SEs clustered by origin polity printed as a check.

Jewish students are reported descriptively but excluded from the regression.

students_final.parquet -> output/figures/heidelberg_religion.png
"""

import matplotlib.pyplot as plt
import polars as pl
import pyfixest as pf
import seaborn as sns

from unis import paths
from unis.geo import haversine_km
from unis.gravity.heidelberg import RELIGION_COLORS, religion_sample, seat
from unis.plotting import save, set_style


def main() -> None:
    df = religion_sample()
    print(f'Heidelberg enrollees with classified religion: {df.height}')
    print(df['rel'].value_counts())

    # --- 1. Descriptives -------------------------------------------------
    desc = df.group_by('rel').agg(
        pl.col.lat.is_not_null().sum().alias('n_geocoded'),
        pl.col.standard_location_name.eq('Baden').mean().alias('share_baden'),
        pl.len().alias('n'),
    )
    print(desc)

    hd_lat, hd_lon = seat(df)
    dfp = df.filter(pl.col.lat.is_not_null()).with_columns(
        dist_km=haversine_km(hd_lat, hd_lon, pl.col.lat, pl.col.lon)
    )
    plot_df = dfp.select(['rel', 'dist_km']).to_pandas()

    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    sns.kdeplot(data=plot_df, x='dist_km', hue='rel', common_norm=False,
                fill=True, alpha=0.25, ax=axes[0], palette=RELIGION_COLORS, clip=(0, None))
    axes[0].set_title('Distance to Heidelberg, by student religion')
    axes[0].set_xlabel('Distance (km)')
    axes[0].set_ylabel('Density')
    axes[0].set_xlim(0, 800)

    share = dfp.group_by('rel').agg(pl.col.standard_location_name.eq('Baden').mean(),
                                    pl.len()).to_pandas()
    sns.barplot(data=share, x='rel', y='standard_location_name', ax=axes[1],
                palette=RELIGION_COLORS, hue='rel', legend=False)
    axes[1].set_title('Share of enrollees from Baden')
    axes[1].set_xlabel('')
    axes[1].set_ylabel('Share from Baden')
    save(fig, paths.FIGURES / 'heidelberg_religion.png')

    # --- 2. PPML: origin-polity x religion counts ------------------------
    cells = (df.filter(pl.col.rel.is_in(['catholic', 'protestant']))
             .group_by('standard_location_name', 'rel').agg(
                 pl.col.lat.first(), pl.col.lon.first(), pl.len().alias('flow'),
             ).filter(pl.col.lat.is_not_null()))
    cells = cells.with_columns(
        log_dist=haversine_km(hd_lat, hd_lon, pl.col.lat, pl.col.lon).log(),
        baden=(pl.col.standard_location_name == 'Baden'),
    ).with_columns(
        inter=(pl.col.rel == 'catholic') & (pl.col.standard_location_name == 'Baden'),
    )
    print('cells:', cells.height, '| total flow:', cells['flow'].sum())

    # The religion FE absorbs the Catholic/Protestant level, so there is no
    # separate rel_catholic term (it would be exactly collinear).
    fml = 'flow ~ log_dist + baden + inter | rel'
    fit = pf.fepois(fml, data=cells.to_pandas(), vcov='hetero')
    print(fit.tidy().round(4))
    fit_c = pf.fepois(fml, data=cells.to_pandas(), vcov={'CRV1': 'standard_location_name'})
    print('\nsame model, SEs clustered by origin polity:')
    print(fit_c.tidy().round(4))


if __name__ == '__main__':
    main()
