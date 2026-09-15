"""Map of every enrollment spell's hometown, coloured by university.

Points are jittered so that places with many students read as clouds rather
than single dots: Gaussian noise scaled by log(students at that exact
location), with longitude noise widened by 1/cos(latitude) so the cloud is
round on the map. Places with one student get no jitter (log 1 = 0).

students_final.parquet -> output/figures/student_map.html, output/figures/student_map.png
"""

import numpy as np
import plotly.express as px
import polars as pl

from unis import paths

BASE_JITTER = 0.002  # degrees per unit of log(city size)
SEED = 42


def main() -> None:
    df = pl.read_parquet(paths.STUDENTS_FINAL).unique('spell_id')
    rng = np.random.default_rng(seed=SEED)

    lats = df['lat'].to_numpy()
    lons = df['lon'].to_numpy()
    densities = df.with_columns(pl.len().over(['lat', 'lon']).alias('n'))['n'].to_numpy()

    raw_noise_lat = rng.normal(loc=0.0, scale=1.0, size=df.height)
    raw_noise_lon = rng.normal(loc=0.0, scale=1.0, size=df.height)
    scaled_noise_lat = raw_noise_lat * BASE_JITTER * np.log(densities)
    scaled_noise_lon = (raw_noise_lon * BASE_JITTER * np.log(densities)) / np.cos(np.radians(lats))

    jittered = df.with_columns([
        pl.Series('lat', lats + scaled_noise_lat),
        pl.Series('lon', lons + scaled_noise_lon),
    ])

    fig = px.scatter_map(
        jittered,
        lat='lat',
        lon='lon',
        color='school',
        hover_data=['hometown', 'region'],
        zoom=4,
        # Explicit centre: plotly's default is the mean coordinate, which is NaN
        # because ungeocoded students have null lat/lon, so it fell back to (0, 0).
        center=dict(lat=float(np.nanmean(lats)), lon=float(np.nanmean(lons))),
        color_discrete_sequence=px.colors.qualitative.Alphabet,
        category_orders={'school': df['school'].unique().sort().to_list()},
    )
    fig.update_traces(marker=dict(size=2.5, opacity=0.01), selector=dict(type='scattermap'))
    fig.update_layout(legend=dict(itemsizing='constant'), map_style='carto-positron')
    for trace in fig.data:
        trace.marker.opacity = 1.0

    paths.FIGURES.mkdir(parents=True, exist_ok=True)
    html = paths.FIGURES / 'student_map.html'
    fig.write_html(html, include_plotlyjs='cdn')
    print(f'wrote {html}')
    png = paths.FIGURES / 'student_map.png'
    fig.write_image(png, width=1400, height=1200, scale=2)
    print(f'wrote {png}')


if __name__ == '__main__':
    main()
