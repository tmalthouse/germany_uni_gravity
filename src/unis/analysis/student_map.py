"""Map of every enrollment spell's hometown, coloured by university.

Points are jittered so that places with many students read as clouds rather
than single dots: Gaussian noise scaled by log(students at that exact
location), with longitude noise widened by 1/cos(latitude) so the cloud is
round on the map. Places with one student get no jitter (log 1 = 0).

The map opens on Germany with East Prussia; the interactive version can be
panned and zoomed out from there.

students_final.parquet -> output/figures/student_map.html, output/figures/student_map.png
"""

import math

import numpy as np
import plotly.express as px
import polars as pl

from unis import paths

BASE_JITTER = 0.002  # degrees per unit of log(city size)
SEED = 42

# Germany with East Prussia: Aachen to Memel, the Alps to the Danish border.
BOUNDS = dict(west=5.8, east=22.9, south=47.2, north=55.9)
WIDTH, HEIGHT = 1600, 1200  # figure size in pixels
MARGIN = 10
LEGEND_WIDTH = 170  # approximate space the legend takes beside the map
TILE_SIZE = 512  # MapLibre renders the world 512 px wide at zoom 0


def _mercator_y(lat: float) -> float:
    return math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))


def _inverse_mercator_y(y: float) -> float:
    return math.degrees(2 * math.atan(math.exp(y)) - math.pi / 2)


def fit_view(west: float, east: float, south: float, north: float,
             width_px: float, height_px: float) -> tuple[dict, float]:
    """Centre and zoom at which a lon/lat box just fits a map of the given size.

    Web Mercator: at zoom z the world is TILE_SIZE * 2**z pixels across, for
    2 pi of longitude and 2 pi of Mercator y.
    """
    y_south, y_north = _mercator_y(south), _mercator_y(north)
    zoom_x = math.log2(width_px * 2 * math.pi / (TILE_SIZE * math.radians(east - west)))
    zoom_y = math.log2(height_px * 2 * math.pi / (TILE_SIZE * (y_north - y_south)))
    center = dict(lat=_inverse_mercator_y((y_south + y_north) / 2), lon=(west + east) / 2)
    return center, min(zoom_x, zoom_y)


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

    center, zoom = fit_view(**BOUNDS, width_px=WIDTH - 2 * MARGIN - LEGEND_WIDTH,
                            height_px=HEIGHT - 2 * MARGIN)
    fig = px.scatter_map(
        jittered,
        lat='lat',
        lon='lon',
        color='school',
        hover_data=['hometown', 'region'],
        center=center,
        zoom=zoom,
        color_discrete_sequence=px.colors.qualitative.Alphabet,
        category_orders={'school': df['school'].unique().sort().to_list()},
    )
    fig.update_traces(marker=dict(size=2.5, opacity=0.01), selector=dict(type='scattermap'))
    fig.update_layout(
        width=WIDTH, height=HEIGHT,
        margin=dict(l=MARGIN, r=MARGIN, t=MARGIN, b=MARGIN),
        legend=dict(itemsizing='constant'),
        map_style='carto-positron',
    )
    for trace in fig.data:
        trace.marker.opacity = 1.0

    paths.FIGURES.mkdir(parents=True, exist_ok=True)
    html = paths.FIGURES / 'student_map.html'
    fig.write_html(html, include_plotlyjs='cdn')
    print(f'wrote {html}')
    png = paths.FIGURES / 'student_map.png'
    fig.write_image(png, scale=2)
    print(f'wrote {png}')


if __name__ == '__main__':
    main()
