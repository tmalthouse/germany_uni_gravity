import math

import pytest

from unis.analysis.student_map import (
    BASEMAP_OPACITY,
    MAP_SCHOOLS,
    TILE_SIZE,
    UNIVERSITY_COLOURS,
    basemap_layout,
    fit_view,
)
from unis.gravity.constants import UNIVERSITY_NAMES


def test_every_map_university_has_its_own_colour():
    names = {UNIVERSITY_NAMES[code] for code in MAP_SCHOOLS}
    assert set(UNIVERSITY_COLOURS) == names
    assert len(set(UNIVERSITY_COLOURS.values())) == len(names)


def test_jena_and_kiel_avoid_the_pale_palette_entries():
    pale = {"#F7E1A0", "#E2E2E2"}
    assert UNIVERSITY_COLOURS["Jena"] not in pale
    assert UNIVERSITY_COLOURS["Kiel"] not in pale


def test_basemap_falls_back_to_carto_without_a_tile_url():
    assert basemap_layout(None) == {"map_style": "carto-positron"}
    assert basemap_layout("") == {"map_style": "carto-positron"}


def test_historic_basemap_is_a_faded_raster_layer_below_the_points():
    layout = basemap_layout("https://tiles.example/{z}/{x}/{y}.png?key=abc")
    assert layout["map_style"] == "white-bg"
    (layer,) = layout["map_layers"]
    assert layer["sourcetype"] == "raster" and layer["below"] == "traces"
    assert layer["opacity"] == BASEMAP_OPACITY < 1
    assert "Huntington" in layer["sourceattribution"]


def test_whole_world_width_fits_at_zoom_zero():
    # 360 degrees of longitude on a TILE_SIZE-wide map, with height to spare
    center, zoom = fit_view(-180, 180, -10, 10, width_px=TILE_SIZE, height_px=10_000)
    assert zoom == pytest.approx(0.0)
    assert center == pytest.approx(dict(lat=0.0, lon=0.0))


def test_doubling_the_map_size_adds_one_zoom_level():
    _, z1 = fit_view(5.8, 22.9, 47.2, 55.9, width_px=700, height_px=600)
    _, z2 = fit_view(5.8, 22.9, 47.2, 55.9, width_px=1400, height_px=1200)
    assert z2 - z1 == pytest.approx(1.0)


def test_center_is_the_mercator_midpoint():
    center, _ = fit_view(0, 10, 0, 60, width_px=1000, height_px=1000)
    assert center["lon"] == 5
    assert 30 < center["lat"] < 60  # Mercator stretches the north, so above 30
    assert not math.isnan(center["lat"])
