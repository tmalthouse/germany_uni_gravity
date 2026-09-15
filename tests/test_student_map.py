import math

import pytest

from unis.analysis.student_map import TILE_SIZE, fit_view


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
