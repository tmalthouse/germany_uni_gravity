import numpy as np
import polars as pl
import pytest

from unis.geo import EARTH_RADIUS_KM, haversine_km


def test_one_degree_of_longitude_on_the_equator():
    assert haversine_km(0.0, 0.0, 0.0, 1.0) == pytest.approx(2 * np.pi * EARTH_RADIUS_KM / 360)


def test_zero_and_symmetric():
    assert haversine_km(51.5, 9.9, 51.5, 9.9) == 0.0
    assert haversine_km(51.5, 9.9, 49.4, 8.7) == pytest.approx(haversine_km(49.4, 8.7, 51.5, 9.9))


def test_numpy_and_polars_agree_exactly():
    lat = np.array([48.1, 52.5, 54.3])
    lon = np.array([11.6, 13.4, 10.1])
    from_numpy = haversine_km(51.533333, 9.933333, lat, lon)
    from_polars = pl.DataFrame({"lat": lat, "lon": lon}).select(
        haversine_km(51.533333, 9.933333, pl.col.lat, pl.col.lon)
    ).to_series().to_numpy()
    np.testing.assert_array_equal(from_numpy, from_polars)
