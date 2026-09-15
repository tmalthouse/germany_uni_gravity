"""Design constants shared by the gravity analyses, and loaders built on them."""

import numpy as np
import polars as pl

from unis import paths
from unis.geo import haversine_km

ERA_LABELS = {1: '1800s', 2: '1810s', 3: '1820s', 4: '1830s', 5: '1840s'}

NEW_UNIVERSITIES = ['berlin', 'bonn']

# --- Göttingen Seven (§8) -----------------------------------------------------
# King Ernst August dismissed seven professors in December 1837; 1837 is the
# omitted reference year in every event study.
TREAT_YEAR = 1837
# Ends 1847: Berlin, Bonn and Jena have no 1848 records, so including 1848
# changes the comparison set (docs/findings.md §8a).
WINDOW = (1834, 1847)
GOETTINGEN_LAT, GOETTINGEN_LON = 51.533333, 9.933333
FAR_KM = 200
GOETTINGEN_SEVEN_LABEL = 'Göttingen Seven\n(Dec 1837)'


def load_od_year_window(window: tuple[int, int] = WINDOW) -> pl.DataFrame:
    """The year grid restricted to first_year in [window[0], window[1]]."""
    return pl.read_parquet(paths.OD_YEAR).filter(
        (pl.col.first_year >= window[0]) & (pl.col.first_year <= window[1])
    )


def distance_to_goettingen(od: pl.DataFrame) -> np.ndarray:
    """Great-circle km from each row's origin to Göttingen."""
    return haversine_km(GOETTINGEN_LAT, GOETTINGEN_LON, od['lat'].to_numpy(), od['lon'].to_numpy())
