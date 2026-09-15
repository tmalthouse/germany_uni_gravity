import polars as pl
import pytest

from unis import paths
from unis.registers.common import location_full


def test_location_full_combines_hometown_and_region():
    df = pl.DataFrame({
        "hometown": ["Kassel", "Kassel", None, None],
        "region": ["Hessen", None, "Hessen", None],
    })
    assert df.select(location_full()).to_series().to_list() == [
        "Kassel, Hessen", "Kassel", "Hessen", None,
    ]


@pytest.mark.skipif(not paths.OUT_FINAL.exists(), reason="raw data not imported")
def test_order_manifest_lists_exactly_the_extraction_files():
    listed = set(paths.OUT_FINAL_ORDER.read_text().split())
    present = {str(p.relative_to(paths.OUT_FINAL)) for p in paths.OUT_FINAL.glob("*/*.csv")}
    assert listed == present
