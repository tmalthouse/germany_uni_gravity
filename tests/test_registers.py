import polars as pl
import pytest

from unis import paths
from unis.registers import berlin, bonn, jena, tuebingen
from unis.registers.common import location_full

GROUPS = ["law_admin", "theology", "medicine", "sciences", "humanities"]


@pytest.mark.parametrize("module, field, group", [
    (berlin, "Theol.", "theology"),
    (berlin, "Kam.", "law_admin"),
    (bonn, "kath. Theologie", "theology"),
    (bonn, "Jura u. Cameral.", "law_admin"),
    (jena, "M.", "medicine"),
    (tuebingen, "Rechtswissenschaft", "law_admin"),
    (tuebingen, "Philosophie", "humanities"),
])
def test_field_flags_assign_one_group(module, field, group):
    row = pl.DataFrame({"field": [field]}).select(**module.field_flags()).row(0, named=True)
    assert [g for g in GROUPS if row[g]] == [group]


def test_berlin_unknown_abbreviation_gets_no_group():
    row = pl.DataFrame({"field": ["s."]}).select(**berlin.field_flags()).row(0, named=True)
    assert not any(row[g] for g in GROUPS)


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
