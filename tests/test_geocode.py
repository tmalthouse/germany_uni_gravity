"""Behavioural expectations for the geocoder over the synthetic fixture.

These assert the substantive behaviour the geocoder exists to deliver, not
just that its contract checks pass. The fixture is built by
tests/fixtures/make_geocode_fixtures.py; tests run on a temporary copy because
the pipeline writes its cache next to its inputs.
"""

import shutil
from pathlib import Path

import polars as pl
import pytest

from unis.geocode import Config, Paths, candidates, resolve, run, tgn_index
from unis.geocode.candidates import _jaro_filter
from unis.geocode.config import Tuning
from unis.geocode.distance import bin_geometry, fit_prior
from unis.geocode.normalize import fold_expr, koelner_phonetik, strip_qualifiers

FIXTURE = Path(__file__).parent / "fixtures" / "geocode"


@pytest.fixture(scope="module")
def fixture_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("geocode")
    shutil.copytree(FIXTURE, root, dirs_exist_ok=True)
    return root


@pytest.fixture(scope="module")
def cfg(fixture_root):
    return Config(paths=Paths.under(fixture_root), use_cache=False, verbose=False)


@pytest.fixture(scope="module")
def out(cfg):
    return run(cfg, write=False)


def resolved(out, region=None, hometown=None):
    f = out.filter(pl.col("region") == region) if region is not None else out.filter(
        pl.col("region").is_null()
    )
    if hometown is not None:
        f = f.filter(pl.col("hometown") == hometown)
    return f.row(0, named=True) if f.height else None


# --- normalization units ------------------------------------------------------

def test_koelner_variants_agree():
    assert koelner_phonetik("marburg") == koelner_phonetik("marpurg")
    assert koelner_phonetik("cassel") == koelner_phonetik("kassel")


def test_strip_qualifiers_returns_the_hint():
    assert strip_qualifiers("herzogtum bremen") == ("bremen", "herzogtum")
    assert strip_qualifiers("greifenberg in schlesien") == ("greifenberg", "schlesien")


def test_fold_expands_umlauts():
    assert pl.select(fold_expr(pl.lit("Köln"))).item() == "koeln"


def test_phonetic_collision_guard():
    # Koelner codes are short and collide freely: Potsdam and Boston share one.
    assert koelner_phonetik("potsdam") == koelner_phonetik("boston")
    probe = pl.DataFrame({
        "src_raw": ["a", "b"],
        "_ls": ["potsdam", "marburg"],
        "_rs": ["boston", "marpurg"],
    })
    assert _jaro_filter(probe, "_ls", "_rs", 0.75)["src_raw"].to_list() == ["b"]


# --- resolution behaviour -----------------------------------------------------

@pytest.mark.parametrize("region, hometown, location_id, why", [
    ("Marburg", None, 7012330, "Marburg resolves to Hessen, not Maribor"),
    ("Marpurg", None, 7012330, "archaic spelling reaches Marburg without a correction"),
    ("Cöln", None, 7003820, "uncorrected archaic Coeln reaches Koeln via the ladder"),
    ("Boston", None, 7013962, "prominent distant Boston beats the nearby hamlet"),
    ("Philadelphia", None, 7007567, "genuine distant origin survives"),
    ("Privatunterricht", None, None, "non-place stays unresolved"),
    ("Zzzznotaplace", None, None, "unresolvable string stays unresolved"),
    ("Mecklenburg-Schwerin", None, 2001001, "territory maps to its seat, not a namesake suburb"),
    ("Frankfurt a. M.", None, 7005245, "Frankfurt am Main"),
    ("Frankfurt a. d. O.", None, 7004601, "Frankfurt an der Oder"),
    ("Gottingn", None, 7003712, "typo recovered by a fallback rung"),
    ("Mittelmarck", None, None, "phonetic match to a far homonym is gated out"),
    ("Mittelmark", None, 7005421, "territory resolves to its seat via correction"),
    ("Schwedisch Pommern", None, 7007778, "dead correction falls back to the raw string"),
    ("Rheinpreußen", None, 7005588, "live territory correction resolves to the seat"),
    ("Herz. Braunschweig", None, 7004434, "abbreviated administrative prefix"),
    ("in Pommern", None, 7009999, "leading preposition"),
    ("Königsberg i. Pr.", None, 7008888, "trailing abbreviation expanded"),
    ("Frankfurth a. M.", None, 7005245, "archaic spelling + abbreviation"),
    (None, "Heidelberg", 7013296, "physical feature with a colliding name is excluded"),
    ("Mecklenburg", "Neustadt", 7013948, "nesting/coherence pick the Neustadt in Mecklenburg"),
])
def test_resolution(out, region, hometown, location_id, why):
    r = resolved(out, region, hometown)
    assert r is not None, why
    assert r["location_id"] == location_id, why


def test_marburg_is_in_hessen(out):
    assert resolved(out, "Marburg")["lat"] > 50


def test_distant_origins_keep_their_distance(out):
    assert resolved(out, "Philadelphia")["dist"] > 5000
    assert resolved(out, "Koblenz")["location_id"] == 7005588  # city, not the nearer hamlet
    assert resolved(out, "Koblenz")["dist"] > 400


def test_matched_field(out):
    assert resolved(out, None, "Heidelberg")["matched"] == 2  # hometown fallback
    assert resolved(out, "Mecklenburg", "Neustadt")["matched"] == 2  # territory defers to settlement


@pytest.mark.parametrize("region, location_id, why", [
    ("Heidelerg", 7013296, "left string shorter than the gazetteer term"),
    ("Wuerzburgs", 7005560, "left string longer than the gazetteer term"),
])
def test_fuzzy_stage(out, region, location_id, why):
    # Exercises the DuckDB length window in both directions: an unsigned
    # length column underflows on the shorter-left case specifically.
    r = resolved(out, region)
    assert (r["location_id"], r["match_method"]) == (location_id, "fuzzy"), why


def test_coincident_duplicates_collapse_to_the_city(out):
    han = out.filter(pl.col("region") == "Hannover")
    assert han["location_id"].unique().to_list() == [1002450]
    assert han.row(0, named=True)["is_settlement"]


def test_merged_record_is_excluded(out):
    assert out.filter(pl.col("location_id") == 8000002).height == 0


def test_place_types_populate_geo_precision(out):
    known = out.filter(pl.col("location_id").is_not_null()).select(
        (pl.col("geo_precision") != "unknown").mean()
    ).item()
    assert known > 0.9


def test_no_resolved_row_lacks_coordinates(out):
    assert out.filter(pl.col("location_id").is_not_null() & pl.col("lat").is_null()).height == 0


def test_deterministic(cfg, out):
    assert out.equals(run(cfg, write=False))


# --- distance prior -----------------------------------------------------------

def test_bin_widths_are_sane(cfg):
    geo = bin_geometry(cfg)
    assert geo["width_km"].max() < 21_000
    assert geo["width_km"][-1] > 1_000


def test_prior_compares_density_not_mass(cfg):
    # Log-spaced bins differ in width by two orders of magnitude, so scoring
    # mass instead of density would make a 12 km candidate lose to a 335 km one.
    fit = pl.DataFrame({"school_fixed": ["s"] * 200, "dist": [10.0] * 100 + [400.0] * 100})
    prior = fit_prior(fit, cfg)
    near = prior.filter(pl.col("dist_bin") == 3)["log_density"][0]
    far = prior.filter(pl.col("dist_bin") == 8)["log_density"][0]
    assert near > far


# --- territories --------------------------------------------------------------

def test_territory_mode_self(fixture_root, out):
    self_cfg = Config(paths=Paths.under(fixture_root), tuning=Tuning(territory_mode="self"),
                      use_cache=False, verbose=False)
    self_out = run(self_cfg, write=False)
    assert resolved(self_out, "Mecklenburg-Schwerin")["location_id"] == 2001002
    assert resolved(self_out, "Cassel")["location_id"] == resolved(out, "Cassel")["location_id"]


def test_context_constraint_removes_the_decoy(cfg):
    idx = tgn_index.build(cfg)
    cand = candidates.build(resolve.load_students(cfg).lazy(), idx,
                            resolve.load_university_coords(cfg), cfg)
    ids = cand.filter(pl.col("src_raw") == "Mecklenburg-Schwerin")["SUBJECT_ID"].to_list()
    assert 7666002 not in ids and 2001001 in ids


# --- prominence ---------------------------------------------------------------

def test_prominence_separates_city_from_hamlet(cfg, out):
    idx = pl.read_parquet(cfg.paths.tgn_index)
    comp = idx.filter(pl.col("SUBJECT_ID").is_in([7005588, 7777002])).select(
        "SUBJECT_ID", "prominence", "is_sub_settlement"
    ).unique()
    city = comp.filter(pl.col("SUBJECT_ID") == 7005588).row(0, named=True)
    hamlet = comp.filter(pl.col("SUBJECT_ID") == 7777002).row(0, named=True)
    assert city["prominence"] > hamlet["prominence"] + 3
    assert hamlet["is_sub_settlement"] and not city["is_sub_settlement"]


@pytest.mark.parametrize("column", ["n_terms", "n_languages", "n_children", "n_placetypes",
                                    "n_sources"])
def test_prominence_components_vary(cfg, out, column):
    subs = pl.read_parquet(cfg.paths.tgn_index).unique(subset=["SUBJECT_ID"])
    assert (subs[column].fill_null(0) > 1).any()


def test_bounding_boxes_present(cfg, out):
    subs = pl.read_parquet(cfg.paths.tgn_index).unique(subset=["SUBJECT_ID"])
    assert subs["bbox_area"].drop_nulls().len() > 0
