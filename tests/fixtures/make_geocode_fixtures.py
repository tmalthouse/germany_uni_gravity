"""Build the synthetic geocoder fixture in tests/fixtures/geocode/.

    uv run python tests/fixtures/make_geocode_fixtures.py

Deliberately includes the cases the pipeline is supposed to get right:
  - archaic spellings that only the ladder resolves
  - a homonym where the nearby candidate is obscure and the far one is famous
  - a genuine distant foreign origin
  - a territory name in `region` with a settlement in `hometown`
  - a non-place value
  - the Marburg/Maribor identifier collision
"""

import random
from pathlib import Path

import polars as pl

ROOT = Path("fixtures/data")
(ROOT / "universities/clean").mkdir(parents=True, exist_ok=True)

# subject_id, term(s), lat, lon, placetype
PLACES = [
    (7012330, ["Marburg", "Marburg an der Lahn", "Marpurg"], 50.802, 8.766, "inhabited place"),
    (7005682, ["Hessen"], 50.600, 9.000, "province"),
    (2134102, ["Hessen"], 50.600, 9.000, "province"),   # duplicate id, remapped away
    (7012987, ["Marburg"], 46.554, 15.646, "inhabited place"),          # Maribor, the trap
    (7003712, ["Göttingen", "Goettingen", "Gottingen"], 51.533, 9.936, "inhabited place"),
    (7004334, ["Halle", "Halle an der Saale"], 51.482, 11.970, "inhabited place"),
    (7004434, ["Braunschweig"], 52.269, 10.521, "inhabited place"),
    (7005289, ["Berlin"], 52.520, 13.405, "inhabited place"),
    (7004333, ["Kassel", "Cassel"], 51.313, 9.480, "inhabited place"),
    (7005245, ["Frankfurt am Main"], 50.110, 8.682, "inhabited place"),
    (7004601, ["Frankfurt an der Oder"], 52.347, 14.551, "inhabited place"),
    (7003820, ["Köln", "Koeln", "Cologne"], 50.937, 6.960, "inhabited place"),
    (7013296, ["Heidelberg"], 49.398, 8.672, "inhabited place"),
    (7005560, ["Würzburg", "Wuerzburg"], 49.792, 9.951, "inhabited place"),
        # Near Mecklenburg, FAR from Halle: the coherent answer, which the
    # distance-to-university term alone would reject.
    (7013948, ["Neustadt"], 53.400, 11.300, "village"),
        # Near Halle, FAR from Mecklenburg: the decoy the distance term prefers.
    (7014952, ["Neustadt"], 51.000, 12.000, "inhabited place"),
    (2001000, ["Mecklenburg"], 53.600, 11.700, "historical region"),
    (2001001, ["Schwerin"], 53.629, 11.412, "inhabited place"),
    # The Schwerin trap: a Bochum suburb of the same name, closer to most
    # universities and with no context to rule it out.
    (7666001, ["Bochum"], 51.481, 7.216, "inhabited place"),
    (7666002, ["Schwerin"], 51.470, 7.300, "village"),
    (7666003, ["Mecklenburg-Vorpommern"], 53.800, 12.500, "state"),
    # The duchy itself, which is what territory_mode="self" resolves to.
    (2001002, ["Mecklenburg-Schwerin"], 53.600, 11.500, "duchy"),
    (7666004, ["Nordrhein-Westfalen"], 51.400, 7.500, "state"),
    (7013962, ["Boston"], 42.358, -71.060, "inhabited place"),          # far, prominent
    (7010101, ["Boston"], 52.978, -0.026, "village"),                   # far, obscure
    (7007567, ["Philadelphia"], 39.952, -75.165, "inhabited place"),
    (7003330, ["München", "Muenchen"], 48.137, 11.575, "inhabited place"),
    (7004044, ["Ohrdruf"], 50.828, 10.735, "village"),
    (7004045, ["Bad Kissingen"], 50.198, 10.077, "inhabited place"),
    # Phonetic trap: shares a Koelner key with "Mittelmark", 18,000 km away.
    (7654321, ["Middlemarch"], -45.512, 170.103, "village"),
    # The same city held twice by TGN, ~4 km apart: city and province.
    (1002450, ["Hannover"], 52.374, 9.738, "inhabited place"),
    (7013260, ["Hannover"], 52.404, 9.702, "province"),
    (7009999, ["Pommern"], 53.800, 15.200, "historical region"),
    (7005421, ["Potsdam"], 52.396, 13.058, "inhabited place"),
    # The Koblenz case: a real city far from Berlin, and a same-named hamlet
    # much closer. Nothing but prominence and the Ortsteil structure separates
    # them.
    (7005588, ["Koblenz"], 50.356, 7.588, "inhabited place"),
    (7777001, ["Wittichenau"], 51.386, 14.245, "inhabited place"),
    (7777002, ["Koblenz"], 51.360, 14.280, "village"),
    (7008888, ["Königsberg", "Koenigsberg"], 54.710, 20.452, "inhabited place"),
    # Present under its archaic name only; "Rheinprovinz" is absent, so the
    # correction pointing there is a dead end.
    (7007777, ["Rheinpreußen"], 50.700, 7.100, "province"),
    # Present under its archaic name only. corrections.csv sends it to
    # Stralsund, which is absent here, so the correction is a dead end and the
    # raw string has to be retried.
    (7007778, ["Schwedisch Pommern"], 54.000, 13.500, "historical region"),
]

# Hierarchy: SUBJECT_ID -> preferred parent. Countries sit under a world root.
COUNTRIES = [
    (9000001, "Deutschland", 51.0, 10.0, "nation"),
    (9000002, "New Zealand", -41.0, 174.0, "nation"),
    (9000003, "United States", 39.0, -98.0, "nation"),
    (9000004, "United Kingdom", 54.0, -2.0, "nation"),
]
PLACES = PLACES + [(sid, [name], la, lo, pt) for sid, name, la, lo, pt in COUNTRIES]

PARENT = {
    9000001: 1000000, 9000002: 1000000, 9000003: 1000000, 9000004: 1000000,
    # The hamlet is an Ortsteil of Wittichenau; the city answers to a region.
    7777002: 7777001, 7777001: 9000001, 7005588: 9000001,
    7666003: 9000001, 7666004: 9000001, 2001002: 7666003,
    7003820: 7666004,  # Koeln under Nordrhein-Westfalen
    7666001: 7666004, 7666002: 7666001,
    # Hannover the city sits inside Hannover the province: the collapse should
    # use that relation, not the 4 km between their coordinates.
    7013260: 9000001, 1002450: 7013260,
    # Schwerin -> Mecklenburg -> Mecklenburg-Vorpommern, so the curated
    # context on the correction can actually be satisfied.
    2001000: 7666003, 2001001: 2001000,
    # One Neustadt is genuinely inside Mecklenburg, the other is not.
    7013948: 2001000, 7014952: 9000001,
    7654321: 9000002, 7013962: 9000003, 7007567: 9000003, 7010101: 9000004,
}
for _sid, *_ in PLACES:
    PARENT.setdefault(_sid, 9000001)

# A physical feature and a merged record, both of which must be filtered out.
EXTRA_SUBJECTS = [
    (8000001, ["Heidelberg"], 49.410, 8.710, "mountain", "P", "N"),
    (8000002, ["Göttingen"], 10.000, 10.000, "inhabited place", "A", "M"),
]

term_rows = []
for sid, terms, lat, lon, ptype in PLACES:
    for i, t in enumerate(terms):
        term_rows.append(
            {"TERM": t, "TERM_ID": sid * 10 + i, "SUBJECT_ID": sid, "PLACETYPE": ptype}
        )
    # A prominence signal: give the famous places extra alias rows.
    if sid in (7005289, 7003820, 7013962, 7007567, 7005245, 7003330, 7005588):
        for j in range(6):
            term_rows.append(
                {
                    "TERM": f"{terms[0]}_alias{j}",
                    "TERM_ID": sid * 10 + 100 + j,
                    "SUBJECT_ID": sid,
                    "PLACETYPE": ptype,
                }
            )

for sid, terms, lat, lon, ptype, _rt, _ms in EXTRA_SUBJECTS:
    for i, t in enumerate(terms):
        term_rows.append({"TERM": t, "TERM_ID": sid * 10 + i, "SUBJECT_ID": sid,
                          "PLACETYPE": ptype})

# A machine-code term of the kind that was matching truncated strings.
term_rows.append({"TERM": "DEU", "TERM_ID": 90000011, "SUBJECT_ID": 9000001,
                  "PLACETYPE": "nation"})

# Flags are written PADDED, exactly as the relational release stores them.
# Anything that compares a flag without stripping will fail the suite.
terms_df = pl.DataFrame(term_rows).with_columns(
    PREFERRED=pl.when(pl.col("TERM_ID") % 10 == 0).then(pl.lit("P ")).otherwise(pl.lit("V ")),
    VERNACULAR=pl.lit("U "),
    HISTORIC_FLAG=pl.lit("C "),
    OTHER_FLAGS=pl.when(pl.col("TERM") == "DEU").then(pl.lit("ISO3L")).otherwise(pl.lit("NA")),
)
terms_df.write_parquet(ROOT / "tgn_terms.parquet")

all_subjects = [(s_, la, lo, pt, "A", "N") for s_, _, la, lo, pt in PLACES] + [
    (s_, la, lo, pt, rt, ms) for s_, _, la, lo, pt, rt, ms in EXTRA_SUBJECTS
]
pl.DataFrame(
    [{"SUBJECT_ID": s_, "PARENT_KEY": PARENT.get(s_, 9000001),
      "RECORD_TYPE": rt + " ", "MERGED_STAT": ms + " "}
     for s_, _la, _lo, _pt, rt, ms in all_subjects]
).write_parquet(ROOT / "tgn_subjects.parquet")

ptype_rows = [
    {"SUBJECT_ID": s_, "PTYPE_ROLE_ID": 1, "PREFERRED": "P ", "HISTORIC_FLAG": "C ",
     "START_DATE": None, "END_DATE": None, "DISPLAY_ORDER": 1, "PLACETYPE": pt}
    for s_, _la, _lo, pt, _rt, _ms in all_subjects
]
# Cities accumulate extra roles; hamlets do not.
for s_, extra in [(7005588, "port"), (7005588, "spa"), (7005289, "national capital")]:
    ptype_rows.append(
        {"SUBJECT_ID": s_, "PTYPE_ROLE_ID": 2 if extra == "port" else 3,
         "PREFERRED": "N ", "HISTORIC_FLAG": "C ", "START_DATE": None,
         "END_DATE": None, "DISPLAY_ORDER": 2, "PLACETYPE": extra}
    )
pl.DataFrame(ptype_rows).write_parquet(ROOT / "tgn_placetypes.parquet")

# Languages, sources and scope notes: the remaining prominence components.
lang_rows = []
for sid, terms, *_ in PLACES:
    n_lang = 5 if sid in (7005588, 7005289, 7003820, 7013962, 7007567) else 1
    for j in range(n_lang):
        lang_rows.append({"SUBJECT_ID": sid, "TERM_ID": sid * 10,
                          "LANGUAGE_CODE": f"L{j}", "LANG_PREFERRED": "P",
                          "TERM_TYPE": "D"})
pl.DataFrame(lang_rows).write_parquet(ROOT / "tgn_languages.parquet")

pl.DataFrame(
    [{"SUBJECT_ID": sid,
      "n_sources": 9 if sid in (7005588, 7005289, 7013962, 7007567) else 1}
     for sid, *_ in PLACES]
).write_parquet(ROOT / "tgn_sources.parquet")

pl.DataFrame(
    [{"SUBJECT_ID": sid, "n_notes": 1}
     for sid in (7005588, 7005289, 7003820, 7013962, 7007567)]
).write_parquet(ROOT / "tgn_scope_notes.parquet")

pl.DataFrame(
    [{"PARENT_ID": PARENT.get(s_, 9000001), "CHILD_ID": s_, "PREFERRED": "P ",
      "HIER_REL_TYPE": "P ", "HISTORIC_FLAG": "C ", "START_DATE": None, "END_DATE": None}
     for s_, _la, _lo, _pt, _rt, _ms in all_subjects]
).write_parquet(ROOT / "tgn_rels.parquet")

# Cities get a bounding box; hamlets get a bare point, exactly as TGN records
# them. Having a box at all is a prominence signal before you measure it.
BOXED = {7005588, 7005289, 7003820, 7013962, 7007567, 7005245, 7003330, 7013296}


def _coord_row(sid, lat, lon):
    row = {"SUBJECT_ID": sid, "LAT": lat, "LON": lon}
    half = 0.12 if sid in BOXED else None
    row["LAT_MIN_BOX"] = lat - half if half else None
    row["LAT_MAX_BOX"] = lat + half if half else None
    row["LON_MIN_BOX"] = lon - half if half else None
    row["LON_MAX_BOX"] = lon + half if half else None
    return row


pl.DataFrame(
    [_coord_row(p[0], p[2], p[3]) for p in PLACES]
    + [_coord_row(s_, la, lo) for s_, _t, la, lo, _pt, _rt, _ms in EXTRA_SUBJECTS]
).write_parquet(ROOT / "tgn_coords.parquet")

pl.DataFrame(
    [
        {"school": "berlin", "SUBJECT_ID": 7005289},
        {"school": "goettingen", "SUBJECT_ID": 7003712},
        {"school": "halle", "SUBJECT_ID": 7004334},
        {"school": "marburg", "SUBJECT_ID": 7012330},
        {"school": "muenchen", "SUBJECT_ID": 7003330},
        {"school": "muenchen_old", "SUBJECT_ID": 7003330},
    ]
).write_csv(ROOT / "uni_tgn_mapping.csv")

CASES = [
    # region, hometown, school, first_year, note
    ("Cassel", None, "goettingen", 1830, "archaic c/k, should reach Kassel"),
    ("Coeln", None, "goettingen", 1835, "archaic, should reach Köln"),
    ("Wirceburgensis", None, "halle", 1810, "Latin form, should reach Würzburg"),
    ("Ohrdruff", None, "halle", 1840, "doubled consonant"),
    ("Kissingen", None, "halle", 1845, "later Bad prefix"),
    ("Marburg", None, "goettingen", 1820, "must NOT go to Maribor"),
    ("Marpurg", None, "goettingen", 1822, "archaic Marburg, ladder-only"),
    ("Cöln", None, "halle", 1836, "uncorrected archaic, ladder must handle"),
    ("Muenchen", None, "halle", 1870, "ue digraph"),
    ("Wuerzburg", None, "halle", 1872, "ue digraph"),
    ("Gottingn", None, "halle", 1874, "typo, fuzzy fallback"),
    # Consonant substitutions, which defeat Koelner Phonetik (it ignores
    # vowels), so these are the cases that genuinely reach the fuzzy stage.
    ("Heidelerg", None, "halle", 1876, "fuzzy; left string SHORTER than the term"),
    ("Wuerzburgs", None, "halle", 1878, "fuzzy; left string LONGER than the term"),
    ("Mittelmark", None, "halle", 1882, "territory -> seat via correction"),
    ("Mittelmarck", None, "halle", 1883, "phonetic trap; gate must reject New Zealand"),
    ("Hannover", None, "halle", 1884, "coincident TGN duplicates, school A"),
    ("Hannover", None, "goettingen", 1886, "coincident TGN duplicates, school B"),
    ("Rheinpreußen", None, "halle", 1888, "territory -> seat, correction now live"),
    ("Schwedisch Pommern", None, "halle", 1889, "dead correction; must fall back to raw"),
    ("Herz. Braunschweig", None, "halle", 1890, "abbreviated administrative prefix"),
    ("in Pommern", None, "halle", 1892, "leading preposition"),
    ("Königsberg i. Pr.", None, "halle", 1894, "trailing abbreviation expanded"),
    ("Frankfurth a. M.", None, "halle", 1896, "archaic + abbreviation together"),
    ("Koblenz", None, "berlin", 1898, "prominent city beats a nearer same-named hamlet"),
    ("Boston", None, "goettingen", 1860, "prominent distant city beats nearby hamlet"),
    ("Philadelphia", None, "goettingen", 1865, "genuine distant origin"),
    ("Mecklenburg", "Neustadt", "halle", 1850, "territory + settlement, coherence should pick"),
    ("Mecklenburg-Schwerin", None, "halle", 1852, "territory -> seat, context must pick the right Schwerin"),
    ("Frankfurt a. M.", None, "marburg", 1830, "abbreviation"),
    ("Frankfurt a. d. O.", None, "halle", 1832, "the other Frankfurt"),
    ("Privatunterricht", None, "halle", 1840, "not a place, must stay null"),
    (None, "Heidelberg", "halle", 1841, "hometown fallback"),
    ("Gottingen", None, "muenchen", 1820, "pre-1826 Landshut: school_fixed -> muenchen_old"),
    ("Goettingen", None, "muenchen", 1830, "post-1826 Munich: school_fixed stays muenchen"),
    ("Zzzznotaplace", None, "halle", 1850, "unresolvable"),
    (None, None, "halle", 1850, "both fields null"),
]

random.seed(0)
rows = []
for i in range(600):
    region, hometown, school, year, note = CASES[i % len(CASES)]
    rows.append(
        {
            "student_id": f"S{i:05d}",
            "region": region,
            "hometown": hometown,
            "school": school,
            "first_year": year + (i % 5),
            "faculty": random.choice(["law", "medicine", "theology"]),
            "_note": note,
        }
    )

pl.DataFrame(rows).write_parquet(ROOT / "universities/clean/all_students_unlinked.parquet")
print(f"fixtures written to {ROOT.resolve()}")
