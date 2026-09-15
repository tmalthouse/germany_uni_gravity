# unis: German university registers, 1800–1848

Cleaning, geocoding, record linkage and gravity analyses of student enrollment
at 13 German universities, 1800–1848. Results, caveats and the history of
corrections are in [`docs/findings.md`](docs/findings.md).

This project ports the live parts of the original `~/Dropbox/1848_unis`:
the cleaning chain that builds `students_final.parquet`, the analyses in
`analysis/gravity/`, and `analysis/map_test.py`.

## Setup

You need [uv](https://docs.astral.sh/uv/) (0.5 or newer) and make (the macOS
system make works). The static map export uses Chrome through kaleido; if it
is missing, run `uv run plotly_get_chrome` once.

```bash
uv sync               # .venv from uv.lock (Python 3.14)
make import-raw       # once: copy sources from ~/Dropbox/1848_unis (override with RAW_SRC=...)
make check-raw        # verify data/raw against data/raw_manifest.sha256
make                  # build everything
make test             # unit tests
```

Make rebuilds only what is out of date: editing an analysis module reruns that
analysis, editing a register cleaner reruns everything downstream of it. Useful
partial targets are `make data`, `make tables`, `make figures` and `make map`.
Console output from every step is kept in `output/logs/`.

## Layout

```
data/raw/         frozen sources, copied once (gitignored; checksums in data/raw_manifest.sha256)
data/manual/      small hand-curated inputs (tracked)
data/interim/     pipeline intermediates (rebuilt)
data/processed/   analysis datasets: students_final, od, od_year (rebuilt)
output/           tables/ (CSV), figures/ (PNG, map HTML), logs/ (rebuilt)
src/unis/         the package: one module per pipeline stage or published output
tests/            pytest suite, plus the synthetic geocoder fixture and its generator
docs/findings.md  research log
```

## Pipeline

| Stage | Module | Reads | Writes |
|---|---|---|---|
| TGN load | `unis.tgn.load` | `raw/tgn_rel_0126/*.out` | `interim/tgn/tgn_*.parquet` |
| Heidelberg page years | `unis.registers.heidelberg_dates` | `manual/heidelberg_heads/*.json` | `interim/heidelberg_date_fixer.parquet` |
| Registers | `unis.registers.combine` (one module per school) | `raw/universities/`, `manual/out_final_order.txt`, `manual/munich_date_helper.csv` | `interim/registers/*.parquet`, `interim/all_students_unlinked.parquet` |
| Geocoding | `unis.geocode` | unlinked students, TGN tables, `manual/uni_tgn_mapping.csv`, `manual/geocode/*.csv` | `interim/students_with_tgn.parquet` (+ review queue in `interim/geocode_cache/`) |
| Hometown polity | `unis.polity.hometowns` | GHGIS layers, `manual/guild_dates_ogilvie.csv` | `interim/students_geocoded.parquet` |
| Linkage | `unis.linkage.linkage` | geocoded students | `interim/students_linked.parquet` |
| Derived attributes | `unis.finalize` | geocoded + linked | `interim/students_clean.parquet` |
| University polity | `unis.polity.universities` | GHGIS layers, TGN coordinates | `processed/students_final.parquet` |
| OD grids | `unis.gravity.build_od` | `students_final` | `processed/od.parquet`, `processed/od_year.parquet` |
| Analyses | `unis.analysis.*` | OD grids, `students_final` | `output/tables/`, `output/figures/` |

Shared helpers: `unis.paths` (every file location), `unis.geo` (great-circle
distance), `unis.plotting` (figure style), `unis.gravity.constants` (design
constants), `unis.gravity.eventstudy` (the single-destination event study used
by the Göttingen Seven analyses), `unis.gravity.heidelberg` (religion sample).

## Sources

Everything under `data/raw/` is treated as frozen input. It is the output of
steps that are expensive or cannot be rerun here (LLM extraction of the
register scans, PDF parsing, the Getty release). None of it is regenerated.

| Source | Location | Origin |
|---|---|---|
| Register extractions (117 volumes) | `raw/universities/out_final/` | matricula LLM extraction of the scans |
| Freiburg, Giessen | `raw/universities/{freiburg,giessen}.csv` | parsed from the printed registers |
| Getty TGN relational release (Jan 2026) | `raw/tgn_rel_0126/` | the 10 tables the loader reads |
| GHGIS boundaries | `raw/ghgis/` | 12 layers: CORE and PROVINCES for 1820/26/30/34/48, 1820 DISTRICTS, 1848 GERMANCONFED |
| University TGN ids | `manual/uni_tgn_mapping.csv` | hand-made |
| Guild abolition dates | `manual/guild_dates_ogilvie.csv` | hand-made, after Ogilvie |
| Geocoding corrections, overrides, remaps | `manual/geocode/` | hand-curated (review queue verdicts) |
| Heidelberg running heads | `manual/heidelberg_heads/` | page-header extraction |
| Munich page years | `manual/munich_date_helper.csv` | page-header extraction |
| Extraction file order | `manual/out_final_order.txt` | see below |

Mapping tables written as code (field-of-study maps, father occupation classes)
live in `src/unis/mappings/`.

## Where the old scripts went

| Original | Here | Output |
|---|---|---|
| `cleaning/geocode_pipeline/load_tgn.py` | `unis.tgn.load` | |
| `clean_uni_lists/clean_and_combine/*.py` | `unis.registers.*` | |
| `.../heidelberg_date_fix/heidelberg_page_fix.py` | `unis.registers.heidelberg_dates` | |
| `geocode_pipeline/run_geocode.py`, `geocode/` | `unis.geocode` | |
| `geocode/classify_guilds.py` | `unis.polity.hometowns` | |
| `link_records/linkage.py`, `names.py` | `unis.linkage` | |
| `cleaning/clean_final.py` | `unis.finalize` | |
| `geocode/uni_polity.py` | `unis.polity.universities` | |
| `analysis/gravity/build_od.py` | `unis.gravity.build_od` | |
| `estimate_gravity.py` (`main`) | `unis.analysis.gravity_baseline` | `gravity_results.csv` |
| `estimate_gravity.py` (`main_event_study`) | `unis.analysis.event_study_1819` | `event_study_1819.csv` |
| `plot_era_coeffs.py` | `unis.analysis.plot_era_coefficients` | `era_coefficients.png` |
| `plot_event_study.py` | `unis.analysis.plot_event_study_1819` | `event_study_1819.png` |
| `design_b_composition.py` | `unis.analysis.design_b_composition` | `design_b_composition.png` |
| `goettingen7_{event_study,heterogeneity,reallocation,matched,placebo}.py` | `unis.analysis.goettingen7_*` | `.csv` + `.png` each |
| `heidelberg_tests.py` | `unis.analysis.heidelberg_tests` | `heidelberg_religion_tests.png` |
| `robustness_student_religion.py` | `unis.analysis.heidelberg_religion` | `heidelberg_religion.png` |
| `analysis/map_test.py` | `unis.analysis.student_map` | `student_map.html`, `student_map.png` |
| `run_all.py`, `clean_all.sh` | `Makefile` | |

Not ported, because nothing live depends on them: `design_a_substitution.py`
(withdrawn, findings §6); `student_counts.parquet` (read only by the
parliament code); the matricula extraction tool and other PDF/OCR helpers;
the geocoder's diagnostic scripts (`diagnose.py`, `explain.py`, `scripts/`);
the linkage synthetic evaluation (`make_synthetic.py`,
`linkage_diagnostics.py`); `make_curated.py` (its CSVs are now the source).

## Reproducibility notes

- **Extraction file order is pinned.** The original read the 117 extraction
  CSVs in unsorted directory order, and Berlin, Bonn, Jena and Marburg
  forward-fill ditto marks across file boundaries, so the order changes the
  data. `manual/out_final_order.txt` records the original order.
- **Student linkage is not deterministic.** The linkage stage estimates u
  from an unseeded random sample of pairs (`USING SAMPLE`). Two runs of the
  original code on identical input gave 98,782 and 98,476 students.
  Enrollment spells (`spell_id`) are unaffected, and every live analysis uses
  spells, not students.
- **The map jitter is not deterministic** either: `unique('spell_id')` does
  not fix row order, so the seeded noise lands on different points each run.
- **The map is centred explicitly.** Plotly's default centre is the mean
  coordinate, which is NaN because ungeocoded students have null lat/lon, so
  the original opened on (0, 0) in the Gulf of Guinea.

## Verification against the original project

The pipeline was rebuilt from `data/raw` and every stage compared with the
files in `~/Dropbox/1848_unis` (2026-09-14):

- **Identical:** all TGN tables, the Heidelberg date fixer, 12 of 13 school
  registers, `od_year.parquet`, `goettingen7_heterogeneity.csv`.
- **Same rows, different order of tied rows:** the Tübingen register and
  everything built from it (`all_students_unlinked`, `students_with_tgn`,
  `students_geocoded`, `students_clean`, `students_final`). The original's
  sort on (first_year, last_name, first_names) is unstable, so rows tied on
  that key come out in arbitrary order.
- **Spells:** identical except two namesake Tübingen spells (Jäger, 1830–35)
  that trade two rows, a consequence of the tie order above. Students differ
  run to run (see the linkage note).
- **`od.parquet`:** identical except the modal `first_year` in 3,460 cells
  where two years tie for the mode; no analysis uses that column.
- **Tables:** point estimates and SEs agree to within about 1e-6. The largest
  gap is the Erlangen placebo path (SE differs by 1.6e-3), a near-unidentified
  placebo excluded from the figure. Refitting the original OD grid with the
  original code moves SEs by the same order of magnitude (about 2e-7), so
  these are numerical rather than port differences. Three destinations with
  zero share change are listed in a different order in the reallocation table.
- Headline numbers in `docs/findings.md` (Göttingen path, Design B Wald tests,
  Heidelberg religion PPML, placebo summary) reproduce as written.

## Known issues carried over unchanged

This is a behaviour-preserving port. These issues are left as they were, so
they can be fixed one at a time with their effect on results visible.

- **No field groups for Berlin, Bonn, Jena and Tübingen.** Their raw field is
  recorded (`field_raw`, 93–100% of rows). The original cleaners computed the
  `law_admin`…`humanities` flags only on per-student summary tables that were
  never used, so `field` is null for these four schools. Findings §4–5
  describes fields there as unrecorded; they are recorded but unclassified.
- **Unmapped origin confession is `''`, not `unknown`** (findings, Data).
- **Linkage u-sampling is unseeded** (above).
