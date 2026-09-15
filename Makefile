# Build for the 1848 universities project.
#
#   make               everything: data pipeline, tables, figures, map
#   make data          up to data/processed/students_final.parquet
#   make analysis      tables and figures (building data as needed)
#   make map           the interactive and static student maps
#   make test          unit tests
#   make import-raw    copy frozen sources from RAW_SRC into data/raw (once)
#   make check-raw     verify data/raw against data/raw_manifest.sha256
#
# Every step tees its console output to output/logs/<step>.log. The estimation
# logs hold the Wald tests and diagnostics quoted in docs/findings.md.

SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c
.DELETE_ON_ERROR:
.SUFFIXES:
.DEFAULT_GOAL := all

PY := uv run --frozen python -m
RAW_SRC ?= $(HOME)/Dropbox/1848_unis

# uv always uses the project's .venv; an activated venv from elsewhere would
# only add a warning to every log.
unexport VIRTUAL_ENV

# Local, uncommitted settings such as HISTORIC_BASEMAP_URL (see .env.example).
-include .env
export HISTORIC_BASEMAP_URL

RAW := data/raw
MANUAL := data/manual
INTERIM := data/interim
PROCESSED := data/processed
TABLES := output/tables
FIGURES := output/figures
LOGS := output/logs
SRC := src/unis

# --- sources -------------------------------------------------------------------
OUT_FINAL_CSV := $(shell find $(RAW)/universities/out_final -name '*.csv' 2>/dev/null)
TGN_TABLES := TERM SUBJECT SUBJECT_RELS PTYPE_ROLE PTYPE_ROLE_RELS COORDINATES \
              LANGUAGE_RELS SOURCE_RELS_SUBJECT SCOPE_NOTES SUBJECT_MERGE
TGN_OUT := $(TGN_TABLES:%=$(RAW)/tgn_rel_0126/%.out)
GHGIS_LAYERS := 1820CORE 1826CORE 1830CORE 1834CORE 1848CORE \
                1820PROVINCES 1826PROVINCES 1830PROVINCES 1834PROVINCES 1848PROVINCES \
                1820DISTRICTS 1848GERMANCONFED
GHGIS_SHP := $(foreach l,$(GHGIS_LAYERS),$(RAW)/ghgis/GHGIS$(l)/GHGIS$(l).shp)

# --- pipeline products ---------------------------------------------------------
TGN_STAMP := $(INTERIM)/tgn/tgn.stamp
HEIDELBERG_FIXER := $(INTERIM)/heidelberg_date_fixer.parquet
UNLINKED := $(INTERIM)/all_students_unlinked.parquet
WITH_TGN := $(INTERIM)/students_with_tgn.parquet
GEOCODED := $(INTERIM)/students_geocoded.parquet
LINKED := $(INTERIM)/students_linked.parquet
CLEAN := $(INTERIM)/students_clean.parquet
FINAL := $(PROCESSED)/students_final.parquet
OD := $(PROCESSED)/od.parquet
OD_YEAR := $(PROCESSED)/od_year.parquet

# Linkage parameters (see src/unis/linkage/linkage.py)
LINK_ANNUAL := berlin,tuebingen,jena,bonn
LINK_MEAN_SCHOOLS := 1.45
LINK_THRESHOLD := 0.95

# --- analysis products ---------------------------------------------------------
GOT7 := goettingen7_event_study goettingen7_heterogeneity goettingen7_reallocation \
        goettingen7_matched
TABLE_FILES := $(TABLES)/gravity_results.csv $(TABLES)/event_study_1819.csv \
               $(TABLES)/design_b_berlin_checks.csv \
               $(GOT7:%=$(TABLES)/%.csv) $(TABLES)/goettingen7_placebo.csv
# .tex tables named after their CSV; the gravity script instead writes two
# differently named tables from gravity_results.csv's specifications.
CSV_TEX := $(filter-out $(TABLES)/gravity_results.tex,$(TABLE_FILES:.csv=.tex))
TEX_FILES := $(CSV_TEX) $(TABLES)/gravity_structure.tex $(TABLES)/gravity_by_decade.tex \
             $(TABLES)/goettingen7_reallocation_ppml.tex \
             $(TABLES)/design_b_composition.tex $(TABLES)/heidelberg_religion.tex
FIGURE_FILES := $(FIGURES)/era_coefficients.png $(FIGURES)/event_study_1819.png \
                $(FIGURES)/design_b_composition.png \
                $(GOT7:%=$(FIGURES)/%.png) $(FIGURES)/goettingen7_placebo.png \
                $(FIGURES)/heidelberg_religion_tests.png $(FIGURES)/heidelberg_religion.png
MAP_FILES := $(FIGURES)/student_map.html $(FIGURES)/student_map.png

.PHONY: all data analysis tables figures map test lint import-raw check-raw \
        clean clean-output
all: analysis map
data: $(FINAL)
analysis: tables figures
tables: $(TABLE_FILES) $(TEX_FILES)
figures: $(FIGURE_FILES)
map: $(MAP_FILES)

$(LOGS):
	mkdir -p $@

$(RAW)/%:
	@echo "Missing source file $@. Run 'make import-raw' first (see README)." >&2; exit 1

# --- data pipeline -------------------------------------------------------------

$(TGN_STAMP): $(TGN_OUT) $(SRC)/tgn/load.py | $(LOGS)
	$(PY) unis.tgn.load 2>&1 | tee $(LOGS)/tgn_load.log
	touch $@

$(HEIDELBERG_FIXER): $(wildcard $(MANUAL)/heidelberg_heads/*.json) \
                     $(SRC)/registers/heidelberg_dates.py | $(LOGS)
	$(PY) unis.registers.heidelberg_dates 2>&1 | tee $(LOGS)/heidelberg_dates.log

$(UNLINKED): $(OUT_FINAL_CSV) $(RAW)/universities/freiburg.csv $(RAW)/universities/giessen.csv \
             $(MANUAL)/out_final_order.txt $(MANUAL)/munich_date_helper.csv $(HEIDELBERG_FIXER) \
             $(wildcard $(SRC)/registers/*.py) $(wildcard $(SRC)/mappings/*.py) | $(LOGS)
	$(PY) unis.registers.combine 2>&1 | tee $(LOGS)/registers.log

$(WITH_TGN): $(UNLINKED) $(TGN_STAMP) $(MANUAL)/uni_tgn_mapping.csv \
             $(wildcard $(MANUAL)/geocode/*.csv) $(wildcard $(SRC)/geocode/*.py) | $(LOGS)
	$(PY) unis.geocode --no-cache 2>&1 | tee $(LOGS)/geocode.log

$(GEOCODED): $(WITH_TGN) $(GHGIS_SHP) $(MANUAL)/guild_dates_ogilvie.csv \
             $(SRC)/polity/hometowns.py $(SRC)/polity/ghgis.py | $(LOGS)
	$(PY) unis.polity.hometowns 2>&1 | tee $(LOGS)/polity_hometowns.log

$(LINKED): $(GEOCODED) $(wildcard $(SRC)/linkage/*.py) | $(LOGS)
	rm -f $@
	$(PY) unis.linkage.linkage $(GEOCODED) --annual $(LINK_ANNUAL) \
	    --mean-schools $(LINK_MEAN_SCHOOLS) --threshold $(LINK_THRESHOLD) --out $@ \
	    2>&1 | tee $(LOGS)/linkage.log

$(CLEAN): $(GEOCODED) $(LINKED) $(SRC)/finalize.py $(SRC)/mappings/fathers.py | $(LOGS)
	$(PY) unis.finalize 2>&1 | tee $(LOGS)/finalize.log

$(FINAL): $(CLEAN) $(TGN_STAMP) $(GHGIS_SHP) $(MANUAL)/uni_tgn_mapping.csv \
          $(SRC)/polity/universities.py $(SRC)/polity/ghgis.py | $(LOGS)
	$(PY) unis.polity.universities 2>&1 | tee $(LOGS)/polity_universities.log

# build_od writes od.parquet, then od_year.parquet; give both the same mtime
# so od.parquet never looks stale.
$(OD_YEAR): $(FINAL) $(SRC)/gravity/build_od.py $(SRC)/geo.py | $(LOGS)
	$(PY) unis.gravity.build_od 2>&1 | tee $(LOGS)/build_od.log
	touch -r $(OD_YEAR) $(OD)
$(OD): $(OD_YEAR)
	@test -f $@ || { rm -f $<; $(MAKE) --no-print-directory $<; }

# --- analyses ------------------------------------------------------------------
PLOTTING := $(SRC)/plotting.py
LATEX := $(SRC)/latex.py $(SRC)/gravity/constants.py
EVENTSTUDY := $(SRC)/gravity/eventstudy.py $(LATEX) $(PLOTTING)
HEIDELBERG := $(SRC)/gravity/heidelberg.py $(SRC)/geo.py $(PLOTTING)

# Every table script writes its .csv, then its booktabs .tex.
$(CSV_TEX): $(TABLES)/%.tex: $(TABLES)/%.csv
	@test -f $@ || { rm -f $<; $(MAKE) --no-print-directory $<; }
$(TABLES)/gravity_structure.tex $(TABLES)/gravity_by_decade.tex: $(TABLES)/gravity_results.csv
	@test -f $@ || { rm -f $<; $(MAKE) --no-print-directory $<; }
$(TABLES)/goettingen7_reallocation_ppml.tex: $(TABLES)/goettingen7_reallocation.csv
	@test -f $@ || { rm -f $<; $(MAKE) --no-print-directory $<; }

$(TABLES)/gravity_results.csv: $(OD) $(SRC)/analysis/gravity_baseline.py $(LATEX) | $(LOGS)
	$(PY) unis.analysis.gravity_baseline 2>&1 | tee $(LOGS)/gravity_baseline.log

$(TABLES)/event_study_1819.csv: $(OD_YEAR) $(SRC)/analysis/event_study_1819.py $(LATEX) | $(LOGS)
	$(PY) unis.analysis.event_study_1819 2>&1 | tee $(LOGS)/event_study_1819.log

$(FIGURES)/era_coefficients.png: $(TABLES)/gravity_results.csv \
                                 $(SRC)/analysis/plot_era_coefficients.py $(PLOTTING) | $(LOGS)
	$(PY) unis.analysis.plot_era_coefficients 2>&1 | tee $(LOGS)/plot_era_coefficients.log

$(FIGURES)/event_study_1819.png: $(TABLES)/event_study_1819.csv \
                                 $(SRC)/analysis/plot_event_study_1819.py $(PLOTTING) | $(LOGS)
	$(PY) unis.analysis.plot_event_study_1819 2>&1 | tee $(LOGS)/plot_event_study_1819.log

$(FIGURES)/design_b_composition.png: $(OD) $(OD_YEAR) $(SRC)/analysis/design_b_composition.py \
                                     $(PLOTTING) $(LATEX) | $(LOGS)
	$(PY) unis.analysis.design_b_composition 2>&1 | tee $(LOGS)/design_b_composition.log

$(TABLES)/design_b_berlin_checks.csv: $(OD) $(FINAL) $(SRC)/analysis/design_b_berlin_checks.py \
                                      $(SRC)/gravity/build_od.py $(LATEX) | $(LOGS)
	$(PY) unis.analysis.design_b_berlin_checks 2>&1 | tee $(LOGS)/design_b_berlin_checks.log

# These two scripts write their figure, then their regression table.
$(TABLES)/design_b_composition.tex: $(FIGURES)/design_b_composition.png
	@test -f $@ || { rm -f $<; $(MAKE) --no-print-directory $<; }
$(TABLES)/heidelberg_religion.tex: $(FIGURES)/heidelberg_religion.png
	@test -f $@ || { rm -f $<; $(MAKE) --no-print-directory $<; }

# Göttingen Seven: each script writes its table, then its figure.
$(GOT7:%=$(TABLES)/%.csv): $(TABLES)/%.csv: $(OD_YEAR) $(SRC)/analysis/%.py $(EVENTSTUDY) | $(LOGS)
	$(PY) unis.analysis.$* 2>&1 | tee $(LOGS)/$*.log

$(TABLES)/goettingen7_placebo.csv: $(OD_YEAR) $(TABLES)/goettingen7_event_study.csv \
                                   $(SRC)/analysis/goettingen7_placebo.py $(EVENTSTUDY) | $(LOGS)
	$(PY) unis.analysis.goettingen7_placebo 2>&1 | tee $(LOGS)/goettingen7_placebo.log

$(GOT7:%=$(FIGURES)/%.png) $(FIGURES)/goettingen7_placebo.png: $(FIGURES)/%.png: $(TABLES)/%.csv
	@test -f $@ || { rm -f $<; $(MAKE) --no-print-directory $<; }

$(FIGURES)/heidelberg_religion_tests.png: $(FINAL) $(SRC)/analysis/heidelberg_tests.py \
                                          $(HEIDELBERG) | $(LOGS)
	$(PY) unis.analysis.heidelberg_tests 2>&1 | tee $(LOGS)/heidelberg_tests.log

$(FIGURES)/heidelberg_religion.png: $(FINAL) $(SRC)/analysis/heidelberg_religion.py \
                                    $(HEIDELBERG) $(LATEX) | $(LOGS)
	$(PY) unis.analysis.heidelberg_religion 2>&1 | tee $(LOGS)/heidelberg_religion.log

$(FIGURES)/student_map.html: $(FINAL) $(SRC)/analysis/student_map.py \
                             $(SRC)/gravity/constants.py $(wildcard .env) | $(LOGS)
	$(PY) unis.analysis.student_map 2>&1 | tee $(LOGS)/student_map.log
$(FIGURES)/student_map.png: $(FIGURES)/student_map.html
	@test -f $@ || { rm -f $<; $(MAKE) --no-print-directory $<; }

# --- housekeeping --------------------------------------------------------------

test:
	uv run --frozen pytest

lint:
	uv run --frozen ruff check src tests

import-raw:
	scripts/import_raw.sh "$(RAW_SRC)"

check-raw:
	cd $(RAW) && shasum -a 256 --quiet -c ../raw_manifest.sha256 && echo "data/raw matches manifest"

clean-output:
	rm -rf output

clean: clean-output
	rm -rf $(INTERIM) $(PROCESSED)
