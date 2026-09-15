"""Build a cached, ladder-normalized index of the gazetteer.

Consumes up to five TGN tables. Only terms and coordinates are required; each
of the others sharpens disambiguation, and the build reports which were found.

  tgn_terms       names, with preferred / vernacular / historic flags and the
                  OTHER_FLAGS marker identifying machine codes
  tgn_coords      point coordinates
  tgn_subjects    RECORD_TYPE (administrative vs physical vs structural),
                  MERGED_STAT, and the preferred parent key
  tgn_placetypes  place types per subject, with preferred flag and dates
  tgn_rels        the full parent-child hierarchy

Subject-level attributes derived here:

  geo_precision  settlement / district / territory / country, from place types
  is_settlement  true if ANY place type is a settlement type. A city that is
                 also a capital carries several types; the administrative
                 region of the same name carries none of them. This is what
                 separates Hannover the city from Hannover the Regierungsbezirk
  ancestors      full preferred-parent chain, for containment tests
  depth          distance from the hierarchy root
  n_terms        prominence proxy, excluding machine-code terms
  term_quality   per-term multiplier on match strength
"""

from __future__ import annotations

import polars as pl

from .config import (
    CODE_TERM_FLAGS,
    PLACETYPE_PRECISION,
    PROMINENCE_WEIGHTS,
    TERM_QUALITY,
    Config,
)
from .normalize import add_ladder

MAX_HIERARCHY_DEPTH = 20
_PRECISION_RANK = {"settlement": 0, "district": 1, "territory": 2, "country": 3}


def _precision_expr(col: pl.Expr) -> pl.Expr:
    lowered = col.str.to_lowercase()
    expr = pl.when(lowered.is_null()).then(pl.lit(None, dtype=pl.Utf8))
    for needle, level in PLACETYPE_PRECISION:
        expr = expr.when(lowered.str.contains(needle, literal=True)).then(pl.lit(level))
    return expr.otherwise(pl.lit(None, dtype=pl.Utf8))


def _load_remap(cfg: Config) -> dict[int, int]:
    if not cfg.paths.subject_remap.exists():
        return {}
    df = pl.read_csv(
        cfg.paths.subject_remap,
        schema_overrides={"from_subject_id": pl.Int64, "to_subject_id": pl.Int64},
    ).drop_nulls(["from_subject_id", "to_subject_id"])
    return dict(zip(df["from_subject_id"].to_list(), df["to_subject_id"].to_list()))


def _merge_survivors(cfg: Config, verbose: bool) -> set[int] | None:
    """Subject ids that MERGED_STAT='M' does NOT lose, from tgn_merges.

    In this TGN release MERGED_STAT='M' does not mean "superseded". It marks
    a record that is part of, or resulted from, a merge. SUBJECT_MERGE.out
    records each cluster as (MERGE_ID = merge event, DOMINANT_ID = surviving
    record, NEW_ID = resulting record), and the merge-event ids are NOT
    present in SUBJECT.out at all. Filtering M subjects out wholesale
    therefore deleted the canonical record of 80,043 places -- Munich
    (7004333), Warsaw, Württemberg (7003691) -- and their names then fell to
    the phonetic rung and attached to unrelated homonyms ("Württemberg" ->
    Wartenberg, same Kölner code 3726174).

    Verified against the release:
      - all 89,115 M subjects in SUBJECT.out carry live terms; 80,777 are
        self-dominant (DOMINANT_ID == MERGE_ID) and the rest appear only as
        the NEW_ID result of a merge. None of them is an absorbed party:
        the absorbed records are merge-event ids that never reach
        SUBJECT.out.
      - so every M subject that exists in SUBJECT.out is a survivor.

    Returns the set of M subject ids to keep (all of them here), or None
    when the merge table is unavailable -- in which case the historical
    drop-all behaviour is retained, because without SUBJECT_MERGE.out there
    is no way to tell survivors from absorbed records.
    """
    if not cfg.paths.tgn_merges.exists():
        if verbose:
            print("[tgn_index] tgn_merges absent: MERGED_STAT='M' subjects cannot "
                  "be split into survivors and absorbed; keeping the old "
                  "drop-all behaviour")
        return None

    mg = pl.read_parquet(cfg.paths.tgn_merges).drop_nulls(
        ["MERGE_ID", "DOMINANT_ID", "NEW_ID"]
    ).unique()

    if not cfg.paths.tgn_subjects.exists():
        if verbose:
            print("[tgn_index] tgn_subjects absent: cannot identify M subjects")
        return None
    subs = pl.read_parquet(cfg.paths.tgn_subjects).select(
        "SUBJECT_ID", pl.col("MERGED_STAT").str.strip_chars().alias("_m")
    )
    m_ids = set(subs.filter(pl.col("_m") == "M")["SUBJECT_ID"].to_list())
    if not m_ids:
        return set()

    self_dominant = set(
        mg.filter(
            (pl.col("DOMINANT_ID") == pl.col("MERGE_ID"))
            & pl.col("MERGE_ID").is_in(list(m_ids))
        )["MERGE_ID"].unique().to_list()
    )
    # A merge-event id distinct from its dominant would be an absorbed
    # record; none of those exists in SUBJECT.out in this release, but the
    # guard keeps the policy honest if a future release changes that.
    absorbed = set(
        mg.filter(
            (pl.col("DOMINANT_ID") != pl.col("MERGE_ID"))
            & pl.col("MERGE_ID").is_in(list(m_ids))
        )["MERGE_ID"].unique().to_list()
    )
    keep = m_ids & (self_dominant | (m_ids - absorbed))
    if verbose:
        print(f"[tgn_index] merge table: {len(self_dominant):,} M subjects are "
              f"self-dominant, {len(keep - self_dominant):,} are merge results; "
              f"{len(absorbed):,} genuinely absorbed -> dropped")
    return keep


def _term_quality_expr(schema: set[str]) -> pl.Expr:
    """Per-term multiplier from the TERM flags, where present."""
    if "PREFERRED" not in schema:
        return pl.lit(TERM_QUALITY["variant"], dtype=pl.Float64)
    preferred = pl.col("PREFERRED") == "P"
    vernacular = (pl.col("VERNACULAR") == "V") if "VERNACULAR" in schema else pl.lit(False)
    historic = (pl.col("HISTORIC_FLAG") == "H") if "HISTORIC_FLAG" in schema else pl.lit(False)
    return (
        pl.when(preferred).then(pl.lit(TERM_QUALITY["preferred"]))
        .when(vernacular).then(pl.lit(TERM_QUALITY["vernacular"]))
        .when(historic).then(pl.lit(TERM_QUALITY["historic_variant"]))
        .otherwise(pl.lit(TERM_QUALITY["variant"]))
        .cast(pl.Float64)
    )


def _subject_attributes(cfg: Config, verbose: bool) -> pl.DataFrame | None:
    if not cfg.paths.tgn_subjects.exists():
        if verbose:
            print("[tgn_index] tgn_subjects absent: cannot filter physical features "
                  "or merged records")
        return None
    return pl.read_parquet(cfg.paths.tgn_subjects).select(
        "SUBJECT_ID",
        "PARENT_KEY",
        pl.col("RECORD_TYPE").str.strip_chars(),
        pl.col("MERGED_STAT").str.strip_chars(),
    )


def _placetype_attributes(cfg: Config, verbose: bool) -> pl.DataFrame | None:
    if not cfg.paths.tgn_placetypes.exists():
        if verbose:
            print("[tgn_index] tgn_placetypes absent: geo_precision stays 'unknown' "
                  "and the settlement preference cannot fire")
        return None

    pt = pl.read_parquet(cfg.paths.tgn_placetypes).with_columns(
        pl.col("PREFERRED").str.strip_chars(),
        precision=_precision_expr(pl.col("PLACETYPE").str.strip_chars()),
    )

    if verbose:
        unmapped = (
            pt.filter(pl.col("precision").is_null() & pl.col("PLACETYPE").is_not_null())
            .group_by("PLACETYPE").agg(pl.len().alias("n"))
            .sort("n", descending=True).head(25)
        )
        if unmapped.height:
            total = pt.filter(pl.col("precision").is_null()).height
            print(f"[tgn_index] {total:,} place-type rows map to no precision level. "
                  "Most frequent unmapped types (extend PLACETYPE_PRECISION):")
            for label, n in unmapped.iter_rows():
                print(f"             {n:>8,}  {label}")

    settlement = pt.group_by("SUBJECT_ID").agg(
        (pl.col("precision") == "settlement").any().alias("is_settlement")
    )
    best = (
        pt.filter(pl.col("precision").is_not_null())
        .with_columns(
            _pref=(pl.col("PREFERRED") == "P").cast(pl.Int8),
            _rank=pl.col("precision").replace_strict(
                _PRECISION_RANK, default=9, return_dtype=pl.Int32
            ),
        )
        .sort(["SUBJECT_ID", "_pref", "_rank", "PTYPE_ROLE_ID"],
              descending=[False, True, False, False])
        .group_by("SUBJECT_ID", maintain_order=True)
        .agg(pl.col("precision").first().alias("geo_precision"),
             pl.col("PLACETYPE").first().alias("placetype"))
    )
    return settlement.join(best, on="SUBJECT_ID", how="left")


def _ancestors(cfg: Config, subjects: pl.DataFrame | None, verbose: bool):
    """Preferred-parent chain and depth per subject."""
    parent = None
    if cfg.paths.tgn_rels.exists():
        rels = pl.read_parquet(cfg.paths.tgn_rels).with_columns(
            pl.col("PREFERRED").str.strip_chars()
        )
        parent = (
            rels.filter(pl.col("PREFERRED") == "P")
            .select(pl.col("CHILD_ID").alias("SUBJECT_ID"), "PARENT_ID")
            .group_by("SUBJECT_ID").agg(pl.col("PARENT_ID").min())
        )
    elif subjects is not None:
        parent = (
            subjects.select("SUBJECT_ID", pl.col("PARENT_KEY").alias("PARENT_ID"))
            .drop_nulls()
        )
    if parent is None:
        if verbose:
            print("[tgn_index] no hierarchy table: containment and nesting signals "
                  "are unavailable")
        return None

    frontier = parent.select("SUBJECT_ID", pl.col("PARENT_ID").alias("anc"))
    chain, current = [frontier], frontier
    for _ in range(MAX_HIERARCHY_DEPTH):
        current = (
            current.join(parent.rename({"SUBJECT_ID": "anc", "PARENT_ID": "next"}),
                         on="anc", how="inner")
            .select("SUBJECT_ID", pl.col("next").alias("anc"))
            .filter(pl.col("anc") != pl.col("SUBJECT_ID"))
            .unique()
        )
        if current.height == 0:
            break
        chain.append(current)

    out = (
        pl.concat(chain).unique()
        .group_by("SUBJECT_ID")
        .agg(pl.col("anc").alias("ancestors"), pl.len().alias("depth"))
        .join(
            parent.rename({"PARENT_ID": "parent_id"}), on="SUBJECT_ID", how="left"
        )
    )
    n_children = (
        parent.group_by("PARENT_ID").agg(pl.len().alias("n_children"))
        .rename({"PARENT_ID": "SUBJECT_ID"})
    )
    out = out.join(n_children, on="SUBJECT_ID", how="left")
    if verbose:
        print(f"[tgn_index] hierarchy: {out.height:,} subjects with ancestors, "
              f"max depth {out['depth'].max()}")
    return out


def _count_table(path, id_col: str, value_col: str) -> pl.DataFrame | None:
    if not path.exists():
        return None
    return pl.read_parquet(path).select(id_col, value_col)


def _bbox_area_expr() -> pl.Expr:
    """Bounding-box area in km2, null when no box is recorded."""
    dlat = (pl.col("LAT_MAX_BOX") - pl.col("LAT_MIN_BOX")).abs() * 111.0
    dlon = (
        (pl.col("LON_MAX_BOX") - pl.col("LON_MIN_BOX")).abs()
        * 111.0
        * ((pl.col("LAT_MAX_BOX") + pl.col("LAT_MIN_BOX")) / 2 * (3.14159265 / 180)).cos()
    )
    return (dlat * dlon).abs()


def _prominence(indexed: pl.DataFrame, cfg: Config) -> pl.DataFrame:
    """Assemble the prominence composite from whatever components exist.

    Each term is log1p of a count, so a place with ten times the documentation
    scores about 2.3 higher rather than ten times higher. Missing components
    contribute zero, which means a place whose source table was never loaded is
    treated the same as one with no sources -- acceptable, because the
    components are only ever compared between candidates for the same string,
    and they all come from the same tables.
    """
    w = PROMINENCE_WEIGHTS
    parts, present = [], []

    def add(name: str, expr: pl.Expr) -> None:
        if w.get(name):
            parts.append(w[name] * expr)
            present.append(name)

    add("terms", pl.col("n_terms").fill_null(1).log1p())
    if "n_languages" in indexed.columns:
        add("languages", pl.col("n_languages").fill_null(0).log1p())
    if "n_children" in indexed.columns:
        add("children", pl.col("n_children").fill_null(0).log1p())
    if "n_placetypes" in indexed.columns:
        add("placetypes", pl.col("n_placetypes").fill_null(0).log1p())
    if "bbox_area" in indexed.columns:
        add("bbox", pl.col("bbox_area").fill_null(0.0).log1p())
    if "n_sources" in indexed.columns:
        add("sources", pl.col("n_sources").fill_null(0).log1p())
    if "n_notes" in indexed.columns:
        add("scope_note", (pl.col("n_notes").fill_null(0) > 0).cast(pl.Float64))

    total = parts[0]
    for extra in parts[1:]:
        total = total + extra

    if cfg.verbose:
        print(f"[tgn_index] prominence components: {', '.join(present)}")
    return indexed.with_columns(prominence=total)


def build(cfg: Config) -> pl.DataFrame:
    if cfg.use_cache and cfg.paths.tgn_index.exists():
        if cfg.verbose:
            print(f"[tgn_index] loading cache {cfg.paths.tgn_index}")
        return pl.read_parquet(cfg.paths.tgn_index)

    schema = set(pl.scan_parquet(cfg.paths.tgn_terms).collect_schema().names())
    missing = {"TERM", "TERM_ID", "SUBJECT_ID"} - schema
    if missing:
        raise ValueError(f"tgn_terms is missing required columns: {sorted(missing)}")
    flags = sorted(schema & {"PREFERRED", "VERNACULAR", "HISTORIC_FLAG", "OTHER_FLAGS"})
    if cfg.verbose:
        print(f"[tgn_index] term flags available: {flags or 'none'}")

    terms = pl.scan_parquet(cfg.paths.tgn_terms).select(
        ["TERM", "TERM_ID", "SUBJECT_ID"] + flags
    )

    remap = _load_remap(cfg)
    if remap:
        terms = terms.with_columns(pl.col("SUBJECT_ID").replace(remap))
        if cfg.verbose:
            print(f"[tgn_index] applied {len(remap)} subject remappings")

    # Flags in the relational release are fixed-width padded ("C " not "C").
    # If tgn_terms was built without stripping, every equality test against a
    # flag silently fails, so normalize here as well as in the loader.
    if flags:
        terms = terms.with_columns([pl.col(c).str.strip_chars() for c in flags])

    terms = terms.filter(pl.col("TERM").is_not_null())

    # Machine codes are not names. Dropped here, they can never reach any rung.
    if "OTHER_FLAGS" in schema:
        before = terms.select(pl.len()).collect().item()
        terms = terms.filter(
            pl.col("OTHER_FLAGS").is_null()
            | ~pl.col("OTHER_FLAGS").is_in(list(CODE_TERM_FLAGS))
        )
        after = terms.select(pl.len()).collect().item()
        if cfg.verbose:
            print(f"[tgn_index] dropped {before - after:,} machine-code terms "
                  "(ISO / FIPS / USPS)")

    terms = terms.with_columns(term_quality=_term_quality_expr(schema)).collect()

    terms = (
        terms.sort(["TERM", "SUBJECT_ID", "term_quality", "TERM_ID"],
                   descending=[False, False, True, False])
        .group_by(["TERM", "SUBJECT_ID"], maintain_order=True)
        .agg(pl.col("TERM_ID").first(), pl.col("term_quality").first())
    )

    n_terms = terms.group_by("SUBJECT_ID").agg(pl.len().alias("n_terms"))

    subjects = _subject_attributes(cfg, cfg.verbose)
    merge_survivors = _merge_survivors(cfg, cfg.verbose)
    if subjects is not None:
        if merge_survivors is not None:
            keep = subjects.filter(
                (pl.col("MERGED_STAT") != "M")
                | pl.col("SUBJECT_ID").is_in(list(merge_survivors))
            )
        else:
            # No merge table: fall back to the historical drop-all behaviour.
            keep = subjects.filter(pl.col("MERGED_STAT") != "M")
        keep = keep.filter(pl.col("RECORD_TYPE").is_in(["A", "B"])).select("SUBJECT_ID")
        before = terms.height
        terms = terms.join(keep, on="SUBJECT_ID", how="semi")
        if cfg.verbose:
            print(f"[tgn_index] administrative subjects only: kept {terms.height:,} "
                  f"of {before:,} term rows ({before - terms.height:,} dropped as "
                  "physical features, guide terms, facets or absorbed merge members)")

    if cfg.verbose:
        print(f"[tgn_index] normalizing {terms.height:,} (term, subject) pairs")
    indexed = add_ladder(terms, "TERM")

    coords = pl.read_parquet(cfg.paths.tgn_coords)
    box_cols = {"LAT_MIN_BOX", "LAT_MAX_BOX", "LON_MIN_BOX", "LON_MAX_BOX"}
    if box_cols <= set(coords.columns):
        coords = coords.select(
            "SUBJECT_ID", "LAT", "LON", bbox_area=_bbox_area_expr()
        )
    else:
        coords = coords.select("SUBJECT_ID", "LAT", "LON")
        if cfg.verbose:
            print("[tgn_index] tgn_coords has no bounding box columns; reload with "
                  "load_tgn.py to add the box-extent prominence signal")
    dupes = coords.height - coords["SUBJECT_ID"].n_unique()
    if dupes:
        raise ValueError(
            f"tgn_coords has {dupes} duplicate SUBJECT_ID rows; this would multiply "
            "student rows and violate the cardinality guarantee."
        )

    indexed = (
        indexed.join(n_terms, on="SUBJECT_ID", how="left")
        .join(coords, on="SUBJECT_ID", how="left")
    )

    # Distinct languages a place's names span, and how many place types it
    # carries: both track prominence closely and cost one group-by each.
    if cfg.paths.tgn_languages.exists():
        langs = (
            pl.read_parquet(cfg.paths.tgn_languages)
            .select("SUBJECT_ID", "LANGUAGE_CODE")
            .drop_nulls()
            .group_by("SUBJECT_ID")
            .agg(pl.col("LANGUAGE_CODE").n_unique().alias("n_languages"))
        )
        indexed = indexed.join(langs, on="SUBJECT_ID", how="left")
    if cfg.paths.tgn_placetypes.exists():
        npt = (
            pl.read_parquet(cfg.paths.tgn_placetypes)
            .group_by("SUBJECT_ID")
            .agg(pl.col("PTYPE_ROLE_ID").n_unique().alias("n_placetypes"))
        )
        indexed = indexed.join(npt, on="SUBJECT_ID", how="left")
    for path, col in (
        (cfg.paths.tgn_sources, "n_sources"),
        (cfg.paths.tgn_scope_notes, "n_notes"),
    ):
        tbl = _count_table(path, "SUBJECT_ID", col)
        if tbl is not None:
            indexed = indexed.join(tbl, on="SUBJECT_ID", how="left")

    pt = _placetype_attributes(cfg, cfg.verbose)
    if pt is not None:
        indexed = indexed.join(pt, on="SUBJECT_ID", how="left")
    else:
        indexed = indexed.with_columns(
            geo_precision=pl.lit(None, dtype=pl.Utf8),
            is_settlement=pl.lit(None, dtype=pl.Boolean),
            placetype=pl.lit(None, dtype=pl.Utf8),
        )

    anc = _ancestors(cfg, subjects, cfg.verbose)
    if anc is not None:
        indexed = indexed.join(anc, on="SUBJECT_ID", how="left")
    else:
        indexed = indexed.with_columns(
            ancestors=pl.lit(None, dtype=pl.List(pl.Int64)),
            depth=pl.lit(None, dtype=pl.UInt32),
            parent_id=pl.lit(None, dtype=pl.Int64),
        )

    indexed = indexed.with_columns(
        pl.col("n_terms").fill_null(1),
        pl.col("geo_precision").fill_null("unknown"),
        pl.col("is_settlement").fill_null(False),
        pl.col("term_quality").fill_null(TERM_QUALITY["variant"]),
    )

    # A settlement whose parent is itself a settlement is an Ortsteil. This is
    # what separates a hamlet inside a Saxon municipality from Koblenz, whose
    # parent is a Regierungsbezirk -- no size data needed.
    if "parent_id" in indexed.columns:
        parent_flag = (
            indexed.select("SUBJECT_ID", "is_settlement")
            .unique(subset=["SUBJECT_ID"])
            .rename({"SUBJECT_ID": "parent_id", "is_settlement": "parent_is_settlement"})
        )
        indexed = indexed.join(parent_flag, on="parent_id", how="left").with_columns(
            pl.col("parent_is_settlement").fill_null(False)
        )
        indexed = indexed.with_columns(
            is_sub_settlement=pl.col("is_settlement") & pl.col("parent_is_settlement")
        )
    else:
        indexed = indexed.with_columns(is_sub_settlement=pl.lit(False))

    indexed = _prominence(indexed, cfg)

    # A component that never varies contributes nothing: every candidate gets
    # the same value and it cannot separate a city from a hamlet. That is what
    # a silently failed join looks like, so say so rather than let the
    # composite quietly collapse to a constant.
    if cfg.verbose:
        subs = indexed.unique(subset=["SUBJECT_ID"])
        for col, source in [
            ("n_terms", "TERM.out"), ("n_languages", "LANGUAGE_RELS.out"),
            ("n_children", "SUBJECT_RELS.out"), ("n_placetypes", "PTYPE_ROLE_RELS.out"),
            ("bbox_area", "COORDINATES.out bounding box"),
            ("n_sources", "SOURCE_RELS_SUBJECT.out"), ("n_notes", "SCOPE_NOTES.out"),
        ]:
            if col not in subs.columns:
                print(f"[tgn_index] prominence: {col} absent ({source} not loaded)")
                continue
            varies = subs.select((pl.col(col).fill_null(0) > 1).mean()).item()
            if varies < 0.01:
                print(f"[tgn_index] prominence: {col} is >1 for only {varies:.2%} of "
                      f"subjects; check {source}")

    cfg.paths.cache_dir.mkdir(parents=True, exist_ok=True)
    indexed.write_parquet(cfg.paths.tgn_index)
    if cfg.verbose:
        share = indexed.select((pl.col("geo_precision") != "unknown").mean()).item()
        sub = indexed.select(pl.col("is_sub_settlement").mean()).item()
        print(f"[tgn_index] wrote {indexed.height:,} rows ({share:.1%} with a known "
              f"precision, {sub:.1%} sub-settlements) -> {cfg.paths.tgn_index}")
    return indexed
