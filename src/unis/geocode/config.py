"""Paths and tunable parameters.

Every number that encodes a judgement call lives here, not scattered through
the pipeline. Overrides can be supplied per-run by constructing a Config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from unis import paths as project_paths


@dataclass
class Paths:
    """Where the geocoder reads and writes. Defaults are the project layout."""

    students: Path = project_paths.STUDENTS_UNLINKED
    tgn_dir: Path = project_paths.TGN  # the tgn_*.parquet tables
    uni_mapping: Path = project_paths.UNI_TGN_MAPPING
    output: Path = project_paths.STUDENTS_WITH_TGN
    # Cached artifacts. Expensive to build, cheap to reuse; delete to rebuild.
    cache_dir: Path = project_paths.GEOCODE_CACHE
    # Curated corrections, overrides and remaps, edited by hand.
    curated_dir: Path = project_paths.GEOCODE_CURATED

    @classmethod
    def under(cls, root: Path) -> Paths:
        """All inputs and outputs under one directory (the test-fixture layout)."""
        return cls(
            students=root / "universities/clean/all_students_unlinked.parquet",
            tgn_dir=root,
            uni_mapping=root / "uni_tgn_mapping.csv",
            output=root / "students_with_tgn.parquet",
            cache_dir=root / "geocode_cache",
        )

    @property
    def tgn_terms(self) -> Path:
        return self.tgn_dir / "tgn_terms.parquet"

    @property
    def tgn_coords(self) -> Path:
        return self.tgn_dir / "tgn_coords.parquet"

    # Optional TGN tables. Each improves disambiguation; the pipeline runs
    # without any of them, and says so.
    @property
    def tgn_subjects(self) -> Path:
        return self.tgn_dir / "tgn_subjects.parquet"

    @property
    def tgn_placetypes(self) -> Path:
        return self.tgn_dir / "tgn_placetypes.parquet"

    @property
    def tgn_rels(self) -> Path:
        return self.tgn_dir / "tgn_rels.parquet"

    @property
    def tgn_languages(self) -> Path:
        return self.tgn_dir / "tgn_languages.parquet"

    @property
    def tgn_sources(self) -> Path:
        return self.tgn_dir / "tgn_sources.parquet"

    @property
    def tgn_scope_notes(self) -> Path:
        return self.tgn_dir / "tgn_scope_notes.parquet"

    @property
    def tgn_merges(self) -> Path:
        return self.tgn_dir / "tgn_merges.parquet"

    @property
    def tgn_index(self) -> Path:
        return self.cache_dir / "tgn_index.parquet"

    @property
    def candidates(self) -> Path:
        return self.cache_dir / "string_candidates.parquet"

    # Human-facing outputs.
    @property
    def review_queue(self) -> Path:
        return self.cache_dir / "review_queue.csv"

    @property
    def diff_report(self) -> Path:
        return self.cache_dir / "resolution_diff.csv"

    # Curated inputs.
    @property
    def corrections(self) -> Path:
        return self.curated_dir / "corrections.csv"

    @property
    def overrides(self) -> Path:
        return self.curated_dir / "overrides.csv"

    @property
    def subject_remap(self) -> Path:
        return self.curated_dir / "subject_remap.csv"


@dataclass
class Weights:  # noqa: D101 -- documented below
    """Log-scale scoring weights.

    The total score is a sum of log-likelihood-ish terms. Absolute values are
    meaningless; only differences between candidates for the same student
    matter. Raising a weight makes that signal more decisive.
    """

    # Quality of the name match itself. Applied to log(string_score).
    string: float = 3.0
    # Prominence of the candidate place, applied to the composite built in
    # tgn_index (see PROMINENCE_WEIGHTS). Deliberately modest. The composite
    # spans roughly 0-13, so a large value here would let prominence dominate
    # every other signal and pull ambiguous villages onto the nearest city.
    # This is the first weight to calibrate against hand-labelled cases.
    #
    # Raised to 1.0 when the distance term was corrected to use density rather
    # than probability mass. That fix widened the distance term's range from
    # roughly 1.5 nats to roughly 10, so the old 0.4 would have left prominence
    # unable to rescue any genuinely distant origin.
    prominence: float = 1.0
    # Distance from the student's university, under the fitted per-school prior.
    distance: float = 1.0
    # Agreement between the region-derived and hometown-derived candidates.
    coherence: float = 1.5
    # Bonus when one field's candidate is a gazetteer ancestor of the other's.
    # This is proof of consistency rather than an inference from proximity:
    # "Neustadt" inside "Mecklenburg" is the Neustadt whose parent chain
    # actually contains Mecklenburg.
    nesting: float = 2.0
    # Bonus when a qualifier parsed out of the string ("...in Schlesien") names
    # an ancestor of the candidate.
    hint_containment: float = 2.0
    # Term quality: preferred name vs historic variant vs abbreviation.
    term_quality: float = 1.0


@dataclass
class Tuning:
    # --- Candidate generation -------------------------------------------
    # Maximum candidates retained per source string. Binding is recorded.
    max_candidates_per_string: int = 40

    # Coarse rungs are gated on plausibility: a candidate reached only by a
    # phonetic or edit-distance key must lie within this radius of at least one
    # university in the corpus, or it is discarded. A phonetic match to a place
    # 12,000 km away is never the right answer -- that is how "Mittelmark"
    # became Middlemarch, New Zealand. Exact and near-exact rungs are NOT
    # gated, so a genuine Philadelphia still resolves.
    coarse_gate_km: float = 2000.0
    gated_methods: tuple[str, ...] = ("strip", "phonetic", "fuzzy")

    # Candidates closer together than this are treated as the same place and
    # collapsed to one representative. TGN routinely holds a city, its
    # district and its historical state as separate subjects at nearly
    # identical coordinates; left separate they produce near-tied scores that
    # flip on which university the student attended, so the same city ends up
    # under different location_ids in different rows.
    dedupe_radius_km: float = 25.0
    # Clustering is applied to this many top candidates per string. Anything
    # below the cap cannot win anyway.
    dedupe_pool: int = 200
    # Jaro-Winkler floor for the fuzzy fallback stage.
    fuzzy_threshold: float = 0.92
    # Fuzzy blocking: candidates must share this many leading characters.
    fuzzy_prefix: int = 2
    # ...and their lengths must differ by no more than this.
    fuzzy_len_tolerance: int = 3
    # Strings shorter than this are not fuzzy-matched (too noisy).
    fuzzy_min_len: int = 4

    # A phonetic key match must ALSO clear this string similarity. Koelner
    # codes are only four to six digits and collide freely -- "Potsdam" and
    # "Boston" are both 1826 -- so the phonetic rung on its own will happily
    # pair unrelated names. The distance gate cannot catch these when the
    # collision happens to be a nearby European place. Marburg/Marpurg scores
    # about 0.95 here, Potsdam/Boston about 0.5.
    phonetic_min_similarity: float = 0.75

    # --- Match quality by method ----------------------------------------
    # These are the string_score values. Ordered, not calibrated.
    method_scores: dict[str, float] = field(
        default_factory=lambda: {
            "override": 1.00,  # curated forced mapping
            "exact": 1.00,  # raw string equals a TGN term
            "correction": 0.98,  # curated string rewrite, then exact
            "fold": 0.92,  # case/umlaut/punctuation folding
            "modernize": 0.85,  # orthographic modernization (c/k, th/t, y/i)
            "strip": 0.75,  # administrative qualifier removed
            "phonetic": 0.60,  # Koelner Phonetik key match
            "fuzzy": 0.50,  # edit-distance fallback
        }
    )

    # --- Distance prior --------------------------------------------------
    # Log10-km bin edges for the per-school origin-distance distribution.
    # The FIRST edge must be strictly greater than 0: bin 0 is the interval
    # log10(1+d) < edge_0, and with edge_0 = 0.0 no non-negative distance can
    # ever fall in it. The bin stays empty, is fitted as pure tail floor, and
    # the midpoint interpolator then drags a candidate standing AT the
    # university (d = 0, logd = 0) halfway toward that floor -- scoring the
    # most likely origin in the corpus as if it were implausible. 0.3 puts
    # [0, 1) km in bin 0.
    #
    # The last edge must stay below the antipodal distance (~20,015 km, i.e.
    # log10 ~ 4.30), or the open-ended final bin gets a nonsensically small
    # width and becomes the most attractive place on Earth to come from.
    dist_bin_edges: tuple[float, ...] = (
        0.3, 0.5, 1.0, 1.3, 1.6, 1.9, 2.2, 2.5, 3.0, 3.5, 4.0,
    )
    # Dirichlet smoothing toward the pooled distribution. Higher = more
    # shrinkage toward the global shape, which guards against overfitting a
    # school with few unambiguous students.
    dist_prior_strength: float = 50.0
    # Floor mixed into every bin, as a fraction of uniform. This is what keeps
    # the long tail alive: it guarantees a distant candidate is never assigned
    # a vanishing prior just because that school rarely draws from there.
    dist_tail_floor: float = 0.05
    # Shape of that floor. A floor spread uniformly across log-spaced bins puts
    # as much protective mass on the 10,000 km bin as on the 50 km bin, which
    # is what let obscure antipodean homonyms outscore the obvious answer. The
    # floor is instead shaped like the pooled distribution, blended with this
    # much uniform so that no bin is ever unreachable.
    dist_floor_uniform_share: float = 0.15

    # --- Cross-field coherence -------------------------------------------
    # Distance scale (km) at which region/hometown disagreement stops mattering.
    coherence_scale_km: float = 75.0
    # Coherence is evaluated between the top-k candidates of each field, at the
    # level of distinct (region, hometown) string pairs rather than per student.
    coherence_topk: int = 8
    # Stand-in distance used when the other field exists but this candidate has
    # no cross-distance. Neutral, deliberately not zero.
    coherence_null_km: float = 500.0

    # --- Coordinates -------------------------------------------------------
    # Flat score penalty for a candidate place that has no coordinates. All
    # else equal, a locatable place is the better answer, and making that
    # explicit is safer than letting it fall out of the distance term.
    no_coords_penalty: float = 1.5

    # --- Execution ---------------------------------------------------------
    # Resolution keys scored per chunk. Peak memory is roughly
    # chunk_size x 2 fields x max_candidates_per_string rows, so this bounds
    # memory regardless of how ambiguous the corpus turns out to be.
    chunk_size: int = 25_000

    # --- Territories -------------------------------------------------------
    # How a territory name in the source data is resolved.
    #   "seat"  apply the territory_seat corrections, mapping e.g.
    #           Mecklenburg-Schwerin to the town of Schwerin
    #   "self"  ignore them, letting the territory resolve to its own TGN
    #           subject (the duchy), whose coordinates are its seat or centroid
    # "self" is the safer default if seat corrections are producing surprises:
    # it never round-trips a precise territory reference through an ambiguous
    # town name.
    territory_mode: str = "seat"

    # --- Sub-settlement demotion ------------------------------------------
    # A settlement whose parent is itself a settlement is an Ortsteil: a
    # hamlet inside a municipality. Structurally identifiable without any
    # size data, and almost never what a register means when the name is
    # shared with a town of standing.
    sub_settlement_penalty: float = 1.5

    # --- Decision ---------------------------------------------------------
    # Candidates scoring this far below the leader are not considered rivals
    # when computing the margin.
    margin_cap: float = 10.0
    # Below this margin a resolution is emitted but flagged low-confidence.
    low_confidence_margin: float = 0.75
    # Field preference: emit the region-derived place unless it is coarser than
    # the hometown-derived one by at least this many precision ranks.
    precision_override_gap: int = 1

    # --- Review queue ------------------------------------------------------
    review_queue_size: int = 400


@dataclass
class Config:
    paths: Paths = field(default_factory=Paths)
    weights: Weights = field(default_factory=Weights)
    tuning: Tuning = field(default_factory=Tuning)
    # Set False to force a rebuild of cached artifacts.
    use_cache: bool = True
    verbose: bool = True


# Ordered coarse-to-fine. Used for geo_precision and the field-preference rule.
# TERM.OTHER_FLAGS values that mark a term as a machine code rather than a
# name. These are never what a 19th-century register clerk wrote, and they are
# what "Wm" was matching -- ISO country codes sitting in the term table.
CODE_TERM_FLAGS = {"ISO3L", "ISO2L", "ISO3N", "ISO2N", "USPS", "FIPS"}

# Components of the prominence composite, each applied to log1p(count) except
# has_scope_note which is 0/1. TGN carries no population, so prominence is
# assembled from how richly the gazetteer documents a place -- which for this
# corpus is arguably the better measure anyway: what matters is whether a
# 19th-century clerk could write the name and expect to be understood.
#
# Every component degrades to zero if its source table is absent, so the
# composite works with whatever subset of TGN has been loaded.
PROMINENCE_WEIGHTS = {
    "terms": 1.0,        # distinct recorded names
    "languages": 1.5,    # distinct languages those names span
    "children": 1.0      ,  # places sitting below this one in the hierarchy
    "placetypes": 0.75,  # a city is also a port, a see, a spa
    "bbox": 0.5,         # log1p of bounding-box area in km2
    "sources": 0.75,     # how many sources Getty cites
    "scope_note": 0.5,   # Getty wrote a descriptive note at all
}

# Multiplies string_score. A match on the preferred name is stronger evidence
# than a match on a variant, and a match on the vernacular form is what a local
# writing in German would actually produce.
TERM_QUALITY = {
    "preferred": 1.00,
    "vernacular": 0.95,
    "variant": 0.85,
    "historic_variant": 0.80,
}

PRECISION_RANKS = {
    "settlement": 0,
    "district": 1,
    "territory": 2,
    "country": 3,
    "unknown": 4,
}

# TGN place-type strings mapped to a precision level, matched as case-folded
# substrings in order -- first hit wins, so the list runs specific to general.
#
# This list is a starting point, not an authority. tgn_index prints the most
# frequent place types that fall through to "unknown" on your data; extend the
# list from that report rather than trusting these to be exhaustive.
PLACETYPE_PRECISION = [
    # --- settlement ---
    ("inhabited place", "settlement"),
    ("city", "settlement"),
    ("town", "settlement"),
    ("village", "settlement"),
    ("hamlet", "settlement"),
    ("municipality", "settlement"),
    ("borough", "settlement"),
    ("settlement", "settlement"),
    ("deserted settlement", "settlement"),
    ("abandoned settlement", "settlement"),
    ("ruined settlement", "settlement"),
    ("market town", "settlement"),
    ("royal burgh", "settlement"),
    ("national capital", "settlement"),
    ("state capital", "settlement"),
    ("provincial capital", "settlement"),
    ("capital", "settlement"),
    ("university town", "settlement"),
    ("spa", "settlement"),
    ("resort", "settlement"),
    ("port", "settlement"),
    ("suburb", "settlement"),
    ("neighborhood", "settlement"),
    ("quarter", "settlement"),
    ("commune", "settlement"),
    ("township", "settlement"),
    ("parish", "settlement"),
    # --- district (sub-state administrative) ---
    ("second level subdivision", "district"),
    ("third level subdivision", "district"),
    ("fourth level subdivision", "district"),
    ("arrondissement", "district"),
    ("prefecture", "district"),
    ("kreis", "district"),
    ("district", "district"),
    ("county", "district"),
    ("departement", "district"),
    ("department", "district"),
    ("circle", "district"),
    # --- territory (state / historic polity) ---
    ("first level subdivision", "territory"),
    ("grand duchy", "territory"),
    ("duchy", "territory"),
    ("principality", "territory"),
    ("electorate", "territory"),
    ("landgraviate", "territory"),
    ("margraviate", "territory"),
    ("archbishopric", "territory"),
    ("bishopric", "territory"),
    ("kingdom", "territory"),
    ("empire", "territory"),
    ("province", "territory"),
    ("governorate", "territory"),
    ("oblast", "territory"),
    ("canton", "territory"),
    ("state", "territory"),
    ("region", "territory"),
    ("territory", "territory"),
    ("former administrative", "territory"),
    # --- country ---
    ("sovereign state", "country"),
    ("nation", "country"),
    ("country", "country"),
]

_LEGACY_PLACETYPE_PRECISION = [
    ("inhabited place", "settlement"),
    ("city", "settlement"),
    ("town", "settlement"),
    ("village", "settlement"),
    ("hamlet", "settlement"),
    ("municipality", "settlement"),
    ("neighborhood", "settlement"),
    ("district", "district"),
    ("county", "district"),
    ("kreis", "district"),
    ("department", "district"),
    ("province", "territory"),
    ("state", "territory"),
    ("region", "territory"),
    ("duchy", "territory"),
    ("principality", "territory"),
    ("kingdom", "territory"),
    ("historical region", "territory"),
    ("nation", "country"),
    ("country", "country"),
]
