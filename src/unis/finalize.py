"""Derived student attributes and linkage ids.

Adds a heuristic nobility flag, the 1800-1848 sample restriction, era, the
five field groups, school-type flags, father occupation classes, and attaches
spell_id / student_id from the linkage stage record by record.

students_geocoded.parquet + students_linked.parquet -> students_clean.parquet
"""

import polars as pl

from unis import paths
from unis.mappings import fathers

# German nouns are always capitalized, so these are matched case-sensitively.
# \b word boundaries prevent matching titles embedded inside other words.
NOBLE_TITLE_PATTERN = (
    r"\b(Fürst|Fürstin|Erbprinz\w*|Prinz|Prinzessin|Herzog|Herzogin|"
    r"Markgraf|Markgräfin|Landgraf|Landgräfin|Pfalzgraf|Pfalzgräfin|"
    r"Graf|Gräfin|Freiherr|Freifrau|Freiin|Baron|Baronin|Ritter|Edler|Edle)\b"
)

# von/zu family of particles, spelled out in full. (?i) makes this
# case-insensitive since capitalization at the start of a name field is
# inconsistent across sources.
PARTICLE_PATTERN = r"(?i)\b(von und zu|von der|von dem|vom|von|zur|zum|zu)\b"

# Abbreviated form, e.g. "Louis v. Thümen" -- common in later/more formal
# records (Gotha-style directories abbreviated the particle to "v." precisely
# to mark it as the noble particle rather than the ordinary preposition).
# Matched CASE-SENSITIVELY (lowercase only): a capital "V." followed by a
# period is much more likely to be a given-name initial than the nobility
# marker, since the abbreviation convention keeps it lowercase.
ABBREVIATED_PARTICLE_PATTERN = r"\bv\.\s?(?:d\.|u\.\s?z\.)?\b|\bz\.\b"

# TGN location_ids that name a whole state rather than a place in it.
STATE_ONLY_LOCATIONS = [
    7007552,  # Schlesien
    7203848,  # Braunschweig
    7003675,  # Mecklenburg
    7003678,  # Hessen
    1036938,  # Hessen (second record)
    7161148,  # Holstein
    7005172,  # Baden
    7119483,  # Preußen
    7003669,  # Bayern
]


def classify_german_nobility(
    df: pl.DataFrame,
    first_col: str = "first_names",
    last_col: str = "last_name",
    known_noble_bare: list[str] | None = None,
    known_commoner_von: list[str] | None = None,
) -> pl.DataFrame:
    """Heuristic classifier for German noble surnames (early-19th-century context).

    A first-pass heuristic, not ground truth. It will:
      - produce FALSE NEGATIVES for genuinely noble families that carried no
        particle (some Uradel / old untitled nobility, e.g. Grote, Knigge, Vincke)
      - produce FALSE POSITIVES for commoners bearing a locative "von"/"zu" name,
        especially from Rhineland, Westphalia, Lower Saxony, and German-speaking
        Switzerland, where von/zu historically just meant "from [place]"
      - not distinguish rank (untitled von-nobility vs. Freiherr/Graf/Fürst) from
        the boolean column -- use the granular column for that.

    For real accuracy, cross-reference against the Gothaisches Genealogisches
    Taschenbuch or the Genealogisches Handbuch des Adels, and feed confirmed
    exceptions back in via `known_noble_bare` and `known_commoner_von` (left
    empty by default: an incomplete list is worse than an honest "ambiguous").

    Adds two columns:

    nobility_class (str):
        - "known_noble_bare_exception"   : surname in the verified noble list
        - "known_commoner_von_exception" : particle, but verified not noble
        - "titled_noble"                 : contains a title -- high confidence
        - "particle_untitled_ambiguous"  : von/zu/etc. but no title
        - "no_marker_likely_non_noble"   : no title or particle found

    noble (bool): True for titled_noble, particle_untitled_ambiguous and
        known_noble_bare_exception.
    """
    known_noble_bare = set(known_noble_bare or [])
    known_commoner_von = set(known_commoner_von or [])

    return df.with_columns(
        (pl.col(first_col).fill_null("") + " " + pl.col(last_col).fill_null("")).alias("_full_name")
    ).with_columns(
        pl.col("_full_name").str.contains(NOBLE_TITLE_PATTERN).alias("_has_title"),
        (
            pl.col("_full_name").str.contains(PARTICLE_PATTERN)
            | pl.col("_full_name").str.contains(ABBREVIATED_PARTICLE_PATTERN)
        ).alias("_has_particle"),
        pl.col(last_col).is_in(list(known_commoner_von)).alias("_is_known_commoner"),
        pl.col(last_col).is_in(list(known_noble_bare)).alias("_is_known_noble_bare"),
    ).with_columns(
        pl.when(pl.col("_is_known_commoner"))
        .then(pl.lit("known_commoner_von_exception"))
        .when(pl.col("_is_known_noble_bare"))
        .then(pl.lit("known_noble_bare_exception"))
        .when(pl.col("_has_title"))
        .then(pl.lit("titled_noble"))
        .when(pl.col("_has_particle"))
        .then(pl.lit("particle_untitled_ambiguous"))
        .otherwise(pl.lit("no_marker_likely_non_noble"))
        .alias("nobility_class")
    ).with_columns(
        pl.col("nobility_class")
        .is_in(["titled_noble", "particle_untitled_ambiguous", "known_noble_bare_exception"])
        .alias("noble")
    ).drop(["_full_name", "_has_title", "_has_particle", "_is_known_commoner", "_is_known_noble_bare"])


def main() -> None:
    df = pl.read_parquet(paths.STUDENTS_GEOCODED)

    df = classify_german_nobility(df).filter(
        pl.col.first_year >= 1800
    ).filter(
        pl.col.first_year <= 1848
    )

    df = df.with_columns(
        era=(pl.col.first_year - 1800) // 10 + 1
    ).with_columns(
        field_raw=pl.col.field,
        field=pl.when(pl.col.law_admin).then(pl.lit("law/admin"))
        .when(pl.col.theology).then(pl.lit("theology"))
        .when(pl.col.medicine).then(pl.lit("medicine"))
        .when(pl.col.humanities).then(pl.lit("humanities"))
        .when(pl.col.sciences).then(pl.lit("sciences")),
        new_school=pl.col.school.is_in(["bonn", "berlin"]),
        catholic_uni=pl.col.school.is_in(["muenchen", "freiburg", "wuerzburg"]),
        protestant_uni=pl.col.school.is_in([
            "marburg", "jena", "tuebingen", "heidelberg", "erlangen", "goettingen", "kiel",
            "giessen",
        ]),
        only_state=pl.col.location_id.is_in(STATE_ONLY_LOCATIONS),
    ).drop(
        "TERM_ID"
    ).with_columns(
        pl.col.father_occupation.str.split(r",").list.first().replace_strict(
            fathers.occupations_mapping, default=None
        ).replace_strict(
            fathers.code_mapping, default=None
        )
    )

    # Attach linkage ids record by record. The linkage stage sets rec_id to the
    # row number of students_geocoded, the same key as `index` here. This used
    # to join on (last_name, first_names, school, first_year), which is not
    # unique: n records sharing that key became n*n rows (3,960 extra rows,
    # Kiel +41%), and namesakes picked up each other's spell_ids, so after
    # deduplication 590 spells kept another person's hometown. validate='1:1'
    # makes any future key mismatch fail loudly instead of duplicating rows.
    link_table = pl.read_parquet(paths.STUDENTS_LINKED).select(
        pl.col.rec_id.cast(df.schema["index"]).alias("index"),
        "spell_id",
        "student_id",
    )

    df = df.join(link_table, how="left", on="index", validate="1:1")

    df.write_parquet(paths.STUDENTS_CLEAN)
    print(f"wrote {paths.STUDENTS_CLEAN}: {df.shape}")


if __name__ == "__main__":
    main()
