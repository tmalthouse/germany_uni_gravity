"""The TGN loader on tiny .out files shaped like the relational release.

Includes the fixed-width padding on flag columns and the undocumented flag
values that occur in real data; the loader must accept those and still reject
genuinely misaligned columns.
"""

from pathlib import Path

import polars as pl
import pytest

from unis.tgn.load import main


def write_out(d: Path, name: str, rows: list[list[str]]) -> None:
    (d / name).write_text("\n".join("\t".join(r) for r in rows) + "\n", encoding="utf-8")


def build(d: Path, term_rows=None) -> None:
    # TERM: AACR2, DISPLAY_DATE, DISPLAY_NAME, DISPLAY_ORDER, END_DATE,
    # HISTORIC_FLAG, OTHER_FLAGS, PREFERRED, START_DATE, SUBJECT_ID, TERM,
    # TERM_ID, VERNACULAR.  Flags padded; PREFERRED uses the undocumented N;
    # HISTORIC_FLAG uses the undocumented LU; DISPLAY_NAME uses I.
    write_out(d, "TERM.out", term_rows if term_rows is not None else [
        ["NA", "", "Y", "1", "", "C ", "NA", "P ", "", "7001", "Marburg", "70011", "V "],
        ["NA", "", "I", "2", "", "H ", "NA", "N ", "", "7001", "Marpurg", "70012", "O "],
        ["NA", "", "N", "1", "", "LU", "ISO3L", "V ", "", "9001", "DEU", "90011", "U "],
    ])
    # SUBJECT: LEGACY_ID, MERGED_STAT, PARENT_KEY, RECORD_TYPE, SORT_ORDER,
    # SPECIAL_PROJ, SUBJECT_ID. LEGACY_ID is numeric for longer than any type
    # inference window before "T2084" -- the reason every column is read as text.
    subject_rows = [
        [str(10_000 + i), "N ", "9001", "A ", "1", "", str(100_000 + i)]
        for i in range(12_000)
    ]
    subject_rows += [
        ["T2084", "N ", "9001", "A ", "1", "", "7001"],
        ["", "N ", "1000000", "A ", "1", "", "9001"],
    ]
    write_out(d, "SUBJECT.out", subject_rows)
    write_out(d, "SUBJECT_RELS.out", [["", "", "C ", "P ", "P ", "", "9001", "7001", "P "]])
    write_out(d, "PTYPE_ROLE.out", [["inhabited place", "1"], ["nation", "2"]])
    write_out(d, "PTYPE_ROLE_RELS.out", [
        ["", "1", "", "C ", "P ", "1", "", "7001"],
        ["", "1", "", "C ", "P ", "2", "", "9001"],
    ])
    coords = ["" for _ in range(34)]
    coords[2], coords[17], coords[33] = "50.802", "8.766", "7001"
    write_out(d, "COORDINATES.out", [coords])


@pytest.fixture
def tgn_dir(tmp_path):
    d = tmp_path / "tgn"
    d.mkdir()
    return d


def test_accepts_padded_flags_and_undocumented_values(tgn_dir, capsys):
    build(tgn_dir)
    main(["--tgn-dir", str(tgn_dir), "--check"])
    assert "TERM.out: 3 rows x 13 cols, structure OK" in capsys.readouterr().out


def test_writes_stripped_flags_and_integer_ids(tgn_dir, tmp_path):
    build(tgn_dir)
    out = tmp_path / "out"
    main(["--tgn-dir", str(tgn_dir), "--out-dir", str(out)])

    terms = pl.read_parquet(out / "tgn_terms.parquet")
    assert set(terms["PREFERRED"].to_list()) <= {"P", "N", "V"}
    subs = pl.read_parquet(out / "tgn_subjects.parquet")
    assert set(subs["RECORD_TYPE"].to_list()) == {"A"}
    assert subs["SUBJECT_ID"].null_count() == 0
    assert 7001 in subs["SUBJECT_ID"].to_list()


def test_rejects_a_misaligned_column(tgn_dir):
    # TERM text where an ID belongs.
    build(tgn_dir, term_rows=[
        ["NA", "", "Y", "1", "", "C ", "NA", "P ", "", "Marburg", "7001", "70011", "V "],
    ])
    with pytest.raises(ValueError, match="SUBJECT_ID"):
        main(["--tgn-dir", str(tgn_dir), "--check"])
