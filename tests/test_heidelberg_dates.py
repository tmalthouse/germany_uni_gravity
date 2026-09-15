import json

from unis.registers.heidelberg_dates import start_pages


def test_start_pages_keep_the_first_head_and_repair_an_isolated_misread(tmp_path):
    heads = {
        "1": "1. Mai 1807",
        "2": "3. Juni 1807",
        "3": "1. Januar 1808",
        "4": "2. Februar 1841)",  # misread head: should be 1809
        "5": "3. März 1809",
        "6": "1. Januar 1810",
    }
    f = tmp_path / "heads.json"
    f.write_text(json.dumps(heads))

    table = start_pages("heidelberg_test", f, 1, 6)

    assert table["source_page"].to_list() == [1, 3, 4, 6]
    assert table["year_helper"].to_list() == [1807, 1808, 1809, 1810]


def test_start_pages_respect_the_page_range(tmp_path):
    f = tmp_path / "heads.json"
    f.write_text(json.dumps({"1": "1805", "2": "1806", "3": "1807"}))
    assert start_pages("v", f, 2, 3)["source_page"].to_list() == [2, 3]
