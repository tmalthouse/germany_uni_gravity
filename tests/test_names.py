import pytest

from unis.linkage.names import canon_surname, initials, koelner


@pytest.mark.parametrize("name, code", [
    ("Mueller", "657"), ("Müller", "657"), ("Miller", "657"),
    ("Schmidt", "862"), ("Schmitt", "862"), ("Schmid", "862"),
    ("Meyer", "67"), ("Meier", "67"), ("Maier", "67"), ("Mayr", "67"),
    ("Wikipedia", "3412"), ("Breschnew", "17863"),
])
def test_koelner_phonetik(name, code):
    assert koelner(name) == code


def test_surname_particle_split():
    # umlaut expanded (ü -> ue), then doubled letters collapsed (ll -> l)
    assert canon_surname("von Müller") == ("von", "mueler")


def test_initials_fold_latin_and_c_k():
    assert initials("Carolus Fridericus") == initials("Karl Friedrich") == "kf"
