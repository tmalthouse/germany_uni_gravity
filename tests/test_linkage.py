"""The linkage stage gives identical output on repeated runs.

The u sample is set smaller than the number of spells, so an unseeded draw
(the original behaviour) would make the two runs differ.
"""

import random
import subprocess
import sys

import polars as pl

SURNAMES = ["Müller", "Schmidt", "Meyer", "Weber", "Wagner", "Becker", "Schulz",
            "Hoffmann", "Koch", "Richter", "von Holle", "Jäger"]
GIVEN = ["Carl Friedrich", "Johann", "Wilhelm", "Georg Heinrich", "Friedrich",
         "Ludwig", "August", "Carolus"]
SCHOOLS = ["berlin", "bonn", "goettingen", "heidelberg", "jena", "tuebingen"]


def synthetic_registers(n_people: int = 400, seed: int = 0) -> pl.DataFrame:
    """People who enroll at one or two schools, some recorded in several years."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n_people):
        last, first = rng.choice(SURNAMES), rng.choice(GIVEN)
        location = rng.randrange(1000, 1060)
        year = rng.randrange(1815, 1840)
        for school in rng.sample(SCHOOLS, rng.choice([1, 2])):
            for k in range(rng.choice([1, 1, 2, 3])):
                rows.append({
                    "last_name": last,
                    # abbreviated given names, as in the Berlin registers
                    "first_names": first if rng.random() < 0.7 else f"{first[0]}.",
                    "first_year": year + k,
                    "school": school,
                    "location_id": location,
                })
            year += rng.randrange(1, 3)
    return pl.DataFrame(rows)


def run_linkage(src, out) -> pl.DataFrame:
    r = subprocess.run(
        [sys.executable, "-m", "unis.linkage.linkage", str(src),
         "--annual", "berlin,bonn,jena,tuebingen", "--threshold", "0.9",
         "--u-sample", "60", "--out", str(out)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr[-2000:]
    return pl.read_parquet(out)


def test_linkage_is_deterministic(tmp_path):
    src = tmp_path / "students.parquet"
    registers = synthetic_registers()
    registers.write_parquet(src)

    a = run_linkage(src, tmp_path / "a.parquet")
    b = run_linkage(src, tmp_path / "b.parquet")

    assert a.height == registers.height
    assert a["spell_id"].n_unique() > 60  # the u sample really is a sample
    assert a.equals(b)
