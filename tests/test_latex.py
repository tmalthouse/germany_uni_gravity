import math

import pandas as pd

from unis.latex import (
    Tabular,
    escape,
    integer,
    multicolumn,
    num,
    pvalue,
    regression_table,
    se,
    stars,
)


def test_regression_table_leaves_missing_terms_blank():
    tidy = lambda rows: pd.DataFrame(  # noqa: E731
        rows, columns=["Coefficient", "Estimate", "Std. Error", "Pr(>|t|)"]
    ).set_index("Coefficient")
    m1 = tidy([("x", -0.5, 0.1, 0.001)])
    m2 = tidy([("x", 0.2, 0.3, 0.5), ("z", 1.0, 0.4, 0.02)])
    lines = regression_table(
        ["A", "B"], [m1, m2], [("X", ["x", "x"]), ("Z", [None, "z"])],
        footer=[("Observations", ["10", "12"])],
    ).render().splitlines()
    assert lines[2:] == [
        r" & (1) & (2) \\",
        r" & A & B \\",
        r"\midrule",
        r"X & $-$0.500$^{***}$ & 0.200 \\",
        r" & (0.100) & (0.300) \\",
        r"Z &  & 1.000$^{**}$ \\",
        r" &  & (0.400) \\",
        r"\midrule",
        r"Observations & 10 & 12 \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]


def test_pvalue():
    assert pvalue(0.0000007) == "$<$0.001"
    assert pvalue(0.1234) == "0.123"


def test_num_uses_a_typographic_minus_and_never_prints_negative_zero():
    assert num(-0.5234) == "$-$0.523"
    assert num(0.5236) == "0.524"
    assert num(-0.0001) == "0.000"
    assert num(None) == "" and num(math.nan) == ""


def test_standard_errors_and_stars():
    assert se(0.1084) == "(0.108)"
    assert se(None) == ""
    assert stars(0.001) == "$^{***}$"
    assert stars(0.03) == "$^{**}$"
    assert stars(0.07) == "$^{*}$"
    assert stars(0.2) == ""


def test_escape_and_integer():
    assert escape("a_b & 5%") == r"a\_b \& 5\%"
    assert integer(1022770) == "1,022,770"


def test_tabular_is_a_bare_booktabs_tabular():
    t = Tabular("lc")
    t.row("Year", multicolumn(1, "c", "Estimate")).midrule().row("1838", num(-0.523))
    text = t.render()
    assert text.splitlines() == [
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"Year & \multicolumn{1}{c}{Estimate} \\",
        r"\midrule",
        r"1838 & $-$0.523 \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    assert "caption" not in text and "begin{table}" not in text
