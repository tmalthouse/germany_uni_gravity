r"""booktabs ``tabular`` output for the analysis tables.

Writers produce only a ``tabular`` environment: caption, label, notes and the
float belong to the document that ``\input``s the file. The document needs
``\usepackage{booktabs}``.

Conventions: estimates to three decimals with a typographic minus, standard
errors in parentheses, significance stars only where a table reports p-values
(* p < 0.10, ** p < 0.05, *** p < 0.01).
"""

from __future__ import annotations

import math
from pathlib import Path

_LATEX_SPECIALS = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
    "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
}


def _missing(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def escape(text: str) -> str:
    """Escape LaTeX special characters in plain text."""
    return "".join(_LATEX_SPECIALS.get(ch, ch) for ch in str(text))


def num(x, digits: int = 3) -> str:
    """A number with a typographic minus sign; blank if missing."""
    if _missing(x):
        return ""
    s = f"{x:.{digits}f}"
    if s.startswith("-"):
        s = s[1:]
        return s if float(s) == 0 else f"$-${s}"  # never print -0.000
    return s


def se(x, digits: int = 3) -> str:
    """A standard error in parentheses; blank if missing."""
    return "" if _missing(x) else f"({num(x, digits)})"


def stars(p) -> str:
    if _missing(p):
        return ""
    for threshold, mark in ((0.01, "***"), (0.05, "**"), (0.10, "*")):
        if p < threshold:
            return f"$^{{{mark}}}$"
    return ""


def pvalue(p) -> str:
    """A p-value to three decimals, or '$<$0.001'."""
    if _missing(p):
        return ""
    return "$<$0.001" if p < 0.001 else num(p, 3)


def integer(n) -> str:
    """An integer with thousands separators."""
    return "" if _missing(n) else f"{int(n):,}"


def multicolumn(n: int, align: str, text: str) -> str:
    return rf"\multicolumn{{{n}}}{{{align}}}{{{text}}}"


class Tabular:
    """A booktabs tabular built row by row.

    The top and bottom rules are added on render; call ``midrule`` after the
    header yourself.
    """

    def __init__(self, colspec: str):
        self.colspec = colspec
        self.lines: list[str] = []

    def row(self, *cells: str) -> Tabular:
        self.lines.append(" & ".join(cells) + r" \\")
        return self

    def midrule(self) -> Tabular:
        self.lines.append(r"\midrule")
        return self

    def cmidrule(self, first: int, last: int, trim: str = "lr") -> Tabular:
        self.lines.append(rf"\cmidrule({trim}){{{first}-{last}}}")
        return self

    def space(self) -> Tabular:
        self.lines.append(r"\addlinespace")
        return self

    def render(self) -> str:
        return "\n".join([
            rf"\begin{{tabular}}{{{self.colspec}}}",
            r"\toprule",
            *self.lines,
            r"\bottomrule",
            r"\end{tabular}",
            "",
        ])

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.render(), encoding="utf-8")
        print(f"wrote {path}")
