"""Figure conventions shared by the analysis scripts."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import seaborn as sns  # noqa: E402


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")


def mark_event(ax, year, label, *, y_frac=0.92, x_offset=0.15, color="crimson") -> None:
    """Dotted vertical line at `year`, labelled near the top of the axes."""
    ax.axvline(year, color=color, lw=1.2, ls=":")
    ax.text(year + x_offset, ax.get_ylim()[1] * y_frac, label,
            color=color, fontsize=9, va="top")


def save(fig, path: Path) -> None:
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    print(f"wrote {path}")
