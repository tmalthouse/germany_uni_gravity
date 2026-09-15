"""Distance computation and the per-school origin-distance prior.

The prior is the piece that fixes the failure mode of a naive nearest-match
rule. Instead of asserting that the closest candidate wins, we estimate, per
university, how far its students actually came from, and score candidates by
how plausible their distance is under that distribution.

Fitting set: students whose source string has exactly one candidate. Those are
unambiguous by construction, so they need no resolution to be usable as
evidence.

Two deliberate distortions of the empirical fit:

1. Shrinkage toward the pooled distribution, so a school with few unambiguous
   students does not get a spiky prior.
2. A tail floor mixed into every bin. Unambiguous strings are *not* a random
   sample -- distinctive foreign names like 'Philadelphia' are unambiguous
   while common German village names are not -- so the raw fit understates the
   long tail for some schools and overstates it for others. The floor makes the
   prior heavy-tailed by construction rather than by estimate, which is the
   conservative choice when the thing you most want to avoid is confidently
   relocating a genuine foreign student onto a nearby hamlet.
"""

from __future__ import annotations

import math

import polars as pl

from .config import Config

EARTH_RADIUS_KM = 6371.0


def haversine_expr(
    lat0: str = "LAT0", lon0: str = "LON0", lat1: str = "LAT1", lon1: str = "LON1"
) -> pl.Expr:
    """Great-circle distance in km between two coordinate column pairs."""
    rad = math.pi / 180
    return (2 * EARTH_RADIUS_KM) * (
        (
            ((pl.col(lat1) - pl.col(lat0)) * rad / 2).sin().pow(2)
            + pl.col(lat0).mul(rad).cos()
            * pl.col(lat1).mul(rad).cos()
            * ((pl.col(lon1) - pl.col(lon0)) * rad / 2).sin().pow(2)
        )
        .sqrt()
        .arcsin()
    )


# Bin index used when the distance cannot be computed at all (the candidate or
# the university has no coordinates). It gets its own row in the prior table
# carrying the school's expected log-density -- see fit_prior.
UNKNOWN_BIN = -1

# Smallest bin width in km. Guards the degenerate first bin (the edges start at
# log10(1+d) = 0, which is 0 km) against dividing by zero.
MIN_BIN_WIDTH_KM = 1.0
# Nominal upper bound for the open-ended final bin.
MAX_DIST_KM = 20_000.0


def bin_geometry(cfg: Config) -> pl.DataFrame:
    """Width in km and midpoint in log-space for every distance bin.

    Width is the crux. The bins are log-spaced, so the 9-19 km bin is 10 km
    wide while the 315-999 km bin is 684 km wide. Comparing the probability
    *mass* of those bins asks "is a student more likely to come from somewhere
    in a 10 km band or somewhere in a 684 km band", which the wide band wins by
    construction, and systematically penalises candidates near the university.
    The quantity that actually compares candidates is the density: mass per km.
    """
    edges = cfg.tuning.dist_bin_edges
    km = [10**e - 1 for e in edges]
    rows = []
    lower_log, lower_km = None, 0.0
    for i, (edge_log, edge_km) in enumerate(zip(edges, km)):
        width = max(edge_km - lower_km, MIN_BIN_WIDTH_KM)
        mid = edge_log - 0.25 if lower_log is None else (lower_log + edge_log) / 2
        rows.append({"dist_bin": i, "width_km": width, "mid_log": mid})
        lower_log, lower_km = edge_log, edge_km
    rows.append(
        {
            "dist_bin": len(edges),
            "width_km": max(MAX_DIST_KM - lower_km, MIN_BIN_WIDTH_KM),
            "mid_log": lower_log + 0.25,
        }
    )
    return pl.DataFrame(rows, schema={"dist_bin": pl.Int32, "width_km": pl.Float64,
                                      "mid_log": pl.Float64})


def bin_expr(dist_col: str, cfg: Config) -> pl.Expr:
    """Assign a distance in km to a log-spaced bin index.

    A null distance maps to UNKNOWN_BIN rather than to null. Leaving it null
    and later filling the log-density with 0.0 is a trap: every real
    log-density is negative, so a candidate with no coordinates would score
    strictly better than one with, and unlocatable places would win ties they
    should lose.
    """
    edges = cfg.tuning.dist_bin_edges
    logd = (pl.col(dist_col).clip(lower_bound=0.0) + 1.0).log10()
    expr = pl.when(pl.col(dist_col).is_null()).then(
        pl.lit(UNKNOWN_BIN, dtype=pl.Int32)
    )
    for i, edge in enumerate(edges):
        expr = expr.when(logd < edge).then(pl.lit(i, dtype=pl.Int32))
    return expr.otherwise(pl.lit(len(edges), dtype=pl.Int32))


def n_bins(cfg: Config) -> int:
    return len(cfg.tuning.dist_bin_edges) + 1


def fit_prior(fitting_rows: pl.DataFrame, cfg: Config) -> pl.DataFrame:
    """Estimate log P(distance bin | school).

    `fitting_rows` needs columns: school_fixed, dist.
    Returns: school_fixed, dist_bin, log_density.
    """
    nb = n_bins(cfg)
    t = cfg.tuning

    rows = fitting_rows.filter(pl.col("dist").is_not_null()).with_columns(
        dist_bin=bin_expr("dist", cfg)
    )

    schools = fitting_rows.select("school_fixed").unique()
    grid = schools.join(
        pl.DataFrame({"dist_bin": list(range(nb))}, schema={"dist_bin": pl.Int32}),
        how="cross",
    )

    # Pooled distribution across all schools.
    pooled = (
        rows.group_by("dist_bin")
        .agg(pl.len().alias("n"))
        .with_columns(p_pooled=pl.col("n") / pl.col("n").sum())
        .select("dist_bin", "p_pooled")
    )
    pooled = (
        pl.DataFrame({"dist_bin": list(range(nb))}, schema={"dist_bin": pl.Int32})
        .join(pooled, on="dist_bin", how="left")
        .with_columns(pl.col("p_pooled").fill_null(0.0))
    )
    # Guard against an entirely empty fitting set.
    if pooled["p_pooled"].sum() == 0:
        pooled = pooled.with_columns(p_pooled=pl.lit(1.0 / nb))

    per_school = (
        rows.group_by(["school_fixed", "dist_bin"]).agg(pl.len().alias("n"))
    )

    out = (
        grid.join(per_school, on=["school_fixed", "dist_bin"], how="left")
        .with_columns(pl.col("n").fill_null(0))
        .join(pooled, on="dist_bin", how="left")
        .with_columns(n_school=pl.col("n").sum().over("school_fixed"))
        .with_columns(
            # Dirichlet posterior mean with the pooled distribution as prior.
            p_raw=(pl.col("n") + t.dist_prior_strength * pl.col("p_pooled"))
            / (pl.col("n_school") + t.dist_prior_strength)
        )
        .join(bin_geometry(cfg).select("dist_bin", "width_km"), on="dist_bin", how="left")
        .with_columns(
            # The floor's uniform component is uniform in *distance*, not
            # across bins: its mass is proportional to bin width, so after the
            # mass-to-density conversion it contributes a constant density at
            # every distance. Spreading it evenly across bins instead would put
            # the same mass in a 10 km band as in a 6,800 km one.
            floor_shape=(1 - t.dist_floor_uniform_share) * pl.col("p_pooled")
            + t.dist_floor_uniform_share
            * pl.col("width_km")
            / pl.col("width_km").sum()
        )
        .with_columns(
            p=(1 - t.dist_tail_floor) * pl.col("p_raw")
            + t.dist_tail_floor * pl.col("floor_shape")
        )
        .with_columns(p=pl.col("p") / pl.col("p").sum().over("school_fixed"))
        # Mass -> density. Without this the score rewards wide bins, which are
        # the distant ones.
        .with_columns(log_density=(pl.col("p") / pl.col("width_km")).log())
        .join(bin_geometry(cfg).select("dist_bin", "mid_log"), on="dist_bin", how="left")
        .select("school_fixed", "dist_bin", "p", "log_density", "mid_log")
    )

    # Row for the unknown-distance case: the expected log-density under the
    # school's own prior. This is the value that carries no information either
    # way -- it cannot beat the school's best bin, and cannot lose to its
    # worst.
    unknown = (
        out.group_by("school_fixed")
        .agg((pl.col("p") * pl.col("log_density")).sum().alias("log_density"))
        .with_columns(
            dist_bin=pl.lit(UNKNOWN_BIN, dtype=pl.Int32),
            mid_log=pl.lit(None, dtype=pl.Float64),
        )
        .select("school_fixed", "dist_bin", "log_density", "mid_log")
    )

    return pl.concat([out.drop("p"), unknown])


def interpolated_log_density(
    frame: pl.DataFrame, prior: pl.DataFrame, cfg: Config
) -> pl.DataFrame:
    """Attach dist_ll, interpolating between bin midpoints.

    Evaluating the histogram as a step function makes every candidate inside a
    bin score identically -- two places 335 km and 384 km away come out exactly
    tied, and the tiebreak then falls to SUBJECT_ID order. Linear interpolation
    in log-distance between neighbouring bin midpoints removes those ties and
    smooths the bin edges.
    """
    left = prior.rename({"log_density": "ld_b", "mid_log": "mid_b"})
    right = prior.select(
        "school_fixed",
        pl.col("dist_bin").alias("nb"),
        pl.col("log_density").alias("ld_n"),
        pl.col("mid_log").alias("mid_n"),
    )

    out = (
        frame.with_columns(logd=(pl.col("dist").clip(lower_bound=0.0) + 1.0).log10())
        .join(left, on=["school_fixed", "dist_bin"], how="left")
        .with_columns(
            nb=pl.when(pl.col("mid_b").is_null())
            .then(None)
            .when(pl.col("logd") > pl.col("mid_b"))
            .then(pl.col("dist_bin") + 1)
            .otherwise(pl.col("dist_bin") - 1)
        )
        .join(right, on=["school_fixed", "nb"], how="left")
        .with_columns(
            _t=(
                (pl.col("logd") - pl.col("mid_b"))
                / (pl.col("mid_n") - pl.col("mid_b"))
            ).clip(0.0, 1.0)
        )
        .with_columns(
            dist_ll=pl.when(pl.col("ld_n").is_null() | pl.col("mid_b").is_null())
            .then(pl.col("ld_b"))
            .otherwise(
                pl.col("ld_b") + pl.col("_t") * (pl.col("ld_n") - pl.col("ld_b"))
            )
        )
        .drop("ld_b", "mid_b", "nb", "ld_n", "mid_n", "_t", "logd")
    )
    return out


def coherence_logdensity(dist_col: str, has_other: str, cfg: Config) -> pl.Expr:
    """Score agreement between the two source fields.

    A soft, heavy-tailed decay rather than a hard constraint: the fields
    genuinely disagree sometimes (a student's region and hometown can be
    unrelated administrative levels), so this nudges rather than forces.

    Three cases, and the distinction matters:
      - the other field is absent  -> 0 for every candidate, no signal, fair
      - a cross-distance is known  -> the decay
      - the other field is present but this candidate has no cross-distance
        -> a fixed neutral value, never 0, for the same reason as UNKNOWN_BIN
    """
    scale = cfg.tuning.coherence_scale_km
    neutral = -math.log1p(cfg.tuning.coherence_null_km / scale)
    return (
        pl.when(~pl.col(has_other))
        .then(pl.lit(0.0))
        .when(pl.col(dist_col).is_null())
        .then(pl.lit(neutral))
        .otherwise(-(1.0 + pl.col(dist_col) / scale).log())
    )
