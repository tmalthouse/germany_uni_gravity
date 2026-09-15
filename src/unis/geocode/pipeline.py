"""Orchestration. This is the drop-in replacement for the original script."""

from __future__ import annotations

import polars as pl

from . import candidates, checks, resolve, review, tgn_index
from .config import Config


def run(cfg: Config | None = None, write: bool = True) -> pl.DataFrame:
    cfg = cfg or Config()

    students = resolve.load_students(cfg)
    checks.check_preflight(students, cfg)
    if cfg.verbose:
        print(f"[pipeline] {students.height:,} students")

    index = tgn_index.build(cfg)
    # University coordinates are needed during candidate generation now, for
    # the coarse-rung plausibility gate.
    unis = resolve.load_university_coords(cfg)
    cand = candidates.build(students.lazy(), index, unis, cfg)

    keys = resolve.key_table(students)
    if cfg.verbose:
        print(
            f"[pipeline] {keys.height:,} distinct resolution keys "
            f"(region, hometown, school)"
        )

    coherence = resolve.coherence_table(keys, cand, cfg)
    hints = candidates.hint_subjects(students.lazy(), index, cfg)
    if cfg.verbose and hints.height:
        print(f"[pipeline] {hints['src_raw'].n_unique():,} strings carry a resolvable "
              "qualifier hint")
    prior = resolve.fit_distance_prior(keys, cand, unis, cfg)
    best, review_cand = resolve.resolve_keys(
        keys, cand, unis, coherence, prior, cfg, hints
    )

    chosen = resolve.choose_field(best, cfg, unis)
    chosen = resolve.apply_overrides(chosen, keys, index, cfg)

    out = resolve.assemble(students, keys, chosen, cfg)

    stats = checks.check_output(students, out, index, cfg)
    if cfg.verbose:
        print("[pipeline] contract checks passed")
        for k, v in stats.items():
            print(f"           {k:>20}: {v:,.4f}" if isinstance(v, float) else f"           {k:>20}: {v:,}")

    review.build(students, keys, review_cand, best, cfg)
    if write:
        prior_output = cfg.paths.output
        if prior_output.exists():
            review.diff_against(prior_output, out, cfg)
        cfg.paths.output.parent.mkdir(parents=True, exist_ok=True)
        out.write_parquet(cfg.paths.output)
        if cfg.verbose:
            print(f"[pipeline] wrote {out.height:,} rows -> {cfg.paths.output}")

    return out
