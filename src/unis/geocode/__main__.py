"""Geocode student hometowns against TGN.

    python -m unis.geocode --no-cache     # what `make` runs
    python -m unis.geocode --dry-run      # checks + review queue, writes no output
"""

import argparse

from unis.geocode import Config, Paths, run
from unis.geocode.config import Tuning


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-cache", action="store_true",
                        help="rebuild the TGN index and candidate table")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--territory-mode", choices=["seat", "self"], default=None,
                        help="'seat' maps territories to their capital (default); "
                             "'self' resolves them to the territory's own TGN subject")
    parser.add_argument("--dry-run", action="store_true",
                        help="run everything, including checks, but do not write output")
    args = parser.parse_args()

    tuning = Tuning()
    if args.territory_mode:
        tuning.territory_mode = args.territory_mode

    cfg = Config(
        paths=Paths(),
        tuning=tuning,
        use_cache=not args.no_cache,
        verbose=not args.quiet,
    )
    run(cfg, write=not args.dry_run)


if __name__ == "__main__":
    main()
