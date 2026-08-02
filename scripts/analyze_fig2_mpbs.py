#!/usr/bin/env python3
"""MPBS analysis entrypoint for Fig. 2.

The public data release ships the processed tables under ``analysis/fig2_mpbs``.
Regenerating those tables from raw wavefunctions is an internal compute step and
is not required to rebuild the published figure from the shipped CSVs.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "fig2_mpbs"


def main() -> None:
    required = [
        OUT / "aggregate_metrics.csv",
        OUT / "per_seed_metrics.csv",
        OUT / "band_metrics.csv",
    ]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
    if missing:
        raise SystemExit("Missing MPBS tables:\n  - " + "\n  - ".join(missing))
    print(f"MPBS tables ready under {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
