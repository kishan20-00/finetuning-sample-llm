#!/usr/bin/env python3
"""Per-family wrapper example — driving the library directly (no CLI).

This is the "thin wrapper" pattern of the hybrid layout: family-specific code
lives here, but all the real logic is shared in `core/`, `techniques/`, and
`backends/`. Every other family can copy this file and swap the model key.

    python families/qwen/example.py            # dry-run plan
    python families/qwen/example.py --train    # actually train
"""

import argparse
import sys
from pathlib import Path

# Repo root on path so this runs without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from techniques import sft  # noqa: E402

FAMILY = "qwen"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--technique", default="lora", choices=["full", "lora", "qlora"])
    ap.add_argument("--data", default="sample")
    ap.add_argument("--train", action="store_true", help="train instead of dry-run")
    args = ap.parse_args()

    sft.run(
        model=FAMILY,
        technique=args.technique,
        dataset=args.data,
        dry_run=not args.train,
    )


if __name__ == "__main__":
    main()
