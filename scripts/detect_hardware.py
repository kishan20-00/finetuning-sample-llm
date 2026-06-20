#!/usr/bin/env python3
"""Print this machine's hardware profile and a suggested training plan.

Safe to run on a fresh checkout *before* installing the ML stack — it only uses
the stdlib (plus rich/psutil if present). Examples:

    python scripts/detect_hardware.py
    python scripts/detect_hardware.py --technique qlora --model-params-b 1.5
    python scripts/detect_hardware.py --json
"""

import sys
from pathlib import Path

# Allow running as a plain script (no install) by adding the repo root to path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.hardware import main  # noqa: E402

if __name__ == "__main__":
    main()
