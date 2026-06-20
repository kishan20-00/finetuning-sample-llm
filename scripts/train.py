#!/usr/bin/env python3
"""Single entrypoint for all SFT runs.

Combines a model family, a technique, and a dataset — the hardware layer adapts
the rest. The same command runs on an RTX 4070 or an Apple M-series Mac.

    # See what the planner decides, without training:
    python scripts/train.py --model qwen --technique qlora --data sample --dry-run

    # Actually train (needs the matching extra installed):
    python scripts/train.py --model qwen --technique lora --data sample

    # Force a backend instead of auto-detect:
    python scripts/train.py --model gemma --technique lora --data sample --backend hf
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import available_datasets, available_models, available_techniques  # noqa: E402
from techniques import sft  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Hardware-adaptive SFT/LoRA/QLoRA runner.")
    parser.add_argument("--model", required=True,
                        help=f"model family (architecture profile). available: {available_models()}")
    parser.add_argument("--model-id", default=None,
                        help="any Hugging Face model id to fine-tune instead of the family default; "
                             "reuses the family's LoRA targets / chat template")
    parser.add_argument("--params-b", type=float, default=None,
                        help="model size in billions (pair with --model-id so the planner sizes correctly)")
    parser.add_argument("--technique", required=True,
                        help=f"technique. available: {available_techniques()}")
    parser.add_argument("--data", required=True,
                        help=f"dataset. available: {available_datasets()}")
    parser.add_argument("--backend", default="auto", choices=["auto", "hf", "mlx"],
                        help="force a backend (default: auto-detect from hardware)")
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--effective-batch", type=int, default=16,
                        help="target effective batch size (filled via grad accumulation)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="cap training steps/iterations (great for a quick smoke test)")
    parser.add_argument("--hf-token", default=None,
                        help="Hugging Face token for gated models / faster downloads. "
                             "Prefer the HF_TOKEN env var (safer than passing it on the CLI).")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the hardware plan and exit without training")
    args = parser.parse_args()

    sft.run(
        model=args.model,
        technique=args.technique,
        dataset=args.data,
        backend=args.backend,
        max_seq_len=args.max_seq_len,
        target_effective_batch=args.effective_batch,
        max_steps=args.max_steps,
        model_id=args.model_id,
        params_b=args.params_b,
        hf_token=args.hf_token,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
