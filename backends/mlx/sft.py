"""MLX SFT backend for Apple Silicon (LoRA / full / "QLoRA").

Apple's MLX runs natively on the unified-memory GPU and is the fastest practical
way to fine-tune on an M-series Mac. We drive ``mlx-lm``'s training entry point
rather than re-implementing the loop, because it already handles quantized bases,
LoRA injection, prompt masking, and checkpointing.

Technique mapping:
- ``lora``  -> ``--fine-tune-type lora`` on the fp16 HF base (mlx-lm converts it
  to MLX format in-memory).
- ``qlora`` -> LoRA on a *quantized* MLX base. There's no bitsandbytes here;
  instead ``mlx_model_id`` points at a 4-bit conversion (``*-4bit``). Training
  LoRA on a 4-bit base ≈ QLoRA.
- ``full``  -> ``--fine-tune-type full`` (only sensible for the smallest models).

We shell out to ``python -m mlx_lm lora`` so we're robust to mlx-lm's internal
API churn. The LoRA *shape* (rank/scale/dropout) is passed via mlx-lm's ``-c``
config file; everything operational is passed as CLI flags.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

from backends.base import TrainResult
from core.config import REPO_ROOT, RunConfig, slugify
from core.data import load_examples
from core.hardware import TrainingPlan

# Iterations to run when the technique config doesn't pin max_steps.
_DEFAULT_ITERS = 200


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _prepare_data_dir(run: RunConfig) -> tuple[Path, int, int]:
    """Materialize train/valid jsonl in mlx-lm's chat format.

    Returns (data_dir, n_train, n_val).
    """
    examples = load_examples(run.data)
    data_dir = Path(run.output_dir) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # mlx-lm needs a valid.jsonl. Hold out a small slice (>=2 so validation can
    # form at least one batch).
    n_val = max(2, min(len(examples) // 10, 50)) if len(examples) > 2 else 1
    val_rows = examples[:n_val]
    train_rows = examples[n_val:] or examples  # never leave train empty

    _write_jsonl(data_dir / "train.jsonl", train_rows)
    _write_jsonl(data_dir / "valid.jsonl", val_rows)
    return data_dir, len(train_rows), len(val_rows)


def _write_lora_config(run: RunConfig, path: Path) -> None:
    """Write mlx-lm's LoRA-shape config (consumed via ``-c``).

    Note: mlx-lm's ``scale`` is a *direct* multiplier on the adapter output,
    whereas PEFT scales by ``alpha / r``. We map ``scale = alpha / r`` as the
    closest analog so the technique YAML stays meaningful across backends.
    """
    cfg = {
        "lora_parameters": {
            "rank": run.technique.lora_r,
            "scale": round(run.technique.lora_alpha / max(run.technique.lora_r, 1), 4),
            "dropout": run.technique.lora_dropout,
        }
    }
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh)


def _mlx_has_weights(path: Path) -> bool:
    """True if `path` holds a modern-format MLX model (model*.safetensors)."""
    return path.exists() and any(path.glob("model*.safetensors"))


def _ensure_mlx_4bit_base(run: RunConfig) -> str:
    """Return a local 4-bit MLX model dir for QLoRA, quantizing on first use.

    We deliberately do NOT depend on community ``*-4bit`` repos (formats drift
    and many are stale). Instead we quantize the *official* ``hf_model_id`` with
    ``mlx_lm convert`` into a cached ``models/<slug>-mlx-4bit`` folder. This runs
    only when the cache is missing, and never uploads anything.
    """
    # Honor an explicit local override if it's already a usable MLX model.
    override = run.model.mlx_model_id
    if override and _mlx_has_weights(Path(override)):
        return override

    out = REPO_ROOT / "models" / f"{slugify(run.model.hf_model_id)}-mlx-4bit"
    if _mlx_has_weights(out):
        return str(out)

    cmd = [
        sys.executable, "-m", "mlx_lm", "convert",
        "--hf-path", run.model.hf_model_id,
        "-q", "--q-bits", "4", "--q-group-size", "64",
        "--mlx-path", str(out),
    ]
    if run.model.trust_remote_code:
        cmd.append("--trust-remote-code")
    print("[mlx] one-time 4-bit quantization of the official base:", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0 or not _mlx_has_weights(out):
        raise RuntimeError(
            f"mlx_lm convert failed (code {result.returncode}). Quantize manually "
            "with `python -m mlx_lm convert --hf-path <id> -q ...`, or use --technique lora."
        )
    return str(out)


def _resolve_model_id(run: RunConfig) -> str:
    if run.technique.technique == "qlora":
        return _ensure_mlx_4bit_base(run)
    # lora / full: train on the fp16 HF weights (mlx-lm converts in-memory).
    return run.model.hf_model_id


def train(run: RunConfig, plan: TrainingPlan) -> TrainResult:
    import importlib.util

    if importlib.util.find_spec("mlx_lm") is None:
        raise RuntimeError("mlx-lm is not installed. On Apple Silicon run: uv sync --extra mlx")

    data_dir, n_train, n_val = _prepare_data_dir(run)
    model_id = _resolve_model_id(run)
    adapter_path = Path(run.output_dir) / "adapters"
    adapter_path.mkdir(parents=True, exist_ok=True)

    # Clamp batch to the dataset size so tiny sets still form train/val batches.
    batch_size = max(1, min(plan.per_device_batch_size, n_train, n_val))

    iters = (
        run.technique.max_steps
        if run.technique.max_steps and run.technique.max_steps > 0
        else _DEFAULT_ITERS
    )
    is_lora = run.technique.technique != "full"
    fine_tune_type = "lora" if is_lora else "full"

    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", model_id,
        "--train",
        "--data", str(data_dir),
        "--fine-tune-type", fine_tune_type,
        "--num-layers", "-1",                 # adapt all layers (parity with HF/PEFT)
        "--batch-size", str(batch_size),
        "--iters", str(iters),
        "--grad-accumulation-steps", str(plan.gradient_accumulation_steps),
        "--learning-rate", str(run.technique.learning_rate),
        "--max-seq-length", str(plan.max_seq_len),
        "--adapter-path", str(adapter_path),
        "--steps-per-report", str(run.technique.logging_steps),
        "--save-every", str(run.technique.save_steps),
        "--val-batches", "-1",                # use the whole (small) val set
        "--mask-prompt",                      # learn from assistant tokens only
        "--seed", str(run.technique.seed),
    ]
    if is_lora:
        cfg_path = Path(run.output_dir) / "mlx_lora_config.yaml"
        _write_lora_config(run, cfg_path)
        cmd += ["-c", str(cfg_path)]
    if plan.gradient_checkpointing:
        cmd += ["--grad-checkpoint"]

    print("[mlx] launching:", " ".join(cmd))
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"mlx_lm lora exited with code {completed.returncode}")

    return TrainResult(
        backend="mlx",
        output_dir=run.output_dir,
        adapter_path=str(adapter_path) if is_lora else None,
        steps=iters,
        extra={
            "model_id": model_id,
            "iters": iters,
            "batch_size": batch_size,
            "data_dir": str(data_dir),
        },
    )
