"""Supervised fine-tuning orchestration (full / LoRA / QLoRA).

One function, :func:`run`, ties the whole pipeline together:

1. Assemble a :class:`~core.config.RunConfig` from the chosen model / technique /
   dataset YAMLs.
2. Detect the machine and compute a hardware-adapted
   :class:`~core.hardware.TrainingPlan`.
3. Dispatch to the resolved backend (HF or MLX).

The same call works on an RTX 4070 and an Apple M-series Mac — only the plan and
backend differ. Pass ``dry_run=True`` to print the plan without training, which
is the recommended way to *study* what the hardware layer decides.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from backends import get_sft_trainer
from backends.base import TrainResult
from core.auth import configure_hf_token
from core.config import RunConfig
from core.hardware import Backend, TrainingPlan, detect, recommend


@dataclass
class RunOutcome:
    run: RunConfig
    plan: TrainingPlan
    result: Optional[TrainResult]  # None on dry-run


def _print_plan(run: RunConfig, plan: TrainingPlan) -> None:
    print(f"\n=== Run: {run.run_name} ===")
    print(f"  model        : {run.model.display_name} ({run.model.params_b}B)")
    print(f"  technique    : {plan.technique}  (requested: {run.technique.technique})")
    print(f"  backend      : {plan.backend.value}   device: {plan.device}")
    print(f"  dtype        : {plan.torch_dtype}   4-bit: {plan.load_in_4bit}")
    print(f"  batch/accum  : {plan.per_device_batch_size} x {plan.gradient_accumulation_steps}"
          f"  (effective {plan.effective_batch_size})")
    print(f"  grad ckpt    : {plan.gradient_checkpointing}   max_seq_len: {plan.max_seq_len}")
    print(f"  est. weights : {plan.est_weight_memory_gb} GiB of {plan.accel_memory_gb} GiB"
          f"   fits: {plan.fits}")
    for note in plan.notes:
        print(f"  ! {note}")
    print()


def run(
    model: str,
    technique: str,
    dataset: str,
    *,
    backend: str = "auto",
    max_seq_len: int = 1024,
    target_effective_batch: int = 16,
    max_steps: Optional[int] = None,
    model_id: Optional[str] = None,
    params_b: Optional[float] = None,
    hf_token: Optional[str] = None,
    dry_run: bool = False,
) -> RunOutcome:
    # Make a HF token available to every download (HF backend, MLX subprocess,
    # 4-bit convert). Falls back to the HF_TOKEN env var / cached login.
    if configure_hf_token(hf_token):
        print("[auth] Hugging Face token configured.")

    run_config = RunConfig.assemble(model, technique, dataset, model_id=model_id, params_b=params_b)
    if max_steps is not None:
        # CLI override — handy for quick smoke tests (e.g. --max-steps 30).
        run_config.technique.max_steps = max_steps

    profile = detect()
    prefer = None if backend == "auto" else Backend(backend)
    plan = recommend(
        profile,
        run_config.model.params_b,
        run_config.technique.technique,
        target_effective_batch=target_effective_batch,
        max_seq_len=max_seq_len,
        prefer_backend=prefer,
    )

    _print_plan(run_config, plan)

    if dry_run:
        print("[dry-run] skipping training. Re-run without --dry-run to train.")
        return RunOutcome(run=run_config, plan=plan, result=None)

    trainer = get_sft_trainer(plan.backend)
    result = trainer(run_config, plan)
    print(f"\n[done] artifacts in: {result.output_dir}")
    return RunOutcome(run=run_config, plan=plan, result=result)
