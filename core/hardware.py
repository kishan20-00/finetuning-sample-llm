"""Hardware detection + training adaptation.

This module is the heart of the repo's "analyze the PC, then adapt" idea.
It answers two questions for any machine:

1. *What do I have?*  -> :func:`detect` returns a :class:`HardwareProfile`
   (accelerator, VRAM/RAM, dtype support, whether 4-bit QLoRA is possible, ...).
2. *Given a model + technique, how should I train?* -> :func:`recommend`
   returns a :class:`TrainingPlan` (backend, dtype, batch size, grad accum,
   gradient checkpointing, whether it fits in memory at all).

Design constraints:
- **Zero hard ML dependencies.** It works with only the Python stdlib, so you
  can run `python scripts/detect_hardware.py` on a brand-new box before
  `uv sync --extra hf/mlx`. `torch`, `psutil`, and `rich` are used *if present*
  for better accuracy / output, otherwise it falls back to OS tools.

The numeric heuristics are deliberately conservative starting points, not a
memory simulator. Treat the suggested batch size as "try this, watch for OOM".
"""

from __future__ import annotations

import math
import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional

# ---------------------------------------------------------------------------
# Optional imports — never required for detection to work.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - environment dependent
    import torch  # type: ignore

    _HAS_TORCH = True
except Exception:  # noqa: BLE001
    torch = None  # type: ignore
    _HAS_TORCH = False

try:  # pragma: no cover - environment dependent
    import psutil  # type: ignore

    _HAS_PSUTIL = True
except Exception:  # noqa: BLE001
    psutil = None  # type: ignore
    _HAS_PSUTIL = False


class Accelerator(str, Enum):
    CUDA = "cuda"          # NVIDIA GPU (e.g. RTX 4070)
    MPS = "mps"            # Apple Silicon GPU via Metal
    CPU = "cpu"            # no accelerator


class Backend(str, Enum):
    HF = "hf"              # transformers + trl + peft (CUDA / CPU / MPS)
    MLX = "mlx"            # Apple's MLX framework (Apple Silicon only)


GIB = 1024 ** 3


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------
@dataclass
class HardwareProfile:
    """A snapshot of what the current machine can do."""

    os: str
    arch: str
    accelerator: Accelerator
    is_apple_silicon: bool

    cpu_cores_logical: int
    ram_total_gb: float

    # GPU specifics (CUDA only — MPS shares system RAM)
    gpu_name: Optional[str] = None
    gpu_count: int = 0
    vram_total_gb: float = 0.0
    cuda_compute_capability: Optional[str] = None  # e.g. "8.9" for Ada / 4070

    # Capabilities derived from the above
    supports_bf16: bool = False
    supports_4bit_qlora: bool = False   # bitsandbytes 4-bit, CUDA only
    mlx_available: bool = False
    torch_available: bool = field(default=_HAS_TORCH)

    @property
    def accel_memory_gb(self) -> float:
        """Memory available to the accelerator.

        CUDA has dedicated VRAM; Apple Silicon shares unified memory with the
        OS, so we expose total RAM and let the planner reserve headroom.
        """
        if self.accelerator is Accelerator.CUDA:
            return self.vram_total_gb
        return self.ram_total_gb

    def to_dict(self) -> dict:
        d = asdict(self)
        d["accelerator"] = self.accelerator.value
        return d


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def _detect_ram_gb() -> float:
    if _HAS_PSUTIL:
        return psutil.virtual_memory().total / GIB
    # POSIX fallback
    try:
        if hasattr(os, "sysconf") and "SC_PHYS_PAGES" in os.sysconf_names:
            return (os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")) / GIB
    except Exception:  # noqa: BLE001
        pass
    # macOS fallback
    if platform.system() == "Darwin":
        try:
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
            return int(out.strip()) / GIB
        except Exception:  # noqa: BLE001
            pass
    return 0.0


def _nvidia_smi_query() -> list[dict]:
    """Return per-GPU info via nvidia-smi, or [] if unavailable."""
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:  # noqa: BLE001
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            name = parts[0]
            mem_mb = float(parts[1]) if parts[1].replace(".", "").isdigit() else 0.0
            cc = parts[2] if len(parts) > 2 else None
            gpus.append({"name": name, "vram_gb": mem_mb / 1024, "compute_cap": cc})
    return gpus


def _detect_cuda() -> dict:
    """Detect CUDA GPUs via torch first, then nvidia-smi as a fallback."""
    info = {"present": False, "name": None, "count": 0, "vram_gb": 0.0, "cc": None}
    if _HAS_TORCH and torch.cuda.is_available():  # type: ignore[union-attr]
        info["present"] = True
        info["count"] = torch.cuda.device_count()  # type: ignore[union-attr]
        props = torch.cuda.get_device_properties(0)  # type: ignore[union-attr]
        info["name"] = props.name
        info["vram_gb"] = props.total_memory / GIB
        info["cc"] = f"{props.major}.{props.minor}"
        return info
    gpus = _nvidia_smi_query()
    if gpus:
        info["present"] = True
        info["count"] = len(gpus)
        info["name"] = gpus[0]["name"]
        info["vram_gb"] = gpus[0]["vram_gb"]
        info["cc"] = gpus[0]["compute_cap"]
    return info


def _mlx_importable() -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec("mlx") is not None
    except Exception:  # noqa: BLE001
        return False


def _bitsandbytes_importable() -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec("bitsandbytes") is not None
    except Exception:  # noqa: BLE001
        return False


def detect() -> HardwareProfile:
    """Inspect the current machine and return a :class:`HardwareProfile`."""
    os_name = platform.system()
    arch = platform.machine()
    is_apple_silicon = os_name == "Darwin" and arch in ("arm64", "aarch64")

    cuda = _detect_cuda()
    mps_available = bool(
        _HAS_TORCH
        and getattr(torch.backends, "mps", None)  # type: ignore[union-attr]
        and torch.backends.mps.is_available()  # type: ignore[union-attr]
    )
    # On Apple Silicon, MPS is effectively always present even if torch isn't
    # installed yet; treat the chip as an accelerator we can plan for.
    mps_present = mps_available or is_apple_silicon

    if cuda["present"]:
        accelerator = Accelerator.CUDA
    elif mps_present:
        accelerator = Accelerator.MPS
    else:
        accelerator = Accelerator.CPU

    # bf16 support: Ampere+ on CUDA (compute capability >= 8.0); Apple Silicon
    # supports bf16 on the MPS backend in recent torch builds.
    supports_bf16 = False
    if accelerator is Accelerator.CUDA and cuda["cc"]:
        try:
            supports_bf16 = float(cuda["cc"]) >= 8.0
        except ValueError:
            supports_bf16 = False
    elif accelerator is Accelerator.MPS:
        supports_bf16 = True

    mlx_available = is_apple_silicon and _mlx_importable()
    supports_4bit = accelerator is Accelerator.CUDA and _bitsandbytes_importable()

    return HardwareProfile(
        os=os_name,
        arch=arch,
        accelerator=accelerator,
        is_apple_silicon=is_apple_silicon,
        cpu_cores_logical=os.cpu_count() or 1,
        ram_total_gb=round(_detect_ram_gb(), 1),
        gpu_name=cuda["name"],
        gpu_count=cuda["count"],
        vram_total_gb=round(cuda["vram_gb"], 1),
        cuda_compute_capability=cuda["cc"],
        supports_bf16=supports_bf16,
        supports_4bit_qlora=supports_4bit,
        mlx_available=mlx_available,
        torch_available=_HAS_TORCH,
    )


# ---------------------------------------------------------------------------
# Memory estimation + training plan
# ---------------------------------------------------------------------------
# Rough bytes-of-accelerator-memory per parameter, for the *model + optimizer*
# state alone (activations handled separately, very roughly, via batch size).
#
#   full  : fp16 weights (2) + fp16 grads (2) + fp32 Adam m,v (8) + fp32 master (4)
#   lora  : fp16 frozen base (2) + tiny adapter/opt overhead (~0.3)
#   qlora : 4-bit frozen base (0.5) + tiny adapter/opt overhead (~0.3)
_BYTES_PER_PARAM = {
    "full": 16.0,
    "lora": 2.3,
    "qlora": 0.8,
}


def estimate_weight_memory_gb(model_params_b: float, technique: str) -> float:
    """Estimate accelerator memory for weights + optimizer state, in GiB."""
    bpp = _BYTES_PER_PARAM.get(technique, _BYTES_PER_PARAM["lora"])
    return model_params_b * 1e9 * bpp / GIB


@dataclass
class TrainingPlan:
    """Concrete, hardware-adapted knobs for a training run."""

    backend: Backend
    device: str
    technique: str
    model_params_b: float

    torch_dtype: str            # "bfloat16" | "float16" | "float32"
    load_in_4bit: bool
    gradient_checkpointing: bool
    per_device_batch_size: int
    gradient_accumulation_steps: int
    max_seq_len: int
    dataloader_num_workers: int

    fits: bool                  # does it plausibly fit in memory?
    est_weight_memory_gb: float
    accel_memory_gb: float
    notes: list[str] = field(default_factory=list)

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_batch_size * self.gradient_accumulation_steps

    def to_dict(self) -> dict:
        d = asdict(self)
        d["backend"] = self.backend.value
        d["effective_batch_size"] = self.effective_batch_size
        return d


def recommend(
    profile: HardwareProfile,
    model_params_b: float,
    technique: str = "lora",
    *,
    target_effective_batch: int = 16,
    max_seq_len: int = 1024,
    prefer_backend: Optional[Backend] = None,
) -> TrainingPlan:
    """Turn a hardware profile + model/technique into a concrete training plan.

    Parameters
    ----------
    model_params_b:
        Model size in *billions* of parameters (e.g. 1.5 for a 1.5B model).
    technique:
        One of ``"full"``, ``"lora"``, ``"qlora"``.
    target_effective_batch:
        Desired effective batch size; grad accumulation fills the gap when the
        per-device batch must stay small.
    """
    technique = technique.lower()
    notes: list[str] = []

    # --- pick backend ------------------------------------------------------
    if prefer_backend is not None:
        backend = prefer_backend
    elif profile.is_apple_silicon and profile.mlx_available:
        backend = Backend.MLX
    else:
        backend = Backend.HF

    if technique == "qlora" and not profile.supports_4bit_qlora and backend is Backend.HF:
        notes.append(
            "QLoRA requested but bitsandbytes 4-bit is unavailable here "
            "(CUDA-only). Falling back to plain LoRA (fp16/bf16 base weights)."
        )
        technique = "lora"

    # --- dtype -------------------------------------------------------------
    if profile.accelerator is Accelerator.CPU:
        dtype = "float32"
        notes.append("No accelerator detected — training on CPU will be very slow.")
    elif profile.supports_bf16:
        dtype = "bfloat16"
    else:
        dtype = "float16"

    load_in_4bit = technique == "qlora" and profile.supports_4bit_qlora

    # --- memory feasibility ------------------------------------------------
    weight_mem = estimate_weight_memory_gb(model_params_b, technique)
    # Reserve headroom: CUDA keeps VRAM for the GPU only; Apple shares unified
    # memory with the OS, so reserve more.
    if profile.accelerator is Accelerator.MPS:
        usable = profile.accel_memory_gb * 0.65
    elif profile.accelerator is Accelerator.CUDA:
        usable = profile.accel_memory_gb * 0.90
    else:
        usable = profile.accel_memory_gb * 0.50

    activation_budget = max(usable - weight_mem, 0.0)
    fits = activation_budget > 0.5  # need at least some room for activations

    # --- batch size from leftover memory ----------------------------------
    # Extremely rough: per-sample activation cost grows with model size and
    # sequence length. ~0.5 GiB per sample per 1B params at seq_len 1024.
    per_sample_gb = max(model_params_b, 0.1) * 0.5 * (max_seq_len / 1024)
    if per_sample_gb <= 0:
        max_batch = 1
    else:
        max_batch = int(activation_budget // per_sample_gb)
    per_device_batch = max(1, min(8, max_batch))

    gradient_checkpointing = False
    if not fits or per_device_batch == 1:
        # Tight on memory: trade compute for memory.
        gradient_checkpointing = True
        if not fits:
            notes.append(
                f"Estimated weights ({weight_mem:.1f} GiB) exceed usable memory "
                f"({usable:.1f} GiB). Consider QLoRA, a smaller model, or shorter "
                f"sequences. Plan is generated anyway for study/dry-run."
            )

    grad_accum = max(1, math.ceil(target_effective_batch / per_device_batch))

    num_workers = min(4, max(0, profile.cpu_cores_logical // 2))

    return TrainingPlan(
        backend=backend,
        device=profile.accelerator.value,
        technique=technique,
        model_params_b=model_params_b,
        torch_dtype=dtype,
        load_in_4bit=load_in_4bit,
        gradient_checkpointing=gradient_checkpointing,
        per_device_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum,
        max_seq_len=max_seq_len,
        dataloader_num_workers=num_workers,
        fits=fits,
        est_weight_memory_gb=round(weight_mem, 2),
        accel_memory_gb=round(profile.accel_memory_gb, 1),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Reporting (CLI)
# ---------------------------------------------------------------------------
def _print_plain(profile: HardwareProfile) -> None:
    print("=== Hardware Profile ===")
    for k, v in profile.to_dict().items():
        print(f"  {k:24s}: {v}")


def _print_rich(profile: HardwareProfile) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="Hardware Profile", show_header=True, header_style="bold cyan")
    table.add_column("Property")
    table.add_column("Value", style="green")
    for k, v in profile.to_dict().items():
        table.add_row(k, str(v))
    console.print(table)


def report(profile: Optional[HardwareProfile] = None) -> HardwareProfile:
    """Print a human-friendly hardware report and return the profile."""
    profile = profile or detect()
    try:
        _print_rich(profile)
    except Exception:  # noqa: BLE001 - rich missing or non-tty
        _print_plain(profile)
    return profile


def main() -> None:  # entry point for `ft-detect` / scripts/detect_hardware.py
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Detect hardware and suggest a training plan.")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    parser.add_argument("--model-params-b", type=float, default=1.5,
                        help="model size in billions of params for the sample plan")
    parser.add_argument("--technique", default="qlora",
                        choices=["full", "lora", "qlora"])
    args = parser.parse_args()

    profile = detect()
    plan = recommend(profile, args.model_params_b, args.technique)

    if args.json:
        print(json.dumps({"profile": profile.to_dict(), "plan": plan.to_dict()}, indent=2))
        return

    report(profile)
    print()
    print(f"=== Suggested plan: {args.technique} on a {args.model_params_b}B model ===")
    for k, v in plan.to_dict().items():
        if k == "notes":
            continue
        print(f"  {k:28s}: {v}")
    if plan.notes:
        print("  notes:")
        for n in plan.notes:
            print(f"    - {n}")


if __name__ == "__main__":
    main()
