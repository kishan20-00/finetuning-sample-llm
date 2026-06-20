"""Config loading + merging.

The repo keeps three orthogonal pieces of configuration as small YAML files:

- ``configs/models/<family>.yaml``     — *what* to fine-tune (model id, arch hints)
- ``configs/techniques/<name>.yaml``   — *how* to fine-tune (sft/lora/qlora knobs)
- ``configs/datasets/<name>.yaml``     — *what data* to fine-tune on

A run combines one of each, then the :mod:`core.hardware` planner overrides the
hardware-sensitive fields (dtype, batch size, grad accum, gradient checkpointing,
4-bit). This keeps the human-authored configs portable across very different
machines — you don't hand-tune batch sizes per GPU.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

# Repo root = parent of this file's parent (core/ -> repo root).
REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def slugify(model_id: str) -> str:
    """Filesystem-safe short name from a HF id ('Qwen/Qwen2.5-0.5B' -> 'qwen2.5-0.5b')."""
    base = model_id.rstrip("/").split("/")[-1]
    return re.sub(r"[^A-Za-z0-9._-]", "-", base).lower()


# ---------------------------------------------------------------------------
# Spec dataclasses (mirror the YAML files)
# ---------------------------------------------------------------------------
@dataclass
class ModelSpec:
    family: str
    display_name: str
    params_b: float
    hf_model_id: str
    # Optional: a local path (or modern-format repo) of a pre-converted 4-bit MLX
    # model. Leave empty — the MLX QLoRA path auto-quantizes hf_model_id instead.
    mlx_model_id: Optional[str] = None
    chat_template: str = "auto"        # "auto" = use tokenizer's built-in template
    lora_target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    trust_remote_code: bool = False
    notes: str = ""
    variants: list[dict] = field(default_factory=list)

    @classmethod
    def load(cls, family: str) -> "ModelSpec":
        data = load_yaml(CONFIG_DIR / "models" / f"{family}.yaml")
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class TechniqueSpec:
    technique: str                     # "full" | "lora" | "qlora"
    learning_rate: float = 2e-4
    num_train_epochs: float = 1.0
    max_steps: int = -1
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    save_steps: int = 200
    seed: int = 42
    # PEFT / LoRA (ignored for full fine-tuning)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # QLoRA / bitsandbytes (ignored unless technique == qlora and CUDA present)
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    @classmethod
    def load(cls, name: str) -> "TechniqueSpec":
        data = load_yaml(CONFIG_DIR / "techniques" / f"{name}.yaml")
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class DataSpec:
    name: str
    source: str = "local"              # "local" | "hf"
    path: Optional[str] = None         # local file (jsonl) when source == local
    hf_id: Optional[str] = None        # dataset id when source == hf
    hf_subset: Optional[str] = None
    split: str = "train"
    format: str = "chat"               # "chat" | "instruction" | "text"
    text_field: str = "messages"       # field holding the data per `format`
    max_samples: Optional[int] = None

    @classmethod
    def load(cls, name: str) -> "DataSpec":
        data = load_yaml(CONFIG_DIR / "datasets" / f"{name}.yaml")
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


# ---------------------------------------------------------------------------
# Assembled run config
# ---------------------------------------------------------------------------
@dataclass
class RunConfig:
    model: ModelSpec
    technique: TechniqueSpec
    data: DataSpec
    output_dir: str
    run_name: str

    @staticmethod
    def assemble(
        model_family: str,
        technique_name: str,
        dataset_name: str,
        output_root: str = "outputs",
        *,
        model_id: Optional[str] = None,
        params_b: Optional[float] = None,
    ) -> "RunConfig":
        """Combine a model family + technique + dataset into a run.

        ``model_id`` overrides the family's ``hf_model_id`` so you can fine-tune
        any Hugging Face checkpoint while reusing the family's architecture
        settings (LoRA targets, chat template). Pair it with ``params_b`` so the
        hardware planner sizes the run for the real model.
        """
        model = ModelSpec.load(model_family)
        if model_id:
            model.hf_model_id = model_id
            model.mlx_model_id = None       # force re-quantize from the new base
            model.display_name = model_id
        if params_b is not None:
            model.params_b = params_b
        technique = TechniqueSpec.load(technique_name)
        data = DataSpec.load(dataset_name)
        run_name = f"{model_family}-{technique_name}-{dataset_name}"
        if model_id:
            run_name += f"-{slugify(model_id)}"
        output_dir = str(REPO_ROOT / output_root / run_name)
        return RunConfig(
            model=model,
            technique=technique,
            data=data,
            output_dir=output_dir,
            run_name=run_name,
        )


# ---------------------------------------------------------------------------
# Discovery helpers (used by the CLI to list options)
# ---------------------------------------------------------------------------
def _stems(subdir: str) -> list[str]:
    d = CONFIG_DIR / subdir
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.yaml"))


def available_models() -> list[str]:
    return _stems("models")


def available_techniques() -> list[str]:
    return _stems("techniques")


def available_datasets() -> list[str]:
    return _stems("datasets")
