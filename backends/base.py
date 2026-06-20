"""Shared result type + the contract every backend implements.

Every backend exposes::

    def train(run_config: RunConfig, plan: TrainingPlan) -> TrainResult

`run_config` carries the human-authored intent (model / technique / data),
`plan` carries the hardware-adapted knobs from :func:`core.hardware.recommend`.
Backends must honor ``plan`` for anything memory-sensitive (dtype, batch size,
gradient accumulation, gradient checkpointing, 4-bit).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainResult:
    backend: str
    output_dir: str
    adapter_path: Optional[str] = None     # where LoRA adapters were written
    final_loss: Optional[float] = None
    steps: Optional[int] = None
    extra: dict = field(default_factory=dict)
