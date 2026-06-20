# The hardware-adaptation layer

The shared idea across every script: **detect the machine, then adapt the run to
it** — instead of hand-tuning batch sizes and dtypes per GPU. All of this lives in
[`core/hardware.py`](../core/hardware.py) and is pure-stdlib, so it runs on a fresh
checkout before any ML packages are installed.

## Two functions

```python
from core.hardware import detect, recommend

profile = detect()                      # what do I have?
plan    = recommend(profile, 1.5, "qlora")   # given a 1.5B model + QLoRA, how should I train?
```

- **`detect() -> HardwareProfile`** — OS/arch, accelerator (CUDA / MPS / CPU),
  GPU name + VRAM + compute capability (via `torch`, falling back to `nvidia-smi`),
  RAM, CPU cores, and derived capabilities: `supports_bf16`, `supports_4bit_qlora`
  (bitsandbytes present + CUDA), `mlx_available`.
- **`recommend(profile, model_params_b, technique) -> TrainingPlan`** — chooses the
  **backend**, **dtype**, **4-bit on/off**, **per-device batch size**,
  **gradient accumulation**, **gradient checkpointing**, and **max sequence length**,
  and estimates whether it all **fits** in memory.

Run it directly anytime:

```bash
python scripts/detect_hardware.py                                  # table
python scripts/detect_hardware.py --technique qlora --model-params-b 7
python scripts/detect_hardware.py --json                           # machine-readable
```

## How decisions are made

| Decision | Rule |
|---|---|
| **Backend** | Apple Silicon + `mlx` installed → MLX; otherwise HuggingFace (CUDA/MPS/CPU). |
| **dtype** | CUDA Ampere+ (cc ≥ 8.0) or Apple Silicon → bf16; older CUDA → fp16; CPU → fp32. |
| **4-bit (QLoRA)** | Only on CUDA with bitsandbytes. Elsewhere QLoRA **downgrades to LoRA** (with a note). |
| **Fit estimate** | `bytes/param × P` vs usable memory (90% of VRAM on CUDA, ~65% of unified RAM on Apple, 50% of RAM on CPU). |
| **Batch size** | Filled from leftover memory after weights; capped at 8, floored at 1. |
| **Grad accumulation** | `ceil(target_effective_batch / per_device_batch)`. |
| **Gradient checkpointing** | Enabled when memory is tight or batch is forced to 1. |

The numbers are deliberately conservative heuristics, not a simulator — see the
`_BYTES_PER_PARAM` constants and comments in `core/hardware.py`.

## The NVIDIA vs Apple split

This is the reason the repo has two backends:

- **NVIDIA / CUDA (e.g. RTX 4070, 12 GB):** full HuggingFace stack. Real 4-bit
  QLoRA via bitsandbytes. This is where `--technique qlora` shines — a 7B model
  becomes trainable.
- **Apple Silicon (M-series):** bitsandbytes doesn't exist. Two viable paths:
  1. **MLX** (default when installed) — native, fast, low memory; LoRA and LoRA
     over a 4-bit MLX base (the QLoRA equivalent).
  2. **HuggingFace on MPS** — works for LoRA/full, but no 4-bit; slower than MLX.

You normally don't choose — `recommend()` does. Override with `--backend hf|mlx`
on `scripts/train.py` when you want to study a specific path.

## Overriding the plan

The plan is advisory. To pin values, edit the technique YAML (e.g. `max_steps`,
`learning_rate`) or pass CLI flags (`--max-seq-len`, `--effective-batch`). If you
hit OOM, lower `--effective-batch`, shorten `--max-seq-len`, switch to `qlora`, or
drop to a smaller model variant.
