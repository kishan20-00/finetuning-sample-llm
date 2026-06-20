# finetuning-sample-llm

A study repo for **fine-tuning small LLMs** across families — Qwen, Llama,
DeepSeek, Gemma, Kimi — using techniques that fit modest hardware (an **RTX 4070**
or **Apple M-series** Mac). Every run **detects your machine and adapts** the dtype,
batch size, quantization, and backend automatically.

> It's fine if a given script can't run on your current machine — the scripts are
> written to be kept and run later when bigger hardware is available. The point is
> to study the concepts (SFT / LoRA / QLoRA) with a clean, shared structure.

## Why two backends?

The NVIDIA and Apple stacks genuinely differ, so the repo carries both behind one
interface and picks for you:

- **HuggingFace** (`transformers` + `trl` + `peft`) → CUDA (incl. real 4-bit QLoRA
  via bitsandbytes), and also CPU / Apple-MPS.
- **MLX** (`mlx` + `mlx-lm`) → native, fast, low-memory fine-tuning on Apple Silicon.

See [`docs/hardware.md`](docs/hardware.md) for exactly how the choice is made.

## Quickstart

```bash
# 0. Detect hardware (works with ZERO ML deps installed):
python scripts/detect_hardware.py

# 1. Install the stack for YOUR machine:
uv sync --extra hf      # NVIDIA / CUDA (RTX 4070)
uv sync --extra mlx     # Apple Silicon (M-series)

# 2. Preview the hardware-adapted plan without training:
python scripts/train.py --model qwen --technique qlora --data sample --dry-run

# 3. Train a LoRA adapter on the tiny sample dataset:
python scripts/train.py --model qwen --technique lora --data sample
```

`Makefile` shortcuts: `make detect`, `make dry-run MODEL=gemma TECH=qlora`, `make train`.

## Running any Hugging Face model

The five family configs are *architecture profiles* (LoRA targets, chat template).
To fine-tune **any other checkpoint on the Hub** without writing a new config, pass
`--model-id` and reuse the closest family:

```bash
# A different Llama-architecture checkpoint, borrowing the `llama` family profile:
python scripts/train.py --model llama --model-id NousResearch/Hermes-3-Llama-3.2-3B \
  --params-b 3.0 --technique lora --data sample --dry-run
```

`--model` picks the architecture settings; `--model-id` swaps the actual weights;
`--params-b` lets the hardware planner size the run. (If a model is something you'll
reuse often, promote it to its own `configs/models/<name>.yaml`.)

## QLoRA uses official repos + local quantization

We don't depend on community `*-4bit` repos (their formats drift and many are stale).
Each config points at the **official full-precision repo**, and the QLoRA path
**quantizes it to 4-bit locally on first use** (`mlx_lm convert` on Apple; bitsandbytes
on CUDA), caching the result under `models/` (gitignored). Nothing is ever uploaded.
To pre-quantize by hand on Apple Silicon:

```bash
python -m mlx_lm convert --hf-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  -q --q-bits 4 --q-group-size 64 --mlx-path models/tinyllama-1.1b-mlx-4bit
```

## Gated models / Hugging Face token

Official `meta-llama/*` and `google/gemma-*` repos are gated, and any download is
faster/rate-limit-friendly when authenticated. Provide a token two ways (env var
preferred — CLI args can leak via shell history / process list):

```bash
export HF_TOKEN=hf_xxx                              # preferred
python scripts/train.py --model llama --technique lora --data sample

python scripts/train.py ... --hf-token hf_xxx       # or pass inline
```

The token reaches every download — HF backend, the MLX subprocess, and the 4-bit
convert step. A prior `huggingface-cli login` is also honored. The token is never
printed or uploaded.

## The two axes

Mix any **model family** with any **technique** — one command, the hardware layer
adapts the rest:

| | `full` | `lora` | `qlora` |
|---|---|---|---|
| **qwen** (0.5B) | ✅ | ✅ | ✅ |
| **llama** (1B) | ⚠️ tight | ✅ | ✅ |
| **deepseek** (1.5B, R1 distill) | ❌ 4070 | ✅ | ✅ |
| **gemma** (2B) | ❌ 4070 | ✅ | ✅ |
| **kimi** (16B MoE) | ❌ | ❌ | ⚠️ stretch |

✅ comfortable · ⚠️ tight / needs care · ❌ won't fit a 12GB 4070 (keep the script
for later). New concepts in [`docs/concepts.md`](docs/concepts.md).

## Layout (hybrid: shared core + thin family wrappers)

```
core/                 # SHARED config used by every script
  hardware.py         #   ⭐ detect machine + recommend a training plan (pure stdlib)
  config.py           #   load/merge model + technique + dataset YAML -> RunConfig
  data.py             #   load + normalize datasets to chat messages
backends/             # interchangeable trainers behind one contract
  hf/sft.py           #   HuggingFace: full / LoRA / QLoRA (CUDA / MPS / CPU)
  mlx/sft.py          #   MLX: LoRA / full on Apple Silicon
techniques/sft.py     # orchestration: config -> hardware plan -> backend
configs/
  models/             # one YAML per family (ids, params, LoRA targets, variants)
  techniques/         # full.yaml / lora.yaml / qlora.yaml hyperparameters
  datasets/           # dataset definitions (local jsonl or HF Hub)
families/             # thin per-family notes + an example wrapper (qwen/example.py)
scripts/
  detect_hardware.py  # CLI: hardware report
  train.py            # CLI: the single training entrypoint
data/samples/         # tiny in-repo dataset for end-to-end smoke tests
docs/                 # concepts.md (study) + hardware.md (the adaptation layer)
outputs/              # checkpoints / adapters (gitignored)
```

## Design principles

1. **Detect, then adapt.** No machine-specific batch sizes in configs; `core/hardware.py`
   derives them. The human configs stay portable.
2. **Config-driven, not copy-pasted.** Family = a YAML key. The same technique
   script runs across Qwen/Llama/Gemma/DeepSeek/Kimi.
3. **Graceful when deps are missing.** Detection and config loading need only the
   stdlib + pyyaml; the heavy ML stack is imported lazily inside the backends.
4. **Keep-and-run-later.** Plans are generated (and printed) even when a model
   won't fit, so the scripts are ready for better hardware.

## Roadmap (room left in the layout)

SFT (full/LoRA/QLoRA) is scaffolded now. Natural next additions, same structure:
preference tuning (DPO/ORPO), reward modeling + PPO, and continued pretraining —
add a `configs/techniques/*.yaml` and a `techniques/<name>.py` orchestrator.

## Requirements

- Python 3.10–3.12 · [`uv`](https://docs.astral.sh/uv/)
- NVIDIA path: a CUDA GPU for QLoRA (bitsandbytes is CUDA-only).
- Apple path: an M-series Mac for MLX.
