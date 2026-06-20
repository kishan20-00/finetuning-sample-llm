# Fine-tuning concepts

A study companion for the techniques in this repo. The goal is intuition + the
memory math that decides what runs on your hardware.

## The big picture

Fine-tuning continues training a pretrained model on your data so it behaves the
way you want. The pipeline most teams use:

```
(pretrained base) -> [continued pretraining] -> [SFT] -> [preference tuning] -> deploy
                          raw domain text       instructions   DPO/ORPO/RLHF
```

This repo currently scaffolds **SFT** in three flavors: **full**, **LoRA**, and
**QLoRA**. (Preference tuning and continued pretraining are natural next steps —
the layout leaves room for them.)

## SFT — Supervised Fine-Tuning

Train on `(prompt, ideal response)` pairs with the standard next-token loss, but
only the **response** tokens are learned from (the prompt is masked). This is how
a base model becomes an instruction-follower or adopts a persona/format.

Our data is normalized to chat messages and rendered with each model's **own chat
template** (`tokenizer.apply_chat_template`). Using the wrong template is the
single most common reason a fine-tune "doesn't take".

## PEFT — Parameter-Efficient Fine-Tuning

Instead of updating all weights, freeze the base and train a tiny number of new
parameters. Cheaper, faster, and you get a small portable **adapter** instead of a
full model copy. LoRA is the dominant PEFT method.

### Full vs LoRA vs QLoRA

| | Full | LoRA | QLoRA |
|---|---|---|---|
| Base weights | trained | frozen (fp16/bf16) | frozen (**4-bit**) |
| Trainable params | 100% | ~0.1–1% | ~0.1–1% |
| Memory/param (≈) | ~16 B | ~2.3 B | ~0.8 B |
| Output | full model | adapter | adapter |
| Best when | tiny models, max quality | the default | big model, small GPU |

### LoRA in one paragraph

A weight update `ΔW` is approximated by a low-rank product `B·A` where `A` is
`r×k` and `B` is `d×r`, with `r` small (8–64). You train only `A` and `B`; the
original `W` stays frozen. At inference you can keep the adapter separate or merge
`W + BA` back into the weights. Key knobs (in `configs/techniques/lora.yaml`):

- **`r` (rank)** — capacity. Higher = more expressive, more params. 8–32 typical.
- **`alpha`** — scaling; the update is scaled by `alpha/r`. Convention: `alpha = 2r`.
- **`dropout`** — regularization on the adapter.
- **target modules** — which projections get adapters (attention `q/k/v/o`, MLP
  `gate/up/down`). Set per architecture in the model config.

### QLoRA in one paragraph

Same trainable LoRA adapters, but the frozen base is **quantized to 4-bit (NF4)**,
roughly quartering base-weight memory. That's what makes a 7B model trainable on a
12GB card. The adapters and computation stay in 16-bit. On NVIDIA this uses
`bitsandbytes` (CUDA-only); on Apple Silicon there's no bitsandbytes, so we train
LoRA on a model that's **already** a 4-bit MLX conversion — same idea, native to
MLX. See [hardware.md](hardware.md) for how the planner handles the difference.

## The memory math (why hardware matters)

For a model with **P** parameters, peak training memory ≈ weights + gradients +
optimizer state + activations. With Adam in mixed precision:

- **Full:** `2P` (fp16 weights) + `2P` (grads) + `12P` (fp32 Adam m, v + master) ≈ **16P bytes**.
  → a 1.5B model needs ~24 GB *before* activations. Only the smallest models fit consumer GPUs.
- **LoRA:** `~2P` for the frozen fp16 base + a sliver for adapters/optimizer ≈ **~2.3P bytes**.
- **QLoRA:** `~0.5P` for the 4-bit base + adapter sliver ≈ **~0.8P bytes**.

`core/hardware.py` encodes exactly these constants (`_BYTES_PER_PARAM`) to estimate
fit and pick batch size. They're rough — treat the suggested batch size as a
starting point and watch for OOM.

## Choosing a technique (rule of thumb)

- **≤ 0.5B model, want max quality, have the memory** → full.
- **Default for almost everything** → LoRA.
- **Model too big to fit in fp16** → QLoRA.
- **On a Mac** → MLX LoRA, or MLX LoRA over a 4-bit base for the QLoRA effect.

## Glossary

- **Adapter** — the small trained LoRA weights; portable, stackable, mergeable.
- **NF4** — 4-bit "normal float" quantization used by QLoRA.
- **Gradient checkpointing** — recompute activations in the backward pass to trade
  compute for memory; the planner enables it when memory is tight.
- **Effective batch size** — `per_device_batch × gradient_accumulation_steps`.
- **Chat template** — model-specific formatting of roles/turns into one string.
