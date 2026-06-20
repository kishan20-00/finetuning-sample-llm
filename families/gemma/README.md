# Gemma

Config: [`configs/models/gemma.yaml`](../../configs/models/gemma.yaml) ·
Default: **Gemma-2-2B-it** (Unsloth ungated mirror)

```bash
python scripts/train.py --model gemma --technique lora --data sample --dry-run
python scripts/train.py --model gemma --technique qlora --data sample
```

**Family notes**
- Official `google/gemma-*` weights are **gated** — accept the license on the Hub,
  then authenticate via `export HF_TOKEN=hf_xxx` (or `--hf-token`), or use the
  ungated `unsloth/*` mirror set as default.
- **Prefer bf16.** Gemma can be numerically unstable in fp16; the hardware planner
  picks bf16 automatically when the device supports it (Ampere+ / Apple Silicon),
  otherwise it warns. Watch the loss for NaNs on fp16-only hardware.
- 9B variant is QLoRA-only on a 12GB 4070.
