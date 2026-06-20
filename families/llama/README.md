# Llama

Config: [`configs/models/llama.yaml`](../../configs/models/llama.yaml) ·
Default: **Llama-3.2-1B-Instruct** (Unsloth ungated mirror)

```bash
python scripts/train.py --model llama --technique lora --data sample --dry-run
python scripts/train.py --model llama --technique qlora --data sample
```

**Family notes**
- Official `meta-llama/*` weights are **gated** — request access on the Hub, then
  authenticate via `export HF_TOKEN=hf_xxx` (or `--hf-token`), or just use the
  ungated `unsloth/*` mirror set as default.
- Architecture `Llama` → standard attention + MLP LoRA targets.
- To run the exact official weights, switch `hf_model_id` to the gated variant
  listed under `variants:` in the model config.

Copy `../qwen/example.py`, set `FAMILY = "llama"` for a direct-library wrapper.
