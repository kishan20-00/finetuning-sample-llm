# Kimi (Moonshot AI)

Config: [`configs/models/kimi.yaml`](../../configs/models/kimi.yaml) ·
Default: **Moonlight-16B-A3B-Instruct** (MoE)

> ⚠️ **Stretch target.** Moonshot's open models are large Mixture-of-Experts nets.
> The smallest, Moonlight-16B-A3B, has 16B total params (~3B active). It will
> **not** fit a 12GB 4070 for training. This is exactly the "keep the script,
> run it later on bigger hardware" case from the project goals.

```bash
# Always start with a dry-run here — the planner will tell you it doesn't fit:
python scripts/train.py --model kimi --technique qlora --data sample --dry-run
```

**Family notes**
- MoE architecture ships **custom modeling code** → `trust_remote_code: true`.
- Frozen 4-bit base is ~8–9 GiB; QLoRA is the only realistic path, and it's tight
  even on a 24 GB Mac. Best run on a 24GB+ NVIDIA card or a large-memory Mac.
- There is no smaller dense Kimi model published; this is the family's entry point.
