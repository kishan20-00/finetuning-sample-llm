# Qwen

Config: [`configs/models/qwen.yaml`](../../configs/models/qwen.yaml) ·
Default: **Qwen2.5-0.5B-Instruct**

Qwen2.5 is the recommended starting point — strong quality at tiny sizes, ungated,
and the 0.5B variant even fits **full** fine-tuning on a 4070.

```bash
# See the hardware-adapted plan (no training):
python scripts/train.py --model qwen --technique qlora --data sample --dry-run

# Train a LoRA adapter on the sample data:
python scripts/train.py --model qwen --technique lora --data sample

# Full fine-tune the 0.5B (small enough to update every weight):
python scripts/train.py --model qwen --technique full --data sample
```

See [`example.py`](example.py) for calling the library directly instead of the CLI.

**Family notes**
- Architecture `Qwen2` → LoRA targets all attention + MLP projections.
- Larger variants (1.5B/3B/7B) are listed in the model config; 7B is QLoRA-only on 12GB.
- The DeepSeek R1 distill in this repo is Qwen2-based, so its LoRA setup matches.
