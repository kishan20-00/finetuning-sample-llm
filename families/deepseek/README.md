# DeepSeek

Config: [`configs/models/deepseek.yaml`](../../configs/models/deepseek.yaml) ·
Default: **DeepSeek-R1-Distill-Qwen-1.5B**

```bash
python scripts/train.py --model deepseek --technique lora --data sample --dry-run
python scripts/train.py --model deepseek --technique qlora --data sample
```

**Family notes**
- The small, runnable DeepSeek here is the **R1 reasoning distill** onto a 1.5B
  Qwen2 base — ungated and ideal for studying reasoning-style SFT.
- Because the base is Qwen2, LoRA targets match the Qwen family.
- Pairs nicely with chain-of-thought datasets (responses that show their work).
- 7B/8B distills exist in `variants:` — QLoRA-only on a 12GB 4070.
