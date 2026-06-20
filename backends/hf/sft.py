"""HuggingFace SFT backend: full / LoRA / QLoRA.

Uses ``transformers`` + ``trl.SFTTrainer`` + ``peft``. Works on:
- CUDA (RTX 4070): full fine-tune (small models), LoRA, and 4-bit QLoRA.
- Apple MPS / CPU: full + LoRA (QLoRA auto-downgrades, since bitsandbytes is
  CUDA-only — the hardware planner handles that downgrade upstream).

All heavy imports are local to :func:`train` so this module imports cleanly even
when the ML stack isn't installed (e.g. on a fresh machine running detection).

The three techniques differ only in how the base model is loaded and whether a
PEFT adapter is attached:

    full   -> load fp16/bf16 weights, train *all* parameters.
    lora   -> load fp16/bf16 weights (frozen), train small LoRA adapters.
    qlora  -> load 4-bit quantized weights (frozen), train LoRA adapters on top.
"""

from __future__ import annotations

import os

from backends.base import TrainResult
from core.config import RunConfig
from core.data import load_examples
from core.hardware import TrainingPlan


def _build_quant_config(plan: TrainingPlan, run: RunConfig):
    """BitsAndBytes 4-bit config for QLoRA (CUDA only). Returns None otherwise."""
    if not plan.load_in_4bit:
        return None
    import torch
    from transformers import BitsAndBytesConfig

    compute_dtype = torch.bfloat16 if plan.torch_dtype == "bfloat16" else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=run.technique.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=run.technique.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def _build_peft_config(run: RunConfig):
    """LoRA config for lora/qlora; None for full fine-tuning."""
    if run.technique.technique == "full":
        return None
    from peft import LoraConfig

    return LoraConfig(
        r=run.technique.lora_r,
        lora_alpha=run.technique.lora_alpha,
        lora_dropout=run.technique.lora_dropout,
        target_modules=run.model.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )


def train(run: RunConfig, plan: TrainingPlan) -> TrainResult:
    import torch
    from datasets import Dataset
    from peft import prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    torch_dtype = getattr(torch, plan.torch_dtype)

    # --- tokenizer --------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        run.model.hf_model_id,
        trust_remote_code=run.model.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- model ------------------------------------------------------------
    quant_config = _build_quant_config(plan, run)
    model = AutoModelForCausalLM.from_pretrained(
        run.model.hf_model_id,
        quantization_config=quant_config,
        torch_dtype=torch_dtype,
        device_map="auto" if plan.device == "cuda" else None,
        trust_remote_code=run.model.trust_remote_code,
    )
    if quant_config is not None:
        # Prepares a 4-bit model for adapter training (casts norms, enables grads).
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=plan.gradient_checkpointing
        )

    # --- data -------------------------------------------------------------
    examples = load_examples(run.data)
    dataset = Dataset.from_list(examples)

    def to_text(batch):
        # Render chat messages with the model's own chat template.
        rendered = [
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            for msgs in batch["messages"]
        ]
        return {"text": rendered}

    dataset = dataset.map(to_text, batched=True, remove_columns=dataset.column_names)

    # --- trainer config ---------------------------------------------------
    use_bf16 = plan.torch_dtype == "bfloat16"
    use_fp16 = plan.torch_dtype == "float16"
    sft_config = SFTConfig(
        output_dir=run.output_dir,
        per_device_train_batch_size=plan.per_device_batch_size,
        gradient_accumulation_steps=plan.gradient_accumulation_steps,
        gradient_checkpointing=plan.gradient_checkpointing,
        learning_rate=run.technique.learning_rate,
        num_train_epochs=run.technique.num_train_epochs,
        max_steps=run.technique.max_steps,
        warmup_ratio=run.technique.warmup_ratio,
        weight_decay=run.technique.weight_decay,
        lr_scheduler_type=run.technique.lr_scheduler_type,
        logging_steps=run.technique.logging_steps,
        save_steps=run.technique.save_steps,
        seed=run.technique.seed,
        bf16=use_bf16,
        fp16=use_fp16,
        max_seq_length=plan.max_seq_len,
        dataset_num_proc=max(1, plan.dataloader_num_workers),
        dataset_text_field="text",
        report_to=["tensorboard"] if os.environ.get("FT_REPORT_TB") else [],
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=_build_peft_config(run),
    )

    train_output = trainer.train()
    trainer.save_model(run.output_dir)
    tokenizer.save_pretrained(run.output_dir)

    return TrainResult(
        backend="hf",
        output_dir=run.output_dir,
        adapter_path=run.output_dir if run.technique.technique != "full" else None,
        final_loss=getattr(train_output, "training_loss", None),
        steps=getattr(train_output, "global_step", None),
    )
