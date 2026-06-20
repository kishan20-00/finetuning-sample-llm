"""Training backends.

Two interchangeable implementations of the same SFT contract:

- :mod:`backends.hf`  — HuggingFace ``transformers`` + ``trl`` + ``peft``.
  Runs on CUDA (RTX 4070: full / LoRA / 4-bit QLoRA), and also CPU / Apple MPS
  (LoRA without 4-bit, since bitsandbytes is CUDA-only).
- :mod:`backends.mlx` — Apple's ``mlx`` / ``mlx-lm``. Native, fast, low-memory
  fine-tuning on Apple Silicon (LoRA, and LoRA over a quantized base ≈ QLoRA).

The active backend is chosen by :func:`core.hardware.recommend` based on the
detected machine, then resolved here by :func:`get_sft_trainer`.
"""

from core.hardware import Backend


def get_sft_trainer(backend: Backend):
    """Return the ``train(run_config, plan)`` callable for the given backend."""
    if backend is Backend.HF:
        from backends.hf.sft import train as hf_train

        return hf_train
    if backend is Backend.MLX:
        from backends.mlx.sft import train as mlx_train

        return mlx_train
    raise ValueError(f"Unsupported backend: {backend}")
