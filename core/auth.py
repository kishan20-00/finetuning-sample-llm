"""Hugging Face authentication.

A single helper that makes a HF token available everywhere downloads happen:
- the HF backend (`from_pretrained` reads it from the environment),
- the MLX backend's `mlx_lm` subprocesses (they inherit this process's env),
- the one-time `mlx_lm convert` 4-bit quantization step.

Needed for **gated** models (official `meta-llama/*`, `google/gemma-*`) and for
higher Hub rate limits / faster downloads. The token is never printed or logged.
"""

from __future__ import annotations

import os
from typing import Optional

# Both names are read by various parts of the HF stack; we set both.
_ENV_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")


def configure_hf_token(token: Optional[str] = None) -> bool:
    """Resolve and export a HF token for child downloads.

    Precedence: explicit ``token`` arg > existing ``HF_TOKEN`` /
    ``HUGGING_FACE_HUB_TOKEN`` env var > nothing (anonymous, or a prior
    ``huggingface-cli login`` cache is still honored by the HF stack).

    Returns ``True`` if a token is now configured. Never reveals the value.
    """
    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        return False
    for var in _ENV_VARS:
        os.environ[var] = token
    return True
