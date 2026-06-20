"""Smoke tests for the non-ML layers (no torch / transformers / mlx required).

These validate that the shared config + hardware + data plumbing works on any
machine, which is exactly the part that should always run regardless of GPU.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from core import config
from core.config import RunConfig
from core.data import load_examples
from core.hardware import Accelerator, Backend, detect, estimate_weight_memory_gb, recommend

FAMILIES = config.available_models()
TECHNIQUES = config.available_techniques()


def test_detect_returns_profile():
    p = detect()
    assert isinstance(p.accelerator, Accelerator)
    assert p.cpu_cores_logical >= 1
    assert p.ram_total_gb > 0


def test_config_discovery_nonempty():
    assert {"qwen", "llama", "deepseek", "gemma", "kimi"}.issubset(set(FAMILIES))
    assert {"full", "lora", "qlora"}.issubset(set(TECHNIQUES))


@pytest.mark.parametrize("family", FAMILIES)
def test_model_configs_load(family):
    spec = config.ModelSpec.load(family)
    assert spec.hf_model_id
    assert spec.params_b > 0
    assert spec.lora_target_modules


@pytest.mark.parametrize("technique", TECHNIQUES)
def test_recommend_produces_plan(technique):
    profile = detect()
    plan = recommend(profile, model_params_b=1.5, technique=technique)
    assert isinstance(plan.backend, Backend)
    assert plan.per_device_batch_size >= 1
    assert plan.gradient_accumulation_steps >= 1
    assert plan.effective_batch_size >= 1
    assert plan.torch_dtype in {"bfloat16", "float16", "float32"}


def test_qlora_downgrades_without_4bit():
    """On a machine without CUDA+bitsandbytes, QLoRA should fall back to LoRA."""
    profile = detect()
    plan = recommend(profile, 1.5, "qlora", prefer_backend=Backend.HF)
    if not profile.supports_4bit_qlora:
        assert plan.technique == "lora"
        assert not plan.load_in_4bit


def test_memory_estimates_ordered():
    # full should cost more memory/param than lora, which costs more than qlora.
    full = estimate_weight_memory_gb(1.0, "full")
    lora = estimate_weight_memory_gb(1.0, "lora")
    qlora = estimate_weight_memory_gb(1.0, "qlora")
    assert full > lora > qlora


def test_model_id_override_reuses_family_config():
    """An arbitrary HF id reuses the family's arch settings but swaps the checkpoint."""
    run = RunConfig.assemble(
        "llama", "lora", "sample",
        model_id="NousResearch/Hermes-3-Llama-3.2-3B", params_b=3.0,
    )
    assert run.model.hf_model_id == "NousResearch/Hermes-3-Llama-3.2-3B"
    assert run.model.params_b == 3.0
    # family's LoRA targets are preserved
    assert "q_proj" in run.model.lora_target_modules
    # run name is unique per checkpoint so artifacts don't collide
    assert run.run_name == "llama-lora-sample-hermes-3-llama-3.2-3b"


def test_assemble_runconfig_and_load_data():
    run = RunConfig.assemble("qwen", "lora", "sample")
    assert run.run_name == "qwen-lora-sample"
    examples = load_examples(run.data)
    assert len(examples) > 5
    # every example is normalized to a list of chat messages
    assert all("messages" in ex and isinstance(ex["messages"], list) for ex in examples)
    assert examples[0]["messages"][0]["role"] in {"system", "user"}
