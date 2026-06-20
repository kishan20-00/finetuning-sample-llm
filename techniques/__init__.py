"""Technique orchestration layer.

These modules glue the pieces together for one finetuning *concept*:

    config (model + technique + data)  ->  hardware plan  ->  backend.train()

They are deliberately thin and backend-agnostic. The per-family folders under
``families/`` call into here; the CLI ``scripts/train.py`` does too.
"""
