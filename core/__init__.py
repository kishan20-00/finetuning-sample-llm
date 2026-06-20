"""Shared library for the fine-tuning samples.

Submodules:
- :mod:`core.hardware` — detect the machine, recommend a training plan.
- :mod:`core.config`   — load/merge YAML configs into a :class:`RunConfig`.
- :mod:`core.data`     — load + format datasets (chat / instruction / text).
"""

from core.hardware import Accelerator, Backend, HardwareProfile, detect, recommend

__all__ = ["Accelerator", "Backend", "HardwareProfile", "detect", "recommend"]
