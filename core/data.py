"""Dataset loading + formatting.

Turns a :class:`core.config.DataSpec` into a list of training examples in a
single normalized shape: each example is a list of chat messages
``[{"role": ..., "content": ...}, ...]``. SFT trainers then apply the model's
chat template to render the final training string.

Supported source formats (``DataSpec.format``):
- ``chat``        : records already have a ``messages`` list.
- ``instruction`` : records have ``instruction`` / ``input`` / ``output`` fields
                    (Alpaca-style) — converted to user/assistant messages.
- ``text``        : records have a raw ``text`` field — wrapped as a single
                    user turn (useful for quick smoke tests / continued-pretrain).

Heavy deps (`datasets`) are imported lazily so the config layer stays importable
without the ML stack installed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from core.config import REPO_ROOT, DataSpec

Message = dict[str, str]
Example = dict[str, Any]  # {"messages": list[Message]}


def _read_jsonl(path: Path) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _to_messages(record: dict, spec: DataSpec) -> list[Message]:
    """Normalize a single raw record into a list of chat messages."""
    if spec.format == "chat":
        msgs = record.get(spec.text_field) or record.get("messages")
        if not isinstance(msgs, list):
            raise ValueError(f"Expected a list of messages in field '{spec.text_field}'")
        return msgs
    if spec.format == "instruction":
        instruction = record.get("instruction", "")
        extra = record.get("input", "")
        user = f"{instruction}\n\n{extra}".strip() if extra else instruction
        return [
            {"role": "user", "content": user},
            {"role": "assistant", "content": record.get("output", "")},
        ]
    if spec.format == "text":
        return [{"role": "user", "content": record.get(spec.text_field, record.get("text", ""))}]
    raise ValueError(f"Unknown data format: {spec.format}")


def load_examples(spec: DataSpec) -> list[Example]:
    """Load + normalize a dataset into ``[{"messages": [...]}, ...]``."""
    if spec.source == "local":
        if not spec.path:
            raise ValueError("Local dataset requires `path`.")
        path = Path(spec.path)
        if not path.is_absolute():
            path = REPO_ROOT / path
        raw: Iterable[dict] = _read_jsonl(path)
    elif spec.source == "hf":
        from datasets import load_dataset  # lazy import

        if not spec.hf_id:
            raise ValueError("HF dataset requires `hf_id`.")
        ds = load_dataset(spec.hf_id, spec.hf_subset, split=spec.split)
        raw = (dict(r) for r in ds)
    else:
        raise ValueError(f"Unknown data source: {spec.source}")

    examples: list[Example] = []
    for i, record in enumerate(raw):
        if spec.max_samples is not None and i >= spec.max_samples:
            break
        examples.append({"messages": _to_messages(record, spec)})
    if not examples:
        raise ValueError(f"Dataset '{spec.name}' produced 0 examples.")
    return examples
