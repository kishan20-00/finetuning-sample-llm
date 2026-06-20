# Convenience targets. The repo uses `uv` (https://docs.astral.sh/uv/).
# Pick the extra that matches your hardware: `hf` (NVIDIA/CUDA) or `mlx` (Apple).

MODEL ?= qwen
TECH  ?= lora
DATA  ?= sample

.PHONY: help
help:
	@echo "Targets:"
	@echo "  make detect              - print hardware profile + a sample plan"
	@echo "  make install-hf          - install NVIDIA/CUDA stack (transformers+trl+peft+bnb)"
	@echo "  make install-mlx         - install Apple Silicon stack (mlx + mlx-lm)"
	@echo "  make dry-run             - show the adapted plan without training"
	@echo "  make train               - train (MODEL=$(MODEL) TECH=$(TECH) DATA=$(DATA))"
	@echo "  make lint                - ruff check"
	@echo "  make test                - pytest"
	@echo ""
	@echo "Override vars, e.g.:  make dry-run MODEL=gemma TECH=qlora"

.PHONY: detect
detect:
	python scripts/detect_hardware.py --technique $(TECH)

.PHONY: install-hf
install-hf:
	uv sync --extra hf

.PHONY: install-mlx
install-mlx:
	uv sync --extra mlx

.PHONY: install-dev
install-dev:
	uv sync --extra dev

.PHONY: dry-run
dry-run:
	python scripts/train.py --model $(MODEL) --technique $(TECH) --data $(DATA) --dry-run

.PHONY: train
train:
	python scripts/train.py --model $(MODEL) --technique $(TECH) --data $(DATA)

.PHONY: lint
lint:
	uv run ruff check .

.PHONY: test
test:
	uv run pytest -q
