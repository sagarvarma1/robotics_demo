#!/usr/bin/env bash
# Set up the project on a Kaggle (or Lightning AI) GPU machine.
# Usage (notebook cell, from the repo root):  !bash scripts/kaggle_setup.sh
# Why uv: Kaggle's system Python version can change, and JAX 0.12 needs Python >= 3.12.
# uv builds an isolated 3.12 environment from our lockfile (uv.lock), so every session
# gets exactly the same package versions as the local machine.
set -euo pipefail
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python3 --version
pip install -q uv
uv sync --python 3.12 --extra cuda
uv run python -c "import jax; print('backend:', jax.default_backend(), jax.devices())"
