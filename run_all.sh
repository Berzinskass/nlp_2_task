#!/usr/bin/env bash
# Full pipeline: data -> features -> training -> noisy evaluation -> figures.
set -euo pipefail
cd "$(dirname "$0")"
python -m src.prepare_data
python -m src.features
python -m src.train
python -m src.evaluate
python -m src.visualize
