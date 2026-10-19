from __future__ import annotations

import torch
from pathlib import Path


# Data
DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
ONLY_EVERY_YEAR_DELAY_DAYS = 30 * 6

# Constraints
BOTTOM_ANCHOR_BENCHMARK_RATIO = 0.2
BOTTOM_ANCHOR_BENCHMARK_SCORE = 0.05
BOTTOM_ANCHOR_VALUE = 0.0
TOP_ANCHOR_MODEL = 'gpt-5'
TOP_ANCHOR_VALUE = 100.0
MIN_STD = 1.0

# Optimization
EPOCHS = 50_000
LEARNING_RATE = 0.1
