from __future__ import annotations

import torch
from pathlib import Path


# Data
DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
ONLY_EVERY_YEAR_DELAY_DAYS = 30 * 6

# Constraints
BOTTOM_ANCHOR_EASIEST_BENCHMARK_SCORE = 0.01
BOTTOM_ANCHOR_VALUE = 1.0
TOP_ANCHOR_MODEL = 'gpt-5'
TOP_ANCHOR_VALUE = 100.0
MIN_STD = 1.0
EXTREME_SCORE_THRESHOLD = 0.001
EXTREME_SCORE_REGULARIZATION = 1e-6

# Optimization
EPOCHS = 3_000
LEARNING_RATE = 0.1
ADAM_BETA_1 = 0.9
ADAM_BETA_2 = 0.9

# Device
DEVICE = torch.device('cpu')
