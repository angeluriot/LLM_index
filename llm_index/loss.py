from __future__ import annotations

import math
import torch

from llm_index.constants import BOTTOM_ANCHOR_BENCHMARK_RATIO, BOTTOM_ANCHOR_BENCHMARK_SCORE, BOTTOM_ANCHOR_VALUE, MIN_STD, TOP_ANCHOR_VALUE


def apply_constraints(
	anchor_model_index: int,
	model_intelligences_param: torch.Tensor,
	benchmark_difficulty_means_param: torch.Tensor,
	benchmark_difficulty_stds_param: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	positive_stds_param = torch.nn.functional.softplus(benchmark_difficulty_stds_param)

	nb_easiest_benchmarks = math.ceil(benchmark_difficulty_means_param.size(0) * BOTTOM_ANCHOR_BENCHMARK_RATIO)
	easiest_benchmark_indices = torch.topk(benchmark_difficulty_means_param, k=nb_easiest_benchmarks, largest=False)[1]
	bottom_1p_std = torch.distributions.Normal(0, 1).icdf(torch.tensor(BOTTOM_ANCHOR_BENCHMARK_SCORE))
	bottom_anchor = (benchmark_difficulty_means_param[easiest_benchmark_indices] + bottom_1p_std * positive_stds_param[easiest_benchmark_indices]).mean()
	top_anchor = model_intelligences_param[anchor_model_index]

	scale = (TOP_ANCHOR_VALUE - BOTTOM_ANCHOR_VALUE + bottom_1p_std * MIN_STD) / (top_anchor - bottom_anchor)

	model_intelligences = model_intelligences_param * scale
	benchmark_difficulty_means = benchmark_difficulty_means_param * scale
	benchmark_difficulty_stds = MIN_STD + positive_stds_param * scale

	shift = TOP_ANCHOR_VALUE - top_anchor * scale

	model_intelligences += shift
	benchmark_difficulty_means += shift

	return model_intelligences, benchmark_difficulty_means, benchmark_difficulty_stds


def compute_loss(
	model_intelligences: torch.Tensor,
	benchmark_difficulty_means: torch.Tensor,
	benchmark_difficulty_stds: torch.Tensor,
	results_model_indices: torch.Tensor,
	results_benchmark_indices: torch.Tensor,
	results_scores: torch.Tensor,
) -> torch.Tensor:
	r_model_intelligences = model_intelligences[results_model_indices]
	r_difficulty_means = benchmark_difficulty_means[results_benchmark_indices]
	r_difficulty_stds = benchmark_difficulty_stds[results_benchmark_indices]

	predictions = torch.special.ndtr((r_model_intelligences - r_difficulty_means) / r_difficulty_stds)

	return torch.mean((predictions - results_scores) ** 2)
