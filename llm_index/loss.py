from __future__ import annotations

import torch

from llm_index.constants import BOTTOM_ANCHOR_BENCHMARK_SCORE, BOTTOM_ANCHOR_VALUE, MIN_STD, TOP_ANCHOR_VALUE
from llm_index.distribution import cumulative_split_normal, inverse_cumulative_split_normal


def apply_constraints(
	anchor_model_index: int,
	model_intelligences_param: torch.Tensor,
	benchmark_difficulty_modes_param: torch.Tensor,
	benchmark_difficulty_low_stds_param: torch.Tensor,
	benchmark_difficulty_high_stds_param: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	positive_low_stds_param = torch.nn.functional.softplus(benchmark_difficulty_low_stds_param)
	positive_high_stds_param = torch.nn.functional.softplus(benchmark_difficulty_high_stds_param)

	bottom_score = torch.tensor(BOTTOM_ANCHOR_BENCHMARK_SCORE)
	bottom_std = torch.special.ndtri(bottom_score)
	bottom_anchor = inverse_cumulative_split_normal(bottom_score, benchmark_difficulty_modes_param, positive_low_stds_param, positive_high_stds_param).min()
	top_anchor = model_intelligences_param[anchor_model_index]

	scale = (TOP_ANCHOR_VALUE - BOTTOM_ANCHOR_VALUE + bottom_std * MIN_STD) / (top_anchor - bottom_anchor)

	model_intelligences = model_intelligences_param * scale
	benchmark_difficulty_modes = benchmark_difficulty_modes_param * scale
	benchmark_difficulty_low_stds = MIN_STD + positive_low_stds_param * scale
	benchmark_difficulty_high_stds = MIN_STD + positive_high_stds_param * scale

	shift = TOP_ANCHOR_VALUE - top_anchor * scale

	model_intelligences += shift
	benchmark_difficulty_modes += shift

	return model_intelligences, benchmark_difficulty_modes, benchmark_difficulty_low_stds, benchmark_difficulty_high_stds


def apply_exact_bottom_anchor(
	model_intelligences: torch.Tensor,
	benchmark_difficulty_modes: torch.Tensor,
	benchmark_difficulty_low_stds: torch.Tensor,
	benchmark_difficulty_high_stds: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	bottom_anchor = inverse_cumulative_split_normal(
		torch.tensor(BOTTOM_ANCHOR_BENCHMARK_SCORE),
		benchmark_difficulty_modes,
		benchmark_difficulty_low_stds,
		benchmark_difficulty_high_stds,
	).min()

	scale = (TOP_ANCHOR_VALUE - BOTTOM_ANCHOR_VALUE) / (TOP_ANCHOR_VALUE - bottom_anchor)
	shift = TOP_ANCHOR_VALUE * (1 - scale)

	return (
		model_intelligences * scale + shift,
		benchmark_difficulty_modes * scale + shift,
		benchmark_difficulty_low_stds * scale,
		benchmark_difficulty_high_stds * scale,
	)


def compute_loss(
	model_intelligences: torch.Tensor,
	benchmark_difficulty_modes: torch.Tensor,
	benchmark_difficulty_low_stds: torch.Tensor,
	benchmark_difficulty_high_stds: torch.Tensor,
	results_model_indices: torch.Tensor,
	results_benchmark_indices: torch.Tensor,
	results_scores: torch.Tensor,
) -> torch.Tensor:
	r_model_intelligences = model_intelligences[results_model_indices]
	r_difficulty_modes = benchmark_difficulty_modes[results_benchmark_indices]
	r_difficulty_low_stds = benchmark_difficulty_low_stds[results_benchmark_indices]
	r_difficulty_high_stds = benchmark_difficulty_high_stds[results_benchmark_indices]

	predictions = cumulative_split_normal(r_model_intelligences, r_difficulty_modes, r_difficulty_low_stds, r_difficulty_high_stds)

	return torch.mean((predictions - results_scores) ** 2)
