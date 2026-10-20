from __future__ import annotations

import torch

from llm_index.constants import BOTTOM_ANCHOR_EASIEST_BENCHMARK_SCORE, BOTTOM_ANCHOR_VALUE, EXTREME_SCORE_THRESHOLD, MIN_STD, TOP_ANCHOR_VALUE
from llm_index.distribution import cumulative_normal, cumulative_split_normal, inverse_cumulative_split_normal


def get_bottom_anchor(
	benchmark_difficulty_modes: torch.Tensor,
	benchmark_difficulty_low_stds: torch.Tensor,
	benchmark_difficulty_high_stds: torch.Tensor,
) -> torch.Tensor:
	return inverse_cumulative_split_normal(
		torch.tensor(BOTTOM_ANCHOR_EASIEST_BENCHMARK_SCORE),
		benchmark_difficulty_modes,
		benchmark_difficulty_low_stds,
		benchmark_difficulty_high_stds,
	).min()


def apply_constraints(
	anchor_model_index: int,
	model_intelligences_param: torch.Tensor,
	benchmark_difficulty_modes_param: torch.Tensor,
	benchmark_difficulty_low_stds_param: torch.Tensor,
	benchmark_difficulty_high_stds_param: torch.Tensor,
	benchmark_has_score_below_half: torch.Tensor,
	benchmark_has_score_above_half: torch.Tensor,
	milestone_difficulty_means_param: torch.Tensor,
	milestone_difficulty_stds_param: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	positive_low_stds_param = torch.nn.functional.softplus(benchmark_difficulty_low_stds_param)
	positive_high_stds_param = torch.nn.functional.softplus(benchmark_difficulty_high_stds_param)
	positive_milestone_stds_param = torch.nn.functional.softplus(milestone_difficulty_stds_param)

	positive_high_stds_param = torch.where(benchmark_has_score_above_half, positive_high_stds_param, positive_low_stds_param)
	positive_low_stds_param = torch.where(benchmark_has_score_below_half, positive_low_stds_param, positive_high_stds_param)

	bottom_std = torch.special.ndtri(torch.tensor(BOTTOM_ANCHOR_EASIEST_BENCHMARK_SCORE))
	bottom_anchor = get_bottom_anchor(benchmark_difficulty_modes_param, positive_low_stds_param, positive_high_stds_param)
	top_anchor = model_intelligences_param[anchor_model_index]

	scale = (TOP_ANCHOR_VALUE - BOTTOM_ANCHOR_VALUE + bottom_std * MIN_STD) / (top_anchor - bottom_anchor)

	model_intelligences = model_intelligences_param * scale
	benchmark_difficulty_modes = benchmark_difficulty_modes_param * scale
	benchmark_difficulty_low_stds = MIN_STD + positive_low_stds_param * scale
	benchmark_difficulty_high_stds = MIN_STD + positive_high_stds_param * scale
	milestone_difficulty_means = milestone_difficulty_means_param * scale
	milestone_difficulty_stds = MIN_STD + positive_milestone_stds_param * scale

	shift = TOP_ANCHOR_VALUE - top_anchor * scale

	model_intelligences += shift
	benchmark_difficulty_modes += shift
	milestone_difficulty_means += shift

	return (
		model_intelligences,
		benchmark_difficulty_modes,
		benchmark_difficulty_low_stds,
		benchmark_difficulty_high_stds,
		milestone_difficulty_means,
		milestone_difficulty_stds,
	)


def apply_exact_bottom_anchor(
	model_intelligences: torch.Tensor,
	benchmark_difficulty_modes: torch.Tensor,
	benchmark_difficulty_low_stds: torch.Tensor,
	benchmark_difficulty_high_stds: torch.Tensor,
	milestone_difficulty_means: torch.Tensor,
	milestone_difficulty_stds: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	bottom_anchor = get_bottom_anchor(benchmark_difficulty_modes, benchmark_difficulty_low_stds, benchmark_difficulty_high_stds)

	scale = (TOP_ANCHOR_VALUE - BOTTOM_ANCHOR_VALUE) / (TOP_ANCHOR_VALUE - bottom_anchor)
	shift = TOP_ANCHOR_VALUE * (1 - scale)

	return (
		torch.abs(model_intelligences * scale + shift),
		benchmark_difficulty_modes * scale + shift,
		benchmark_difficulty_low_stds * scale,
		benchmark_difficulty_high_stds * scale,
		milestone_difficulty_means * scale + shift,
		milestone_difficulty_stds * scale,
	)


def compute_loss(
	model_intelligences: torch.Tensor,
	benchmark_difficulty_modes: torch.Tensor,
	benchmark_difficulty_low_stds: torch.Tensor,
	benchmark_difficulty_high_stds: torch.Tensor,
	milestone_difficulty_means: torch.Tensor,
	milestone_difficulty_stds: torch.Tensor,
	benchmark_results_model_indices: torch.Tensor,
	benchmark_results_benchmark_indices: torch.Tensor,
	benchmark_results_scores: torch.Tensor,
	milestone_results_model_indices: torch.Tensor,
	milestone_results_milestone_indices: torch.Tensor,
	milestone_results_scores: torch.Tensor,
) -> torch.Tensor:
	benchmark_predictions = cumulative_split_normal(
		model_intelligences[benchmark_results_model_indices],
		benchmark_difficulty_modes[benchmark_results_benchmark_indices],
		benchmark_difficulty_low_stds[benchmark_results_benchmark_indices],
		benchmark_difficulty_high_stds[benchmark_results_benchmark_indices],
	)

	milestone_predictions = cumulative_normal(
		model_intelligences[milestone_results_model_indices],
		milestone_difficulty_means[milestone_results_milestone_indices],
		milestone_difficulty_stds[milestone_results_milestone_indices],
	)

	predictions = torch.cat([benchmark_predictions, milestone_predictions])
	scores = torch.cat([benchmark_results_scores, milestone_results_scores])

	lower_bounds = torch.where(scores <= EXTREME_SCORE_THRESHOLD, 0.0, scores.clamp(max=1 - EXTREME_SCORE_THRESHOLD))
	upper_bounds = torch.where(scores >= 1 - EXTREME_SCORE_THRESHOLD, 1.0, scores.clamp(min=EXTREME_SCORE_THRESHOLD))

	return torch.mean((predictions - predictions.clamp(lower_bounds, upper_bounds)) ** 2)
