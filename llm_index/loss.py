from __future__ import annotations

import torch

from llm_index.constants import BOTTOM_ANCHOR_EASIEST_BENCHMARK_SCORE, BOTTOM_ANCHOR_VALUE, EXTREME_SCORE_REGULARIZATION, EXTREME_SCORE_THRESHOLD, MIN_STD, TOP_ANCHOR_VALUE
from llm_index.distribution import inverse_cumulative_split_normal, log_cumulative_and_complement_normal, log_cumulative_and_complement_split_normal


def get_bottom_anchor(
	benchmark_difficulty_modes: torch.Tensor,
	benchmark_difficulty_low_stds: torch.Tensor,
	benchmark_difficulty_high_stds: torch.Tensor,
) -> torch.Tensor:
	return inverse_cumulative_split_normal(
		torch.tensor(BOTTOM_ANCHOR_EASIEST_BENCHMARK_SCORE, device=benchmark_difficulty_modes.device),
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
	# STDs must be positive
	positive_low_stds_param = torch.nn.functional.softplus(benchmark_difficulty_low_stds_param)
	positive_high_stds_param = torch.nn.functional.softplus(benchmark_difficulty_high_stds_param)
	positive_milestone_stds_param = torch.nn.functional.softplus(milestone_difficulty_stds_param)

	# A side of a benchmark difficulty curve without any score should copy the other side
	positive_high_stds_param = torch.where(benchmark_has_score_above_half, positive_high_stds_param, positive_low_stds_param)
	positive_low_stds_param = torch.where(benchmark_has_score_below_half, positive_low_stds_param, positive_high_stds_param)

	# Set anchors for scaling
	bottom_std = torch.special.ndtri(torch.tensor(BOTTOM_ANCHOR_EASIEST_BENCHMARK_SCORE, device=model_intelligences_param.device))
	bottom_anchor = get_bottom_anchor(benchmark_difficulty_modes_param, positive_low_stds_param, positive_high_stds_param)
	top_anchor = model_intelligences_param[anchor_model_index]

	# Apply scaling

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


def compute_log_predictions(
	model_intelligences: torch.Tensor,
	benchmark_difficulty_modes: torch.Tensor,
	benchmark_difficulty_low_stds: torch.Tensor,
	benchmark_difficulty_high_stds: torch.Tensor,
	milestone_difficulty_means: torch.Tensor,
	milestone_difficulty_stds: torch.Tensor,
	benchmark_results_model_indices: torch.Tensor,
	benchmark_results_benchmark_indices: torch.Tensor,
	milestone_results_model_indices: torch.Tensor,
	milestone_results_milestone_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
	benchmark_log_predictions, benchmark_log_complements = log_cumulative_and_complement_split_normal(
		model_intelligences[benchmark_results_model_indices],
		benchmark_difficulty_modes[benchmark_results_benchmark_indices],
		benchmark_difficulty_low_stds[benchmark_results_benchmark_indices],
		benchmark_difficulty_high_stds[benchmark_results_benchmark_indices],
	)

	milestone_log_predictions, milestone_log_complements = log_cumulative_and_complement_normal(
		model_intelligences[milestone_results_model_indices],
		milestone_difficulty_means[milestone_results_milestone_indices],
		milestone_difficulty_stds[milestone_results_milestone_indices],
	)

	return (
		torch.cat([benchmark_log_predictions, milestone_log_predictions]),
		torch.cat([benchmark_log_complements, milestone_log_complements]),
	)


def bernoulli_divergence(targets: torch.Tensor, log_predictions: torch.Tensor, log_complements: torch.Tensor) -> torch.Tensor:
	return (
		torch.special.xlogy(targets, targets) - targets * log_predictions
		+ torch.special.xlogy(1 - targets, 1 - targets) - (1 - targets) * log_complements
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
	log_predictions, log_complements = compute_log_predictions(
		model_intelligences,
		benchmark_difficulty_modes,
		benchmark_difficulty_low_stds,
		benchmark_difficulty_high_stds,
		milestone_difficulty_means,
		milestone_difficulty_stds,
		benchmark_results_model_indices,
		benchmark_results_benchmark_indices,
		milestone_results_model_indices,
		milestone_results_milestone_indices,
	)

	frozen_log_predictions, frozen_log_complements = compute_log_predictions(
		model_intelligences,
		benchmark_difficulty_modes.detach(),
		benchmark_difficulty_low_stds.detach(),
		benchmark_difficulty_high_stds.detach(),
		milestone_difficulty_means.detach(),
		milestone_difficulty_stds.detach(),
		benchmark_results_model_indices,
		benchmark_results_benchmark_indices,
		milestone_results_model_indices,
		milestone_results_milestone_indices,
	)

	scores = torch.cat([benchmark_results_scores, milestone_results_scores])

	# No loss for non-matching extreme scores (<0.01 or >0.99)

	lower_bounds = torch.where(scores <= EXTREME_SCORE_THRESHOLD, 0.0, scores.clamp(max=1 - EXTREME_SCORE_THRESHOLD))
	upper_bounds = torch.where(scores >= 1 - EXTREME_SCORE_THRESHOLD, 1.0, scores.clamp(min=EXTREME_SCORE_THRESHOLD))

	predictions = torch.exp(log_predictions).detach()
	targets = predictions.clamp(lower_bounds, upper_bounds)
	divergences = bernoulli_divergence(targets, log_predictions, log_complements)

	is_admissible = (predictions >= lower_bounds) & (predictions <= upper_bounds)

	# Slight regularization for extreme scores towards the center to prevent models from being pushed arbitrarily far

	edges = scores.clamp(EXTREME_SCORE_THRESHOLD, 1 - EXTREME_SCORE_THRESHOLD)
	regularizations = EXTREME_SCORE_REGULARIZATION * bernoulli_divergence(edges, frozen_log_predictions, frozen_log_complements)

	return torch.mean(torch.where(is_admissible, regularizations, divergences))
