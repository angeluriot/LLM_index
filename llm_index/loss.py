from __future__ import annotations

import math
import torch

from llm_index.constants import BOTTOM_ANCHOR_EASIEST_BENCHMARK_SCORE, BOTTOM_ANCHOR_VALUE, EXTREME_SCORE_REGULARIZATION, EXTREME_SCORE_THRESHOLD, MAX_STD_RATIO, MILESTONE_CENTERING_REGULARIZATION, MIN_STD, TOP_ANCHOR_VALUE
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
	milestone_difficulties_param: torch.Tensor,
	elo_stds_param: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	# STDs must be positive
	positive_low_stds_param = torch.nn.functional.softplus(benchmark_difficulty_low_stds_param)
	positive_high_stds_param = torch.nn.functional.softplus(benchmark_difficulty_high_stds_param)
	positive_elo_stds_param = torch.nn.functional.softplus(elo_stds_param)

	# A side of a benchmark difficulty curve without any score should copy the other side
	positive_high_stds_param = torch.where(benchmark_has_score_above_half, positive_high_stds_param, positive_low_stds_param)
	positive_low_stds_param = torch.where(benchmark_has_score_below_half, positive_low_stds_param, positive_high_stds_param)

	# The ratio between the two STDs of a benchmark difficulty curve is smoothly bounded to [1 / MAX_STD_RATIO, MAX_STD_RATIO]
	# (adding MIN_STD to both sides afterwards can only bring it closer to 1)
	log_low_stds_param = torch.log(positive_low_stds_param)
	log_high_stds_param = torch.log(positive_high_stds_param)
	log_geometric_means = (log_low_stds_param + log_high_stds_param) / 2
	max_half_log_ratio = math.log(MAX_STD_RATIO) / 2
	half_log_ratios = max_half_log_ratio * torch.tanh((log_low_stds_param - log_high_stds_param) / 2 / max_half_log_ratio)
	positive_low_stds_param = torch.exp(log_geometric_means + half_log_ratios)
	positive_high_stds_param = torch.exp(log_geometric_means - half_log_ratios)

	# Set anchors for scaling
	top_anchor = model_intelligences_param[anchor_model_index]

	if benchmark_difficulty_modes_param.numel() > 0:
		bottom_std = torch.special.ndtri(torch.tensor(BOTTOM_ANCHOR_EASIEST_BENCHMARK_SCORE, device=model_intelligences_param.device))
		bottom_anchor = get_bottom_anchor(benchmark_difficulty_modes_param, positive_low_stds_param, positive_high_stds_param)
		scale = (TOP_ANCHOR_VALUE - BOTTOM_ANCHOR_VALUE + bottom_std * MIN_STD) / (top_anchor - bottom_anchor)
	else:
		# Without any benchmark, the bottom anchor is the least intelligent model, aimed at the inverse softplus of its value
		# so that it lands exactly on it after the final softplus
		bottom_anchor = model_intelligences_param.min()
		scale = (TOP_ANCHOR_VALUE - math.log(math.expm1(BOTTOM_ANCHOR_VALUE))) / (top_anchor - bottom_anchor)

	# Apply scaling

	model_intelligences = model_intelligences_param * scale
	benchmark_difficulty_modes = benchmark_difficulty_modes_param * scale
	benchmark_difficulty_low_stds = MIN_STD + positive_low_stds_param * scale
	benchmark_difficulty_high_stds = MIN_STD + positive_high_stds_param * scale
	milestone_difficulties = milestone_difficulties_param * scale
	elo_stds = MIN_STD + positive_elo_stds_param * scale

	shift = TOP_ANCHOR_VALUE - top_anchor * scale

	model_intelligences += shift
	benchmark_difficulty_modes += shift
	milestone_difficulties += shift

	return (
		torch.nn.functional.softplus(model_intelligences),
		benchmark_difficulty_modes,
		benchmark_difficulty_low_stds,
		benchmark_difficulty_high_stds,
		milestone_difficulties,
		elo_stds,
	)


def apply_exact_bottom_anchor(
	model_intelligences: torch.Tensor,
	benchmark_difficulty_modes: torch.Tensor,
	benchmark_difficulty_low_stds: torch.Tensor,
	benchmark_difficulty_high_stds: torch.Tensor,
	milestone_difficulties: torch.Tensor,
	elo_stds: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	# Without any benchmark, the bottom anchor (the least intelligent model) is already exact
	if benchmark_difficulty_modes.numel() == 0:
		return (
			model_intelligences,
			benchmark_difficulty_modes,
			benchmark_difficulty_low_stds,
			benchmark_difficulty_high_stds,
			milestone_difficulties,
			elo_stds,
		)

	bottom_anchor = get_bottom_anchor(benchmark_difficulty_modes, benchmark_difficulty_low_stds, benchmark_difficulty_high_stds)

	scale = (TOP_ANCHOR_VALUE - BOTTOM_ANCHOR_VALUE) / (TOP_ANCHOR_VALUE - bottom_anchor)
	shift = TOP_ANCHOR_VALUE * (1 - scale)

	return (
		torch.abs(model_intelligences * scale + shift),
		benchmark_difficulty_modes * scale + shift,
		benchmark_difficulty_low_stds * scale,
		benchmark_difficulty_high_stds * scale,
		milestone_difficulties * scale + shift,
		elo_stds * scale,
	)


def compute_log_predictions(
	model_intelligences: torch.Tensor,
	benchmark_difficulty_modes: torch.Tensor,
	benchmark_difficulty_low_stds: torch.Tensor,
	benchmark_difficulty_high_stds: torch.Tensor,
	milestone_difficulties: torch.Tensor,
	elo_stds: torch.Tensor,
	benchmark_results_model_indices: torch.Tensor,
	benchmark_results_benchmark_indices: torch.Tensor,
	milestone_results_model_indices: torch.Tensor,
	milestone_results_milestone_indices: torch.Tensor,
	elo_results_model_indices: torch.Tensor,
	elo_results_opponent_indices: torch.Tensor,
	elo_results_elo_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
	benchmark_log_predictions, benchmark_log_complements = log_cumulative_and_complement_split_normal(
		model_intelligences[benchmark_results_model_indices],
		benchmark_difficulty_modes[benchmark_results_benchmark_indices],
		benchmark_difficulty_low_stds[benchmark_results_benchmark_indices],
		benchmark_difficulty_high_stds[benchmark_results_benchmark_indices],
	)

	milestone_log_predictions, milestone_log_complements = log_cumulative_and_complement_normal(
		model_intelligences[milestone_results_model_indices],
		milestone_difficulties[milestone_results_milestone_indices],
		torch.tensor(MIN_STD, device=milestone_difficulties.device),
	)

	# Thurstone model: on an Elo, each model performs around its intelligence with a normal noise of the Elo STD,
	# so a model beats its opponent when the difference of two such performances is positive
	elo_log_predictions, elo_log_complements = log_cumulative_and_complement_normal(
		model_intelligences[elo_results_model_indices],
		model_intelligences[elo_results_opponent_indices],
		math.sqrt(2) * elo_stds[elo_results_elo_indices],
	)

	return (
		torch.cat([benchmark_log_predictions, milestone_log_predictions, elo_log_predictions]),
		torch.cat([benchmark_log_complements, milestone_log_complements, elo_log_complements]),
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
	milestone_difficulties: torch.Tensor,
	elo_stds: torch.Tensor,
	benchmark_results_model_indices: torch.Tensor,
	benchmark_results_benchmark_indices: torch.Tensor,
	benchmark_results_scores: torch.Tensor,
	milestone_results_model_indices: torch.Tensor,
	milestone_results_milestone_indices: torch.Tensor,
	milestone_results_scores: torch.Tensor,
	elo_results_model_indices: torch.Tensor,
	elo_results_opponent_indices: torch.Tensor,
	elo_results_elo_indices: torch.Tensor,
	elo_results_scores: torch.Tensor,
	elo_results_weights: torch.Tensor,
) -> torch.Tensor:
	log_predictions, log_complements = compute_log_predictions(
		model_intelligences,
		benchmark_difficulty_modes,
		benchmark_difficulty_low_stds,
		benchmark_difficulty_high_stds,
		milestone_difficulties,
		elo_stds,
		benchmark_results_model_indices,
		benchmark_results_benchmark_indices,
		milestone_results_model_indices,
		milestone_results_milestone_indices,
		elo_results_model_indices,
		elo_results_opponent_indices,
		elo_results_elo_indices,
	)

	frozen_log_predictions, frozen_log_complements = compute_log_predictions(
		model_intelligences,
		benchmark_difficulty_modes.detach(),
		benchmark_difficulty_low_stds.detach(),
		benchmark_difficulty_high_stds.detach(),
		milestone_difficulties.detach(),
		elo_stds.detach(),
		benchmark_results_model_indices,
		benchmark_results_benchmark_indices,
		milestone_results_model_indices,
		milestone_results_milestone_indices,
		elo_results_model_indices,
		elo_results_opponent_indices,
		elo_results_elo_indices,
	)

	scores = torch.cat([benchmark_results_scores, milestone_results_scores, elo_results_scores])
	weights = torch.cat([torch.ones_like(benchmark_results_scores), torch.ones_like(milestone_results_scores), elo_results_weights])

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

	loss = torch.sum(weights * torch.where(is_admissible, regularizations, divergences)) / torch.sum(weights)

	# Slight regularization of milestone difficulties towards the middle of the gap between the best failing model and the worst succeeding one

	milestone_intelligences = model_intelligences[milestone_results_model_indices].detach()
	is_success = milestone_results_scores > 0.5

	best_failures = torch.full_like(milestone_difficulties, -torch.inf).scatter_reduce(
		0, milestone_results_milestone_indices[~is_success], milestone_intelligences[~is_success], 'amax',
	)
	worst_successes = torch.full_like(milestone_difficulties, torch.inf).scatter_reduce(
		0, milestone_results_milestone_indices[is_success], milestone_intelligences[is_success], 'amin',
	)

	has_gap = torch.isfinite(best_failures) & torch.isfinite(worst_successes) & (best_failures < worst_successes)
	gap_centers = torch.where(has_gap, (best_failures + worst_successes) / 2, 0.0)
	centerings = torch.where(has_gap, (milestone_difficulties - gap_centers) ** 2, 0.0)

	if centerings.numel() == 0:
		return loss

	return loss + MILESTONE_CENTERING_REGULARIZATION * torch.mean(centerings)
