from __future__ import annotations

from dataclasses import dataclass, field

import torch

from llm_index.data import Data
from llm_index.distribution import inverse_cumulative_split_normal
from llm_index.loss import apply_constraints, apply_exact_bottom_anchor, compute_loss, get_score_benchmark_std
from llm_index.constants import TOP_ANCHOR_MODEL, EPOCHS, LEARNING_RATE, ADAM_BETA_1, ADAM_BETA_2, DEVICE


@dataclass
class Output:
	model_intelligences: dict[str, float] = field(default_factory=dict)
	accuracy_benchmark_difficulty_modes: dict[str, float] = field(default_factory=dict)
	accuracy_benchmark_difficulty_low_stds: dict[str, float] = field(default_factory=dict)
	accuracy_benchmark_difficulty_high_stds: dict[str, float] = field(default_factory=dict)
	milestone_difficulties: dict[str, float] = field(default_factory=dict)
	elo_stds: dict[str, float] = field(default_factory=dict)
	score_benchmark_offsets: dict[str, float] = field(default_factory=dict)
	score_benchmark_slopes: dict[str, float] = field(default_factory=dict)
	score_benchmark_std: float = 0.0
	loss: float = 0.0


def create_output(
	anchor_model_index: int,
	model_intelligences_param: torch.Tensor,
	accuracy_benchmark_difficulty_modes_param: torch.Tensor,
	accuracy_benchmark_difficulty_low_stds_param: torch.Tensor,
	accuracy_benchmark_difficulty_high_stds_param: torch.Tensor,
	accuracy_benchmark_has_score_below_half: torch.Tensor,
	accuracy_benchmark_has_score_above_half: torch.Tensor,
	milestone_difficulties_param: torch.Tensor,
	elo_stds_param: torch.Tensor,
	score_benchmark_offsets_param: torch.Tensor,
	score_benchmark_slopes_param: torch.Tensor,
	accuracy_benchmark_results_model_indices: torch.Tensor,
	accuracy_benchmark_results_benchmark_indices: torch.Tensor,
	accuracy_benchmark_results_scores: torch.Tensor,
	milestone_results_model_indices: torch.Tensor,
	milestone_results_milestone_indices: torch.Tensor,
	milestone_results_scores: torch.Tensor,
	elo_results_model_indices: torch.Tensor,
	elo_results_opponent_indices: torch.Tensor,
	elo_results_elo_indices: torch.Tensor,
	elo_results_scores: torch.Tensor,
	elo_results_weights: torch.Tensor,
	score_benchmark_results_model_indices: torch.Tensor,
	score_benchmark_results_benchmark_indices: torch.Tensor,
	score_benchmark_results_scores: torch.Tensor,
	model_id_to_index: dict[str, int],
	accuracy_benchmark_id_to_index: dict[str, int],
	milestone_id_to_index: dict[str, int],
	elo_id_to_index: dict[str, int],
	score_benchmark_id_to_index: dict[str, int],
	data: Data,
) -> Output:
	with torch.no_grad():
		model_intelligences, \
		accuracy_benchmark_difficulty_modes, \
		accuracy_benchmark_difficulty_low_stds, \
		accuracy_benchmark_difficulty_high_stds, \
		milestone_difficulties, \
		elo_stds, \
		score_benchmark_offsets, \
		score_benchmark_slopes = apply_constraints(
			anchor_model_index,
			model_intelligences_param,
			accuracy_benchmark_difficulty_modes_param,
			accuracy_benchmark_difficulty_low_stds_param,
			accuracy_benchmark_difficulty_high_stds_param,
			accuracy_benchmark_has_score_below_half,
			accuracy_benchmark_has_score_above_half,
			milestone_difficulties_param,
			elo_stds_param,
			score_benchmark_offsets_param,
			score_benchmark_slopes_param,
		)

		model_intelligences, \
		accuracy_benchmark_difficulty_modes, \
		accuracy_benchmark_difficulty_low_stds, \
		accuracy_benchmark_difficulty_high_stds, \
		milestone_difficulties, \
		elo_stds, \
		score_benchmark_offsets, \
		score_benchmark_slopes = apply_exact_bottom_anchor(
			model_intelligences,
			accuracy_benchmark_difficulty_modes,
			accuracy_benchmark_difficulty_low_stds,
			accuracy_benchmark_difficulty_high_stds,
			milestone_difficulties,
			elo_stds,
			score_benchmark_offsets,
			score_benchmark_slopes,
		)

		loss = compute_loss(
			model_intelligences,
			accuracy_benchmark_difficulty_modes,
			accuracy_benchmark_difficulty_low_stds,
			accuracy_benchmark_difficulty_high_stds,
			milestone_difficulties,
			elo_stds,
			accuracy_benchmark_results_model_indices,
			accuracy_benchmark_results_benchmark_indices,
			accuracy_benchmark_results_scores,
			milestone_results_model_indices,
			milestone_results_milestone_indices,
			milestone_results_scores,
			elo_results_model_indices,
			elo_results_opponent_indices,
			elo_results_elo_indices,
			elo_results_scores,
			elo_results_weights,
			score_benchmark_offsets,
			score_benchmark_slopes,
			score_benchmark_results_model_indices,
			score_benchmark_results_benchmark_indices,
			score_benchmark_results_scores,
		)

	output_model_intelligences = {
		id: model_intelligences[index].item()
		for id, index in sorted(model_id_to_index.items(), key=lambda item: data.models[item[0]].release_date)
	}

	accuracy_benchmark_difficulty_medians = inverse_cumulative_split_normal(
		torch.tensor(0.5, device=accuracy_benchmark_difficulty_modes.device),
		accuracy_benchmark_difficulty_modes,
		accuracy_benchmark_difficulty_low_stds,
		accuracy_benchmark_difficulty_high_stds,
	)

	sorted_accuracy_benchmark_ids = sorted(accuracy_benchmark_id_to_index.items(), key=lambda item: accuracy_benchmark_difficulty_medians[item[1]].item())

	output_accuracy_benchmark_difficulty_modes = {id: accuracy_benchmark_difficulty_modes[index].item() for id, index in sorted_accuracy_benchmark_ids}
	output_accuracy_benchmark_difficulty_low_stds = {id: accuracy_benchmark_difficulty_low_stds[index].item() for id, index in sorted_accuracy_benchmark_ids}
	output_accuracy_benchmark_difficulty_high_stds = {id: accuracy_benchmark_difficulty_high_stds[index].item() for id, index in sorted_accuracy_benchmark_ids}

	sorted_milestone_ids = sorted(milestone_id_to_index.items(), key=lambda item: data.milestones[item[0]].date_first)

	output_milestone_difficulties = {id: milestone_difficulties[index].item() for id, index in sorted_milestone_ids}

	output_elo_stds = {id: elo_stds[index].item() for id, index in elo_id_to_index.items()}

	output_score_benchmark_offsets = {id: score_benchmark_offsets[index].item() for id, index in score_benchmark_id_to_index.items()}
	output_score_benchmark_slopes = {id: score_benchmark_slopes[index].item() for id, index in score_benchmark_id_to_index.items()}
	output_score_benchmark_std = get_score_benchmark_std(accuracy_benchmark_difficulty_low_stds, accuracy_benchmark_difficulty_high_stds).item()

	return Output(
		model_intelligences=output_model_intelligences,
		accuracy_benchmark_difficulty_modes=output_accuracy_benchmark_difficulty_modes,
		accuracy_benchmark_difficulty_low_stds=output_accuracy_benchmark_difficulty_low_stds,
		accuracy_benchmark_difficulty_high_stds=output_accuracy_benchmark_difficulty_high_stds,
		milestone_difficulties=output_milestone_difficulties,
		elo_stds=output_elo_stds,
		score_benchmark_offsets=output_score_benchmark_offsets,
		score_benchmark_slopes=output_score_benchmark_slopes,
		score_benchmark_std=output_score_benchmark_std,
		loss=loss.item(),
	)


def fit(data: Data, verbose: bool = True) -> Output:
	assert len(data.models) >= 2 and TOP_ANCHOR_MODEL in data.models

	accuracy_benchmark_results = data.get_accuracy_benchmark_results()
	milestone_results = data.get_milestone_results()
	elo_results = data.get_elo_results()
	score_benchmark_results = data.get_score_benchmark_results()

	model_id_to_index = {id: index for index, id in enumerate(data.models)}
	accuracy_benchmark_id_to_index = {id: index for index, id in enumerate(data.accuracy_benchmarks)}
	milestone_id_to_index = {id: index for index, id in enumerate(data.milestones)}
	elo_id_to_index = {id: index for index, id in enumerate(data.elos)}
	score_benchmark_id_to_index = {id: index for index, id in enumerate(data.score_benchmarks)}
	anchor_model_index = model_id_to_index[TOP_ANCHOR_MODEL]

	accuracy_benchmark_results_benchmark_indices = torch.tensor([accuracy_benchmark_id_to_index[benchmark] for benchmark, _, _ in accuracy_benchmark_results], dtype=torch.long, device=DEVICE)
	accuracy_benchmark_results_model_indices = torch.tensor([model_id_to_index[model] for _, model, _ in accuracy_benchmark_results], dtype=torch.long, device=DEVICE)
	accuracy_benchmark_results_scores = torch.tensor([score for _, _, score in accuracy_benchmark_results], dtype=torch.float32, device=DEVICE)

	milestone_results_milestone_indices = torch.tensor([milestone_id_to_index[milestone] for milestone, _, _ in milestone_results], dtype=torch.long, device=DEVICE)
	milestone_results_model_indices = torch.tensor([model_id_to_index[model] for _, model, _ in milestone_results], dtype=torch.long, device=DEVICE)
	milestone_results_scores = torch.tensor([score for _, _, score in milestone_results], dtype=torch.float32, device=DEVICE)

	elo_results_elo_indices = torch.tensor([elo_id_to_index[elo] for elo, _, _, _, _ in elo_results], dtype=torch.long, device=DEVICE)
	elo_results_model_indices = torch.tensor([model_id_to_index[model] for _, model, _, _, _ in elo_results], dtype=torch.long, device=DEVICE)
	elo_results_opponent_indices = torch.tensor([model_id_to_index[opponent] for _, _, opponent, _, _ in elo_results], dtype=torch.long, device=DEVICE)
	elo_results_scores = torch.tensor([score for _, _, _, score, _ in elo_results], dtype=torch.float32, device=DEVICE)
	elo_results_weights = torch.tensor([weight for _, _, _, _, weight in elo_results], dtype=torch.float32, device=DEVICE)

	score_benchmark_results_benchmark_indices = torch.tensor([score_benchmark_id_to_index[benchmark] for benchmark, _, _ in score_benchmark_results], dtype=torch.long, device=DEVICE)
	score_benchmark_results_model_indices = torch.tensor([model_id_to_index[model] for _, model, _ in score_benchmark_results], dtype=torch.long, device=DEVICE)
	score_benchmark_results_scores = torch.tensor([score for _, _, score in score_benchmark_results], dtype=torch.float32, device=DEVICE)

	accuracy_benchmark_has_score_below_half = torch.zeros(len(accuracy_benchmark_id_to_index), dtype=torch.bool, device=DEVICE)
	accuracy_benchmark_has_score_below_half[accuracy_benchmark_results_benchmark_indices[accuracy_benchmark_results_scores < 0.5]] = True
	accuracy_benchmark_has_score_above_half = torch.zeros(len(accuracy_benchmark_id_to_index), dtype=torch.bool, device=DEVICE)
	accuracy_benchmark_has_score_above_half[accuracy_benchmark_results_benchmark_indices[accuracy_benchmark_results_scores > 0.5]] = True

	model_intelligences_param = torch.rand(len(data.models), device=DEVICE, requires_grad=True)
	accuracy_benchmark_difficulty_modes_param = torch.rand(len(accuracy_benchmark_id_to_index), device=DEVICE, requires_grad=True)
	accuracy_benchmark_difficulty_low_stds_param = torch.rand(len(accuracy_benchmark_id_to_index), device=DEVICE, requires_grad=True)
	accuracy_benchmark_difficulty_high_stds_param = torch.rand(len(accuracy_benchmark_id_to_index), device=DEVICE, requires_grad=True)
	milestone_difficulties_param = torch.rand(len(data.milestones), device=DEVICE, requires_grad=True)
	elo_stds_param = torch.rand(len(elo_id_to_index), device=DEVICE, requires_grad=True)
	score_benchmark_offsets_param = torch.rand(len(score_benchmark_id_to_index), device=DEVICE, requires_grad=True)
	score_benchmark_slopes_param = torch.rand(len(score_benchmark_id_to_index), device=DEVICE, requires_grad=True)

	optimizer = torch.optim.Adam([
		model_intelligences_param,
		accuracy_benchmark_difficulty_modes_param,
		accuracy_benchmark_difficulty_low_stds_param,
		accuracy_benchmark_difficulty_high_stds_param,
		milestone_difficulties_param,
		elo_stds_param,
		score_benchmark_offsets_param,
		score_benchmark_slopes_param,
	], lr=LEARNING_RATE, betas=(ADAM_BETA_1, ADAM_BETA_2))

	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=0.0)

	for epoch in range(1, EPOCHS + 1):
		optimizer.zero_grad()

		model_intelligences, \
		accuracy_benchmark_difficulty_modes, \
		accuracy_benchmark_difficulty_low_stds, \
		accuracy_benchmark_difficulty_high_stds, \
		milestone_difficulties, \
		elo_stds, \
		score_benchmark_offsets, \
		score_benchmark_slopes = apply_constraints(
			anchor_model_index,
			model_intelligences_param,
			accuracy_benchmark_difficulty_modes_param,
			accuracy_benchmark_difficulty_low_stds_param,
			accuracy_benchmark_difficulty_high_stds_param,
			accuracy_benchmark_has_score_below_half,
			accuracy_benchmark_has_score_above_half,
			milestone_difficulties_param,
			elo_stds_param,
			score_benchmark_offsets_param,
			score_benchmark_slopes_param,
		)

		loss = compute_loss(
			model_intelligences,
			accuracy_benchmark_difficulty_modes,
			accuracy_benchmark_difficulty_low_stds,
			accuracy_benchmark_difficulty_high_stds,
			milestone_difficulties,
			elo_stds,
			accuracy_benchmark_results_model_indices,
			accuracy_benchmark_results_benchmark_indices,
			accuracy_benchmark_results_scores,
			milestone_results_model_indices,
			milestone_results_milestone_indices,
			milestone_results_scores,
			elo_results_model_indices,
			elo_results_opponent_indices,
			elo_results_elo_indices,
			elo_results_scores,
			elo_results_weights,
			score_benchmark_offsets,
			score_benchmark_slopes,
			score_benchmark_results_model_indices,
			score_benchmark_results_benchmark_indices,
			score_benchmark_results_scores,
		)

		loss.backward()
		optimizer.step()
		scheduler.step()

		if verbose and (epoch == 1 or epoch % 500 == 0 or epoch == EPOCHS):
			print(f'epoch {epoch:,} | loss = {loss.item():.7f}')

	return create_output(
		anchor_model_index,
		model_intelligences_param,
		accuracy_benchmark_difficulty_modes_param,
		accuracy_benchmark_difficulty_low_stds_param,
		accuracy_benchmark_difficulty_high_stds_param,
		accuracy_benchmark_has_score_below_half,
		accuracy_benchmark_has_score_above_half,
		milestone_difficulties_param,
		elo_stds_param,
		score_benchmark_offsets_param,
		score_benchmark_slopes_param,
		accuracy_benchmark_results_model_indices,
		accuracy_benchmark_results_benchmark_indices,
		accuracy_benchmark_results_scores,
		milestone_results_model_indices,
		milestone_results_milestone_indices,
		milestone_results_scores,
		elo_results_model_indices,
		elo_results_opponent_indices,
		elo_results_elo_indices,
		elo_results_scores,
		elo_results_weights,
		score_benchmark_results_model_indices,
		score_benchmark_results_benchmark_indices,
		score_benchmark_results_scores,
		model_id_to_index,
		accuracy_benchmark_id_to_index,
		milestone_id_to_index,
		elo_id_to_index,
		score_benchmark_id_to_index,
		data,
	)
