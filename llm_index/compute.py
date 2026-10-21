from __future__ import annotations

from dataclasses import dataclass, field

import torch

from llm_index.data import Data
from llm_index.distribution import inverse_cumulative_split_normal
from llm_index.loss import apply_constraints, apply_exact_bottom_anchor, compute_loss
from llm_index.constants import TOP_ANCHOR_MODEL, EPOCHS, LEARNING_RATE, ADAM_BETA_1, ADAM_BETA_2, DEVICE


@dataclass
class Output:
	model_intelligences: dict[str, float] = field(default_factory=dict)
	benchmark_difficulty_modes: dict[str, float] = field(default_factory=dict)
	benchmark_difficulty_low_stds: dict[str, float] = field(default_factory=dict)
	benchmark_difficulty_high_stds: dict[str, float] = field(default_factory=dict)
	milestone_difficulties: dict[str, float] = field(default_factory=dict)
	loss: float = 0.0


def create_output(
	anchor_model_index: int,
	model_intelligences_param: torch.Tensor,
	benchmark_difficulty_modes_param: torch.Tensor,
	benchmark_difficulty_low_stds_param: torch.Tensor,
	benchmark_difficulty_high_stds_param: torch.Tensor,
	benchmark_has_score_below_half: torch.Tensor,
	benchmark_has_score_above_half: torch.Tensor,
	milestone_difficulties_param: torch.Tensor,
	benchmark_results_model_indices: torch.Tensor,
	benchmark_results_benchmark_indices: torch.Tensor,
	benchmark_results_scores: torch.Tensor,
	milestone_results_model_indices: torch.Tensor,
	milestone_results_milestone_indices: torch.Tensor,
	milestone_results_scores: torch.Tensor,
	model_id_to_index: dict[str, int],
	benchmark_id_to_index: dict[str, int],
	milestone_id_to_index: dict[str, int],
	data: Data,
) -> Output:
	with torch.no_grad():
		model_intelligences, \
		benchmark_difficulty_modes, \
		benchmark_difficulty_low_stds, \
		benchmark_difficulty_high_stds, \
		milestone_difficulties = apply_constraints(
			anchor_model_index,
			model_intelligences_param,
			benchmark_difficulty_modes_param,
			benchmark_difficulty_low_stds_param,
			benchmark_difficulty_high_stds_param,
			benchmark_has_score_below_half,
			benchmark_has_score_above_half,
			milestone_difficulties_param,
		)

		model_intelligences, \
		benchmark_difficulty_modes, \
		benchmark_difficulty_low_stds, \
		benchmark_difficulty_high_stds, \
		milestone_difficulties = apply_exact_bottom_anchor(
			model_intelligences,
			benchmark_difficulty_modes,
			benchmark_difficulty_low_stds,
			benchmark_difficulty_high_stds,
			milestone_difficulties,
		)

		loss = compute_loss(
			model_intelligences,
			benchmark_difficulty_modes,
			benchmark_difficulty_low_stds,
			benchmark_difficulty_high_stds,
			milestone_difficulties,
			benchmark_results_model_indices,
			benchmark_results_benchmark_indices,
			benchmark_results_scores,
			milestone_results_model_indices,
			milestone_results_milestone_indices,
			milestone_results_scores,
		)

	output_model_intelligences = {
		id: model_intelligences[index].item()
		for id, index in sorted(model_id_to_index.items(), key=lambda item: data.models[item[0]].release_date)
	}

	benchmark_difficulty_medians = inverse_cumulative_split_normal(
		torch.tensor(0.5, device=benchmark_difficulty_modes.device),
		benchmark_difficulty_modes,
		benchmark_difficulty_low_stds,
		benchmark_difficulty_high_stds,
	)

	sorted_benchmark_ids = sorted(benchmark_id_to_index.items(), key=lambda item: benchmark_difficulty_medians[item[1]].item())

	output_benchmark_difficulty_modes = {id: benchmark_difficulty_modes[index].item() for id, index in sorted_benchmark_ids}
	output_benchmark_difficulty_low_stds = {id: benchmark_difficulty_low_stds[index].item() for id, index in sorted_benchmark_ids}
	output_benchmark_difficulty_high_stds = {id: benchmark_difficulty_high_stds[index].item() for id, index in sorted_benchmark_ids}

	sorted_milestone_ids = sorted(milestone_id_to_index.items(), key=lambda item: data.milestones[item[0]].date_first)

	output_milestone_difficulties = {id: milestone_difficulties[index].item() for id, index in sorted_milestone_ids}

	return Output(
		model_intelligences=output_model_intelligences,
		benchmark_difficulty_modes=output_benchmark_difficulty_modes,
		benchmark_difficulty_low_stds=output_benchmark_difficulty_low_stds,
		benchmark_difficulty_high_stds=output_benchmark_difficulty_high_stds,
		milestone_difficulties=output_milestone_difficulties,
		loss=loss.item(),
	)


def fit(data: Data, verbose: bool = True) -> Output:
	benchmark_results = data.get_benchmark_results()
	milestone_results = data.get_milestone_results()

	model_id_to_index = {id: index for index, id in enumerate(data.models)}
	benchmark_id_to_index = {id: index for index, id in enumerate(data.benchmarks)}
	milestone_id_to_index = {id: index for index, id in enumerate(data.milestones)}
	anchor_model_index = model_id_to_index[TOP_ANCHOR_MODEL]

	benchmark_results_benchmark_indices = torch.tensor([benchmark_id_to_index[benchmark] for benchmark, _, _ in benchmark_results], dtype=torch.long, device=DEVICE)
	benchmark_results_model_indices = torch.tensor([model_id_to_index[model] for _, model, _ in benchmark_results], dtype=torch.long, device=DEVICE)
	benchmark_results_scores = torch.tensor([score for _, _, score in benchmark_results], dtype=torch.float32, device=DEVICE)

	milestone_results_milestone_indices = torch.tensor([milestone_id_to_index[milestone] for milestone, _, _ in milestone_results], dtype=torch.long, device=DEVICE)
	milestone_results_model_indices = torch.tensor([model_id_to_index[model] for _, model, _ in milestone_results], dtype=torch.long, device=DEVICE)
	milestone_results_scores = torch.tensor([score for _, _, score in milestone_results], dtype=torch.float32, device=DEVICE)

	benchmark_has_score_below_half = torch.zeros(len(data.benchmarks), dtype=torch.bool, device=DEVICE)
	benchmark_has_score_below_half[benchmark_results_benchmark_indices[benchmark_results_scores < 0.5]] = True
	benchmark_has_score_above_half = torch.zeros(len(data.benchmarks), dtype=torch.bool, device=DEVICE)
	benchmark_has_score_above_half[benchmark_results_benchmark_indices[benchmark_results_scores > 0.5]] = True

	model_intelligences_param = torch.rand(len(data.models), device=DEVICE, requires_grad=True)
	benchmark_difficulty_modes_param = torch.rand(len(data.benchmarks), device=DEVICE, requires_grad=True)
	benchmark_difficulty_low_stds_param = torch.rand(len(data.benchmarks), device=DEVICE, requires_grad=True)
	benchmark_difficulty_high_stds_param = torch.rand(len(data.benchmarks), device=DEVICE, requires_grad=True)
	milestone_difficulties_param = torch.rand(len(data.milestones), device=DEVICE, requires_grad=True)

	optimizer = torch.optim.Adam([
		model_intelligences_param,
		benchmark_difficulty_modes_param,
		benchmark_difficulty_low_stds_param,
		benchmark_difficulty_high_stds_param,
		milestone_difficulties_param,
	], lr=LEARNING_RATE, betas=(ADAM_BETA_1, ADAM_BETA_2))

	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=0.0)

	for epoch in range(1, EPOCHS + 1):
		optimizer.zero_grad()

		model_intelligences, \
		benchmark_difficulty_modes, \
		benchmark_difficulty_low_stds, \
		benchmark_difficulty_high_stds, \
		milestone_difficulties = apply_constraints(
			anchor_model_index,
			model_intelligences_param,
			benchmark_difficulty_modes_param,
			benchmark_difficulty_low_stds_param,
			benchmark_difficulty_high_stds_param,
			benchmark_has_score_below_half,
			benchmark_has_score_above_half,
			milestone_difficulties_param,
		)

		loss = compute_loss(
			model_intelligences,
			benchmark_difficulty_modes,
			benchmark_difficulty_low_stds,
			benchmark_difficulty_high_stds,
			milestone_difficulties,
			benchmark_results_model_indices,
			benchmark_results_benchmark_indices,
			benchmark_results_scores,
			milestone_results_model_indices,
			milestone_results_milestone_indices,
			milestone_results_scores,
		)

		loss.backward()
		optimizer.step()
		scheduler.step()

		if verbose and (epoch == 1 or epoch % 500 == 0 or epoch == EPOCHS):
			print(f'epoch {epoch:,} | loss = {loss.item():.7f}')

	return create_output(
		anchor_model_index,
		model_intelligences_param,
		benchmark_difficulty_modes_param,
		benchmark_difficulty_low_stds_param,
		benchmark_difficulty_high_stds_param,
		benchmark_has_score_below_half,
		benchmark_has_score_above_half,
		milestone_difficulties_param,
		benchmark_results_model_indices,
		benchmark_results_benchmark_indices,
		benchmark_results_scores,
		milestone_results_model_indices,
		milestone_results_milestone_indices,
		milestone_results_scores,
		model_id_to_index,
		benchmark_id_to_index,
		milestone_id_to_index,
		data,
	)
