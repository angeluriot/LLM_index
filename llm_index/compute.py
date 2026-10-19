from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import torch

from llm_index.data import Data
from llm_index.loss import apply_constraints, compute_loss
from llm_index.constants import TOP_ANCHOR_MODEL, EPOCHS, LEARNING_RATE
from llm_index.utils import has_prefix, remove_prefix


@dataclass
class Output:
	model_intelligences: dict[str, float] = field(default_factory=dict)
	benchmark_difficulty_means: dict[str, float] = field(default_factory=dict)
	benchmark_difficulty_stds: dict[str, float] = field(default_factory=dict)
	milestone_difficulties: dict[str, float] = field(default_factory=dict)
	loss: float = 0.0


def create_output(
	anchor_model_index: int,
	model_intelligences_param: torch.Tensor,
	benchmark_difficulty_means_param: torch.Tensor,
	benchmark_difficulty_stds_param: torch.Tensor,
	results_model_indices: torch.Tensor,
	results_benchmark_indices: torch.Tensor,
	results_scores: torch.Tensor,
	model_id_to_index: dict[str, int],
	benchmark_id_to_index: dict[str, int],
	data: Data,
) -> Output:
	with torch.no_grad():
		model_intelligences, \
		benchmark_difficulty_means, \
		benchmark_difficulty_stds = apply_constraints(
			anchor_model_index,
			model_intelligences_param,
			benchmark_difficulty_means_param,
			benchmark_difficulty_stds_param,
		)

		loss = compute_loss(
			model_intelligences,
			benchmark_difficulty_means,
			benchmark_difficulty_stds,
			results_model_indices,
			results_benchmark_indices,
			results_scores,
		)

	output_model_intelligences = {
		id: model_intelligences[index].item()
		for id, index in sorted(model_id_to_index.items(), key=lambda item: data.models[item[0]].release_date)
	}

	output_benchmark_difficulty_means = {
		remove_prefix('benchmark-', id): benchmark_difficulty_means[index].item()
		for id, index in sorted(benchmark_id_to_index.items(), key=lambda item: benchmark_difficulty_means[item[1]].item())
		if has_prefix('benchmark-', id)
	}

	output_benchmark_difficulty_stds = {
		remove_prefix('benchmark-', id): benchmark_difficulty_stds[index].item()
		for id, index in sorted(benchmark_id_to_index.items(), key=lambda item: benchmark_difficulty_means[item[1]].item())
		if has_prefix('benchmark-', id)
	}

	output_milestone_difficulties = {
		remove_prefix('milestone-', id): benchmark_difficulty_means[index].item()
		for id, index in sorted(
			benchmark_id_to_index.items(),
			key=lambda item: data.milestones[remove_prefix('milestone-', item[0])].date_first \
				if has_prefix('milestone-', item[0]) \
				else date.min
		)
		if has_prefix('milestone-', id)
	}

	return Output(
		model_intelligences=output_model_intelligences,
		benchmark_difficulty_means=output_benchmark_difficulty_means,
		benchmark_difficulty_stds=output_benchmark_difficulty_stds,
		milestone_difficulties=output_milestone_difficulties,
		loss=loss.item(),
	)


def fit(data: Data, verbose: bool = True) -> Output:
	results = data.get_normalized_results()
	benchmark_ids, model_ids = data.get_results_ids()

	benchmark_id_to_index = {id: index for index, id in enumerate(benchmark_ids)}
	model_id_to_index = {id: index for index, id in enumerate(model_ids)}
	anchor_model_index = model_id_to_index[TOP_ANCHOR_MODEL]

	results_benchmark_indices = torch.tensor([benchmark_id_to_index[benchmark] for benchmark, _, _ in results], dtype=torch.long)
	results_model_indices = torch.tensor([model_id_to_index[model] for _, model, _ in results], dtype=torch.long)
	results_scores = torch.tensor([score for _, _, score in results], dtype=torch.float32)

	model_intelligences_param = torch.rand(len(model_ids), requires_grad=True)
	benchmark_difficulty_means_param = torch.rand(len(benchmark_ids), requires_grad=True)
	benchmark_difficulty_stds_param = torch.rand(len(benchmark_ids), requires_grad=True)

	optimizer = torch.optim.Adam([
		model_intelligences_param,
		benchmark_difficulty_means_param,
		benchmark_difficulty_stds_param,
	], lr=LEARNING_RATE)

	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=0.0)
	best_loss = float('inf')

	for epoch in range(EPOCHS):
		optimizer.zero_grad()

		model_intelligences, \
		benchmark_difficulty_means, \
		benchmark_difficulty_stds = apply_constraints(
			anchor_model_index,
			model_intelligences_param,
			benchmark_difficulty_means_param,
			benchmark_difficulty_stds_param,
		)

		loss = compute_loss(
			model_intelligences,
			benchmark_difficulty_means,
			benchmark_difficulty_stds,
			results_model_indices,
			results_benchmark_indices,
			results_scores,
		)

		loss.backward()
		optimizer.step()
		scheduler.step()

		if verbose and (epoch % 5_000 == 0 or epoch == EPOCHS - 1):
			print(f'epoch {epoch:,} | loss = {loss.item():.7f}')

		if loss.item() < best_loss:
			best_loss = loss.item()
			output = create_output(
				anchor_model_index,
				model_intelligences_param,
				benchmark_difficulty_means_param,
				benchmark_difficulty_stds_param,
				results_model_indices,
				results_benchmark_indices,
				results_scores,
				model_id_to_index,
				benchmark_id_to_index,
				data,
			)

	return output
