from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, timedelta


from llm_index.constants import DATA_DIR, ONLY_EVERY_YEAR_DELAY_DAYS
from llm_index.utils import normalize


@dataclass
class Model:
	id: str
	name: str
	release_date: date
	benchmarks: dict[str, float] = field(default_factory=dict)
	elos: dict[str, float] = field(default_factory=dict)
	milestones: list[str] = field(default_factory=list)


@dataclass
class Benchmark:
	id: str
	name: str
	min_score: float | None
	max_score: float | None
	results: dict[str, float] = field(default_factory=dict)

	def get_normalized_results(self) -> list[tuple[str, str, float]]:
		results: list[tuple[str, str, float]] = []

		# A missing bound is replaced by the most extreme score on its side
		min_score = min(self.results.values()) if self.min_score is None else self.min_score
		max_score = max(self.results.values()) if self.max_score is None else self.max_score

		for model_id, score in self.results.items():
			results.append((self.id, model_id, normalize(min_score, score, max_score)))

		return results


@dataclass
class Elo:
	id: str
	name: str
	base: float
	scale: float
	results: dict[str, float] = field(default_factory=dict)

	def get_normalized_results(self) -> list[tuple[str, str, str, float, float]]:
		results: list[tuple[str, str, str, float, float]] = []
		model_ids = list(self.results)

		# The pairs are weighted so that the whole Elo weighs n - 1 results, its number of degrees of freedom (an Elo is only defined up to a shift)
		weight = 2 / len(model_ids)

		for index, model_id in enumerate(model_ids):
			for opponent_id in model_ids[index + 1:]:
				win_probability = 1 / (1 + self.base ** ((self.results[opponent_id] - self.results[model_id]) / self.scale))
				results.append((self.id, model_id, opponent_id, win_probability, weight))

		return results


@dataclass
class Milestone:
	id: str
	name: str
	date_first: date
	achievers: list[str] = field(default_factory=list)
	only_every_year: bool = False

	def get_normalized_results(self, models: dict[str, Model]) -> list[tuple[str, str, float]]:
		results: list[tuple[str, str, float]] = []
		delay = ONLY_EVERY_YEAR_DELAY_DAYS if self.only_every_year else 1
		last_failure_date = self.date_first - timedelta(days=delay)

		for model_id, model in models.items():
			if model_id in self.achievers:
				normalized_score = 1.0
			elif model.release_date <= last_failure_date:
				normalized_score = 0.0
			else:
				continue

			results.append((self.id, model_id, normalized_score))

		return results


@dataclass
class Data:
	models: dict[str, Model] = field(default_factory=dict)
	benchmarks: dict[str, Benchmark] = field(default_factory=dict)
	elos: dict[str, Elo] = field(default_factory=dict)
	milestones: dict[str, Milestone] = field(default_factory=dict)

	def get_benchmark_results(self) -> list[tuple[str, str, float]]:
		results: list[tuple[str, str, float]] = []

		for benchmark in self.benchmarks.values():
			results.extend(benchmark.get_normalized_results())

		return results

	def get_elo_results(self) -> list[tuple[str, str, str, float, float]]:
		results: list[tuple[str, str, str, float, float]] = []

		for elo in self.elos.values():
			results.extend(elo.get_normalized_results())

		return results

	def get_milestone_results(self) -> list[tuple[str, str, float]]:
		results: list[tuple[str, str, float]] = []

		for milestone in self.milestones.values():
			results.extend(milestone.get_normalized_results(self.models))

		return results


def load_raw_scores(directory: str) -> dict[str, dict]:
	raws: dict[str, dict] = {}

	for path in sorted((DATA_DIR / directory).glob('*.json')):
		with open(path, encoding='utf-8') as file:
			raw = json.load(file)

		raw['results'] = {model_id: score for model_id, score in raw['results'].items() if score is not None}
		raws[path.stem] = raw

	return raws


def load_data() -> Data:
	with open(DATA_DIR / 'models.json', encoding='utf-8') as file:
		raw_models = json.load(file)

	models = {
		model_id: Model(id=model_id, name=raw_model['name'], release_date=date.fromisoformat(raw_model['release_date']))
		for model_id, raw_model in sorted(raw_models.items(), key=lambda item: item[1]['release_date'])
	}

	benchmarks: dict[str, Benchmark] = {}

	for benchmark_id, raw_benchmark in load_raw_scores('benchmarks').items():
		benchmarks[benchmark_id] = Benchmark(
			id=benchmark_id,
			name=raw_benchmark['name'],
			min_score=raw_benchmark['min_score'],
			max_score=raw_benchmark['max_score'],
			results=raw_benchmark['results'],
		)

		for model_id, score in raw_benchmark['results'].items():
			models[model_id].benchmarks[benchmark_id] = score

	elos: dict[str, Elo] = {}

	for elo_id, raw_elo in load_raw_scores('elos').items():
		elos[elo_id] = Elo(
			id=elo_id,
			name=raw_elo['name'],
			base=raw_elo['base'],
			scale=raw_elo['scale'],
			results=raw_elo['results'],
		)

		for model_id, score in raw_elo['results'].items():
			models[model_id].elos[elo_id] = score

	with open(DATA_DIR / 'milestones.json', encoding='utf-8') as file:
		raw_milestones = json.load(file)

	milestones: dict[str, Milestone] = {}

	for milestone_id, raw_milestone in sorted(raw_milestones.items(), key=lambda item: item[1]['date_first']):
		milestones[milestone_id] = Milestone(
			id=milestone_id,
			name=raw_milestone['name'],
			date_first=date.fromisoformat(raw_milestone['date_first']),
			achievers=list(raw_milestone['achievers']),
			only_every_year=raw_milestone['only_every_year'],
		)

		for model_id in raw_milestone['achievers']:
			models[model_id].milestones.append(milestone_id)

	return Data(models=models, benchmarks=benchmarks, elos=elos, milestones=milestones)
