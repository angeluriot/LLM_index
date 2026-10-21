from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum
from typing import Literal


from llm_index.constants import DATA_DIR, ONLY_EVERY_YEAR_DELAY_DAYS
from llm_index.utils import normalize


@dataclass
class Model:
	id: str
	name: str
	release_date: date
	benchmarks: dict[str, float] = field(default_factory=dict)
	milestones: list[str] = field(default_factory=list)


class BenchmarkType(Enum):
	ACCURACY = 'ACCURACY'
	SCORE = 'SCORE'
	ELO = 'ELO'


@dataclass
class BenchmarkAccuracyConfig:
	min_score: float
	max_score: float
	type: Literal[BenchmarkType.ACCURACY] = BenchmarkType.ACCURACY


@dataclass
class BenchmarkScoreConfig:
	type: Literal[BenchmarkType.SCORE] = BenchmarkType.SCORE


@dataclass
class BenchmarkEloConfig:
	base: float
	scale: float
	type: Literal[BenchmarkType.ELO] = BenchmarkType.ELO


@dataclass
class Benchmark:
	id: str
	name: str
	config: BenchmarkAccuracyConfig | BenchmarkScoreConfig | BenchmarkEloConfig
	results: dict[str, float] = field(default_factory=dict)

	def get_normalized_results(self) -> list[tuple[str, str, float]]:
		results: list[tuple[str, str, float]] = []

		for model_id, score in self.results.items():
			if self.config.type == BenchmarkType.ACCURACY:
				normalized_score = normalize(self.config.min_score, score, self.config.max_score)
			else:
				normalized_score = score

			results.append((self.id, model_id, normalized_score))

		return results

	def get_normalized_pair_results(self) -> list[tuple[str, str, str, float, float]]:
		results: list[tuple[str, str, str, float, float]] = []
		model_ids = list(self.results)

		# The pairs are weighted so that the whole benchmark weighs n - 1 results, its number of degrees of freedom (an Elo is only defined up to a shift)
		weight = 2 / len(model_ids)

		for index, model_id in enumerate(model_ids):
			for opponent_id in model_ids[index + 1:]:
				win_probability = 1 / (1 + self.config.base ** ((self.results[opponent_id] - self.results[model_id]) / self.config.scale))
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
	milestones: dict[str, Milestone] = field(default_factory=dict)

	def get_benchmark_results(self) -> list[tuple[str, str, float]]:
		results: list[tuple[str, str, float]] = []

		for benchmark in self.benchmarks.values():
			if benchmark.config.type != BenchmarkType.ELO:
				results.extend(benchmark.get_normalized_results())

		return results

	def get_elo_results(self) -> list[tuple[str, str, str, float, float]]:
		results: list[tuple[str, str, str, float, float]] = []

		for benchmark in self.benchmarks.values():
			if benchmark.config.type == BenchmarkType.ELO:
				results.extend(benchmark.get_normalized_pair_results())

		return results

	def get_milestone_results(self) -> list[tuple[str, str, float]]:
		results: list[tuple[str, str, float]] = []

		for milestone in self.milestones.values():
			results.extend(milestone.get_normalized_results(self.models))

		return results


def load_config(raw: dict) -> BenchmarkAccuracyConfig | BenchmarkScoreConfig | BenchmarkEloConfig:
	benchmark_type = BenchmarkType(raw['type'])

	if benchmark_type == BenchmarkType.ACCURACY:
		return BenchmarkAccuracyConfig(min_score=raw['min_score'], max_score=raw['max_score'])

	if benchmark_type == BenchmarkType.SCORE:
		return BenchmarkScoreConfig()

	return BenchmarkEloConfig(base=raw['base'], scale=raw['scale'])


def load_data() -> Data:
	with open(DATA_DIR / 'models.json', encoding='utf-8') as file:
		raw_models = json.load(file)

	models = {
		model_id: Model(id=model_id, name=raw_model['name'], release_date=date.fromisoformat(raw_model['release_date']))
		for model_id, raw_model in sorted(raw_models.items(), key=lambda item: item[1]['release_date'])
	}

	benchmarks: dict[str, Benchmark] = {}

	for path in sorted((DATA_DIR / 'benchmarks').glob('*.json')):
		with open(path, encoding='utf-8') as file:
			raw_benchmark = json.load(file)

		benchmark_id = path.stem
		results = {model_id: score for model_id, score in raw_benchmark['results'].items() if score is not None}
		benchmarks[benchmark_id] = Benchmark(
			id=benchmark_id,
			name=raw_benchmark['name'],
			config=load_config(raw_benchmark['config']),
			results=results,
		)

		for model_id, score in results.items():
			models[model_id].benchmarks[benchmark_id] = score

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

	return Data(models=models, benchmarks=benchmarks, milestones=milestones)
