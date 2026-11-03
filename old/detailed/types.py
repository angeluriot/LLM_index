from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Literal


@dataclass
class ModelHistory:
	first_mention: date		# Date of first mention from the creator (can be before the "official" announcement)
	access: date | None		# Date of first widely available access to the model (none for internal models or not released publicly yet)
	weights: date | None	# Date of release of the model weights (none for non-open weights models or weights not released yet)


@dataclass
class ModelParameters:
	all: int | None		# Number of parameters in the model (none if unknown)
	active: int | None	# Number of active parameters in the model (none if unknown)


@dataclass
class ModelPrice:
	input: float | None		# Price in dollars per 1M input tokens
	output: float | None	# Price in dollars per 1M output tokens


@dataclass
class RawModel:
	display_name: str			# Shown name of the model
	names: list[str]			# List of all names or aliases used to refer to the model (should include the display name)
	link: str					# Link to the model's official page > announcement article > paper > API doc > GitHub repo > other relevant sources
	history: ModelHistory		# Important dates related to the model's release and availability
	parameters: ModelParameters	# Parameter information for the model
	price: ModelPrice			# Pricing information for the model (creator's API pricing or most relevant other pricing)
	efforts: list[str]			# Effort modes (like "", ""/"Reasoning", ""/"Thinking", "Instant"/"", "None"/"Low"/"Medium"/"High"/"Xhigh"/"Max", etc...)


@dataclass
class RawCompany:
	display_name: str
	country: str


@dataclass
class Company:
	id: str
	display_name: str
	country: str


@dataclass
class Model:
	id: str						# Unique identifier for the model (name of the file without extension)
	display_name: str
	names: list[str]
	company: Company			# The company that created the model (by id from the name of the directory)
	link: str
	history: ModelHistory
	parameters: ModelParameters
	price: ModelPrice
	efforts: list[str]


class EvaluationType(Enum):
	SCORE = 'SCORE'
	ELO = 'ELO'


@dataclass
class EvaluationScoreConfig:
	type: Literal[EvaluationType.SCORE]
	min: float | None					# Minimum score achieved in the evaluation (none if no minimum)
	max: float | None					# Maximum score achieved in the evaluation (none if no maximum)
	log: bool							# Whether log should be applied to the scores
	inverted: bool						# Whether the scores should be inverted (where lower is better)


@dataclass
class EvaluationEloConfig:
	type: Literal[EvaluationType.ELO]
	base: float							# Exponent base for the ELO calculation
	scale: float						# Scale factor for the ELO calculation


@dataclass
class RawEvaluation:
	display_name: str									# Shown name of the evaluation
	description: str									# A short description of the evaluation
	link: str											# Link to the evaluation's official page
	last_updated: date									# Date when the evaluation was last updated
	config: EvaluationScoreConfig | EvaluationEloConfig	# Configuration for score processing
	versions: list[str]									# List of versions of the evaluation


@dataclass
class ResultModel:
	id: str				# Id of the model or group of models this result corresponds to
	effort: str | None	# Effort mode used for this result (none if not provided and nb efforts > 1)


class CostType(Enum):
	TASK = 'TASK'
	EVALUATION = 'EVALUATION'


@dataclass
class ResultCost:
	cost: float		# Cost in dollars
	type: CostType	# Cost per task or for the entire evaluation


@dataclass
class Result:
	model: ResultModel		# Model this result corresponds to
	version: str			# Version of the evaluation this result corresponds to
	score: float			# Score achieved in this result
	shots: int				# Number of examples shown to the model (0 by default)
	passes: int				# Number of attempts given to the model (1 by default)
	cost: ResultCost | None	# Cost of the result (none if not available)
	source: str				# Source of the result (url)


@dataclass
class Evaluation:
	id: str												# Id of the evaluation from the directory name
	display_name: str
	description: str
	link: str
	last_updated: date
	config: EvaluationScoreConfig | EvaluationEloConfig
	versions: list[str]


@dataclass
class EvaluationWithResults(Evaluation):
	results: list[Result]	# List of results for this evaluation
