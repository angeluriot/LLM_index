from dataclasses import dataclass
from datetime import date


@dataclass
class ModelPrice:
	input: int		# Price in cent per 1M input tokens in the creator API
	output: int		# Price in cent per 1M output tokens in the creator API


@dataclass
class ModelAccess:
	announcement: date		# Date of first mention from the creator (can be before the "official" announcement)
	api: date | None		# Date of first widely available access to the model (not necessarily an "API", any kind of access counts)
	weights: date | None	# Date of release of the model weights
	portable: bool			# Whether the model can be run with consumer hardware (false whatever the size if the weights are not released)


@dataclass
class ModelVersion:
	name: str					# Name of the model version (like "o1-preview"/"o1", "DeepSeek V4.1 Flash"/"DeepSeek V4.1 Flash 1017", all the "GPT-4o (<date>)", etc...)
	other_names: list[str]		# Other names or aliases for the model version
	access: ModelAccess			# Access information for the model version
	price: ModelPrice | None	# Pricing information for the model (creator's API pricing or most relevant other pricing)


@dataclass
class _Model:
	name: str					# Name of the model
	other_names: list[str]		# Other names or aliases for the model
	company: str				# Id (from the name of the directory) of the company that created the model
	link: str					# Link to the model's official page, article, announcement, paper, API doc or GitHub repo
	efforts: list[str]			# Effort modes (like "", ""/"Reasoning", "Instant"/"", "Low"/"Medium"/"High"/"Xhigh"/"Max", etc...)


@dataclass
class SingleVersionModel(_Model, ModelVersion):
	pass


@dataclass
class MultiVersionsModel(_Model):
	versions: list[ModelVersion]	# List of different versions of the model


Model = SingleVersionModel | MultiVersionsModel
