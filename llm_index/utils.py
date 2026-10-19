def normalize(min: float, value: float, max: float) -> float:
	if value < min:
		return 0.0
	if value > max:
		return 1.0
	if min == max:
		return 0.5
	return (value - min) / (max - min)


def add_prefix(prefix: str, text: str) -> str:
	return f'{prefix}{text}'


def remove_prefix(prefix: str, text: str) -> str:
	if text.startswith(prefix):
		return text[len(prefix):]
	return text


def has_prefix(prefix: str, text: str) -> bool:
	return text.startswith(prefix)
