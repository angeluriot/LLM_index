def normalize(min: float, value: float, max: float) -> float:
	if value < min:
		return 0.0
	if value > max:
		return 1.0
	if min == max:
		return 0.5
	return (value - min) / (max - min)
