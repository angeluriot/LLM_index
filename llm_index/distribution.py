from __future__ import annotations

import math
import torch


def cumulative_normal(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
	return torch.special.ndtr((x - mean) / std)


def inverse_cumulative_normal(p: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
	return mean + std * torch.special.ndtri(p)


def split_normal(x: torch.Tensor, mode: torch.Tensor, std_low: torch.Tensor, std_high: torch.Tensor) -> torch.Tensor:
	std = torch.where(x < mode, std_low, std_high)
	return 2 / (std_low + std_high) * torch.exp(-0.5 * ((x - mode) / std) ** 2) / (2 * torch.pi) ** 0.5


def cumulative_split_normal(x: torch.Tensor, mode: torch.Tensor, std_low: torch.Tensor, std_high: torch.Tensor) -> torch.Tensor:
	total = std_low + std_high
	low = 2 * std_low / total * torch.special.ndtr((x - mode) / std_low)
	high = (std_low - std_high) / total + 2 * std_high / total * torch.special.ndtr((x - mode) / std_high)

	return torch.where(x < mode, low, high)


def log1mexp(x: torch.Tensor) -> torch.Tensor:
	x = x.clamp(max=-1e-7)
	return torch.where(x > -math.log(2), torch.log(-torch.expm1(x)), torch.log1p(-torch.exp(x)))


def log_cumulative_normal(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
	z = (x - mean) / std
	return torch.special.log_ndtr(z), torch.special.log_ndtr(-z)


def log_cumulative_split_normal(x: torch.Tensor, mode: torch.Tensor, std_low: torch.Tensor, std_high: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
	total = std_low + std_high
	is_low = x < mode

	# Each branch is evaluated on its own side of the mode only, so the unused branch stays finite and has no gradient
	log_p_low = torch.log(2 * std_low / total) + torch.special.log_ndtr((torch.minimum(x, mode) - mode) / std_low)
	log_q_high = torch.log(2 * std_high / total) + torch.special.log_ndtr((mode - torch.maximum(x, mode)) / std_high)

	log_p = torch.where(is_low, log_p_low, log1mexp(log_q_high))
	log_q = torch.where(is_low, log1mexp(log_p_low), log_q_high)

	return log_p, log_q


def inverse_cumulative_split_normal(p: torch.Tensor, mode: torch.Tensor, std_low: torch.Tensor, std_high: torch.Tensor) -> torch.Tensor:
	total = std_low + std_high
	is_low = p < std_low / total

	low_p = torch.where(is_low, p * total / (2 * std_low), 0.5)
	high_p = torch.where(is_low, 0.5, (p * total - std_low + std_high) / (2 * std_high))

	low = mode + std_low * torch.special.ndtri(low_p)
	high = mode + std_high * torch.special.ndtri(high_p)

	return torch.where(is_low, low, high)
