"""Generates data/models/mistral/*.json (one file per model).

All the data lives in this script: edit it and run `python scripts/models/gen_mistral.py`.
Existing JSON files are deleted and rewritten, so manual edits to them are lost.

Prices are Mistral's API launch prices in USD (the first USD prices for the Dec 2023 models, billed in EUR at the time).
Pixtral (vision), Mathstral, Saba and Leanstral are versions of the model they are post-trained from.
Magistral (reasoning) and Devstral (coding) are well-known product lines that nobody lumps with Mistral Small / Medium,
so they are separate models even when built on the same weights. Each new version number is also a separate model
(Small 3 / 3.1 / 3.2, Large 2 / 2.1, Magistral Small 1.0 / 1.1 / 1.2...).
"""

from common import B, M, N, S, V, write

M_ = 1_000_000
NH = ['None', 'High']		# reasoning_effort of the hybrid models (Small 4, Medium 3.5, Large 4 preview)
LMH = ['Low', 'Medium', 'High']

NEWS = 'https://mistral.ai/news/'
DOCS = 'https://docs.mistral.ai/models/'
HF = 'https://huggingface.co/'


def hf_names(*repos):
	"""Repo ids, both bare and with the mistralai organisation."""
	return list(dict.fromkeys([r for r in repos] + [f'mistralai/{r}' for r in repos]))


# Mistral 7B (v0.2 / v0.3 are updates of the same model; Mathstral is a fine-tune of Mistral 7B)
P7B = (7_240 * M_, 7_240 * M_)
M('Mistral 7B',
	V('Mistral 7B v0.1', ['Mistral 7B', 'Mistral-7B', 'Mistral 7B Instruct v0.1', 'mistral-7b-v0.1', 'mistral-7b-instruct-v0.1', *hf_names('Mistral-7B-v0.1', 'Mistral-7B-Instruct-v0.1')], NEWS + 'announcing-mistral-7b', '2023-09-27', '2023-09-27', weights='2023-09-27', params=P7B),
	V('Mistral 7B v0.2', ['Mistral 7B Instruct v0.2', 'mistral-7b-instruct-v0.2', 'mistral-tiny', 'mistral-tiny-2312', *hf_names('Mistral-7B-v0.2', 'Mistral-7B-Instruct-v0.2')], DOCS + 'mistral-7b-0-2', '2023-12-11', '2023-12-11', weights='2023-12-11', params=P7B, price=(0.25, 0.25)),
	V('Mistral 7B v0.3', ['Mistral 7B Instruct v0.3', 'mistral-7b-v0.3', 'mistral-7b-instruct-v0.3', *hf_names('Mistral-7B-v0.3', 'Mistral-7B-Instruct-v0.3')], DOCS + 'mistral-7b-0-3', '2024-05-22', '2024-05-22', weights='2024-05-22', params=P7B, price=(0.25, 0.25)),
	V('Mathstral 7B', ['Mathstral', 'MathΣtral', 'Mathstral-7B', 'mathstral-7b-v0.1', *hf_names('Mathstral-7B-v0.1')], NEWS + 'mathstral', '2024-07-16', '2024-07-16', weights='2024-07-16', params=P7B),
)

# Mixtral (sparse MoE)
S('Mixtral 8x7B', ['Mixtral 8x7B Instruct', 'Mixtral 8x7B Instruct v0.1', 'Mixtral-8x7B', 'Mixtral 8x7B v0.1', 'open-mixtral-8x7b', 'mistral-small-2312', 'mixtral-8x7b-instruct', *hf_names('Mixtral-8x7B-v0.1', 'Mixtral-8x7B-Instruct-v0.1')], NEWS + 'mixtral-of-experts', '2023-12-08', '2023-12-08', weights='2023-12-08', params=(46_700 * M_, 12_900 * M_), price=(0.7, 0.7))
S('Mixtral 8x22B', ['Mixtral 8x22B Instruct', 'Mixtral 8x22B Instruct v0.1', 'Mixtral-8x22B', 'Mixtral 8x22B v0.1', 'open-mixtral-8x22b', 'open-mixtral-8x22b-2404', 'mixtral-8x22b-instruct', *hf_names('Mixtral-8x22B-v0.1', 'Mixtral-8x22B-Instruct-v0.1')], NEWS + 'mixtral-8x22b', '2024-04-10', '2024-04-10', weights='2024-04-10', params=(141 * B, 39 * B), price=(2, 6))

# First closed models (La Plateforme, Le Chat)
S('Mistral Medium', ['Mistral Medium 1.0', 'Mistral Medium 1', 'mistral-medium-2312'], NEWS + 'la-plateforme', '2023-12-11', '2023-12-11', price=(2.7, 8.1))
S('Mistral Next', ['mistral-next'], DOCS + 'mistral-next', '2024-02-16', '2024-02-16')
LARGE = NEWS + 'mistral-large'
M('Mistral Large',
	V('Mistral Large (Feb 2024)', ['Mistral Large', 'Mistral Large 1.0', 'Mistral Large 1', "Mistral Large (Feb '24)", 'Mistral Large 24.02', 'mistral-large-2402'], LARGE, '2024-02-26', '2024-02-26', id='mistral-large-1', price=(8, 24)),
)
M('Mistral Small',
	V('Mistral Small (Feb 2024)', ['Mistral Small 1.0', 'Mistral Small 1', "Mistral Small (Feb '24)", 'Mistral Small 24.02', 'mistral-small-2402'], LARGE, '2024-02-26', '2024-02-26', id='mistral-small-1', price=(2, 6)),
)

# Codestral
S('Codestral 22B', ['Codestral', 'Codestral 24.05', 'Codestral-22B', 'codestral-22b', 'codestral-2405', *hf_names('Codestral-22B-v0.1')], NEWS + 'codestral', '2024-05-29', '2024-05-29', weights='2024-05-29', params=(22_200 * M_, 22_200 * M_), price=(1, 3))
S('Codestral Mamba', ['Codestral Mamba 7B', 'Mamba-Codestral-7B', 'open-codestral-mamba', *hf_names('Mamba-Codestral-7B-v0.1')], NEWS + 'codestral-mamba', '2024-07-16', '2024-07-16', weights='2024-07-16', params=(7_290 * M_, 7_290 * M_), price=(0.25, 0.25))

# Mistral NeMo 12B (with NVIDIA) and Pixtral 12B (decoder based on Mistral NeMo)
M('Mistral NeMo',
	V('Mistral NeMo', ['Mistral Nemo', 'Mistral NeMo 12B', 'Mistral Nemo 12B', 'Mistral NeMo Instruct', 'open-mistral-nemo', 'open-mistral-nemo-2407', 'mistral-nemo-instruct-2407', *hf_names('Mistral-Nemo-Base-2407', 'Mistral-Nemo-Instruct-2407', 'Mistral-Nemo-Instruct-FP8-2407')], NEWS + 'mistral-nemo', '2024-07-18', '2024-07-18', weights='2024-07-18', params=(12_200 * M_, 12_200 * M_), price=(0.3, 0.3)),
	V('Pixtral 12B', ['Pixtral', 'Pixtral-12B', 'pixtral-12b', 'pixtral-12b-2409', *hf_names('Pixtral-12B-2409', 'Pixtral-12B-Base-2409')], NEWS + 'pixtral-12b', '2024-09-11', '2024-09-11', weights='2024-09-11', params=(12_400 * M_, 12_400 * M_), price=(0.15, 0.15)),
)

# Mistral Large 2 (123B; Pixtral Large is built on it) and its Nov 2024 update
P_L2 = (123 * B, 123 * B)
M('Mistral Large 2',
	V('Mistral Large 2', ['Mistral Large 2.0', "Mistral Large 2 (Jul '24)", 'Mistral Large 2 (Jul 2024)', 'Mistral Large 24.07', 'Mistral Large 2407', 'mistral-large-2407', 'mistral-large-2-2407', *hf_names('Mistral-Large-Instruct-2407')], NEWS + 'mistral-large-2407', '2024-07-24', '2024-07-24', weights='2024-07-24', params=P_L2, price=(3, 9)),
	V('Pixtral Large', ['Pixtral Large 24.11', 'Pixtral-Large', 'pixtral-large', 'pixtral-large-2411', *hf_names('Pixtral-Large-Instruct-2411')], NEWS + 'pixtral-large', '2024-11-18', '2024-11-18', weights='2024-11-18', params=(124 * B, 124 * B), price=(2, 6)),
)
S('Mistral Large 2.1', ["Mistral Large 2 (Nov '24)", 'Mistral Large 2 (Nov 2024)', 'Mistral Large 24.11', 'Mistral Large 2411', 'mistral-large-2411', *hf_names('Mistral-Large-Instruct-2411')], NEWS + 'pixtral-large', '2024-11-18', '2024-11-18', weights='2024-11-18', params=P_L2, price=(2, 6))

M('Mistral Small 2',
	V('Mistral Small v24.09', ['Mistral Small 2.0', 'Mistral Small 2', "Mistral Small (Sep '24)", 'Mistral Small 24.09', 'Mistral Small 22B', 'mistral-small-2409', *hf_names('Mistral-Small-Instruct-2409')], NEWS + 'september-24-release', '2024-09-17', '2024-09-17', weights='2024-09-17', params=(22 * B, 22 * B), price=(0.2, 0.6)),
)

# Ministral (Les Ministraux, Oct 2024)
MINISTRAUX = NEWS + 'ministraux'
S('Ministral 8B', ['Ministral 8B Instruct', 'Ministral 8B 24.10', 'ministral-8b-2410', 'ministral-8b-instruct-2410', *hf_names('Ministral-8B-Instruct-2410')], MINISTRAUX, '2024-10-16', '2024-10-16', weights='2024-10-16', params=(8_020 * M_, 8_020 * M_), price=(0.1, 0.1))
S('Ministral 3B', ['Ministral 3B 24.10', 'ministral-3b-2410'], MINISTRAUX, '2024-10-16', '2024-10-16', params=(3 * B, 3 * B), price=(0.04, 0.04))

# Codestral 25.xx (new architecture, closed weights; the API's current "Codestral", vs. the open-weights Codestral 22B)
M('Codestral',
	V('Codestral 25.01', ['Codestral 2501', "Codestral (Jan '25)", 'codestral-2501'], NEWS + 'codestral-2501', '2025-01-13', '2025-01-13', price=(0.3, 0.9)),
	V('Codestral 25.08', ['Codestral 2508', 'codestral-2508'], NEWS + 'codestral-25-08', '2025-07-30', '2025-07-30', price=(0.3, 0.9)),
)

# 24B generation: Mistral Small 3 (+ Saba, a fine-tuned Small) / 3.1 / 3.2, and Devstral Small / Magistral Small (built on Small 3.1 / 3.2)
P24 = (24 * B, 24 * B)
MAGISTRAL = NEWS + 'magistral'
DEVSTRAL_2507 = NEWS + 'devstral-2507'
M('Mistral Small 3',
	V('Mistral Small 3', ['Mistral Small 3.0', 'Mistral Small 3 24B', 'Mistral Small 24B', 'Mistral Small 3 24B Instruct', 'Mistral Small 3 24B Base', 'Mistral Small 25.01', 'mistral-small-2501', 'mistral-small-24b-instruct-2501', 'mistral-small-24b-base-2501', *hf_names('Mistral-Small-24B-Base-2501', 'Mistral-Small-24B-Instruct-2501')], NEWS + 'mistral-small-3', '2025-01-30', '2025-01-30', weights='2025-01-30', params=P24, price=(0.1, 0.3)),
	V('Mistral Saba', ['Mistral Saba 24B', 'Mistral Saba 25.02', 'mistral-saba-2502'], NEWS + 'mistral-saba', '2025-02-17', '2025-02-17', params=P24, price=(0.2, 0.6)),
)
S('Mistral Small 3.1', ['Mistral Small 3.1 24B', 'Mistral Small 3.1 24B Instruct', 'Mistral Small 3.1 24B Base', 'Mistral Small 25.03', 'mistral-small-2503', 'mistral-small-3.1-24b-instruct-2503', 'mistral-small-3.1-24b-base-2503', *hf_names('Mistral-Small-3.1-24B-Base-2503', 'Mistral-Small-3.1-24B-Instruct-2503')], NEWS + 'mistral-small-3-1', '2025-03-17', '2025-03-17', weights='2025-03-17', params=P24, price=(0.1, 0.3))
S('Devstral Small 1.0', ['Devstral Small', "Devstral Small (May '25)", 'Devstral Small 2505', 'devstral-small-2505', *hf_names('Devstral-Small-2505')], NEWS + 'devstral', '2025-05-21', '2025-05-21', weights='2025-05-21', params=P24, price=(0.1, 0.3))
S('Magistral Small 1.0', ['Magistral Small', 'Magistral Small 1', 'Magistral Small 2506', 'magistral-small-2506', *hf_names('Magistral-Small-2506')], MAGISTRAL, '2025-06-10', '2025-06-10', weights='2025-06-10', params=P24, price=(0.5, 1.5))
S('Mistral Small 3.2', ['Mistral Small 3.2 24B', 'Mistral Small 3.2 24B Instruct', 'Mistral Small 25.06', 'mistral-small-2506', 'mistral-small-3.2-24b-instruct-2506', *hf_names('Mistral-Small-3.2-24B-Instruct-2506')], HF + 'mistralai/Mistral-Small-3.2-24B-Instruct-2506', '2025-06-20', '2025-06-20', weights='2025-06-20', params=P24, price=(0.1, 0.3))
S('Devstral Small 1.1', ["Devstral Small (Jul '25)", 'Devstral Small 2507', 'devstral-small-2507', *hf_names('Devstral-Small-2507')], DEVSTRAL_2507, '2025-07-10', '2025-07-10', weights='2025-07-10', params=P24, price=(0.1, 0.3))
S('Magistral Small 1.1', ['Magistral Small 2507', 'magistral-small-2507', *hf_names('Magistral-Small-2507')], DOCS + 'magistral-small-1-1-25-07', '2025-07-24', '2025-07-24', weights='2025-07-24', params=P24, price=(0.5, 1.5))
S('Magistral Small 1.2', ['Magistral Small 2509', 'magistral-small-2509', *hf_names('Magistral-Small-2509')], DOCS + 'magistral-small-1-2-25-09', '2025-09-17', '2025-09-17', weights='2025-09-17', params=P24, price=(0.5, 1.5))
S('Devstral Small 2', ['Devstral Small 2 24B', 'Devstral 2 (24B)', 'Devstral Small 2512', 'labs-devstral-small-2512', *hf_names('Devstral-Small-2-24B-Instruct-2512')], NEWS + 'devstral-2-vibe-cli', '2025-12-09', '2025-12-09', weights='2025-12-09', params=P24, price=(0.1, 0.3))

# Mistral Medium 3 / 3.1 and Magistral Medium (trained on top of Medium 3)
S('Mistral Medium 3', ['Mistral Medium 3.0', 'Mistral Medium 25.05', 'mistral-medium-2505'], NEWS + 'mistral-medium-3', '2025-05-07', '2025-05-07', price=(0.4, 2))
S('Magistral Medium 1.0', ['Magistral Medium', 'Magistral Medium 1', 'Magistral Medium 2506', 'magistral-medium-2506'], MAGISTRAL, '2025-06-10', '2025-06-10', price=(2, 5))
S('Magistral Medium 1.1', ['Magistral Medium 2507', 'magistral-medium-2507'], DOCS + 'magistral-medium-1-1-25-07', '2025-07-24', '2025-07-24', price=(2, 5))
S('Mistral Medium 3.1', ['Mistral Medium 25.08', 'mistral-medium-2508'], DOCS + 'mistral-medium-3-1-25-08', '2025-08-12', '2025-08-12', price=(0.4, 2))
S('Magistral Medium 1.2', ['Magistral Medium 2509', 'magistral-medium-2509'], DOCS + 'magistral-medium-1-2-25-09', '2025-09-17', '2025-09-17', price=(2, 5))
S('Devstral Medium', ['Devstral Medium 1.0', 'Devstral Medium 2507', 'devstral-medium-2507'], DEVSTRAL_2507, '2025-07-10', '2025-07-10', price=(0.4, 2))

# Mistral 3 (Dec 2025): Mistral Large 3 and the Ministral 3 sizes (each with base, instruct and reasoning weights)
MISTRAL3 = NEWS + 'mistral-3'
S('Mistral Large 3', ['Mistral Large 3 675B', 'Mistral Large 3 (675B Instruct 2512)', 'Mistral Large 3 (675B Base)', 'Mistral Large 3 (675B Instruct 2512 NVFP4)', 'Mistral Large 25.12', 'mistral-large-2512', 'mistral-large-3-675b-base-2512', 'mistral-large-3-675b-instruct-2512-nvfp4', *hf_names('Mistral-Large-3-675B-Base-2512', 'Mistral-Large-3-675B-Instruct-2512', 'Mistral-Large-3-675B-Instruct-2512-BF16', 'Mistral-Large-3-675B-Instruct-2512-NVFP4')], MISTRAL3, '2025-12-02', '2025-12-02', weights='2025-12-02', params=(675 * B, 41 * B), price=(0.5, 1.5))
for size, p, price in [('14B', 14 * B, 0.2), ('8B', 8 * B, 0.15), ('3B', 3 * B, 0.1)]:
	s = size.lower()
	M(f'Ministral 3 {size}',
		V(f'Ministral 3 {size}', [f'Ministral 3 {size} Instruct', f'Ministral 3 ({size} Instruct 2512)', f'Ministral 3 ({size} Base 2512)', f'Ministral 3 {size} 25.12', f'ministral-{s}-2512', f'ministral-3-{s}-instruct-2512', f'ministral-3-{s}-base-2512', *hf_names(f'Ministral-3-{size}-Base-2512', f'Ministral-3-{size}-Instruct-2512', f'Ministral-3-{size}-Instruct-2512-BF16')], MISTRAL3, '2025-12-02', '2025-12-02', weights='2025-12-02', params=(p, p), price=(price, price)),
		V(f'Ministral 3 {size} Reasoning', [f'Ministral 3 ({size} Reasoning 2512)', f'ministral-3-{s}-reasoning-2512', *hf_names(f'Ministral-3-{size}-Reasoning-2512')], MISTRAL3, '2025-12-02', '2025-12-02', weights='2025-12-02', params=(p, p)),
	)

# Devstral 2 (123B dense, base model not disclosed) and Mistral Small Creative (Labs, base model not disclosed)
S('Devstral 2', ['Devstral 2 123B', 'Devstral 2 (123B)', 'Devstral 2512', 'devstral-2512', *hf_names('Devstral-2-123B-Instruct-2512')], NEWS + 'devstral-2-vibe-cli', '2025-12-09', '2025-12-09', weights='2025-12-09', params=(123 * B, 123 * B), price=(0.4, 2))
S('Mistral Small Creative', ['Mistral Small Creative 25.12', 'mistral-small-creative', 'labs-mistral-small-creative'], DOCS + 'mistral-small-creative-25-12', '2025-12-16', '2025-12-16', price=(0.1, 0.3))

# Mistral Small 4 (119B MoE) and Leanstral (Lean 4 prover of the Small 4 family, free Labs API)
P_S4 = (119 * B, 6_500 * M_)
M('Mistral Small 4',
	V('Mistral Small 4', ['Mistral Small 4 119B', 'Mistral Small 4 (Reasoning)', 'Mistral Small 4 (Non-reasoning)', 'Mistral Small 26.03', 'mistral-small-2603', *hf_names('Mistral-Small-4-119B-2603', 'Mistral-Small-4-119B-2603-NVFP4')], NEWS + 'mistral-small-4', '2026-03-16', '2026-03-16', weights='2026-03-16', params=P_S4, price=(0.15, 0.6), efforts=NH),
	V('Leanstral', ['Leanstral 26.03', 'Leanstral 1.0', 'Leanstral-120B-A6B', 'labs-leanstral-2603', *hf_names('Leanstral-2603')], NEWS + 'leanstral', '2026-03-16', '2026-03-16', weights='2026-03-16', params=P_S4, price=(0, 0), efforts=NH),
	V('Leanstral 1.5', ['Leanstral-1.5', 'Leanstral 1.5 119B', 'labs-leanstral-1-5', *hf_names('Leanstral-1.5-119B-A6B')], NEWS + 'leanstral-1-5', '2026-06-30', '2026-06-30', weights='2026-07-02', params=P_S4, price=(0, 0), efforts=NH),
)

# Mistral Medium 3.5 (128B dense, open weights, base model not disclosed)
S('Mistral Medium 3.5', ['Mistral Medium 3.5 128B', 'Mistral Medium 26.04', 'mistral-medium-3-5', *hf_names('Mistral-Medium-3.5-128B')], NEWS + 'vibe-remote-agents-mistral-medium-3-5', '2026-04-28', '2026-04-28', weights='2026-04-28', params=(128 * B, 128 * B), price=(1.5, 7.5), efforts=NH)

# Mistral Large 4 ("Le Chonk"; the open-weights release is the preview with three more weeks of RL)
P_L4 = (1_050 * B, 52 * B)
M('Mistral Large 4',
	V('Mistral Large 4 Preview', ['Mistral Large 4 (Preview)', 'ML4 Preview', 'Le Chonk Preview'], NEWS + 'mistral-large-4', '2026-10-06', '2026-10-06', params=P_L4, price=(0.7, 2.1), efforts=NH),
	V('Mistral Large 4', ['ML4', 'Le Chonk', 'Mistral Large 4 (Le Chonk)', 'Mistral Large 4 (High)', 'Mistral Large 4 (Medium)', 'Mistral Large 4 (Low)', 'Mistral Large 26.10', 'mistral-large-4', *hf_names('Mistral-Large-4-1T-A52B', 'Mistral-Large-4-1T-A52B-NVFP4')], NEWS + 'mistral-large-4-oss', '2026-10-06', '2026-10-29', weights='2026-10-29', params=P_L4, price=(0.7, 2.1), efforts=LMH),
)

write('mistral')
