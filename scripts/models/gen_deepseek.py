"""Generates data/models/deepseek/*.json (one file per model).

All the data lives in this script: edit it and run `python scripts/models/gen_deepseek.py`.
Existing JSON files are deleted and rewritten, so manual edits to them are lost.

Each open-weights size is a model. Little-known variants post-trained from the same weights are its versions (Coder v1.5,
Math, VL, Prover...), while DeepSeek-Coder-V2 and DeepSeek-R1 are well-known product lines that are separate models even
when built on the same base. Each new version number is also a separate model (V2 / V2.5, V3 / V3.1 / V3.2, V4 / V4.1),
and dated updates (0628, 0324, 0528, 0731, 0813, 1017...) are versions of their model.
Prices are DeepSeek's API list prices in USD (cache miss, no promotion, peak rate since the Aug 2026 peak / off-peak
pricing), at the time each version was served.
"""

from common import B, M, N, S, V, write

M_ = 1_000_000
T = ['', 'Thinking']							# deepseek-chat / deepseek-reasoner (or the thinking switch) of the hybrid V3.x models
NHM = ['None', 'High', 'Max']					# V4 preview: non-thinking, Think High, Think Max
NLHM = ['None', 'Low', 'High', 'Max']			# reasoning_effort since the V4 GA (Aug 2026)

ARXIV = 'https://arxiv.org/abs/'
NEWS = 'https://api-docs.deepseek.com/news/'
UPDATES = 'https://api-docs.deepseek.com/updates'
GH = 'https://github.com/deepseek-ai/'


def names(*items):
	"""Flattens names and lists of names, without duplicates."""
	out = []
	for i in items:
		out += [i] if isinstance(i, str) else list(i)
	return list(dict.fromkeys(out))


def hf(*repos):
	"""Repo ids, both bare and with the deepseek-ai organisation."""
	return names(repos, [f'deepseek-ai/{r}' for r in repos])


def aa(name, *modes):
	"""Artificial Analysis labels: plain and with each mode suffix."""
	return [name, *[f'{name} ({m})' for m in modes]]


# DeepSeek Coder (trained from scratch, separate from DeepSeek LLM)
CODER = GH + 'DeepSeek-Coder'
for size, p in [('1.3B', 1_346_471_936), ('6.7B', 6_740_512_768), ('33B', 33_342_991_360)]:
	s = size.lower()
	S(f'DeepSeek Coder {size}', names(f'DeepSeek-Coder-{size}', f'DeepSeek Coder {size} Instruct', f'deepseek-coder-{s}', hf(f'deepseek-coder-{s}-base', f'deepseek-coder-{s}-instruct')),
		CODER, '2023-11-02', '2023-11-02', weights='2023-11-02', params=(p, p))
S('DeepSeek Coder 5.7B MQA', names('DeepSeek-Coder-5.7B-MQA', hf('deepseek-coder-5.7bmqa-base')), CODER, '2023-11-02', '2023-11-02', weights='2023-11-02', params=(5_700 * M_, 5_700 * M_))

# DeepSeek LLM (Coder v1.5 is DeepSeek LLM 7B with continued pre-training on code, DeepSeekMath and the 7B Provers are built on
# top of it, DeepSeek-VL 7B on DeepSeek LLM 7B Base)
LLM = GH + 'DeepSeek-LLM'
P7 = (6_910 * M_, 6_910 * M_)
M('DeepSeek LLM 7B',
	V('DeepSeek LLM 7B', names('DeepSeek-LLM-7B', 'DeepSeek LLM 7B Chat', 'DeepSeek LLM 7B Base', 'deepseek-llm-7b', hf('deepseek-llm-7b-base', 'deepseek-llm-7b-chat')), LLM, '2023-11-29', '2023-11-29', weights='2023-11-29', params=P7),
	V('DeepSeek Coder 7B v1.5', names('DeepSeek-Coder-v1.5 7B', 'DeepSeek Coder v1.5', 'DeepSeek-Coder-7B-v1.5', 'deepseek-coder-7b-v1.5', hf('deepseek-coder-7b-base-v1.5', 'deepseek-coder-7b-instruct-v1.5')), CODER, '2024-01-25', '2024-01-25', weights='2024-01-25', params=P7),
	V('DeepSeekMath 7B', names('DeepSeekMath', 'DeepSeek-Math-7B', 'DeepSeekMath 7B RL', 'DeepSeekMath 7B Instruct', 'DeepSeekMath-RL 7B', hf('deepseek-math-7b-base', 'deepseek-math-7b-instruct', 'deepseek-math-7b-rl')), ARXIV + '2402.03300', '2024-02-05', '2024-02-05', weights='2024-02-05', params=P7),
	V('DeepSeek-VL 7B', names('DeepSeek-VL-7B', 'DeepSeek VL 7B', hf('deepseek-vl-7b-base', 'deepseek-vl-7b-chat')), ARXIV + '2403.05525', '2024-03-08', '2024-03-08', weights='2024-03-08', params=(7_344 * M_, 7_344 * M_)),
	V('DeepSeek-Prover-V1', names('DeepSeek-Prover', 'DeepSeek Prover V1', hf('DeepSeek-Prover-V1')), ARXIV + '2405.14333', '2024-05-23', '2024-08-16', weights='2024-08-16', params=P7),
	V('DeepSeek-Prover-V1.5', names('DeepSeek Prover V1.5', 'DeepSeek-Prover-V1.5-RL', 'DeepSeek-Prover-V1.5-SFT', hf('DeepSeek-Prover-V1.5-Base', 'DeepSeek-Prover-V1.5-SFT', 'DeepSeek-Prover-V1.5-RL')), ARXIV + '2408.08152', '2024-08-15', '2024-08-15', weights='2024-08-15', params=P7),
	V('DeepSeek-Prover-V2-7B', names('DeepSeek Prover V2 7B', hf('DeepSeek-Prover-V2-7B')), ARXIV + '2504.21801', '2025-04-30', '2025-04-30', weights='2025-04-30', params=P7),
)
S('DeepSeek LLM 67B', names('DeepSeek-LLM-67B', 'DeepSeek LLM 67B Chat', 'DeepSeek LLM 67B Chat (V1)', 'DeepSeek LLM 67B Base', 'deepseek-llm-67b', hf('deepseek-llm-67b-base', 'deepseek-llm-67b-chat')), LLM, '2023-11-29', '2023-11-29', weights='2023-11-29', params=(67 * B, 67 * B))
S('DeepSeekMoE 16B', names('DeepSeek-MoE-16B', 'DeepSeekMoE-16B', 'DeepSeekMoE 16B Chat', hf('deepseek-moe-16b-base', 'deepseek-moe-16b-chat')), ARXIV + '2401.06066', '2024-01-11', '2024-01-11', weights='2024-01-11', params=(16_376 * M_, 2_800 * M_))
S('DeepSeek-VL 1.3B', names('DeepSeek-VL-1.3B', 'DeepSeek VL 1.3B', hf('deepseek-vl-1.3b-base', 'deepseek-vl-1.3b-chat')), ARXIV + '2403.05525', '2024-03-08', '2024-03-08', weights='2024-03-08', params=(1_975 * M_, 1_975 * M_))

# DeepSeek-V2 (236B MoE; the 0628 chat is post-trained from the Coder-V2 base) and DeepSeek-V2-Lite (16B MoE)
V2 = GH + 'DeepSeek-V2'
P_V2 = (236 * B, 21 * B)
P_V2_LITE = (15_706 * M_, 2_400 * M_)
M('DeepSeek-V2',
	V('DeepSeek-V2 (May 2024)', names('DeepSeek-V2', 'DeepSeek V2', 'DeepSeek-V2-Chat', 'DeepSeek-V2 Chat', 'DeepSeek-V2-Chat (RL)', 'DeepSeek-V2 (MoE-236B, May 2024)', hf('DeepSeek-V2', 'DeepSeek-V2-Chat')), ARXIV + '2405.04434', '2024-05-06', '2024-05-06', weights='2024-05-06', params=P_V2, price=(0.14, 0.28)),
	V('DeepSeek-V2 (Jun 2024)', names('DeepSeek-V2-0628', 'DeepSeek-V2-Chat-0628', 'DeepSeek V2 Chat 0628', hf('DeepSeek-V2-Chat-0628')), UPDATES, '2024-06-28', '2024-06-28', weights='2024-07-18', params=P_V2, price=(0.14, 0.28)),
)
S('DeepSeek-V2-Lite', names('DeepSeek V2 Lite', 'DeepSeek-V2-Lite-Chat', hf('DeepSeek-V2-Lite', 'DeepSeek-V2-Lite-Chat')), V2, '2024-05-16', '2024-05-16', weights='2024-05-16', params=P_V2_LITE)

# DeepSeek-Coder-V2 (continued pre-training of an intermediate DeepSeek-V2 checkpoint, and of V2-Lite for the Lite model)
CODER_V2 = GH + 'DeepSeek-Coder-V2'
M('DeepSeek-Coder-V2',
	V('DeepSeek-Coder-V2 (Jun 2024)', names('DeepSeek-Coder-V2', 'DeepSeek Coder V2', 'DeepSeek-Coder-V2-0614', 'DeepSeek-Coder-V2-Instruct', 'DeepSeek-Coder-V2-Base', hf('DeepSeek-Coder-V2-Base', 'DeepSeek-Coder-V2-Instruct')), CODER_V2, '2024-06-14', '2024-06-14', weights='2024-06-17', params=P_V2, price=(0.14, 0.28)),
	V('DeepSeek-Coder-V2 (Jul 2024)', names('DeepSeek-Coder-V2-0724', 'DeepSeek-Coder-V2-Instruct-0724', hf('DeepSeek-Coder-V2-Instruct-0724')), UPDATES, '2024-07-24', '2024-07-24', weights='2024-09-05', params=P_V2, price=(0.14, 0.28)),
)
S('DeepSeek-Coder-V2-Lite', names('DeepSeek Coder V2 Lite', 'DeepSeek Coder V2 Lite Instruct', 'DeepSeek-Coder-V2-Lite-Instruct', 'DeepSeek-Coder-V2-Lite-Base', hf('DeepSeek-Coder-V2-Lite-Base', 'DeepSeek-Coder-V2-Lite-Instruct')), CODER_V2, '2024-06-17', '2024-06-17', weights='2024-06-17', params=P_V2_LITE)

# DeepSeek-V2.5 (merge of DeepSeek-V2-0628 and DeepSeek-Coder-V2-0724)
M('DeepSeek-V2.5',
	V('DeepSeek-V2.5 (Sep 2024)', names('DeepSeek-V2.5', 'DeepSeek V2.5', hf('DeepSeek-V2.5')), NEWS + 'news0905', '2024-09-05', '2024-09-05', weights='2024-09-05', params=P_V2, price=(0.14, 0.28)),
	V('DeepSeek-V2.5 (Dec 2024)', names("DeepSeek-V2.5 (Dec '24)", 'DeepSeek-V2.5-1210', 'DeepSeek V2.5 1210', hf('DeepSeek-V2.5-1210')), NEWS + 'news1210', '2024-12-10', '2024-12-10', weights='2024-12-10', params=P_V2, price=(0.14, 0.28)),
)

# First reasoning preview (chat only, base model not disclosed)
S('DeepSeek-R1-Lite-Preview', ['DeepSeek-R1-Lite', 'DeepSeek R1 Lite Preview', 'DeepSeek-R1-Lite (Preview)'], NEWS + 'news1120', '2024-11-20', '2024-11-20')

# DeepSeek-VL2 (MoE vision-language models built on DeepSeekMoE-3B / 16B / 27B)
VL2 = ARXIV + '2412.10302'
S('DeepSeek-VL2-Tiny', names('DeepSeek VL2 Tiny', hf('deepseek-vl2-tiny')), VL2, '2024-12-13', '2024-12-13', weights='2024-12-13', params=(3_370 * M_, 1_000 * M_))
S('DeepSeek-VL2-Small', names('DeepSeek VL2 Small', hf('deepseek-vl2-small')), VL2, '2024-12-13', '2024-12-13', weights='2024-12-13', params=(16_148 * M_, 2_800 * M_))
S('DeepSeek-VL2', names('DeepSeek VL2', hf('deepseek-vl2')), VL2, '2024-12-13', '2024-12-13', weights='2024-12-13', params=(27_480 * M_, 4_500 * M_))

# DeepSeek-V3 (671B MoE; DeepSeek-Prover-V2-671B is trained from DeepSeek-V3-Base)
P_V3 = (671 * B, 37 * B)
M('DeepSeek-V3',
	V('DeepSeek-V3 (Dec 2024)', names('DeepSeek-V3', 'DeepSeek V3', "DeepSeek V3 (Dec '24)", 'DeepSeek-V3-Base', hf('DeepSeek-V3', 'DeepSeek-V3-Base')), NEWS + 'news1226', '2024-12-26', '2024-12-26', weights='2024-12-26', params=P_V3, price=(0.27, 1.1)),
	V('DeepSeek-V3 (Mar 2025)', names('DeepSeek-V3-0324', 'DeepSeek V3 0324', 'DeepSeek-V3 0324', "DeepSeek V3 0324 (Mar '25)", 'chutes/DeepSeek-V3-0324', hf('DeepSeek-V3-0324')), NEWS + 'news250325', '2025-03-24', '2025-03-24', weights='2025-03-24', params=P_V3, price=(0.27, 1.1)),
	V('DeepSeek-Prover-V2-671B', names('DeepSeek-Prover-V2', 'DeepSeek Prover V2 671B', hf('DeepSeek-Prover-V2-671B')), ARXIV + '2504.21801', '2025-04-30', '2025-04-30', weights='2025-04-30', params=P_V3),
)

# DeepSeek-R1 (reasoning models trained from DeepSeek-V3-Base)
M('DeepSeek-R1',
	V('DeepSeek-R1-Zero', names('DeepSeek R1 Zero', hf('DeepSeek-R1-Zero')), NEWS + 'news250120', '2025-01-20', '2025-01-20', weights='2025-01-20', params=P_V3),
	V('DeepSeek-R1 (Jan 2025)', names('DeepSeek-R1', 'DeepSeek R1', "DeepSeek R1 (Jan '25)", 'DeepSeek-R1-0120', hf('DeepSeek-R1')), NEWS + 'news250120', '2025-01-20', '2025-01-20', weights='2025-01-20', params=P_V3, price=(0.55, 2.19)),
	V('DeepSeek-R1 (May 2025)', names('DeepSeek-R1-0528', 'DeepSeek R1 0528', "DeepSeek R1 0528 (May '25)", 'chutes/DeepSeek-R1-0528', hf('DeepSeek-R1-0528')), NEWS + 'news250528', '2025-05-28', '2025-05-28', weights='2025-05-28', params=P_V3, price=(0.55, 2.19)),
)

# DeepSeek-R1 distilled models (Qwen2.5 / Llama 3 / Qwen3 bases fine-tuned on DeepSeek-R1 outputs)
R1_PAPER = ARXIV + '2501.12948'
for base, size, p, price in [('Qwen', '1.5B', 1_777, (None, None)), ('Qwen', '7B', 7_616, (None, None)), ('Llama', '8B', 8_030, (None, None)),
		('Qwen', '14B', 14_770, (None, None)), ('Qwen', '32B', 32_764, (None, None)), ('Llama', '70B', 70_554, (0.7, 1.1))]:
	S(f'DeepSeek-R1-Distill-{base}-{size}', names(f'DeepSeek R1 Distill {base} {size}', f'deepseek-r1-distill-{base.lower()}-{size.lower()}', hf(f'DeepSeek-R1-Distill-{base}-{size}')),
		R1_PAPER, '2025-01-20', '2025-01-20', weights='2025-01-20', params=(p * M_, p * M_), price=price)
S('DeepSeek-R1-0528-Qwen3-8B', names('DeepSeek R1 0528 Qwen3 8B', 'deepseek-r1-0528-qwen3-8b', hf('DeepSeek-R1-0528-Qwen3-8B')), NEWS + 'news250528', '2025-05-29', '2025-05-29', weights='2025-05-29', params=(8_191 * M_, 8_191 * M_))

# DeepSeek-V3.1 (hybrid thinking, V3 base with 840B more tokens of long-context pre-training)
M('DeepSeek-V3.1',
	V('DeepSeek-V3.1', names(aa('DeepSeek V3.1', 'Reasoning', 'Non-reasoning'), 'DeepSeek-V3.1-Think', 'DeepSeek-V3.1_thinking', 'DeepSeek-V3.1-Base', hf('DeepSeek-V3.1', 'DeepSeek-V3.1-Base')), NEWS + 'news250821', '2025-08-19', '2025-08-19', weights='2025-08-21', params=P_V3, price=(0.56, 1.68), efforts=T),
	V('DeepSeek-V3.1-Terminus', names(aa('DeepSeek V3.1 Terminus', 'Reasoning', 'Non-reasoning'), hf('DeepSeek-V3.1-Terminus')), NEWS + 'news250922', '2025-09-22', '2025-09-22', weights='2025-09-22', params=P_V3, price=(0.56, 1.68), efforts=T),
)

# DeepSeek-V3.2 (V3.1-Terminus with DeepSeek Sparse Attention; DeepSeekMath-V2 is built on DeepSeek-V3.2-Exp-Base)
V32_PAPER = 'https://huggingface.co/deepseek-ai/DeepSeek-V3.2/resolve/main/assets/paper.pdf'
M('DeepSeek-V3.2',
	V('DeepSeek-V3.2-Exp', names(aa('DeepSeek V3.2 Exp', 'Reasoning', 'Non-reasoning', 'non-reasoning'), 'DeepSeek-V3.2-Exp_thinking', 'DeepSeek-V3.2-Exp-Base', hf('DeepSeek-V3.2-Exp', 'DeepSeek-V3.2-Exp-Base')), NEWS + 'news250929', '2025-09-29', '2025-09-29', weights='2025-09-29', params=P_V3, price=(0.28, 0.42), efforts=T),
	V('DeepSeekMath-V2', names('DeepSeek-Math-V2', 'DeepSeekMath V2', hf('DeepSeek-Math-V2')), ARXIV + '2511.22570', '2025-11-27', '2025-11-27', weights='2025-11-27', params=P_V3),
	V('DeepSeek-V3.2', names(aa('DeepSeek V3.2', 'Reasoning', 'Non-reasoning'), 'DeepSeek-V3.2 (Thinking)', 'DeepSeek-V3.2 (Non-thinking)', 'deepseek/deepseek-v3.2', 'fireworks/deepseek-v3p2', hf('DeepSeek-V3.2')), NEWS + 'news251201', '2025-12-01', '2025-12-01', weights='2025-12-01', params=P_V3, price=(0.28, 0.42), efforts=T),
	V('DeepSeek-V3.2-Speciale', names('DeepSeek V3.2 Speciale', hf('DeepSeek-V3.2-Speciale')), NEWS + 'news251201', '2025-12-01', '2025-12-01', weights='2025-12-01', params=P_V3, price=(0.28, 0.42)),
)

# DeepSeek-V4 (Apr 2026 preview, then re-post-trained GA versions; V4-Flash-Vision-Exp matches V4-Flash on text)
V4_PAPER = 'https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf'
P_V4_PRO = (1_600 * B, 49 * B)
P_V4_FLASH = (284 * B, 13 * B)
M('DeepSeek-V4-Pro',
	V('DeepSeek-V4-Pro (Apr 2026)', names('DeepSeek-V4-Pro', 'DeepSeek V4 Pro', 'DeepSeek-V4-Pro-Preview', 'DeepSeek-V4-Pro Preview', 'DeepSeek-V4-Pro-Max', aa('DeepSeek V4 Pro 0424', 'Max', 'High', 'Non-reasoning'), hf('DeepSeek-V4-Pro', 'DeepSeek-V4-Pro-Base')), NEWS + 'news260424', '2026-04-24', '2026-04-24', weights='2026-04-24', params=P_V4_PRO, price=(1.74, 3.48), efforts=NHM),
	V('DeepSeek-V4-Pro (Aug 2026)', names('DeepSeek-V4-Pro-0813', 'DeepSeek V4 Pro 0813', 'DeepSeek-V4-Pro GA', aa('DeepSeek V4 Pro 0813', 'Max', 'max', 'Non-reasoning', 'non-reasoning'), hf('DeepSeek-V4-Pro-0813')), NEWS + 'news260813', '2026-08-13', '2026-08-13', weights='2026-08-13', params=P_V4_PRO, price=(1.32, 3.96), efforts=NLHM),
)
M('DeepSeek-V4-Flash',
	V('DeepSeek-V4-Flash (Apr 2026)', names('DeepSeek-V4-Flash', 'DeepSeek V4 Flash', 'DeepSeek-V4-Flash-Preview', 'DeepSeek-V4-Flash Preview', 'DeepSeek-V4-Flash-Max', 'DeepSeek-V4-Flash-0423', aa('DeepSeek V4 Flash 0420', 'Max', 'High', 'Non-reasoning'), hf('DeepSeek-V4-Flash', 'DeepSeek-V4-Flash-Base')), NEWS + 'news260424', '2026-04-24', '2026-04-24', weights='2026-04-24', params=P_V4_FLASH, price=(0.14, 0.28), efforts=NHM),
	V('DeepSeek-V4-Flash (Jul 2026)', names('DeepSeek-V4-Flash-0731', 'DeepSeek V4 Flash 0731', aa('DeepSeek V4 Flash 0731', 'Max'), hf('DeepSeek-V4-Flash-0731')), UPDATES, '2026-07-31', '2026-07-31', weights='2026-07-31', params=P_V4_FLASH, price=(0.14, 0.28), efforts=NLHM),
	V('DeepSeek-V4-Flash-Vision-Exp', names('DeepSeek V4 Flash Vision', 'DeepSeek V4 Flash Vision (Max)', 'deepseek-v4-flash-vision-exp', hf('DeepSeek-V4-Flash-Vision-Exp')), NEWS + 'news260821', '2026-08-21', '2026-08-21', weights='2026-08-31', params=P_V4_FLASH, price=(0.44, 1.32), efforts=NLHM),
)

# DeepSeek-V4.1 (new causal encoder-decoder MoE architecture with native vision; 8B active parameters for the input, 16B for the output)
V41_FLASH = NEWS + 'news260910'
P_V41_FLASH = (552 * B, 16 * B)
M('DeepSeek-V4.1-Flash',
	V('DeepSeek-V4.1-Flash (Sep 2026)', names('DeepSeek-V4.1-Flash', 'DeepSeek V4.1 Flash', aa('DeepSeek V4.1 Flash', 'Max', 'Non-reasoning'), 'deepseek-v4.1-flash-20260910', hf('DeepSeek-V4.1-Flash')), V41_FLASH, '2026-09-10', '2026-09-10', weights='2026-09-10', params=P_V41_FLASH, price=(0.3, 1.2), efforts=NLHM),
	V('DeepSeek-V4.1-Flash (Oct 2026)', names('DeepSeek-V4.1-Flash-1017', 'DeepSeek V4.1 Flash 1017', aa('DeepSeek V4.1 Flash 1017', 'Max', 'max', 'High', 'high', 'Non-reasoning', 'non-reasoning'), hf('DeepSeek-V4.1-Flash-1017')), NEWS + 'news261017', '2026-10-17', '2026-10-17', weights='2026-10-17', params=P_V41_FLASH, price=(0.3, 1.2), efforts=NLHM),
)
S('DeepSeek-V4.1-Pro', ['DeepSeek V4.1 Pro'], V41_FLASH, '2026-09-10', None)

write('deepseek')
