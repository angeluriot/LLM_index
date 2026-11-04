"""Generates data/models/xiaomi/*.json (one file per model).

Covers Xiaomi's language models: the MiLM models of the XiaoAI assistant and the MiMo family of the Xiaomi LLM Core / MiMo team.
All the data lives in this script: edit it and run `python scripts/models/gen_xiaomi.py`.
Existing JSON files are deleted and rewritten, so manual edits to them are lost.

Base checkpoints are folded into the version of their post-trained model (names of both). MiMo-VL and MiMo-Embodied are
versions of MiMo-7B, the language model they are built on. Each new version number is a separate model, even when it reuses
the previous backbone (MiMo-V2.5 is built on MiMo-V2-Flash, V2.6-Flash on V2.5), and the dated updates of a model (V2-Flash
0112 / 0204, V2.6 MOPD checkpoints) are its versions. The UltraSpeed endpoints serve the same weights faster, so they are only
names of the version they serve. MiMo-Audio (speech) and the TTS / ASR models are out of scope.
Prices are Xiaomi MiMo API overseas USD list prices (cache miss, base context tier); for the V2.5 series, the permanent prices
since the 2026-05-27 price cut, which V2.6 kept.
"""

from common import B, M, N, S, V, write

M_ = 1_000_000
T = ['', 'Thinking']			# thinking.type switch (disabled / enabled)
P_7B = (7_833 * M_, 7_833 * M_)	# MiMo-7B (with its MTP layer)
P_VL = (8_306 * M_, 8_306 * M_)	# MiMo-VL-7B (MiMo-7B + ViT)
P_FLASH = (309 * B, 15 * B)		# MiMo-V2-Flash architecture
P_V25 = (310 * B, 15 * B)		# MiMo-V2.5 / V2.6-Flash (V2-Flash backbone + omni encoders)
P_PRO = (1_020 * B, 42 * B)		# MiMo-V2-Pro / V2.5-Pro / V2.6-Pro architecture

ARXIV = 'https://arxiv.org/abs/'
SITE = 'https://mimo.xiaomi.com/'
NEWS = 'https://mimo.mi.com/docs/en-US/news/'
HF = 'https://huggingface.co/XiaomiMiMo/'


def names(*items):
	"""Flattens names and lists of names, without duplicates."""
	out = []
	for i in items:
		out += [i] if isinstance(i, str) else list(i)
	return list(dict.fromkeys(out))


def hf(*repos):
	"""Xiaomi Hugging Face repo ids, both bare and with the XiaomiMiMo organisation."""
	return list(repos) + ['XiaomiMiMo/' + r for r in repos]


# MiLM (XiaoAI assistant models, never released)
S('MiLM-6B', ['MiLM 6B', 'Xiaomi MiLM-6B', 'MiLM'], 'https://github.com/XiaoMi/MiLM-6B', '2023-08-09', None, params=(6_400 * M_, 6_400 * M_))
S('MiLM-1.3B', ['MiLM 1.3B', 'Xiaomi MiLM-1.3B'], 'https://github.com/XiaoMi/MiLM-6B', '2023-08-11', None, params=(1_300 * M_, 1_300 * M_))

# MiMo-7B (reasoning model, and the VL / embodied models built on it)
MIMO7B = ARXIV + '2505.07608'
MIMO_VL = ARXIV + '2506.03569'
M('MiMo-7B',
	V('MiMo-7B-Base', names('MiMo 7B Base', hf('MiMo-7B-Base')), MIMO7B, '2025-04-30', '2025-04-30', weights='2025-04-30', params=P_7B),
	V('MiMo-7B-SFT', names('MiMo 7B SFT', hf('MiMo-7B-SFT')), MIMO7B, '2025-04-30', '2025-04-30', weights='2025-04-30', params=P_7B),
	V('MiMo-7B-RL-Zero', names('MiMo 7B RL Zero', 'MiMo-7B-RL Zero', hf('MiMo-7B-RL-Zero')), MIMO7B, '2025-04-30', '2025-04-30', weights='2025-04-30', params=P_7B),
	V('MiMo-7B-RL', names('MiMo-7B', 'MiMo 7B', 'MiMo 7B RL', 'Xiaomi MiMo-7B', hf('MiMo-7B-RL')), MIMO7B, '2025-04-30', '2025-04-30', weights='2025-04-30', params=P_7B),
	V('MiMo-7B-RL-0530', names('MiMo 7B RL 0530', 'MiMo-7B-RL (May 2025)', hf('MiMo-7B-RL-0530')), HF + 'MiMo-7B-RL-0530', '2025-05-30', '2025-05-30', weights='2025-05-30', params=P_7B),
	V('MiMo-VL-7B-SFT', names('MiMo VL 7B SFT', hf('MiMo-VL-7B-SFT')), MIMO_VL, '2025-05-30', '2025-05-30', weights='2025-05-30', params=P_VL),
	V('MiMo-VL-7B-RL', names('MiMo-VL-7B', 'MiMo VL 7B', 'MiMo VL 7B RL', hf('MiMo-VL-7B-RL')), MIMO_VL, '2025-05-30', '2025-05-30', weights='2025-05-30', params=P_VL),
	V('MiMo-VL-7B-SFT-2508', names('MiMo VL 7B SFT 2508', hf('MiMo-VL-7B-SFT-2508')), HF + 'MiMo-VL-7B-SFT-2508', '2025-08-07', '2025-08-07', weights='2025-08-07', params=P_VL, efforts=T),
	V('MiMo-VL-7B-RL-2508', names('MiMo VL 7B RL 2508', hf('MiMo-VL-7B-RL-2508')), HF + 'MiMo-VL-7B-RL-2508', '2025-08-07', '2025-08-07', weights='2025-08-07', params=P_VL, efforts=T),
	V('MiMo-Embodied-7B', names('MiMo-Embodied', 'MiMo Embodied 7B', hf('MiMo-Embodied-7B')), ARXIV + '2511.16518', '2025-11-19', '2025-11-19', weights='2025-11-19', params=P_VL),
)

# MiMo-V2 (Dec 2025 - Mar 2026)
M('MiMo-V2-Flash',
	V('MiMo-V2-Flash (Dec 2025)', names('MiMo-V2-Flash', 'MiMo V2 Flash', 'mimo-v2-flash', 'MiMo-V2-Flash (Reasoning)', 'MiMo-V2-Flash (Non-reasoning)', 'MiMo-V2-Flash (non-reasoning)', 'xiaomi/mimo-v2-flash', hf('MiMo-V2-Flash', 'MiMo-V2-Flash-Base')),
		NEWS + 'previous-news/news20251216', '2025-12-16', '2025-12-16', weights='2025-12-16', params=P_FLASH, price=(0.1, 0.3), efforts=T),
	V('MiMo-V2-Flash (Jan 2026)', ['MiMo-V2-Flash (January 2026)', 'mimo-v2-flash-0112'], 'https://mimo.mi.com/docs/en-US/updates/model', '2026-01-12', '2026-01-12', params=P_FLASH, price=(0.1, 0.3), efforts=T),
	V('MiMo-V2-Flash (Feb 2026)', ['MiMo-V2-Flash (February 2026)', 'mimo-v2-flash-0204', 'mimo-v2-0206'], 'https://mimo.mi.com/docs/en-US/updates/model', '2026-02-04', '2026-02-04', params=P_FLASH, price=(0.1, 0.3), efforts=T),
)
S('MiMo-V2-Pro', ['MiMo V2 Pro', 'mimo-v2-pro', 'MiMo v2 Pro (High)', 'xiaomi/mimo-v2-pro', 'Hunter Alpha'], SITE + 'mimo-v2-pro', '2026-03-18', '2026-03-18', params=P_PRO, price=(1, 3), efforts=T)
S('MiMo-V2-Omni', ['MiMo V2 Omni', 'mimo-v2-omni', 'xiaomi/mimo-v2-omni', 'Healer Alpha'], SITE + 'mimo-v2-omni', '2026-03-18', '2026-03-18', price=(0.4, 2), efforts=T)

# MiMo-V2.5 (Apr 2026, open weights)
S('MiMo-V2.5', names('MiMo V2.5', 'mimo-v2.5', 'mimo-v2-5', 'mimo-v2-5-0424', 'xiaomi/mimo-v2.5', hf('MiMo-V2.5', 'MiMo-V2.5-Base')),
	HF + 'MiMo-V2.5', '2026-04-22', '2026-04-23', weights='2026-04-27', params=P_V25, price=(0.14, 0.28), efforts=T)
S('MiMo-V2.5-Pro', names('MiMo V2.5 Pro', 'MiMo v2.5 Pro', 'mimo-v2.5-pro', 'mimo-v2-5-pro', 'xiaomi/mimo-v2.5-pro', 'MiMo-V2.5-Pro-UltraSpeed', hf('MiMo-V2.5-Pro', 'MiMo-V2.5-Pro-Base')),
	SITE + 'mimo-v2-5-pro/', '2026-04-22', '2026-04-23', weights='2026-04-27', params=P_PRO, price=(0.435, 0.87), efforts=T)

# MiMo-V2.6 (Sep 2026, open weights, natively omnimodal; the MOPD checkpoints fix tool-call repetition)
V26 = SITE + 'mimo-v2-6'
V26_NEWS = NEWS + 'latest/v2-6'
M('MiMo-V2.6-Pro',
	V('MiMo-V2.6-Pro', names('MiMo V2.6 Pro', 'mimo-v2.6-pro', 'mimo-v2-6-pro', 'xiaomi/mimo-v2.6-pro', 'MiMo-V2.6-Pro-UltraSpeed', 'mimo-v2.6-pro-ultraspeed', 'xiaomi/mimo-v2.6-pro-ultraspeed', hf('MiMo-V2.6-Pro-RL')),
		V26, '2026-09-21', '2026-09-21', weights='2026-09-21', params=P_PRO, price=(0.435, 0.87), efforts=T),
	V('MiMo-V2.6-Pro-MOPD', names('MiMo V2.6 Pro MOPD', hf('MiMo-V2.6-Pro-MOPD')), HF + 'MiMo-V2.6-Pro-MOPD', '2026-09-27', '2026-09-27', weights='2026-09-27', params=P_PRO, price=(0.435, 0.87), efforts=T),
)
M('MiMo-V2.6-Flash',
	V('MiMo-V2.6-Flash', names('MiMo V2.6 Flash', 'mimo-v2.6-flash', 'mimo-v2-6-flash', 'xiaomi/mimo-v2.6-flash', hf('MiMo-V2.6-Flash-RL')),
		V26, '2026-09-21', '2026-09-21', weights='2026-09-21', params=P_V25, price=(0.14, 0.28), efforts=T),
	V('MiMo-V2.6-Flash-MOPD', names('MiMo V2.6 Flash MOPD', hf('MiMo-V2.6-Flash-MOPD')), HF + 'MiMo-V2.6-Flash-MOPD', '2026-09-27', '2026-09-27', weights='2026-09-27', params=P_V25, price=(0.14, 0.28), efforts=T),
)
S('MiMo-V2.6-Distill-Qwen-9B', names('MiMo V2.6 Distill Qwen 9B', hf('MiMo-V2.6-Distill-Qwen-9B')), HF + 'MiMo-V2.6-Distill-Qwen-9B', '2026-09-21', '2026-09-21', weights='2026-09-21', params=(9_410 * M_, 9_410 * M_), efforts=T)

write('xiaomi')
