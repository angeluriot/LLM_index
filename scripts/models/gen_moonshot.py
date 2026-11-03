"""Generates data/models/moonshot/*.json (one file per model).

All the data lives in this script: edit it and run `python scripts/models/gen_moonshot.py`.
Existing JSON files are deleted and rewritten, so manual edits to them are lost.

Prices are Moonshot's USD list prices on the global Kimi API platform (cache miss, base context tier for the tiered
moonshot-v1 / kimi-latest models). The Kimi-VL models are versions of Moonlight-16B-A3B, the language model they are
built on, and K2 Thinking is a version of Kimi K2. Each new version number is a separate model, even when post-trained
from the previous one (K2.7 Code is built on K2.6, K3.1 Pro on K3). High-speed / turbo endpoints serve the same weights
faster, so they are only names of the version they serve.
"""

from common import B, M, N, S, V, write

M_ = 1_000_000
T = ['', 'Thinking']					# thinking switch of K2.5 / K2.6
LHM = ['Low', 'High', 'Max']			# reasoning_effort of the K3 series
P_K2 = (1_040 * B, 32_600 * M_)			# Kimi K2 architecture (K2, K2.5, K2.6, K2.7 Code)
P_K3 = (2_800 * B, 104 * B)				# Kimi K3 architecture (K3, K3.1 Pro)

ARXIV = 'https://arxiv.org/abs/'
BLOG = 'https://www.kimi.com/blog/'
API_BLOG = 'https://platform.kimi.com/blog/posts/'
DOCS = 'https://platform.kimi.ai/docs/'


def names(*items):
	"""Flattens names and lists of names, without duplicates."""
	out = []
	for i in items:
		out += [i] if isinstance(i, str) else list(i)
	return list(dict.fromkeys(out))


def hf(*repos):
	"""Repo ids, both bare and with the moonshotai organisation."""
	return names(repos, [f'moonshotai/{r}' for r in repos])


def aa(name, *modes):
	"""Artificial Analysis labels: plain and with each mode suffix."""
	return [name, *[f'{name} ({m})' for m in modes]]


# Moonshot-v1 (the model behind Kimi Chat, served as moonshot-v1-8k / 32k / 128k, plus a vision preview)
M('Moonshot-v1',
	V('Moonshot-v1', ['moonshot-v1', 'Moonshot v1', 'moonshot-v1-8k', 'moonshot-v1-32k', 'moonshot-v1-128k', 'moonshot-v1-auto', 'Kimi Chat'], DOCS + 'models', '2023-10-09', '2023-11-16', price=(0.2, 2)),
	V('Moonshot-v1 Vision Preview', ['moonshot-v1-vision-preview', 'Moonshot-V1-Vision-Preview', 'moonshot-v1-8k-vision-preview', 'moonshot-v1-32k-vision-preview', 'moonshot-v1-128k-vision-preview'], DOCS + 'models', '2025-01-15', '2025-01-15', price=(0.2, 2)),
)

# k-series reasoning models (chat app only until kimi-thinking-preview; bases not disclosed)
S('k0-math', ['Kimi k0-math', 'Kimi k0 math', 'k0 math'], 'https://www.globaltimes.cn/page/202411/1323248.shtml', '2024-11-16', '2024-11-16')
S('Kimi k1', ['k1', 'Kimi K1', 'Kimi k1 visual thinking', 'k1 visual thinking model'], 'https://en.tmtpost.com/news/7383172', '2024-12-16', '2024-12-16')
K15 = ARXIV + '2501.12599'
M('Kimi k1.5',
	V('Kimi k1.5', ['Kimi-k1.5', 'Kimi K1.5', 'kimi-k1.5', 'k1.5', 'Kimi k1.5 (long-CoT)', 'Kimi k1.5 long-CoT', 'Kimi k1.5 Long-CoT'], K15, '2025-01-20', '2025-01-20'),
	V('Kimi k1.5 (short-CoT)', ['Kimi k1.5 short-CoT', 'Kimi k1.5 Short-CoT', 'Kimi-k1.5-short-CoT'], K15, '2025-01-20', '2025-01-20', id='kimi-k1-5-short-cot'),
)
S('Kimi k1.6', ['Kimi 1.6', 'k1.6', 'Kimi-k1.6', 'Kimi-k1.6-IOI', 'Kimi-k1.6-IOI-high', 'Kimi k1.6 IOI', 'Kimi k1.6 IOI high'], 'https://livecodebench.github.io/leaderboard.html', '2025-02-27', None)
S('kimi-latest', ['Kimi Latest', 'kimi-latest-8k', 'kimi-latest-32k', 'kimi-latest-128k'], API_BLOG + 'kimi-latest', '2025-02-17', '2025-02-17', price=(0.2, 2))
S('kimi-thinking-preview', ['Kimi Thinking Preview', 'Kimi-Thinking-Preview'], API_BLOG + 'kimi-thinking', '2025-05-06', '2025-05-06', price=(30, 30))

# Moonlight (16B MoE trained with Muon) and the Kimi-VL vision-language models built on it
P_MOON = (16 * B, 3 * B)
KIMI_VL = ARXIV + '2504.07491'
M('Moonlight-16B-A3B',
	V('Moonlight-16B-A3B', names('Moonlight', 'Moonlight 16B A3B', 'Moonlight-16B-A3B-Instruct', 'Moonlight 16B A3B Instruct', hf('Moonlight-16B-A3B', 'Moonlight-16B-A3B-Instruct')), ARXIV + '2502.16982', '2025-02-22', '2025-02-22', weights='2025-02-22', params=P_MOON),
	V('Kimi-VL-A3B-Instruct', names('Kimi-VL', 'Kimi-VL-A3B', 'Kimi VL A3B', 'Kimi VL A3B Instruct', 'kimi-vl-a3b', hf('Kimi-VL-A3B-Instruct')), KIMI_VL, '2025-04-10', '2025-04-10', weights='2025-04-10', params=P_MOON),
	V('Kimi-VL-A3B-Thinking', names('Kimi VL A3B Thinking', hf('Kimi-VL-A3B-Thinking')), KIMI_VL, '2025-04-10', '2025-04-10', weights='2025-04-10', params=P_MOON),
	V('Kimi-VL-A3B-Thinking-2506', names('Kimi VL A3B Thinking 2506', 'Kimi-VL-A3B-Thinking (Jun 2025)', hf('Kimi-VL-A3B-Thinking-2506')), 'https://huggingface.co/moonshotai/Kimi-VL-A3B-Thinking-2506', '2025-06-21', '2025-06-21', weights='2025-06-21', params=P_MOON),
)

# Kimi-Dev-72B (coding model RL-trained from Qwen2.5-72B)
S('Kimi-Dev-72B', names('Kimi Dev 72B', 'Kimi Dev 72b', 'Kimi-Dev', hf('Kimi-Dev-72B')), 'https://moonshotai.github.io/Kimi-Dev/', '2025-06-16', '2025-06-16', weights='2025-06-16', params=(72_700 * M_, 72_700 * M_))

# Kimi K2 (1T MoE; K2 Thinking is post-trained from the same base; turbo endpoints serve the same weights faster)
M('Kimi K2',
	V('Kimi K2 (Jul 2025)', names('Kimi K2', 'Kimi-K2', 'Kimi K2 Instruct', 'Kimi-K2-Instruct', 'kimi-k2-instruct', 'Kimi K2 0711', 'kimi-k2-0711-preview', 'moonshotai/kimi-k2', 'Kimi K2 Base', 'Kimi-K2-Base', 'kimi-k2-base', hf('Kimi-K2-Instruct', 'Kimi-K2-Base')), 'https://moonshotai.github.io/Kimi-K2/', '2025-07-11', '2025-07-11', weights='2025-07-11', params=P_K2, price=(0.6, 2.5)),
	V('Kimi K2 (Sep 2025)', names('Kimi K2 0905', 'Kimi K2-Instruct-0905', 'Kimi K2 Instruct 0905', 'kimi-k2-instruct-0905', 'kimi-k2-0905', 'kimi-k2-0905-preview', 'moonshotai/kimi-k2-0905', 'fireworks/Kimi-K2-Instruct-0905', 'kimi-k2-turbo-preview', 'Kimi K2 Turbo', hf('Kimi-K2-Instruct-0905')), API_BLOG + 'kimi-k2-0905', '2025-09-05', '2025-09-05', weights='2025-09-05', params=P_K2, price=(0.6, 2.5)),
	V('Kimi K2 Thinking', names('Kimi-K2-Thinking', 'kimi-k2-thinking', 'moonshotai/kimi-k2-thinking', 'Kimi K2-Thinking-0905', 'kimi-k2-thinking-0905', 'kimi-k2-thinking-turbo', 'Kimi K2 Thinking Turbo', hf('Kimi-K2-Thinking')), 'https://moonshotai.github.io/Kimi-K2/thinking.html', '2025-11-06', '2025-11-06', weights='2025-11-06', params=P_K2, price=(0.6, 2.5)),
)

# Kimi Linear (48B MoE with Kimi Delta Attention)
S('Kimi Linear 48B A3B', names('Kimi Linear', 'Kimi-Linear', 'Kimi-Linear-48B-A3B', 'Kimi Linear 48B A3B Instruct', 'Kimi-Linear-48B-A3B-Instruct', 'Kimi-Linear-48B-A3B-Base', hf('Kimi-Linear-48B-A3B-Instruct', 'Kimi-Linear-48B-A3B-Base')), ARXIV + '2510.26692', '2025-10-30', '2025-10-30', weights='2025-10-30', params=(48 * B, 3 * B))

# Kimi K2.x (native multimodal, continued pre-training of a near-end Kimi K2 checkpoint)
S('Kimi K2.5', names(aa('Kimi K2.5', 'Reasoning', 'Non-reasoning', 'non-reasoning'), 'Kimi-K2.5', 'kimi-k2.5', 'kimi-k2-5', 'moonshotai/kimi-k2.5', 'fireworks/kimi-k2p5', 'Kimi K2.5 Thinking', 'Kimi K2.5 Instant', hf('Kimi-K2.5')),
	BLOG + 'kimi-k2-5.html', '2026-01-27', '2026-01-27', weights='2026-01-27', params=P_K2, price=(0.6, 3), efforts=T)
S('Kimi K2.6', names(aa('Kimi K2.6', 'Reasoning', 'Non-reasoning', 'non-reasoning'), 'Kimi-K2.6', 'kimi-k2.6', 'kimi-k2-6', 'moonshotai/kimi-k2.6', 'kimi/kimi-k2.6', 'Kimi K2.6 Thinking', 'Kimi K2.6 Instant', hf('Kimi-K2.6')),
	BLOG + 'kimi-k2-6.html', '2026-04-20', '2026-04-20', weights='2026-04-20', params=P_K2, price=(0.95, 4), efforts=T)
S('Kimi K2.7 Code', names('Kimi K2.7', 'Kimi-K2.7-Code', 'kimi-k2.7-code', 'kimi-k2-7-code', 'moonshotai/kimi-k2.7-code', 'kimi-k2.7-code-highspeed', 'Kimi K2.7 Code HighSpeed', 'Kimi K2.7 Code Highspeed', hf('Kimi-K2.7-Code')),
	DOCS + 'guide/kimi-k2-7-code-quickstart', '2026-06-12', '2026-06-12', weights='2026-06-12', params=P_K2, price=(0.95, 4))

# Kimi K3 (2.8T MoE with Kimi Delta Attention and Attention Residuals; K3.1 Pro is a post-trained version of K3)
S('Kimi K3', names(aa('Kimi K3', 'Max', 'max', 'High', 'high', 'Low', 'low'), 'Kimi-K3', 'kimi-k3', 'kimi-k3-max', 'kimi-k3-low', 'moonshotai/kimi-k3', 'kimi/kimi-k3', hf('Kimi-K3')),
	BLOG + 'kimi-k3', '2026-07-16', '2026-07-16', weights='2026-07-27', params=P_K3, price=(3, 15), efforts=LHM)

# Kimi K2.8 Preview (Kimi Code only, served as kimi-for-coding; architecture not disclosed)
S('Kimi K2.8 Preview', ['Kimi K2.8', 'Kimi-K2.8-Preview', 'kimi-k2.8', 'kimi-k2.8-preview', 'kimi-k2-8-preview', 'moonshotai/kimi-k2.8-preview'],
	'https://www.kimi.com/code/docs/en/kimi-code/whats-new.html', '2026-09-11', '2026-09-11', efforts=['', 'Low', 'High', 'Max'])

# Kimi K3.1 Pro (weights announced for Nov 4, 2026)
S('Kimi K3.1 Pro', names(aa('Kimi K3.1 Pro', 'Max', 'max'), 'Kimi-K3.1-Pro', 'kimi-k3.1-pro', 'kimi-k3-1-pro', 'kimi-k3.1-pro-max', 'moonshotai/kimi-k3.1-pro', 'kimi/kimi-k3.1-pro', 'kimi-k3.1-pro-highspeed', 'Kimi K3.1 Pro Highspeed', 'Kimi K3.1 Pro HighSpeed', hf('Kimi-K3.1-Pro')),
	BLOG + 'kimi-k3-1-pro', '2026-10-23', '2026-10-23', params=P_K3, price=(2, 10), efforts=LHM)

write('moonshot')
