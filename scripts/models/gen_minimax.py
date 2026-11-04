"""Generates data/models/minimax/*.json (one file per model).

All the data lives in this script: edit it and run `python scripts/models/gen_minimax.py`.
Existing JSON files are deleted and rewritten, so manual edits to them are lost.

Each abab generation and size tier is a model (abab6.5 is the ~1T MoE, abab6.5s its faster smaller sibling), and the
dated updates of a rolling alias are its versions. MiniMax-VL-01 is a version of MiniMax-Text-01 (the same LLM with a
vision encoder), and the 40k / 80k thinking budgets are versions of MiniMax-M1. Each new M-series version number is a
separate model, even on the same architecture (M2 to M2.7 are all 230B-A10B, M3 and M3.1 Flash are both 428B-A23B).
The highspeed / Lightning endpoints serve the same weights faster, so they are only names of the version they serve.
Prices are MiniMax's USD list prices (base context tier, after the permanent 50% discount on M3). The abab models only
sold on the Chinese platform have their CNY price converted at 7.2.
"""

from common import B, M, N, S, V, write

M_ = 1_000_000
LMHXM = ['Low', 'Medium', 'High', 'Xhigh', 'Max']	# M3.1 Flash effort (thinking cannot be disabled)
TA = ['', 'Adaptive', 'Thinking']					# M3 thinking: disabled / adaptive / enabled
P_01 = (456 * B, 45_900 * M_)						# MiniMax-01 architecture (Text-01, M1)
P_M2 = (230 * B, 10 * B)							# M2 architecture (M2 to M2.7)
P_M3 = (428 * B, 23 * B)							# M3 architecture (M3, M3.1 Flash)

IO = 'https://www.minimax.io/'
CN_NEWS = 'https://www.minimaxi.com/news/'
DOCS = 'https://platform.minimax.io/docs/'
ARXIV = 'https://arxiv.org/abs/'


def cny(x):
	"""A CNY price per 1M tokens converted to USD."""
	return round(x / 7.2, 4)


def names(*items):
	"""Flattens names and lists of names, without duplicates."""
	out = []
	for i in items:
		out += [i] if isinstance(i, str) else list(i)
	return list(dict.fromkeys(out))


def hf(*repos):
	"""Repo ids with the MiniMaxAI organisation."""
	return [f'MiniMaxAI/{r}' for r in repos]


# abab series (closed models of the Chinese open platform)
M('abab5.5',
	V('abab5.5 (Jul 2023)', ['abab5.5 (2023)'], 'https://www.volcengine.com/docs/82379/1113739', '2023-07-18', '2023-07-18', price=(cny(15), cny(15))),
	V('abab5.5 (Jan 2024)', ['abab5.5-chat-240123'], CN_NEWS + 'abab5.5-chat-240123', '2024-01-23', '2024-01-23', price=(cny(15), cny(15))),
)
S('abab5.5s', ['abab5.5s-chat', 'abab5.5s-chat-240119', 'abab 5.5s'], CN_NEWS + 'abab5.5s-chat-240119', '2024-01-19', '2024-01-19', price=(cny(5), cny(5)))
S('abab6', ['abab6-chat', 'abab 6'], 'https://news.qq.com/rain/a/20240116A04EVV00', '2024-01-16', '2024-01-16', price=(cny(100), cny(100)))
ABAB65 = IO + 'news/abab65-series'
S('abab6.5', ['abab6.5-chat', 'abab 6.5'], ABAB65, '2024-04-17', '2024-04-17', params=(1_000 * B, None), price=(cny(30), cny(30)))
S('abab6.5s', ['abab6.5s-chat', 'abab 6.5s'], ABAB65, '2024-04-17', '2024-04-17', price=(cny(10), cny(10)))
S('abab7-preview', ['abab7', 'abab7 preview', 'abab 7 preview', 'abab7-chat-preview'], IO + 'news/abab7-preview-release', '2024-11-06', '2024-11-06')

# MiniMax-01 (456B MoE with Lightning Attention; VL-01 adds a vision encoder to Text-01)
MM01 = IO + 'news/minimax-01-series-2'
M('MiniMax-Text-01',
	V('MiniMax-Text-01', names('MiniMax-01', 'MiniMax Text 01', 'MiniMax Text-01', 'minimax-text-01', 'minimax/minimax-01', hf('MiniMax-Text-01', 'MiniMax-Text-01-hf')),
		MM01, '2025-01-15', '2025-01-15', weights='2025-01-15', params=P_01, price=(0.2, 1.1)),
	V('MiniMax-VL-01', names('MiniMax VL 01', 'MiniMax VL-01', 'minimax-vl-01', hf('MiniMax-VL-01')),
		MM01, '2025-01-15', '2025-01-15', weights='2025-01-15', params=(456_400 * M_, None), price=(0.2, 1.1)),
)

# SynLogic (logical reasoning RL research models on Qwen2.5 bases)
SYNLOGIC = ARXIV + '2505.19641'
M('SynLogic-32B',
	V('SynLogic-32B', names('SynLogic 32B', 'synlogic-32b', hf('SynLogic-32B')), SYNLOGIC, '2025-05-26', '2025-05-30', weights='2025-05-30', params=(32_800 * M_, 32_800 * M_)),
	V('SynLogic-Mix-3-32B', names('SynLogic Zero-Mix-3', 'Zero-Mix-3', 'SynLogic Mix-3 32B', hf('SynLogic-Mix-3-32B')), 'https://huggingface.co/MiniMaxAI/SynLogic-Mix-3-32B', '2025-05-30', '2025-05-30', weights='2025-05-30', params=(32_800 * M_, 32_800 * M_)),
)
S('SynLogic-7B', names('SynLogic 7B', 'synlogic-7b', hf('SynLogic-7B')), SYNLOGIC, '2025-05-26', '2025-06-03', weights='2025-06-03', params=(7_600 * M_, 7_600 * M_))

# MiniMax-M1 (reasoning model continued from the MiniMax-Text-01 base, with 40k and 80k thinking budgets)
M1 = IO + 'news/minimaxm1'
M('MiniMax-M1',
	V('MiniMax-M1-40k', names('MiniMax M1 40k', 'MiniMax-M1-40K', 'MiniMax M1 40K', 'minimax-m1-40k', hf('MiniMax-M1-40k', 'MiniMax-M1-40k-hf')),
		M1, '2025-06-16', '2025-06-16', weights='2025-06-16', params=P_01),
	V('MiniMax-M1-80k', names('MiniMax-M1', 'MiniMax M1', 'MiniMax M1 80k', 'MiniMax-M1-80K', 'MiniMax M1 80K', 'minimax-m1-80k', 'minimax/minimax-m1', hf('MiniMax-M1-80k', 'MiniMax-M1-80k-hf')),
		M1, '2025-06-16', '2025-06-16', weights='2025-06-16', params=P_01, price=(0.4, 2.2)),
)

# M2 series (230B-A10B MoE with full attention and interleaved thinking, always on)
S('MiniMax-M2', names('MiniMax M2', 'minimax-m2', 'minimax/minimax-m2', hf('MiniMax-M2')),
	IO + 'news/minimax-m2', '2025-10-27', '2025-10-27', weights='2025-10-27', params=P_M2, price=(0.3, 1.2))
S('MiniMax-M2.1', names('MiniMax M2.1', 'minimax-m2.1', 'minimax-m2-1', 'minimax/minimax-m2.1', 'MiniMax-M2.1-highspeed', 'MiniMax M2.1 highspeed', hf('MiniMax-M2.1')),
	IO + 'news/minimax-m21', '2025-12-23', '2025-12-23', weights='2025-12-25', params=P_M2, price=(0.3, 1.2))
S('MiniMax-M2-her', ['M2-her', 'MiniMax M2-her', 'MiniMax M2 her', 'minimax-m2-her', 'minimax/minimax-m2-her'],
	IO + 'news/a-deep-dive-into-the-minimax-m2-her-2', '2026-01-23', '2026-01-23', price=(0.3, 1.2))
S('MiniMax-M2.5', names('MiniMax M2.5', 'minimax-m2.5', 'minimax-m2-5', 'minimax/minimax-m2.5', 'minimax/minimax-m2.5-20260211',
		'MiniMax-M2.5-highspeed', 'MiniMax M2.5 highspeed', 'MiniMax-M2.5-Lightning', 'MiniMax M2.5 Lightning', 'M2.5-Lightning', hf('MiniMax-M2.5')),
	IO + 'news/minimax-m25', '2026-02-12', '2026-02-12', weights='2026-02-13', params=P_M2, price=(0.3, 1.2))
S('MiniMax-M2.7', names('MiniMax M2.7', 'minimax-m2.7', 'minimax-m2-7', 'minimax/minimax-m2.7', 'minimax/minimax-m2.7-20260318', 'MiniMax-M2.7-highspeed', 'MiniMax M2.7 highspeed', hf('MiniMax-M2.7')),
	IO + 'news/minimax-m27-en', '2026-03-18', '2026-03-18', weights='2026-04-11', params=P_M2, price=(0.3, 1.2))

# M3 (428B-A23B native multimodal MoE with MiniMax Sparse Attention and 1M context)
S('MiniMax-M3', names('MiniMax M3', 'minimax-m3', 'minimax/minimax-m3', 'minimax/minimax-m3-20260531', hf('MiniMax-M3', 'MiniMax-M3-MXFP8')),
	IO + 'models/text/m3', '2026-06-01', '2026-06-01', weights='2026-06-11', params=P_M3, price=(0.3, 1.2), efforts=TA)

# M3.1 Flash (built on M3 with sparse attention throughout and quantization-aware training; tested anonymously as Space Bunny)
M31F = IO + 'blog/minimax-m31f'
M('MiniMax-M3.1-Flash',
	V('MiniMax-M3.1-Flash-Preview', ['MiniMax M3.1 Flash Preview', 'M3.1-Flash-Preview', 'M3.1 Flash Preview', 'Space Bunny', 'Space Bunny Alpha'],
		M31F, '2026-09-27', '2026-09-27', efforts=LMHXM),
	V('MiniMax-M3.1-Flash', names('MiniMax M3.1 Flash', 'M3.1 Flash', 'minimax-m3.1-flash', 'minimax-m3-1-flash', 'minimax/minimax-m3.1-flash', hf('MiniMax-M3.1-Flash')),
		M31F, '2026-10-19', '2026-10-19', weights='2026-10-23', params=P_M3, price=(0.15, 0.6), efforts=LMHXM),
)

write('minimax')
