"""Generates data/models/zai/*.json (one file per model).

Covers Z.ai (formerly Zhipu AI) and the Tsinghua KEG / THUDM models it co-developed and now hosts under zai-org.
All the data lives in this script: edit it and run `python scripts/models/gen_zai.py`.
Existing JSON files are deleted and rewritten, so manual edits to them are lost.

Each open-weights size is a model, and the variants post-trained from that size are its versions (chat, long-context,
CodeGeeX, vision, Z1 reasoning...). The closed API tiers (GLM-4, GLM-4-Air, GLM-4-Flash...) are rolling aliases whose
dated snapshots are versions of one model. Each new version number is a separate model, even when post-trained from
the previous base (GLM-5.3 is built on the GLM-5.2 base). High-speed X endpoints (GLM-4-AirX, GLM-4.5-X, GLM-5.3-FlashX...)
serve the same weights faster, so they are only names of the version they serve.
Prices are Z.ai's USD list prices (cache miss); for the free Flash models, the price of their paid FlashX endpoint
when there is one. The models only sold on the Chinese BigModel platform have their launch CNY price converted at 7.2.
"""

from common import B, M, N, S, V, write

M_ = 1_000_000
T = ['', 'Thinking']			# thinking.type switch (disabled / enabled)
NHM = ['None', 'High', 'Max']	# GLM-5.2 reasoning_effort (none / minimal stop thinking, low / medium map to high, xhigh to max)
LHM = ['Low', 'High', 'Max']	# GLM-5.3 series reasoning_effort (thinking always on)
P_45 = (355 * B, 32 * B)		# GLM-4.5 architecture (GLM-4.5, 4.6, 4.7)
P_AIR = (106 * B, 12 * B)		# GLM-4.5-Air architecture
P_5 = (744 * B, 40 * B)			# GLM-5 architecture (GLM-5 to GLM-5.3)
P_FLASH = (320 * B, 18 * B)		# GLM-5.3-Flash architecture

ARXIV = 'https://arxiv.org/abs/'
GH = 'https://github.com/THUDM/'
HF = 'https://huggingface.co/zai-org/'
BLOG = 'https://z.ai/blog/'
DOCS = 'https://docs.z.ai/guides/'
BIGMODEL = 'https://docs.bigmodel.cn/cn/guide/start/model-overview'
CHATGLM = ARXIV + '2406.12793'	# ChatGLM: A Family of Large Language Models from GLM-130B to GLM-4 All Tools


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
	"""Repo ids with the zai-org organisation and the former THUDM one."""
	return names([f'zai-org/{r}' for r in repos], [f'THUDM/{r}' for r in repos])


def hfz(*repos):
	"""Repo ids with the zai-org organisation (models released after the THUDM rename)."""
	return [f'zai-org/{r}' for r in repos]


def aa(name, *modes):
	"""Artificial Analysis labels: plain and with each mode suffix."""
	return [name, *[f'{name} ({m})' for m in modes]]


RNR = ('Reasoning', 'Non-reasoning')

# GLM-130B and ChatGLM (the chat model of the first ChatGLM beta, post-trained from GLM-130B)
M('GLM-130B',
	V('GLM-130B', ['GLM (130B)', 'GLM 130B', 'glm-130b', 'THUDM/GLM-130B'], 'https://keg.cs.tsinghua.edu.cn/glm-130b/posts/glm-130b/', '2022-08-04', '2022-08-04', weights='2022-08-04', params=(130 * B, 130 * B)),
	V('ChatGLM (130B)', ['ChatGLM', 'ChatGLM-130B', 'ChatGLM 130B'], CHATGLM, '2023-03-14', None, id='chatglm-130b', params=(130 * B, 130 * B)),
)

# CodeGeeX (13B code model)
S('CodeGeeX', ['CodeGeeX 13B', 'CodeGeeX-13B', 'codegeex-13b'], ARXIV + '2303.17568', '2022-09-19', '2022-09-19', weights='2022-09-30', params=(13 * B, 13 * B))

# ChatGLM 6B open models (each generation is a new base; CodeGeeX2 and VisualGLM are built on them)
M('ChatGLM-6B',
	V('ChatGLM-6B', names('ChatGLM 6B', 'ChatGLM-6B-int4', 'ChatGLM-6B-int8', hf('chatglm-6b')), GH + 'ChatGLM-6B', '2023-03-14', '2023-03-14', weights='2023-03-14', params=(6_200 * M_, 6_200 * M_)),
	V('VisualGLM-6B', names('VisualGLM', 'VisualGLM 6B', hf('visualglm-6b')), GH + 'VisualGLM-6B', '2023-05-17', '2023-05-17', weights='2023-05-17', params=(7_800 * M_, 7_800 * M_)),
)
M('ChatGLM2-6B',
	V('ChatGLM2-6B', names('ChatGLM2', 'ChatGLM2 6B', 'ChatGLM 2-6B', 'ChatGLM2-6B (base)', 'ChatGLM 2-6B (base)', 'chatglm2-6b', hf('chatglm2-6b')), GH + 'ChatGLM2-6B', '2023-06-25', '2023-06-25', weights='2023-06-25', params=(6_200 * M_, 6_200 * M_)),
	V('CodeGeeX2-6B', names('CodeGeeX2', 'CodeGeeX2 6B', 'codegeex2-6b', hf('codegeex2-6b')), GH + 'CodeGeeX2', '2023-07-24', '2023-07-24', weights='2023-07-24', params=(6_200 * M_, 6_200 * M_)),
	V('ChatGLM2-6B-32K', names('ChatGLM2 6B 32K', 'chatglm2-6b-32k', hf('chatglm2-6b-32k')), GH + 'ChatGLM2-6B', '2023-07-31', '2023-07-31', weights='2023-07-31', params=(6_200 * M_, 6_200 * M_)),
)
M('ChatGLM3-6B',
	V('ChatGLM3-6B', names('ChatGLM3', 'ChatGLM3 6B', 'chatglm3-6b', 'ChatGLM3-6B-Base', 'ChatGLM3 6B Base', hf('chatglm3-6b', 'chatglm3-6b-base')), GH + 'ChatGLM3', '2023-10-27', '2023-10-27', weights='2023-10-27', params=(6_200 * M_, 6_200 * M_)),
	V('ChatGLM3-6B-32K', names('ChatGLM3 6B 32K', 'chatglm3-6b-32k', hf('chatglm3-6b-32k')), GH + 'ChatGLM3', '2023-10-27', '2023-10-27', weights='2023-10-27', params=(6_200 * M_, 6_200 * M_)),
	V('ChatGLM3-6B-128K', names('ChatGLM3 6B 128K', 'chatglm3-6b-128k', hf('chatglm3-6b-128k')), GH + 'ChatGLM3', '2024-01-30', '2024-01-30', weights='2024-01-30', params=(6_200 * M_, 6_200 * M_)),
)

# GLM-4 closed API models (BigModel platform)
M('GLM-4',
	V('GLM-4 (Jan 2024)', ['GLM-4 (0116)', 'GLM-4-0116', 'glm-4-0116'], CHATGLM, '2024-01-16', '2024-01-16', price=(cny(100), cny(100))),
	V('GLM-4 (May 2024)', ['GLM-4 (0520)', 'GLM-4-0520', 'glm-4-0520'], CHATGLM, '2024-05-20', '2024-05-20', price=(cny(100), cny(100))),
)
S('GLM-3-Turbo', ['GLM-3 Turbo', 'glm-3-turbo'], CHATGLM, '2024-01-16', '2024-01-16', price=(cny(5), cny(5)))
S('GLM-4V', ['glm-4v'], CHATGLM, '2024-01-16', '2024-01-16', price=(cny(50), cny(50)))
M('GLM-4-Air',
	V('GLM-4-Air (Jun 2024)', ['GLM-4-Air', 'GLM-4 Air', 'glm-4-air', 'GLM-4-AirX', 'glm-4-airx'], BIGMODEL, '2024-06-05', '2024-06-05', price=(cny(1), cny(1))),
	V('GLM-4-Air (Apr 2025)', ['GLM-4-Air-250414', 'glm-4-air-250414'], BIGMODEL, '2025-04-14', '2025-04-14', price=(cny(0.5), cny(0.5))),
)
M('GLM-4-Flash',
	V('GLM-4-Flash (Jun 2024)', ['GLM-4-Flash', 'GLM-4 Flash', 'glm-4-flash', 'GLM-4-FlashX', 'glm-4-flashx'], BIGMODEL, '2024-06-05', '2024-06-05', price=(cny(0.1), cny(0.1))),
	V('GLM-4-Flash (Apr 2025)', ['GLM-4-Flash-250414', 'glm-4-flash-250414'], BIGMODEL, '2025-04-14', '2025-04-14', price=(0, 0)),
)
S('GLM-4-Long', ['GLM-4 Long', 'glm-4-long'], BIGMODEL, '2024-08-10', '2024-08-10', price=(cny(1), cny(1)))
S('GLM-4-Plus', ['GLM-4 Plus', 'glm-4-plus'], BIGMODEL, '2024-08-29', '2024-08-29', price=(cny(50), cny(50)))
S('GLM-4V-Plus', ['GLM-4V Plus', 'glm-4v-plus'], BIGMODEL, '2024-08-29', '2024-08-29', price=(cny(10), cny(10)))
S('GLM-Zero-Preview', ['GLM-Zero', 'GLM Zero Preview', 'glm-zero-preview'], BIGMODEL, '2024-12-31', '2024-12-31')

# GLM-4-9B open models (CodeGeeX4, GLM-4V-9B and the LongWriter / LongCite research models are built on it)
M('GLM-4-9B',
	V('GLM-4-9B-Chat', names('GLM-4-9B', 'GLM-4 9B', 'GLM-4-9B Chat', 'glm-4-9b', 'glm-4-9b-chat', 'GLM-4-9B-Base', hf('glm-4-9b', 'glm-4-9b-chat', 'glm-4-9b-chat-hf', 'glm-4-9b-hf')), CHATGLM, '2024-06-05', '2024-06-05', weights='2024-06-05', params=(9_400 * M_, 9_400 * M_)),
	V('GLM-4-9B-Chat-1M', names('GLM-4-9B Chat 1M', 'glm-4-9b-chat-1m', hf('glm-4-9b-chat-1m', 'glm-4-9b-chat-1m-hf')), CHATGLM, '2024-06-05', '2024-06-05', weights='2024-06-05', params=(9_400 * M_, 9_400 * M_)),
	V('GLM-4V-9B', names('GLM-4V 9B', 'glm-4v-9b', hf('glm-4v-9b')), HF + 'glm-4v-9b', '2024-06-05', '2024-06-05', weights='2024-06-05', params=(9 * B, 9 * B)),
	V('CodeGeeX4-ALL-9B', names('CodeGeeX4', 'CodeGeeX4 9B', 'CodeGeeX4-ALL', 'codegeex4-all-9b', 'codegeex4', hf('codegeex4-all-9b')), GH + 'CodeGeeX4', '2024-07-05', '2024-07-05', weights='2024-07-05', params=(9_400 * M_, 9_400 * M_)),
	V('LongWriter-GLM4-9B', names('LongWriter-glm4-9b', 'LongWriter GLM4 9B', hf('LongWriter-glm4-9b')), ARXIV + '2408.07055', '2024-08-13', '2024-08-13', weights='2024-08-13', params=(9_400 * M_, 9_400 * M_)),
	V('LongCite-GLM4-9B', names('LongCite-glm4-9b', 'LongCite GLM4 9B', hf('LongCite-glm4-9b')), ARXIV + '2409.02897', '2024-09-04', '2024-09-04', weights='2024-09-04', params=(9_400 * M_, 9_400 * M_)),
)

# GLM-Edge (on-device chat models)
GLM_EDGE = 'https://github.com/THUDM/GLM-Edge'
S('GLM-Edge-1.5B-Chat', names('GLM-Edge 1.5B', 'GLM-Edge-1.5B', 'glm-edge-1.5b-chat', hf('glm-edge-1.5b-chat')), GLM_EDGE, '2024-11-29', '2024-11-29', weights='2024-11-29', params=(1_600 * M_, 1_600 * M_))
S('GLM-Edge-4B-Chat', names('GLM-Edge 4B', 'GLM-Edge-4B', 'glm-edge-4b-chat', hf('glm-edge-4b-chat')), GLM_EDGE, '2024-11-29', '2024-11-29', weights='2024-11-29', params=(4 * B, 4 * B))

# GLM-4-0414 open models and their GLM-Z1 reasoning versions (GLM-Z1-Air / AirX serve GLM-Z1-32B-0414)
GLM_0414 = 'https://github.com/zai-org/GLM-4'
M('GLM-4-32B-0414',
	V('GLM-4-32B-0414', names('GLM-4-32B', 'GLM-4 32B 0414', 'glm-4-32b-0414', 'GLM-4-32B-Base-0414', 'GLM-4-32B-0414-128K', 'glm-4-32b-0414-128k', hf('GLM-4-32B-0414', 'GLM-4-32B-Base-0414')), HF + 'GLM-4-32B-0414', '2025-04-14', '2025-04-14', weights='2025-04-14', params=(32 * B, 32 * B), price=(0.1, 0.1)),
	V('GLM-Z1-32B-0414', names('GLM-Z1-32B', 'GLM-Z1 32B 0414', 'glm-z1-32b-0414', 'GLM-Z1-Air', 'glm-z1-air', 'GLM-Z1-AirX', 'glm-z1-airx', hf('GLM-Z1-32B-0414')), HF + 'GLM-Z1-32B-0414', '2025-04-14', '2025-04-14', weights='2025-04-14', params=(32 * B, 32 * B), price=(cny(0.5), cny(0.5))),
	V('GLM-Z1-Rumination-32B-0414', names('GLM-Z1-Rumination-32B', 'GLM-Z1 Rumination 32B 0414', 'glm-z1-rumination-32b-0414', 'GLM-Z1-Rumination', hf('GLM-Z1-Rumination-32B-0414')), HF + 'GLM-Z1-Rumination-32B-0414', '2025-04-14', '2025-04-14', weights='2025-04-14', params=(32 * B, 32 * B)),
)
M('GLM-4-9B-0414',
	V('GLM-4-9B-0414', names('GLM-4 9B 0414', 'glm-4-9b-0414', hf('GLM-4-9B-0414')), HF + 'GLM-4-9B-0414', '2025-04-14', '2025-04-14', weights='2025-04-14', params=(9_400 * M_, 9_400 * M_)),
	V('GLM-Z1-9B-0414', names('GLM-Z1-9B', 'GLM-Z1 9B 0414', 'glm-z1-9b-0414', hf('GLM-Z1-9B-0414')), HF + 'GLM-Z1-9B-0414', '2025-04-14', '2025-04-14', weights='2025-04-14', params=(9_400 * M_, 9_400 * M_)),
)
S('GLM-Z1-Flash', ['GLM-Z1 Flash', 'glm-z1-flash'], BIGMODEL, '2025-04-14', '2025-04-14', price=(0, 0))
S('GLM-4.1V-9B-Thinking', names('GLM-4.1V-Thinking', 'GLM-4.1V 9B Thinking', 'glm-4.1v-9b-thinking', 'GLM-4.1V-Thinking-Flash', 'glm-4.1v-thinking-flash', 'GLM-4.1V-Thinking-FlashX', 'glm-4.1v-thinking-flashx', 'GLM-4.1V-9B-Base', hf('GLM-4.1V-9B-Thinking', 'GLM-4.1V-9B-Base')),
	ARXIV + '2507.01006', '2025-07-01', '2025-07-01', weights='2025-07-01', params=(9 * B, 9 * B))

# GLM-4.5 (355B MoE) and GLM-4.5-Air (106B MoE, base of GLM-4.5V)
M('GLM-4.5',
	V('GLM-4.5', names(aa('GLM-4.5', *RNR), 'GLM 4.5', 'glm-4.5', 'glm-4p5', 'z-ai/glm-4.5', 'GLM-4.5-X', 'glm-4.5-x', 'GLM-4.5-Base', 'GLM-4.5-FP8', hfz('GLM-4.5', 'GLM-4.5-Base', 'GLM-4.5-FP8')),
		BLOG + 'glm-4.5', '2025-07-28', '2025-07-28', weights='2025-07-28', params=P_45, price=(0.6, 2.2), efforts=T),
)
M('GLM-4.5-Air',
	V('GLM-4.5-Air', names(aa('GLM-4.5-Air', *RNR), 'GLM 4.5 Air', 'glm-4.5-air', 'glm-4p5-air', 'z-ai/glm-4.5-air', 'GLM-4.5-AirX', 'glm-4.5-airx', 'GLM-4.5-Air-Base', 'GLM-4.5-Air-FP8', hfz('GLM-4.5-Air', 'GLM-4.5-Air-Base', 'GLM-4.5-Air-FP8')),
		BLOG + 'glm-4.5', '2025-07-28', '2025-07-28', weights='2025-07-28', params=P_AIR, price=(0.2, 1.1), efforts=T),
	V('GLM-4.5V', names(aa('GLM-4.5V', *RNR), 'GLM 4.5V', 'glm-4.5v', 'z-ai/glm-4.5v', hfz('GLM-4.5V', 'GLM-4.5V-FP8')),
		DOCS + 'vlm/glm-4.5v', '2025-08-11', '2025-08-11', weights='2025-08-11', params=(108 * B, 12 * B), price=(0.6, 1.8), efforts=T),
)
S('GLM-4.5-Flash', ['GLM 4.5 Flash', 'glm-4.5-flash'], DOCS + 'llm/glm-4.5', '2025-07-28', '2025-07-28', price=(0, 0), efforts=T)

# GLM-4.6 and the GLM-4.6V vision models
S('GLM-4.6', names(aa('GLM-4.6', *RNR), 'GLM 4.6', 'glm-4.6', 'glm-4-6', 'glm-4p6', 'glm-4-6-reasoning', 'z-ai/glm-4.6', 'accounts/fireworks/models/glm-4p6', hfz('GLM-4.6', 'glm-4.6', 'GLM-4.6-FP8')),
	BLOG + 'glm-4.6', '2025-09-30', '2025-09-30', weights='2025-09-30', params=P_45, price=(0.6, 2.2), efforts=T)
S('GLM-4.6V', names(aa('GLM-4.6V', *RNR), 'GLM 4.6V', 'glm-4.6v', 'z-ai/glm-4.6v', hfz('GLM-4.6V', 'GLM-4.6V-FP8')),
	DOCS + 'vlm/glm-4.6v', '2025-12-08', '2025-12-08', weights='2025-12-08', params=P_AIR, price=(0.3, 0.9), efforts=T)
S('GLM-4.6V-Flash', names('GLM 4.6V Flash', 'glm-4.6v-flash', 'GLM-4.6V-FlashX', 'glm-4.6v-flashx', hfz('GLM-4.6V-Flash')),
	DOCS + 'vlm/glm-4.6v', '2025-12-08', '2025-12-08', weights='2025-12-08', params=(9 * B, 9 * B), price=(0.04, 0.4), efforts=T)

# GLM-4.7
S('GLM-4.7', names(aa('GLM-4.7', *RNR, 'High'), 'GLM 4.7', 'glm-4.7', 'glm-4-7', 'glm-4p7', 'z-ai/glm-4.7', 'zai/glm-4.7', 'fireworks/glm-4p7', hfz('GLM-4.7', 'glm-4.7', 'GLM-4.7-FP8')),
	BLOG + 'glm-4.7', '2025-12-22', '2025-12-22', weights='2025-12-22', params=P_45, price=(0.6, 2.2), efforts=T)
S('GLM-4.7-Flash', names(aa('GLM-4.7-Flash', *RNR), 'GLM 4.7 Flash', 'glm-4.7-flash', 'z-ai/glm-4.7-flash', 'GLM-4.7-FlashX', 'glm-4.7-flashx', hfz('GLM-4.7-Flash')),
	DOCS + 'llm/glm-4.7', '2026-01-19', '2026-01-19', weights='2026-01-19', params=(30 * B, 3 * B), price=(0.07, 0.4), efforts=T)

# GLM-5 (744B MoE with DeepSeek Sparse Attention; tested anonymously on OpenRouter as Pony Alpha)
S('GLM-5', names(aa('GLM-5', *RNR), 'GLM 5', 'GLM 5.0', 'GLM-5.0', 'glm-5', 'z-ai/glm-5', 'zai/glm-5', 'Pony Alpha', hfz('GLM-5', 'glm-5', 'GLM-5-FP8')),
	BLOG + 'glm-5', '2026-02-11', '2026-02-11', weights='2026-02-11', params=P_5, price=(1, 3.2), efforts=T)

# GLM-5-Turbo (closed agent-optimised variant, size not disclosed) and GLM-5V-Turbo (its vision version)
M('GLM-5-Turbo',
	V('GLM-5-Turbo', ['GLM 5 Turbo', 'glm-5-turbo', 'z-ai/glm-5-turbo'], DOCS + 'llm/glm-5-turbo', '2026-03-15', '2026-03-15', price=(1.2, 4), efforts=T),
	V('GLM-5V-Turbo', names(aa('GLM 5V Turbo', 'Reasoning', 'Non-reasoning'), 'GLM-5V Turbo', 'glm-5v-turbo', 'z-ai/glm-5v-turbo'), DOCS + 'vlm/glm-5v-turbo', '2026-04-01', '2026-04-01', price=(1.2, 4), efforts=T),
)

# GLM-5.x (same 744B architecture)
S('GLM-5.1', names(aa('GLM-5.1', *RNR), 'GLM 5.1', 'Glm 5.1', 'glm-5.1', 'glm-5-1', 'z-ai/glm-5.1', 'zai/glm-5.1', hfz('GLM-5.1', 'GLM-5.1-FP8')),
	DOCS + 'llm/glm-5.1', '2026-03-27', '2026-03-27', weights='2026-04-07', params=P_5, price=(1.4, 4.4), efforts=T)
S('GLM-5.2', names(aa('GLM-5.2', 'Max', 'Non-reasoning', 'High'), 'GLM 5.2', 'GLM 5.2 (Max)', 'glm-5.2', 'glm-5-2', 'glm-5p2', 'glm-5.2[1m]', 'z-ai/glm-5.2', 'zai/glm-5.2', hfz('GLM-5.2', 'glm-5.2', 'GLM-5.2-FP8')),
	DOCS + 'llm/glm-5.2', '2026-06-13', '2026-06-13', weights='2026-06-16', params=P_5, price=(1.4, 4.4), efforts=NHM)
S('GLM-5.3', names(aa('GLM-5.3', 'Low', 'High', 'Max'), 'GLM 5.3', 'GLM 5.3 Low', 'GLM 5.3 High', 'GLM 5.3 Max', 'glm-5.3', 'glm-5-3', 'z-ai/glm-5.3', 'zai/glm-5.3', hfz('GLM-5.3', 'GLM-5.3-BF16')),
	DOCS + 'llm/glm-5.3', '2026-08-14', '2026-08-14', weights='2026-08-28', params=P_5, price=(1.4, 4.4), efforts=LHM)

# GLM-5.3-Flash (320B native multimodal MoE with sparse + linear attention; tested on OpenRouter as Ox Alpha) and GLM-Legal
FLASH53 = DOCS + 'vlm/glm-5.3-flash'
M('GLM-5.3-Flash',
	V('GLM-5.3-Flash', names(aa('GLM-5.3-Flash', 'Low', 'High', 'Max'), 'GLM 5.3 Flash', 'GLM 5.3 Flash Low', 'GLM 5.3 Flash High', 'GLM 5.3 Flash Max', 'glm-5.3-flash', 'glm-5-3-flash', 'z-ai/glm-5.3-flash', 'zai/glm-5.3-flash', 'GLM-5.3-FlashX', 'GLM 5.3 FlashX', 'glm-5.3-flashx', 'Ox Alpha', hfz('GLM-5.3-Flash', 'glm-5.3-flash', 'GLM-5.3-Flash-BF16')),
		FLASH53, '2026-08-26', '2026-08-26', weights='2026-08-26', params=P_FLASH, price=(0.15, 0.5), efforts=LHM),
	V('GLM-5.3-FlashX-Legal', ['GLM-Legal', 'GLM Legal', 'GLM 5.3 FlashX Legal', 'glm-5.3-flashx-legal'], FLASH53, '2026-10-12', None, params=P_FLASH, price=(0.37, 1.25), efforts=LHM),
)

write('zai')
