"""Generates data/models/alibaba/*.json (one file per model).

Covers every Alibaba lab: the Qwen team (Alibaba Cloud / Tongyi Lab), DAMO Academy and AIDC-AI.
All the data lives in this script: edit it and run `python scripts/models/gen_alibaba.py`.
Existing JSON files are deleted and rewritten, so manual edits to them are lost.

Each open-weights size is a model, and the variants post-trained from that size are its versions (Coder, Math, VL, Omni,
1M, QwQ, QVQ, 2507 updates, hosted Plus / Flash API versions...). The closed API tiers (Qwen-Max, Qwen-Plus, Qwen-Turbo,
Qwen-VL-Max / Plus) are rolling aliases: their dated snapshots are versions of one model, and the flagships launched
under a new name (Qwen2.5-Max, Qwen3-Max, Qwen3.x-Max / Plus / Flash) are separate models.
Prices are Alibaba Cloud Model Studio / QwenCloud list prices in USD (International deployment when listed, otherwise the
USD price Alibaba lists for the Chinese mainland), base context tier; for hybrid models the non-thinking output price.
"""

from common import B, M, N, S, V, write

M_ = 1_000_000
T = ['', 'Thinking']						# enable_thinking switch (or /think in the open-weights chat template)
LMX = ['Low', 'Medium', 'Xhigh']			# reasoning_effort of the thinking-only Qwen3.8 models
TLMX = ['', 'Low', 'Medium', 'Xhigh']		# Qwen3.8: enable_thinking + reasoning_effort

ARXIV = 'https://arxiv.org/abs/'
BLOG = 'https://qwenlm.github.io/blog/'
QBLOG = 'https://qwen.ai/blog?id='
HF = 'https://huggingface.co/Qwen/'
DOCS = 'https://www.alibabacloud.com/help/en/model-studio/models'
CHANGELOG = 'https://docs.qwencloud.com/changelog/models'
QC = 'https://www.qwencloud.com/models/'


def hf(*repos):
	"""Repo ids, both bare and with the Qwen organisation."""
	return list(dict.fromkeys([*repos, *[f'Qwen/{r}' for r in repos]]))


def aa(name):
	"""Artificial Analysis labels of a hybrid model: plain, reasoning and non-reasoning."""
	return [name, f'{name} (Reasoning)', f'{name} (Non-reasoning)']


# DAMO Academy (2021)
S('PLUG', ['PLUG 27B', 'PLUG-27B'], 'https://mp.weixin.qq.com/s/DAQomIkDa52Sef-ruyH5qg', '2021-04-19', None, params=(27 * B, 27 * B))

# Tongyi Qianwen and the Qwen-Max / Qwen-Plus / Qwen-Turbo API tiers (rolling aliases, generation changes included)
S('Tongyi Qianwen', ['Tongyi Qianwen 1.0', '通义千问'], 'https://en.wikipedia.org/wiki/Qwen', '2023-04-07', '2023-09-13')
M('Qwen-Max',
	V('Tongyi Qianwen 2.0', ['通义千问2.0'], 'https://www.alibabacloud.com/blog/alibaba-cloud-launches-tongyi-qianwen-2-0-and-industry-specific-models-to-support-customers-reap-benefits-of-generative-ai_600526', '2023-10-31', '2023-10-31'),
	V('Qwen-Max (Apr 2024)', ['Qwen-Max-0428', 'qwen-max-0428', 'Tongyi Qianwen 2.5'], DOCS, '2024-04-28', '2024-04-28'),
	V('Qwen-Max (Sep 2024)', ['Qwen-Max-0919', 'qwen-max-0919', 'qwen-max-2024-09-19'], BLOG + 'qwen2.5/', '2024-09-19', '2024-09-19', price=(2.868, 8.602)),
)
M('Qwen-Plus',
	V('Qwen-Plus (Feb 2024)', [], DOCS, '2024-02-06', '2024-02-06'),
	V('Qwen-Plus (Aug 2024)', ['qwen-plus-0806', 'qwen-plus-2024-08-06'], DOCS, '2024-08-06', '2024-08-06', price=(0.574, 1.721)),
	V('Qwen-Plus (Sep 2024)', ['qwen-plus-0919', 'qwen-plus-2024-09-19'], BLOG + 'qwen2.5/', '2024-09-19', '2024-09-19', price=(0.115, 0.287)),
	V('Qwen-Plus (Nov 2024)', ['Qwen2.5-Plus (Nov 2024)', 'qwen2.5-plus-1127', 'qwen-plus-2024-11-25', 'qwen-plus-2024-11-27'], DOCS, '2024-11-25', '2024-11-25', price=(0.115, 0.287)),
	V('Qwen-Plus (Dec 2024)', ['qwen-plus-2024-12-20'], DOCS, '2024-12-20', '2024-12-20', price=(0.115, 0.287)),
	V('Qwen-Plus (Jan 2025)', ['qwen-plus-0125', 'qwen-plus-2025-01-12', 'qwen-plus-2025-01-25'], DOCS, '2025-01-12', '2025-01-12', price=(0.4, 1.2)),
	V('Qwen-Plus (Apr 2025)', ['qwen-plus-2025-04-28'], BLOG + 'qwen3/', '2025-04-28', '2025-04-28', price=(0.4, 1.2), efforts=T),
	V('Qwen-Plus (Jul 2025)', ['qwen-plus-2025-07-14', 'qwen-plus-2025-07-28'], DOCS, '2025-07-14', '2025-07-14', price=(0.4, 1.2), efforts=T),
	V('Qwen-Plus (Sep 2025)', ['qwen-plus-2025-09-11'], DOCS, '2025-09-11', '2025-09-11', price=(0.4, 1.2), efforts=T),
	V('Qwen-Plus (Dec 2025)', ['qwen-plus-2025-12-01'], DOCS, '2025-12-01', '2025-12-01', price=(0.4, 1.2), efforts=T),
)
M('Qwen-Turbo',
	V('Qwen-Turbo (Feb 2024)', [], DOCS, '2024-02-06', '2024-02-06'),
	V('Qwen-Turbo (Sep 2024)', ['qwen-turbo-0919', 'qwen-turbo-2024-09-19'], BLOG + 'qwen2.5/', '2024-09-19', '2024-09-19'),
	V('Qwen-Turbo (Apr 2025)', ['qwen-turbo-2025-04-28'], BLOG + 'qwen3/', '2025-04-28', '2025-04-28', price=(0.05, 0.2), efforts=T),
	V('Qwen-Turbo (Jul 2025)', ['qwen-turbo-2025-07-15'], DOCS, '2025-07-15', '2025-07-15', price=(0.044, 0.087), efforts=T),
)
S('Qwen-Long', ['qwen-long', 'qwen-long-latest', 'qwen-long-2025-01-25'], DOCS, '2024-05-21', '2024-05-21', price=(0.072, 0.287))
S('Qwen2.5-Turbo', ['Qwen2.5 Turbo', 'Qwen2.5-Turbo-1M', 'qwen-turbo-1101', 'qwen-turbo-2024-11-01'], BLOG + 'qwen2.5-turbo/', '2024-11-15', '2024-11-15', price=(0.05, 0.2))
S('Qwen-Flash', ['Qwen Flash', 'qwen-flash', 'qwen-flash-2025-07-28'], DOCS, '2025-07-28', '2025-07-28', price=(0.05, 0.4), efforts=T)

# Qwen-VL-Plus / Qwen-VL-Max (closed vision-language API tiers, rolling aliases)
QWEN_VL = BLOG + 'qwen-vl/'
M('Qwen-VL-Plus',
	V('Qwen-VL-Plus (Jan 2024)', [], QWEN_VL, '2024-01-25', '2024-01-25'),
	V('Qwen-VL-Plus (Aug 2024)', ['qwen-vl-plus-0809', 'qwen-vl-plus-2024-08-09'], DOCS, '2024-08-09', '2024-08-09', price=(0.216, 0.646)),
	V('Qwen-VL-Plus (Jan 2025)', ['qwen-vl-plus-2025-01-02', 'qwen-vl-plus-2025-01-25'], DOCS, '2025-01-02', '2025-01-02', price=(0.21, 0.63)),
	V('Qwen-VL-Plus (May 2025)', ['qwen-vl-plus-2025-05-07'], DOCS, '2025-05-07', '2025-05-07', price=(0.21, 0.63)),
	V('Qwen-VL-Plus (Jul 2025)', ['qwen-vl-plus-2025-07-10'], DOCS, '2025-07-10', '2025-07-10', price=(0.022, 0.216)),
	V('Qwen-VL-Plus (Aug 2025)', ['qwen-vl-plus-2025-08-15'], DOCS, '2025-08-15', '2025-08-15', price=(0.21, 0.63)),
)
M('Qwen-VL-Max',
	V('Qwen-VL-Max (Jan 2024)', [], QWEN_VL, '2024-01-25', '2024-01-25'),
	V('Qwen-VL-Max (Aug 2024)', ['qwen-vl-max-0809', 'qwen-vl-max-2024-08-09'], DOCS, '2024-08-09', '2024-08-09', price=(2.868, 2.868)),
	V('Qwen-VL-Max (Oct 2024)', ['qwen-vl-max-2024-10-30'], DOCS, '2024-10-30', '2024-10-30', price=(2.868, 2.868)),
	V('Qwen-VL-Max (Nov 2024)', ['qwen-vl-max-2024-11-19'], DOCS, '2024-11-19', '2024-11-19', price=(0.431, 1.291)),
	V('Qwen-VL-Max (Dec 2024)', ['qwen-vl-max-2024-12-30'], DOCS, '2024-12-30', '2024-12-30', price=(0.431, 1.291)),
	V('Qwen-VL-Max (Jan 2025)', ['qwen-vl-max-2025-01-25'], DOCS, '2025-01-25', '2025-01-25', price=(0.431, 1.291)),
	V('Qwen-VL-Max (Apr 2025)', ['qwen-vl-max-2025-04-02', 'qwen-vl-max-2025-04-08'], DOCS, '2025-04-02', '2025-04-02', price=(0.8, 3.2)),
	V('Qwen-VL-Max (Aug 2025)', ['qwen-vl-max-2025-08-13'], DOCS, '2025-08-13', '2025-08-13', price=(0.8, 3.2)),
)

# Qwen (first open-weights generation; Qwen-VL is built on Qwen-7B)
QWEN = ARXIV + '2309.16609'
S('Qwen-1.8B', ['Qwen 1.8B', 'Qwen-1.8B-Chat', *hf('Qwen-1_8B', 'Qwen-1_8B-Chat')], QWEN, '2023-11-30', '2023-11-30', weights='2023-11-30', params=(1_837 * M_, 1_837 * M_))
M('Qwen-7B',
	V('Qwen-7B', ['Qwen 7B', 'Qwen Chat 7B', *hf('Qwen-7B', 'Qwen-7B-Chat')], QWEN, '2023-08-03', '2023-08-03', weights='2023-08-03', params=(7_721 * M_, 7_721 * M_)),
	V('Qwen-VL', ['Qwen-VL 7B', *hf('Qwen-VL', 'Qwen-VL-Chat')], ARXIV + '2308.12966', '2023-08-24', '2023-08-24', weights='2023-08-24', params=(9_600 * M_, 9_600 * M_)),
)
S('Qwen-14B', ['Qwen 14B', 'Qwen Chat 14B', *hf('Qwen-14B', 'Qwen-14B-Chat')], QWEN, '2023-09-25', '2023-09-25', weights='2023-09-25', params=(14_167 * M_, 14_167 * M_))
S('Qwen-72B', ['Qwen 72B', 'Qwen Chat 72B', *hf('Qwen-72B', 'Qwen-72B-Chat')], QWEN, '2023-11-30', '2023-11-30', weights='2023-11-30', params=(72_288 * M_, 72_288 * M_))

# Qwen1.5 (CodeQwen1.5 has its own architecture and vocabulary, so it is not a Qwen1.5-7B version)
QWEN15 = BLOG + 'qwen1.5/'
for size, p, link, date in [('0.5B', 620, QWEN15, '2024-02-05'), ('1.8B', 1_837, QWEN15, '2024-02-05'), ('4B', 3_950, QWEN15, '2024-02-05'), ('7B', 7_721, QWEN15, '2024-02-05'), ('14B', 14_167, QWEN15, '2024-02-05'),
		('32B', 32_512, BLOG + 'qwen1.5-32b/', '2024-04-02'), ('72B', 72_288, QWEN15, '2024-02-05'), ('110B', 111_210, BLOG + 'qwen1.5-110b/', '2024-04-25')]:
	S(f'Qwen1.5-{size}', [f'Qwen1.5 {size}', f'Qwen1.5 Chat {size}', f'qwen1.5-{size.lower()}-chat', *hf(f'Qwen1.5-{size}', f'Qwen1.5-{size}-Chat')], link, date, date, weights=date, params=(p * M_, p * M_))
S('Qwen1.5-MoE-A2.7B', ['Qwen1.5 MoE A2.7B', *hf('Qwen1.5-MoE-A2.7B', 'Qwen1.5-MoE-A2.7B-Chat')], BLOG + 'qwen-moe/', '2024-03-28', '2024-03-28', weights='2024-03-28', params=(14_316 * M_, 2_700 * M_))
S('CodeQwen1.5-7B', ['CodeQwen1.5', 'CodeQwen 1.5 7B', *hf('CodeQwen1.5-7B', 'CodeQwen1.5-7B-Chat')], BLOG + 'codeqwen1.5/', '2024-04-16', '2024-04-16', weights='2024-04-16', params=(7_250 * M_, 7_250 * M_))

# Qwen2 (Qwen2-Math, Qwen2-VL and QVQ are built on the matching size; Qwen2-VL-2B is built on Qwen2-1.5B)
QWEN2 = BLOG + 'qwen2/'
QWEN2_MATH = BLOG + 'qwen2-math/'
QWEN2_VL = BLOG + 'qwen2-vl/'


def q2(size):
	s = size.lower()
	return [f'Qwen2 {size}', f'Qwen2 Instruct {size}', f'Qwen2 {size} Instruct', f'qwen2-{s}-instruct', *hf(f'Qwen2-{size}', f'Qwen2-{size}-Instruct')]


def math2(size):
	return [f'Qwen2-Math {size}', f'qwen2-math-{size.lower()}-instruct', *hf(f'Qwen2-Math-{size}', f'Qwen2-Math-{size}-Instruct')]


S('Qwen2-0.5B', q2('0.5B'), QWEN2, '2024-06-07', '2024-06-07', weights='2024-06-07', params=(494 * M_, 494 * M_))
M('Qwen2-1.5B',
	V('Qwen2-1.5B', q2('1.5B'), QWEN2, '2024-06-07', '2024-06-07', weights='2024-06-07', params=(1_544 * M_, 1_544 * M_)),
	V('Qwen2-Math-1.5B', math2('1.5B'), QWEN2_MATH, '2024-08-08', '2024-08-08', weights='2024-08-08', params=(1_544 * M_, 1_544 * M_)),
	V('Qwen2-VL-2B', ['Qwen2-VL 2B', *hf('Qwen2-VL-2B', 'Qwen2-VL-2B-Instruct')], QWEN2_VL, '2024-08-29', '2024-08-29', weights='2024-08-29', params=(2_209 * M_, 2_209 * M_)),
)
P2_7B = (7_616 * M_, 7_616 * M_)
M('Qwen2-7B',
	V('Qwen2-7B', q2('7B'), QWEN2, '2024-06-07', '2024-06-07', weights='2024-06-07', params=P2_7B),
	V('Qwen2-Math-7B', math2('7B'), QWEN2_MATH, '2024-08-08', '2024-08-08', weights='2024-08-08', params=P2_7B),
	V('Qwen2-VL-7B', ['Qwen2-VL 7B', *hf('Qwen2-VL-7B', 'Qwen2-VL-7B-Instruct')], QWEN2_VL, '2024-08-29', '2024-08-29', weights='2024-08-29', params=(8_291 * M_, 8_291 * M_)),
	V('Marco-o1', ['Marco o1', 'AIDC-AI/Marco-o1'], ARXIV + '2411.14405', '2024-11-21', '2024-11-21', weights='2024-11-21', params=P2_7B),
)
S('Qwen2-57B-A14B', q2('57B-A14B'), QWEN2, '2024-06-07', '2024-06-07', weights='2024-06-07', params=(57_409 * M_, 14 * B))
P2_72B = (72_706 * M_, 72_706 * M_)
P2_VL72B = (73_406 * M_, 73_406 * M_)
M('Qwen2-72B',
	V('Qwen2-72B', q2('72B'), QWEN2, '2024-06-07', '2024-06-07', weights='2024-06-07', params=P2_72B),
	V('Qwen2-Math-72B', math2('72B'), QWEN2_MATH, '2024-08-08', '2024-08-08', weights='2024-08-08', params=P2_72B),
	V('Qwen2-VL-72B', ['Qwen2-VL 72B', 'qwen2-vl-72b', 'qwen2-vl-72b-instruct', *hf('Qwen2-VL-72B', 'Qwen2-VL-72B-Instruct')], QWEN2_VL, '2024-08-29', '2024-08-29', weights='2024-09-19', params=P2_VL72B, price=(2.294, 6.881)),
	V('QVQ-72B-Preview', ['QVQ', 'QVQ 72B Preview', 'QvQ-72B-Preview', 'qvq-72b-preview', *hf('QVQ-72B-Preview')], BLOG + 'qvq-72b-preview/', '2024-12-25', '2024-12-25', weights='2024-12-25', params=P2_VL72B, price=(1.721, 5.161)),
)

# Qwen2.5 (Coder, Math, VL, Omni, 1M, QwQ and QVQ are built on the matching size)
QWEN25 = BLOG + 'qwen2.5/'
CODER25 = BLOG + 'qwen2.5-coder-family/'
MATH25 = BLOG + 'qwen2.5-math/'
VL25 = BLOG + 'qwen2.5-vl/'
OMNI25 = BLOG + 'qwen2.5-omni/'
QWEN25_1M = BLOG + 'qwen2.5-1m/'


def q25(size):
	s = size.lower()
	return [f'Qwen2.5 {size}', f'Qwen2.5 Instruct ({size})', f'Qwen2.5 Instruct {size}', f'Qwen2.5 {size} Instruct', f'qwen2.5-{s}-instruct', f'qwen-2.5-{s}-instruct', *hf(f'Qwen2.5-{size}', f'Qwen2.5-{size}-Instruct')]


def coder25(size):
	s = size.lower()
	return [f'Qwen2.5-Coder ({size})', f'Qwen2.5-Coder {size}', f'Qwen2.5 Coder Instruct {size}', f'Qwen2.5-Coder {size} Instruct', f'qwen2.5-coder-{s}-instruct', f'qwen-2.5-coder-{s}-instruct', *hf(f'Qwen2.5-Coder-{size}', f'Qwen2.5-Coder-{size}-Instruct')]


def math25(size):
	return [f'Qwen2.5-Math {size}', f'Qwen2.5-Math-{size}-Base', f'qwen2.5-math-{size.lower()}-instruct', *hf(f'Qwen2.5-Math-{size}', f'Qwen2.5-Math-{size}-Instruct')]


def vl25(size):
	s = size.lower()
	return [f'Qwen2.5-VL {size}', f'Qwen2.5 VL {size} Instruct', f'qwen2.5-vl-{s}', f'qwen2.5-vl-{s}-instruct', *hf(f'Qwen2.5-VL-{size}-Instruct')]


def qwen25(size, p, *versions, price=(None, None)):
	"""A Qwen2.5 size: the base / instruct version first, then the variants built on it."""
	M(f'Qwen2.5-{size}', V(f'Qwen2.5-{size}', q25(size), QWEN25, '2024-09-19', '2024-09-19', weights='2024-09-19', params=(p * M_, p * M_), price=price), *versions)


qwen25('0.5B', 494,
	V('Qwen2.5-Coder-0.5B', coder25('0.5B'), CODER25, '2024-11-12', '2024-11-12', weights='2024-11-12', params=(494 * M_, 494 * M_)),
)
qwen25('1.5B', 1_544,
	V('Qwen2.5-Coder-1.5B', coder25('1.5B'), CODER25, '2024-09-19', '2024-09-19', weights='2024-09-19', params=(1_544 * M_, 1_544 * M_)),
	V('Qwen2.5-Math-1.5B', math25('1.5B'), MATH25, '2024-09-19', '2024-09-19', weights='2024-09-19', params=(1_544 * M_, 1_544 * M_)),
)
qwen25('3B', 3_086,
	V('Qwen2.5-Coder-3B', coder25('3B'), CODER25, '2024-11-12', '2024-11-12', weights='2024-11-12', params=(3_086 * M_, 3_086 * M_)),
	V('Qwen2.5-VL-3B', vl25('3B'), VL25, '2025-01-26', '2025-01-26', weights='2025-01-26', params=(3_755 * M_, 3_755 * M_), price=(0.21, 0.63)),
	V('Qwen2.5-Omni-3B', ['Qwen2.5-Omni 3B', 'qwen2.5-omni-3b', *hf('Qwen2.5-Omni-3B')], OMNI25, '2025-04-30', '2025-04-30', weights='2025-04-30', params=(5_537 * M_, 5_537 * M_)),
	price=(0.044, 0.13),
)
P25_7B = (7_616 * M_, 7_616 * M_)
qwen25('7B', 7_616,
	V('Qwen2.5-Coder-7B', coder25('7B'), CODER25, '2024-09-19', '2024-09-19', weights='2024-09-19', params=P25_7B, price=(0.144, 0.287)),
	V('Qwen2.5-Math-7B', math25('7B'), MATH25, '2024-09-19', '2024-09-19', weights='2024-09-19', params=P25_7B, price=(0.144, 0.287)),
	V('Qwen2.5-7B-Instruct-1M', ['Qwen2.5-7B-1M', 'Qwen2.5 7B Instruct 1M', 'qwen2.5-7b-instruct-1m', 'Qwen/Qwen2.5-7B-Instruct-1M'], QWEN25_1M, '2025-01-27', '2025-01-27', weights='2025-01-27', params=P25_7B, price=(0.368, 1.47)),
	V('Qwen2.5-VL-7B', vl25('7B'), VL25, '2025-01-26', '2025-01-26', weights='2025-01-26', params=(8_292 * M_, 8_292 * M_), price=(0.35, 1.05)),
	V('Qwen2.5-Omni-7B', ['Qwen2.5-Omni 7B', 'qwen2.5-omni-7b', *hf('Qwen2.5-Omni-7B')], OMNI25, '2025-03-27', '2025-03-27', weights='2025-03-27', params=(10_732 * M_, 10_732 * M_)),
	price=(0.175, 0.7),
)
P25_14B = (14_770 * M_, 14_770 * M_)
qwen25('14B', 14_770,
	V('Qwen2.5-Coder-14B', coder25('14B'), CODER25, '2024-11-12', '2024-11-12', weights='2024-11-12', params=P25_14B, price=(0.287, 0.861)),
	V('Qwen2.5-14B-Instruct-1M', ['Qwen2.5-14B-1M', 'Qwen2.5 14B Instruct 1M', 'qwen2.5-14b-instruct-1m', 'Qwen/Qwen2.5-14B-Instruct-1M'], QWEN25_1M, '2025-01-27', '2025-01-27', weights='2025-01-27', params=P25_14B, price=(0.805, 3.22)),
	price=(0.35, 1.4),
)
P25_32B = (32_764 * M_, 32_764 * M_)
qwen25('32B', 32_764,
	V('Qwen2.5-Coder-32B', coder25('32B'), CODER25, '2024-11-12', '2024-11-12', weights='2024-11-12', params=P25_32B, price=(0.287, 0.861)),
	V('QwQ-32B-Preview', ['QwQ 32B-Preview', 'QwQ 32B Preview', 'QwQ-32B (Preview)', 'qwq-32b-preview', 'Qwen/QwQ-32B-Preview'], BLOG + 'qwq-32b-preview/', '2024-11-28', '2024-11-28', weights='2024-11-28', params=P25_32B, price=(0.287, 0.861)),
	V('QwQ-32B', ['QwQ 32B', 'QwQ', 'qwq-32b', 'Qwen/QwQ-32B'], BLOG + 'qwq-32b/', '2025-03-06', '2025-03-06', weights='2025-03-06', params=P25_32B, price=(0.287, 0.861)),
	V('Qwen2.5-VL-32B', vl25('32B'), BLOG + 'qwen2.5-vl-32b/', '2025-03-24', '2025-03-24', weights='2025-03-24', params=(33_453 * M_, 33_453 * M_), price=(1.4, 4.2)),
	price=(0.7, 2.8),
)
P25_72B = (72_706 * M_, 72_706 * M_)
qwen25('72B', 72_706,
	V('Qwen2.5-Math-72B', math25('72B'), MATH25, '2024-09-19', '2024-09-19', weights='2024-09-19', params=P25_72B, price=(0.574, 1.721)),
	V('Qwen2.5-VL-72B', vl25('72B'), VL25, '2025-01-26', '2025-01-26', weights='2025-01-26', params=(73_411 * M_, 73_411 * M_), price=(2.8, 8.4)),
	price=(1.4, 5.6),
)

# Qwen2.5 closed models (QwQ-Max-Preview is built on Qwen2.5-Max; the bases of QwQ-Plus and QVQ-Max are not disclosed)
M('Qwen2.5-Max',
	V('Qwen2.5-Max', ['Qwen2.5 Max', 'qwen-max-0125', 'qwen-max-2025-01-25'], BLOG + 'qwen2.5-max/', '2025-01-28', '2025-01-28', price=(1.6, 6.4)),
	V('QwQ-Max-Preview', ['QwQ Max Preview', 'QwQ-Max (Preview)', 'qwq-max-preview'], BLOG + 'qwq-max-preview/', '2025-02-25', '2025-02-25'),
)
S('QwQ-Plus', ['QWQ-Plus', 'QwQ Plus', 'qwq-plus', 'qwq-plus-2025-03-05'], 'https://www.alibabacloud.com/help/en/model-studio/qwq', '2025-03-06', '2025-03-06', price=(0.8, 2.4))
S('QVQ-Max', ['QVQ Max', 'QVQ-Max-Preview', 'qvq-max', 'qvq-max-2025-03-25', 'qvq-max-2025-05-15'], BLOG + 'qvq-max-preview/', '2025-03-28', '2025-03-28', price=(1.2, 4.8))

# Qwen3 (2507 updates, Coder, VL and Tongyi DeepResearch are built on the matching size;
# Qwen3-VL-2B is built on Qwen3-1.7B, Qwen3-Coder-Plus / Flash are the hosted Qwen3-Coder models)
QWEN3 = BLOG + 'qwen3/'
QWEN3_2507 = HF + 'Qwen3-235B-A22B-Instruct-2507'
QWEN3_VL = 'https://github.com/QwenLM/Qwen3-VL'
CODER3 = BLOG + 'qwen3-coder/'


def q3(size, base=True):
	sp = size.replace('-', ' ')
	return [*aa(f'Qwen3 {sp}'), f'qwen3-{size.lower()}', *hf(f'Qwen3-{size}', *([f'Qwen3-{size}-Base'] if base else []))]


def vl3(size, kind):
	sp, s = size.replace('-', ' '), size.lower()
	labels = [f'Qwen3 VL {sp} Instruct'] if kind == 'Instruct' else [f'Qwen3 VL {sp} (Reasoning)', f'Qwen3 VL {sp} Thinking']
	return [*labels, f'qwen3-vl-{s}-{kind.lower()}', *hf(f'Qwen3-VL-{size}-{kind}')]


def vl3_versions(size, date, p, prices=((None, None), (None, None))):
	return [V(f'Qwen3-VL-{size}-{kind}', vl3(size, kind), QWEN3_VL, date, date, weights=date, params=p, price=price) for kind, price in zip(['Instruct', 'Thinking'], prices)]


S('Qwen3-0.6B', q3('0.6B'), QWEN3, '2025-04-29', '2025-04-29', weights='2025-04-29', params=(596 * M_, 596 * M_), price=(0.11, 0.42), efforts=T)
M('Qwen3-1.7B',
	V('Qwen3-1.7B', q3('1.7B'), QWEN3, '2025-04-29', '2025-04-29', weights='2025-04-29', params=(1_721 * M_, 1_721 * M_), price=(0.11, 0.42), efforts=T),
	*vl3_versions('2B', '2025-10-21', (2_128 * M_, 2_128 * M_)),
)
P3_4B = (4_022 * M_, 4_022 * M_)
M('Qwen3-4B',
	V('Qwen3-4B', q3('4B'), QWEN3, '2025-04-29', '2025-04-29', weights='2025-04-29', params=P3_4B, price=(0.11, 0.42), efforts=T),
	V('Qwen3-4B-Instruct-2507', ['Qwen3 4B 2507 Instruct', 'qwen3-4b-instruct-2507', 'Qwen/Qwen3-4B-Instruct-2507'], HF + 'Qwen3-4B-Instruct-2507', '2025-08-06', '2025-08-06', weights='2025-08-06', params=P3_4B),
	V('Qwen3-4B-Thinking-2507', ['Qwen3 4B 2507 (Reasoning)', 'qwen3-4b-thinking-2507', 'Qwen/Qwen3-4B-Thinking-2507'], HF + 'Qwen3-4B-Thinking-2507', '2025-08-06', '2025-08-06', weights='2025-08-06', params=P3_4B),
	*vl3_versions('4B', '2025-10-14', (4_438 * M_, 4_438 * M_)),
)
M('Qwen3-8B',
	V('Qwen3-8B', q3('8B'), QWEN3, '2025-04-29', '2025-04-29', weights='2025-04-29', params=(8_191 * M_, 8_191 * M_), price=(0.18, 0.7), efforts=T),
	*vl3_versions('8B', '2025-10-14', (8_767 * M_, 8_767 * M_), ((0.072, 0.287), (0.072, 0.717))),
)
S('Qwen3-14B', q3('14B'), QWEN3, '2025-04-29', '2025-04-29', weights='2025-04-29', params=(14_768 * M_, 14_768 * M_), price=(0.35, 1.4), efforts=T)
M('Qwen3-32B',
	V('Qwen3-32B', q3('32B', base=False), QWEN3, '2025-04-29', '2025-04-29', weights='2025-04-29', params=(32_762 * M_, 32_762 * M_), price=(0.16, 0.64), efforts=T),
	*vl3_versions('32B', '2025-10-21', (33_357 * M_, 33_357 * M_), ((0.287, 1.147), (0.287, 2.867))),
)
P3_30B = (30_532 * M_, 3_300 * M_)
M('Qwen3-30B-A3B',
	V('Qwen3-30B-A3B', q3('30B-A3B'), QWEN3, '2025-04-29', '2025-04-29', weights='2025-04-29', params=P3_30B, price=(0.2, 0.8), efforts=T),
	V('Qwen3-30B-A3B-Instruct-2507', ['Qwen3-30B-A3B (Jul 2025)', 'Qwen3 30B A3B 2507 Instruct', 'qwen3-30b-a3b-instruct-2507', 'Qwen/Qwen3-30B-A3B-Instruct-2507'], HF + 'Qwen3-30B-A3B-Instruct-2507', '2025-07-29', '2025-07-29', weights='2025-07-29', params=P3_30B, price=(0.2, 0.8)),
	V('Qwen3-30B-A3B-Thinking-2507', ['Qwen3 30B A3B 2507 (Reasoning)', 'qwen3-30b-a3b-thinking-2507', 'Qwen/Qwen3-30B-A3B-Thinking-2507'], HF + 'Qwen3-30B-A3B-Thinking-2507', '2025-07-30', '2025-07-30', weights='2025-07-30', params=P3_30B, price=(0.2, 2.4)),
	V('Qwen3-Coder-30B-A3B', ['Qwen3 Coder 30B A3B Instruct', 'Qwen3-Coder-30B-A3B-Instruct', 'qwen3-coder-30b-a3b-instruct', 'Qwen/Qwen3-Coder-30B-A3B-Instruct'], HF + 'Qwen3-Coder-30B-A3B-Instruct', '2025-07-31', '2025-07-31', weights='2025-07-31', params=P3_30B, price=(0.45, 2.25)),
	V('Qwen3-Coder-Flash', ['Qwen3 Coder Flash', 'qwen3-coder-flash', 'qwen3-coder-flash-2025-07-28'], DOCS, '2025-07-31', '2025-07-31', params=P3_30B, price=(0.3, 1.5)),
	V('Tongyi DeepResearch', ['Tongyi DeepResearch 30B-A3B', 'Tongyi-DeepResearch-30B-A3B', 'Alibaba-NLP/Tongyi-DeepResearch-30B-A3B'], ARXIV + '2510.24701', '2025-09-16', '2025-09-16', weights='2025-09-16', params=P3_30B),
	*vl3_versions('30B-A3B', '2025-10-03', (31_071 * M_, 3_300 * M_), ((0.108, 0.43), (0.108, 1.075))),
)
P3_235B = (235_094 * M_, 22 * B)
M('Qwen3-235B-A22B',
	V('Qwen3-235B-A22B', q3('235B-A22B', base=False), QWEN3, '2025-04-29', '2025-04-29', weights='2025-04-29', params=P3_235B, price=(0.7, 2.8), efforts=T),
	V('Qwen3-235B-A22B-Instruct-2507', ['Qwen3-235B-A22B (Jul 2025)', 'Qwen3 235B A22B 2507 Instruct', 'qwen3-235b-a22b-instruct-2507', 'Qwen/Qwen3-235B-A22B-Instruct-2507'], QWEN3_2507, '2025-07-21', '2025-07-21', weights='2025-07-21', params=P3_235B, price=(0.23, 0.92)),
	V('Qwen3-235B-A22B-Thinking-2507', ['Qwen3-235B-A22B-Thinking (Jul 2025)', 'Qwen3 235B A22B 2507 (Reasoning)', 'qwen3-235b-a22b-thinking-2507', 'Qwen/Qwen3-235B-A22B-Thinking-2507'], HF + 'Qwen3-235B-A22B-Thinking-2507', '2025-07-25', '2025-07-25', weights='2025-07-25', params=P3_235B, price=(0.23, 2.3)),
	*vl3_versions('235B-A22B', '2025-09-23', (235_670 * M_, 22 * B), ((0.287, 1.147), (0.287, 2.867))),
)
P3_480B = (480_155 * M_, 35 * B)
M('Qwen3-Coder-480B-A35B',
	V('Qwen3-Coder-480B-A35B', ['Qwen3-Coder', 'Qwen3 Coder 480B A35B Instruct', 'Qwen3-Coder 480B A35B Instruct', 'Qwen3-Coder-480B-A35B-Instruct', 'qwen3-coder', 'qwen3-coder-480b-a35b-instruct', 'Qwen/Qwen3-Coder-480B-A35B-Instruct'], CODER3, '2025-07-22', '2025-07-22', weights='2025-07-22', params=P3_480B, price=(1.5, 7.5)),
	V('Qwen3-Coder-Plus (Jul 2025)', ['qwen3-coder-plus-2025-07-22'], CODER3, '2025-07-22', '2025-07-22', params=P3_480B, price=(1, 5)),
	V('Qwen3-Coder-Plus (Sep 2025)', ['qwen3-coder-plus-2025-09-23'], DOCS, '2025-09-23', '2025-09-23', params=P3_480B, price=(1, 5)),
)

# Qwen3-Next (hybrid attention; Qwen3-Coder-Next is built on it) and Qwen3-Omni (base not disclosed)
NEXT = 'https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd&from=research.latest-advancements-list'
P3_NEXT = (81_325 * M_, 3 * B)
M('Qwen3-Next-80B-A3B',
	V('Qwen3-Next-80B-A3B-Instruct', ['Qwen3 Next 80B A3B Instruct', 'qwen3-next-80b-a3b-instruct', 'qwen3-next-80b-a3b-base', *hf('Qwen3-Next-80B-A3B-Instruct', 'Qwen3-Next-80B-A3B-Base')], NEXT, '2025-09-11', '2025-09-11', weights='2025-09-11', params=P3_NEXT, price=(0.15, 1.2)),
	V('Qwen3-Next-80B-A3B-Thinking', ['Qwen3 Next 80B A3B (Reasoning)', 'qwen3-next-80b-a3b-thinking', 'Qwen/Qwen3-Next-80B-A3B-Thinking'], NEXT, '2025-09-11', '2025-09-11', weights='2025-09-11', params=P3_NEXT, price=(0.15, 1.2)),
	V('Qwen3-Coder-Next', ['Qwen3 Coder Next', 'qwen3-coder-next', *hf('Qwen3-Coder-Next', 'Qwen3-Coder-Next-Base')], QBLOG + 'qwen3-coder-next', '2026-02-03', '2026-02-03', weights='2026-02-03', params=(79_674 * M_, 3 * B), price=(0.31, 1)),
)
OMNI3 = ARXIV + '2509.17765'
P3_OMNI = (35_260 * M_, 3 * B)
M('Qwen3-Omni-30B-A3B',
	V('Qwen3-Omni-30B-A3B-Instruct', ['Qwen3-Omni-30B-A3B', 'Qwen3 Omni 30B A3B Instruct', 'qwen3-omni-30b-a3b-instruct', 'Qwen/Qwen3-Omni-30B-A3B-Instruct'], OMNI3, '2025-09-22', '2025-09-22', weights='2025-09-22', params=P3_OMNI),
	V('Qwen3-Omni-30B-A3B-Thinking', ['Qwen3 Omni 30B A3B (Reasoning)', 'qwen3-omni-30b-a3b-thinking', 'Qwen/Qwen3-Omni-30B-A3B-Thinking'], OMNI3, '2025-09-22', '2025-09-22', weights='2025-09-22', params=(31_719 * M_, 3 * B)),
	V('Qwen3-Omni-Flash (Sep 2025)', ['Qwen3-Omni-Flash', 'qwen3-omni-flash-2025-09-15'], OMNI3, '2025-09-22', '2025-09-22', price=(0.43, 1.66), efforts=T),
	V('Qwen3-Omni-Flash (Dec 2025)', ['qwen3-omni-flash-2025-12-01'], DOCS, '2025-12-01', '2025-12-01', price=(0.43, 1.66), efforts=T),
)

# Qwen3 closed models (over 1T parameters for Qwen3-Max; bases of Qwen3-VL-Plus / Flash not disclosed)
QMAX = 'https://qwen.ai/blog?id=87dc93fc8a590dc718c77e1f6e84c07b474f6c5a&from=home.latest-research-list'
P3_MAX = (1_000 * B, None)
M('Qwen3-Max',
	V('Qwen3-Max-Preview', ['Qwen3 Max (Preview)', 'Qwen3 Max Thinking (Preview)', 'Qwen3-Max-Thinking-Preview', 'Qwen3-Max-Instruct-Preview', 'qwen3-max-preview'], QMAX, '2025-09-05', '2025-09-05', params=P3_MAX, price=(1.2, 6), efforts=T),
	V('Qwen3-Max', ['Qwen3 Max', 'Qwen3-Max (Sep 2025)', 'Qwen3-Max-Instruct', 'qwen3-max-2025-09-23'], QMAX, '2025-09-05', '2025-09-23', params=P3_MAX, price=(1.2, 6)),
	V('Qwen3-Max-Thinking', ['Qwen3 Max Thinking', 'Qwen3-Max (Jan 2026)', 'qwen3-max-2026-01-23'], QBLOG + 'qwen3-max-thinking', '2025-09-23', '2026-01-26', params=P3_MAX, price=(1.2, 6), efforts=T),
)
S('Qwen3-VL-Plus', ['Qwen3 VL Plus', 'qwen3-vl-plus', 'qwen3-vl-plus-2025-09-23', 'qwen3-vl-plus-2025-12-19'], DOCS, '2025-09-23', '2025-09-23', price=(0.2, 1.6), efforts=T)
S('Qwen3-VL-Flash', ['Qwen3 VL Flash', 'qwen3-vl-flash', 'qwen3-vl-flash-2025-10-15'], DOCS, '2025-10-15', '2025-10-15', price=(0.05, 0.4), efforts=T)

# Qwen3.5 (Qwen3.5-Plus / Flash are the hosted 397B-A17B / 35B-A3B; Qwen-AgentWorld is built on Qwen3.5-35B-A3B-Base)
QWEN35 = QBLOG + 'qwen3.5'


def q35(size, base=True):
	sp = size.replace('-', ' ')
	return list(dict.fromkeys([*aa(f'Qwen3.5 {sp}'), f'Qwen3.5 {size}', f'qwen3.5-{size.lower()}', *hf(f'Qwen3.5-{size}', *([f'Qwen3.5-{size}-Base'] if base else []))]))


P35_397B = (403_398 * M_, 17 * B)
M('Qwen3.5-397B-A17B',
	V('Qwen3.5-397B-A17B', q35('397B-A17B', base=False), QWEN35, '2026-02-16', '2026-02-16', weights='2026-02-16', params=P35_397B, price=(0.6, 3.6), efforts=T),
	V('Qwen3.5-Plus', ['Qwen3.5 Plus', 'Qwen 3.5 Plus (hosted 397B-A17B)', 'Qwen3.5-Plus (Feb 2026)', 'qwen3.5-plus-2026-02-15'], QWEN35, '2026-02-16', '2026-02-16', params=P35_397B, price=(0.4, 2.4), efforts=T),
	V('Qwen3.5-Plus (Apr 2026)', ['qwen3.5-plus-2026-04-20'], CHANGELOG, '2026-04-23', '2026-04-23', params=P35_397B, price=(0.4, 2.4), efforts=T),
)
S('Qwen3.5-122B-A10B', q35('122B-A10B', base=False), HF + 'Qwen3.5-122B-A10B', '2026-02-24', '2026-02-24', weights='2026-02-24', params=(125_086 * M_, 10 * B), price=(0.4, 3.2), efforts=T)
P35_35B = (35_952 * M_, 3 * B)
M('Qwen3.5-35B-A3B',
	V('Qwen3.5-35B-A3B', q35('35B-A3B'), HF + 'Qwen3.5-35B-A3B', '2026-02-24', '2026-02-24', weights='2026-02-24', params=P35_35B, price=(0.25, 2), efforts=T),
	V('Qwen3.5-Flash', ['Qwen3.5 Flash', 'Qwen 3.5 Flash (hosted 35B-A3B)', 'qwen3.5-flash', 'qwen3.5-flash-2026-02-23'], CHANGELOG, '2026-02-24', '2026-02-24', params=P35_35B, price=(0.1, 0.4), efforts=T),
	V('Qwen-AgentWorld-35B-A3B', ['Qwen-AgentWorld', 'Qwen AgentWorld 35B A3B', 'Qwen/Qwen-AgentWorld-35B-A3B'], HF + 'Qwen-AgentWorld-35B-A3B', '2026-06-23', '2026-06-23', weights='2026-06-23', params=(34_661 * M_, 3 * B)),
)
S('Qwen3.5-27B', q35('27B', base=False), HF + 'Qwen3.5-27B', '2026-02-24', '2026-02-24', weights='2026-02-24', params=(27_781 * M_, 27_781 * M_), price=(0.3, 2.4), efforts=T)
for size, p, price in [('9B', 9_653, (0.135, 0.2)), ('4B', 4_660, (0.03, 0.15)), ('2B', 2_274, (None, None)), ('0.8B', 873, (None, None))]:
	S(f'Qwen3.5-{size}', q35(size), HF + f'Qwen3.5-{size}', '2026-03-02', '2026-03-02', weights='2026-03-02', params=(p * M_, p * M_), price=price, efforts=T)
OMNI35 = ARXIV + '2604.15804'
S('Qwen3.5-Omni-Plus', ['Qwen3.5 Omni Plus', 'qwen3.5-omni-plus', 'qwen3.5-omni-plus-2026-03-15'], OMNI35, '2026-03-30', '2026-03-30', price=(1.4, 8.3))
S('Qwen3.5-Omni-Flash', ['Qwen3.5 Omni Flash', 'qwen3.5-omni-flash', 'qwen3.5-omni-flash-2026-03-15'], OMNI35, '2026-03-30', '2026-03-30', price=(0.4, 2.2))

# Qwen3.6 (Qwen3.6-Flash is the hosted 35B-A3B)
M('Qwen3.6-Plus',
	V('Qwen3.6-Plus-Preview', ['Qwen3.6 Plus Preview', 'qwen3.6-plus-preview', 'qwen/qwen3.6-plus-preview'], 'https://openrouter.ai/qwen/qwen3.6-plus-preview:free', '2026-03-30', '2026-03-30'),
	V('Qwen3.6-Plus', ['Qwen3.6 Plus', 'Qwen 3.6 Plus', 'qwen3.6-plus', 'qwen3.6-plus-2026-04-02'], QBLOG + 'qwen3.6', '2026-03-30', '2026-04-02', price=(0.5, 3), efforts=T),
)
P36_35B = (35_952 * M_, 3 * B)
M('Qwen3.6-35B-A3B',
	V('Qwen3.6-35B-A3B', [*aa('Qwen3.6 35B A3B'), 'qwen3.6-35b-a3b', 'Qwen/Qwen3.6-35B-A3B'], HF + 'Qwen3.6-35B-A3B', '2026-04-16', '2026-04-16', weights='2026-04-16', params=P36_35B, price=(0.375, 2.25), efforts=T),
	V('Qwen3.6-Flash', ['Qwen3.6 Flash', 'qwen3.6-flash', 'qwen3.6-flash-2026-04-16'], CHANGELOG, '2026-04-16', '2026-04-16', params=P36_35B, price=(0.25, 1.5), efforts=T),
)
M('Qwen3.6-Max',
	V('Qwen3.6-Max-Preview', ['Qwen3.6 Max Preview', 'qwen3.6-max-preview'], CHANGELOG, '2026-04-20', '2026-04-20', efforts=T),
)
S('Qwen3.6-27B', [*aa('Qwen3.6 27B'), 'Qwen 3.6-27B', 'qwen3.6-27b', 'Qwen/Qwen3.6-27B'], QBLOG + 'qwen3.6-27b', '2026-04-22', '2026-04-22', weights='2026-04-22', params=(27_781 * M_, 27_781 * M_), price=(0.6, 3.6), efforts=T)

# Qwen3.7 (closed; Max-Preview is thinking-only)
M('Qwen3.7-Max',
	V('Qwen3.7-Max-Preview', ['Qwen3.7 Max Preview', 'qwen3.7-max-preview', 'qwen3.7-max-2026-05-17'], CHANGELOG, '2026-05-19', '2026-05-25', price=(2.5, 7.5)),
	V('Qwen3.7-Max', ['Qwen3.7 Max', 'Qwen3.7-Max (May 2026)', 'qwen3.7-max-2026-05-20'], QBLOG + 'qwen3.7', '2026-05-19', '2026-05-20', price=(2.5, 7.5), efforts=T),
	V('Qwen3.7-Max (Jun 2026)', ['Qwen3.7-Max (June 2026)', 'qwen3.7-max-2026-06-08'], CHANGELOG, '2026-06-10', '2026-06-10', price=(2.5, 7.5), efforts=T),
)
S('Qwen3.7-Plus', ['Qwen3.7 Plus', 'qwen3.7-plus', 'qwen3.7-plus-2026-05-26'], QBLOG + 'qwen3.7-plus', '2026-06-01', '2026-06-01', price=(0.4, 1.6), efforts=T)
S('Qwen3.7-Flash', ['Qwen3.7 Flash', 'qwen3.7-flash', 'qwen3.7-flash-2026-07-15'], QC + 'qwen3.7-flash', '2026-07-25', '2026-07-25', price=(0.03, 0.13), efforts=T)

# Qwen3.8 (Qwen3.8-2.4T-A95B is the open-weights Qwen3.8-Max, Qwen3.8-Flash the hosted Qwen3.8-Flash-Next, a preview of the
# Qwen4 architecture; Qwen3.8-Omni-Flash uses that architecture but its base is not disclosed)
QWEN38 = QBLOG + 'qwen3.8'
P38_MAX = (2_446_183 * M_, 95 * B)
M('Qwen3.8-Max',
	V('Qwen3.8-Max-Preview', ['Qwen3.8 Max Preview', 'qwen3.8-max-preview'], QWEN38, '2026-07-19', '2026-07-19', params=P38_MAX),
	V('Qwen3.8-Max', ['Qwen3.8 Max', 'Qwen 3.8 Max', 'Qwen3.8-Max (Aug 2026)'], QWEN38, '2026-07-19', '2026-08-03', params=P38_MAX, price=(2, 6), efforts=TLMX),
	V('Qwen3.8-2.4T-A95B', ['Qwen3.8 2.4T A95B', 'qwen3.8-2.4t-a95b', 'Qwen/Qwen3.8-2.4T-A95B'], HF + 'Qwen3.8-2.4T-A95B', '2026-08-12', '2026-08-12', weights='2026-08-12', params=P38_MAX, price=(2, 6), efforts=LMX),
	V('Qwen3.8-Max (Sep 2026)', ['Qwen3.8 Max (0902)', 'Qwen3.8-Max-0902', 'qwen3.8-max-0902', 'qwen3.8-max-2026-09-02'], CHANGELOG, '2026-09-02', '2026-09-02', params=P38_MAX, price=(2, 6), efforts=TLMX),
)
S('Qwen3.8-27B', ['Qwen3.8 27B', 'Qwen 3.8 27B', 'Qwen3.8 27B (Xhigh)', 'Qwen3.8 27B (Medium)', 'Qwen3.8 27B (Low)', 'Qwen3.8 27B (Non-reasoning)', 'qwen3.8-27b', 'Qwen/Qwen3.8-27B'], HF + 'Qwen3.8-27B', '2026-08-14', '2026-08-14', weights='2026-08-14', params=(27_781 * M_, 27_781 * M_), price=(0.5, 3), efforts=TLMX)
P38_FLASH = (180 * B, 6 * B)
M('Qwen3.8-Flash',
	V('Qwen3.8-Flash-Next', ['Qwen3.8 Flash Next', 'qwen3.8-flash-next', 'Qwen/Qwen3.8-Flash-Next'], HF + 'Qwen3.8-Flash-Next', '2026-08-26', '2026-08-26', weights='2026-08-26', params=P38_FLASH, price=(0.15, 0.47), efforts=TLMX),
	V('Qwen3.8-Flash', ['Qwen3.8 Flash', 'qwen3.8-flash'], QC + 'qwen3.8-flash', '2026-08-26', '2026-08-26', params=P38_FLASH, price=(0.15, 0.47), efforts=TLMX),
)
S('Qwen3.8-Omni-Flash', ['Qwen3.8 Omni Flash', 'qwen3.8-omni-flash'], CHANGELOG, '2026-09-18', '2026-09-18', price=(0.15, 0.47), efforts=TLMX)

# SeaLLMs (DAMO Academy, Southeast Asian languages; v2 is built on Mistral 7B, v2.5 on Gemma 7B)
S('SeaLLM-7B-v2', ['SeaLLM 7B v2', 'SeaLLMs/SeaLLM-7B-v2'], 'https://huggingface.co/SeaLLMs/SeaLLM-7B-v2', '2024-02-02', '2024-02-02', weights='2024-02-02', params=(7_376 * M_, 7_376 * M_))
S('SeaLLM-7B-v2.5', ['SeaLLM 7B v2.5', 'SeaLLMs/SeaLLM-7B-v2.5'], 'https://huggingface.co/SeaLLMs/SeaLLM-7B-v2.5', '2024-04-03', '2024-04-03', weights='2024-04-03', params=(8_538 * M_, 8_538 * M_))

write('alibaba')
