"""Generates data/models/openai/*.json (one file per model).

All the data lives in this script: edit it and run `python scripts/models/gen_openai.py`.
Existing JSON files are deleted and rewritten, so manual edits to them are lost.
"""

from common import B, M, N, S, V, write

LMH = ['Low', 'Medium', 'High']
MLMH = ['Minimal', 'Low', 'Medium', 'High']
NLMH = ['None', 'Low', 'Medium', 'High']
LMHX = ['Low', 'Medium', 'High', 'Xhigh']
MHX = ['Medium', 'High', 'Xhigh']
NLMHX = ['None', 'Low', 'Medium', 'High', 'Xhigh']
LMHXM = ['Low', 'Medium', 'High', 'Xhigh', 'Max']
NLMHXM = ['None', 'Low', 'Medium', 'High', 'Xhigh', 'Max']

NOTES = 'https://help.openai.com/en/articles/9624314-model-release-notes'
CHATGPT_NOTES = 'https://help.openai.com/en/articles/6825453-chatgpt-release-notes'
DOCS = 'https://developers.openai.com/api/docs/models/'

# GPT-2 (all sizes are versions: deliberate exception for GPT-2, GPT-3, T5, PaLM and PaLM 2 only, other sizes are separate models)
GPT2 = 'https://openai.com/index/better-language-models/'
M('GPT-2',
	V('GPT-2 (124M)', ['GPT-2 124M', 'GPT-2 117M', 'GPT-2 Small', 'gpt2'], GPT2, '2019-02-14', '2019-02-14', id='gpt-2-124m', weights='2019-02-14', params=(124_000_000, 124_000_000)),
	V('GPT-2 (355M)', ['GPT-2 355M', 'GPT-2 345M', 'GPT-2 Medium', 'gpt2-medium'], GPT2, '2019-02-14', '2019-05-03', id='gpt-2-355m', weights='2019-05-03', params=(355_000_000, 355_000_000)),
	V('GPT-2 (774M)', ['GPT-2 774M', 'GPT-2 762M', 'GPT-2 Large', 'gpt2-large'], 'https://openai.com/index/gpt-2-6-month-follow-up/', '2019-02-14', '2019-08-20', id='gpt-2-774m', weights='2019-08-20', params=(774_000_000, 774_000_000)),
	V('GPT-2 (1.5B)', ['GPT-2', 'GPT-2 1.5B', 'GPT-2 XL', 'gpt2-xl'], 'https://openai.com/index/gpt-2-1-5b-release/', '2019-02-14', '2019-11-05', id='gpt-2-1-5b', weights='2019-11-05', params=(1_500_000_000, 1_500_000_000)),
)

# GPT-3 (all sizes as versions, same exception as GPT-2, plus Codex, InstructGPT and the 2023 base replacements)
GPT3 = 'https://arxiv.org/abs/2005.14165'
INSTRUCT = 'https://openai.com/index/instruction-following/'
FT_UPDATE = 'https://openai.com/index/gpt-3-5-turbo-fine-tuning-and-api-updates/'
M('GPT-3',
	V('GPT-3 125M', ['GPT-3 Small'], GPT3, '2020-05-28', None, params=(125_000_000, 125_000_000)),
	V('GPT-3 350M (ada)', ['GPT-3 350M', 'GPT-3 Medium', 'GPT-3 ada', 'ada'], GPT3, '2020-05-28', '2020-06-11', id='gpt-3-350m', params=(356_000_000, 356_000_000), price=(0.4, 0.4)),
	V('GPT-3 760M', ['GPT-3 Large'], GPT3, '2020-05-28', None, params=(760_000_000, 760_000_000)),
	V('GPT-3 1.3B (babbage)', ['GPT-3 1.3B', 'GPT-3 XL', 'GPT-3 XL (babbage)', 'GPT-3 babbage', 'babbage'], GPT3, '2020-05-28', '2020-06-11', id='gpt-3-1-3b', params=(1_320_000_000, 1_320_000_000), price=(0.5, 0.5)),
	V('GPT-3 2.7B', [], GPT3, '2020-05-28', None, params=(2_650_000_000, 2_650_000_000)),
	V('GPT-3 6.7B (curie)', ['GPT-3 6.7B', 'GPT-3 curie', 'curie'], GPT3, '2020-05-28', '2020-06-11', id='gpt-3-6-7b', params=(6_660_000_000, 6_660_000_000), price=(2, 2)),
	V('GPT-3 13B', [], GPT3, '2020-05-28', None, params=(12_850_000_000, 12_850_000_000)),
	V('GPT-3 175B (davinci)', ['GPT-3', 'GPT-3 175B', 'GPT-3 davinci', 'davinci'], GPT3, '2020-05-28', '2020-06-11', id='gpt-3-175b', params=(174_600_000_000, 174_600_000_000), price=(20, 20)),
	V('Codex (12B)', ['Codex', 'Codex 12B', 'code-cushman-001', 'code-cushman-002'], 'https://openai.com/index/openai-codex/', '2021-07-07', '2021-08-10', id='codex-12b', params=(12 * B, 12 * B)),
	V('InstructGPT 350M', ['text-ada-001'], INSTRUCT, '2022-01-27', '2022-01-27', params=(350_000_000, 350_000_000), price=(0.4, 0.4)),
	V('InstructGPT 1.3B', ['text-babbage-001'], INSTRUCT, '2022-01-27', '2022-01-27', params=(1_300_000_000, 1_300_000_000), price=(0.5, 0.5)),
	V('InstructGPT 6B', ['text-curie-001'], INSTRUCT, '2022-01-27', '2022-01-27', params=(6 * B, 6 * B), price=(2, 2)),
	V('InstructGPT 175B', ['InstructGPT', 'text-davinci-001'], INSTRUCT, '2022-01-27', '2022-01-27', params=(175 * B, 175 * B), price=(20, 20)),
	V('davinci-002', [], FT_UPDATE, '2023-08-22', '2023-08-22', price=(2, 2)),
	V('babbage-002', [], FT_UPDATE, '2023-08-22', '2023-08-22', price=(0.4, 0.4)),
)

# GPT-3.5 (completions era)
GPT35_DOCS = 'https://platform.openai.com/docs/models/gpt-3-5'
M('GPT-3.5',
	V('code-davinci-002', ['Codex (code-davinci-002)'], GPT35_DOCS, '2022-03-15', '2022-03-15'),
	V('GPT-3.5 (text-davinci-002)', ['GPT-3.5 (davinci-002)', 'text-davinci-002'], GPT35_DOCS, '2022-03-15', '2022-03-15', id='text-davinci-002', price=(20, 20)),
	V('GPT-3.5 (text-davinci-003)', ['GPT-3.5 (davinci-003)', 'text-davinci-003'], GPT35_DOCS, '2022-11-28', '2022-11-28', id='text-davinci-003', price=(20, 20)),
)

# GPT-3.5 Turbo
FUNC = 'https://openai.com/index/function-calling-and-other-api-updates/'
DEVDAY = 'https://openai.com/index/new-models-and-developer-products-announced-at-devday/'
JAN24 = 'https://openai.com/index/new-embedding-models-and-api-updates/'
P35 = (20 * B, None)
M('GPT-3.5 Turbo',
	V('ChatGPT (Nov 2022)', ['ChatGPT', 'text-davinci-002-render-sha'], 'https://openai.com/index/chatgpt/', '2022-11-30', '2022-11-30'),
	V('GPT-3.5 Turbo (Mar 2023)', ['GPT-3.5 Turbo (0301)', 'gpt-3.5-turbo-0301'], 'https://openai.com/index/introducing-chatgpt-and-whisper-apis/', '2023-03-01', '2023-03-01', params=P35, price=(2, 2)),
	V('GPT-3.5 Turbo (Jun 2023)', ['GPT-3.5 Turbo (June 2023)', 'GPT-3.5 Turbo (0613)', 'gpt-3.5-turbo-0613', 'gpt-3.5-turbo-16k-0613', 'gpt-3.5-turbo-16k'], FUNC, '2023-06-13', '2023-06-13', params=P35, price=(1.5, 2)),
	V('GPT-3.5 Turbo Instruct', ['gpt-3.5-turbo-instruct', 'gpt-3.5-turbo-instruct-0914'], DOCS + 'gpt-3.5-turbo', '2023-09-18', '2023-09-18', params=P35, price=(1.5, 2)),
	V('GPT-3.5 Turbo (Nov 2023)', ['GPT-3.5 Turbo (1106)', 'gpt-3.5-turbo-1106'], DEVDAY, '2023-11-06', '2023-11-06', params=P35, price=(1, 2)),
	V('GPT-3.5 Turbo (Jan 2024)', ['GPT-3.5 Turbo (0125)', 'gpt-3.5-turbo-0125'], JAN24, '2024-01-25', '2024-01-25', params=P35, price=(0.5, 1.5)),
)

# GPT-4 family
GPT4 = 'https://openai.com/index/gpt-4-research/'
P4 = (1_800 * B, None)
M('GPT-4',
	V('GPT-4 (Mar 2023)', ['GPT-4 (0314)', 'gpt-4-0314', 'gpt-4-32k-0314'], GPT4, '2023-03-14', '2023-03-14', params=P4, price=(30, 60)),
	V('GPT-4V', ['GPT-4 Vision', 'GPT-4V(ision)'], 'https://openai.com/index/gpt-4v-system-card/', '2023-03-14', '2023-09-25', params=P4),
	V('GPT-4 (Jun 2023)', ['GPT-4 (June 2023)', 'GPT-4 (0613)', 'gpt-4-0613', 'gpt-4-32k-0613'], FUNC, '2023-06-13', '2023-06-13', params=P4, price=(30, 60)),
)
M('GPT-4 Turbo',
	V('GPT-4 Turbo (Nov 2023)', ['GPT-4 Turbo Preview (Nov 2023)', 'gpt-4-1106-preview'], DEVDAY, '2023-11-06', '2023-11-06', price=(10, 30)),
	V('GPT-4 Turbo with Vision (Nov 2023)', ['gpt-4-vision-preview', 'gpt-4-1106-vision-preview'], DEVDAY, '2023-11-06', '2023-11-06', price=(10, 30)),
	V('GPT-4 Turbo (Jan 2024)', ['GPT-4 Turbo Preview (January 2024)', 'GPT-4 Turbo Preview (Jan 2024)', 'gpt-4-0125-preview', 'gpt-4-turbo-preview'], JAN24, '2024-01-25', '2024-01-25', price=(10, 30)),
	V('GPT-4 Turbo (Apr 2024)', ['gpt-4-turbo-2024-04-09'], 'https://x.com/OpenAI/status/1778574613813006610', '2024-04-09', '2024-04-09', price=(10, 30)),
)
M('GPT-4o',
	V('GPT-4o (May 2024)', ["GPT-4o (May '24)", 'gpt-4o-2024-05-13'], 'https://openai.com/index/hello-gpt-4o/', '2024-05-13', '2024-05-13', price=(5, 15)),
	V('GPT-4o (Aug 2024)', ["GPT-4o (Aug '24)", 'gpt-4o-2024-08-06'], 'https://openai.com/index/introducing-structured-outputs-in-the-api/', '2024-08-06', '2024-08-06', price=(2.5, 10)),
	V('GPT-4o (Nov 2024)', ["GPT-4o (Nov '24)", 'gpt-4o-2024-11-20', 'chatgpt-4o-11-20-2024'], NOTES, '2024-11-20', '2024-11-20', price=(2.5, 10)),
	V('GPT-4o (Jan 2025)', ['ChatGPT-4o (Jan 2025)', 'GPT-4o (ChatGPT)', 'chatgpt-4o-01-29-2025'], NOTES, '2025-01-29', '2025-01-29', price=(5, 15)),
	V('GPT-4o (Mar 2025)', ['ChatGPT-4o (Mar 2025)', 'GPT-4o (March 2025, chatgpt-4o-latest)', 'chatgpt-4o-03-27-2025'], CHATGPT_NOTES, '2025-03-27', '2025-03-27', price=(5, 15)),
)
S('GPT-4o mini', ['GPT-4o Mini', 'gpt-4o-mini', 'gpt-4o-mini-2024-07-18'], 'https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/', '2024-07-18', '2024-07-18', price=(0.15, 0.6))
S('GPT-4.5', ['GPT-4.5 (Preview)', 'GPT-4.5 Preview', 'GPT-4.5 Preview (Feb 2025)', 'gpt-4.5-preview', 'gpt-4.5-preview-2025-02-27'], 'https://openai.com/index/introducing-gpt-4-5/', '2025-02-27', '2025-02-27', price=(75, 150))
GPT41 = 'https://openai.com/index/gpt-4-1/'
S('GPT-4.1', ['gpt-4.1', 'gpt-4.1-2025-04-14'], GPT41, '2025-04-14', '2025-04-14', price=(2, 8))
S('GPT-4.1 mini', ['GPT-4.1 Mini', 'gpt-4.1-mini', 'gpt-4.1-mini-2025-04-14'], GPT41, '2025-04-14', '2025-04-14', price=(0.4, 1.6))
S('GPT-4.1 nano', ['GPT-4.1 Nano', 'gpt-4.1-nano', 'gpt-4.1-nano-2025-04-14'], GPT41, '2025-04-14', '2025-04-14', price=(0.1, 0.4))

# o-series
M('o1',
	V('o1-preview', ['o1 Preview', 'o1-preview-2024-09-12'], 'https://openai.com/index/introducing-openai-o1-preview/', '2024-09-12', '2024-09-12', price=(15, 60)),
	V('o1', ['o1-2024-12-17'], 'https://openai.com/index/learning-to-reason-with-llms/', '2024-09-12', '2024-12-05', price=(15, 60), efforts=LMH),
	V('o1-pro', ['o1 Pro', 'o1 pro mode', 'o1-pro-2025-03-19'], 'https://openai.com/index/introducing-chatgpt-pro/', '2024-12-05', '2024-12-05', price=(150, 600), efforts=LMH),
)
S('o1-mini', ['o1 mini', 'o1-mini-2024-09-12'], 'https://openai.com/index/openai-o1-mini-advancing-cost-efficient-reasoning/', '2024-09-12', '2024-09-12', price=(1.1, 4.4))
S('o3-mini', ['o3 mini', 'o3-mini-2025-01-31'], 'https://openai.com/index/openai-o3-mini/', '2024-12-20', '2025-01-31', price=(1.1, 4.4), efforts=LMH)
O3 = 'https://openai.com/index/introducing-o3-and-o4-mini/'
CODEX = 'https://openai.com/index/introducing-codex/'
M('o3',
	V('o3-preview', ['o3 Preview', 'o3 (Preview)'], 'https://arcprize.org/blog/oai-o3-pub-breakthrough', '2024-12-20', None, efforts=['Low', 'High']),
	V('o3', ['o3-2025-04-16'], O3, '2024-12-20', '2025-04-16', price=(2, 8), efforts=LMH),
	V('o3-deep-research', ['o3 Deep Research', 'Deep Research', 'OpenAI Deep Research', 'o3-deep-research-2025-06-26'], 'https://openai.com/index/introducing-deep-research/', '2025-02-02', '2025-02-02', price=(10, 40)),
	V('codex-1', [], CODEX, '2025-05-16', '2025-05-16'),
	V('o3-pro', ['o3 Pro', 'o3-pro-2025-06-10'], NOTES, '2025-06-10', '2025-06-10', price=(20, 80), efforts=LMH),
)
M('o4-mini',
	V('o4-mini', ['o4 mini', 'o4-mini-2025-04-16'], O3, '2025-04-16', '2025-04-16', price=(1.1, 4.4), efforts=LMH),
	V('o4-mini-deep-research', ['o4-mini Deep Research', 'o4-mini-deep-research-2025-06-26'], DOCS + 'o4-mini-deep-research', '2025-04-24', '2025-04-24', price=(2, 8)),
	V('codex-mini', ['codex-mini-latest'], CODEX, '2025-05-16', '2025-05-16', price=(1.5, 6), efforts=LMH),
)

# gpt-oss
OSS = 'https://openai.com/index/introducing-gpt-oss/'
S('gpt-oss-120b', ['GPT OSS 120B', 'openai/gpt-oss-120b'], OSS, '2025-08-05', '2025-08-05', weights='2025-08-05', params=(116_830_000_000, 5_130_000_000), price=(0.15, 0.6), efforts=LMH)
S('gpt-oss-20b', ['GPT OSS 20B', 'openai/gpt-oss-20b'], OSS, '2025-08-05', '2025-08-05', weights='2025-08-05', params=(20_910_000_000, 3_610_000_000), price=(0.05, 0.22), efforts=LMH)

# GPT-5
GPT5 = 'https://openai.com/index/introducing-gpt-5/'
M('GPT-5',
	V('GPT-5', ['GPT-5 Thinking', 'gpt-5', 'gpt-5-2025-08-07'], GPT5, '2025-08-07', '2025-08-07', price=(1.25, 10), efforts=MLMH),
	V('GPT-5 Instant', ['GPT-5 Chat', 'GPT-5 (ChatGPT)', 'gpt-5-chat', 'gpt-5-chat-latest'], GPT5, '2025-08-07', '2025-08-07', price=(1.25, 10)),
	V('GPT-5 Pro', ['gpt-5-pro', 'gpt-5-pro-2025-10-06'], GPT5, '2025-08-07', '2025-08-07', price=(15, 120), efforts=['High']),
	V('GPT-5 Codex', ['GPT-5-Codex', 'gpt-5-codex'], 'https://openai.com/index/introducing-upgrades-to-codex/', '2025-09-15', '2025-09-15', price=(1.25, 10), efforts=LMH),
)
S('GPT-5 mini', ['GPT-5 Mini', 'gpt-5-mini', 'gpt-5-mini-2025-08-07'], GPT5, '2025-08-07', '2025-08-07', price=(0.25, 2), efforts=MLMH)
S('GPT-5 nano', ['GPT-5 Nano', 'gpt-5-nano', 'gpt-5-nano-2025-08-07'], GPT5, '2025-08-07', '2025-08-07', price=(0.05, 0.4), efforts=MLMH)

# GPT-5.1
GPT51 = 'https://openai.com/index/gpt-5-1/'
GPT51_DEV = 'https://openai.com/index/gpt-5-1-for-developers/'
M('GPT-5.1',
	V('GPT-5.1', ['GPT-5.1 Thinking', 'gpt-5.1', 'gpt-5.1-2025-11-13'], GPT51, '2025-11-12', '2025-11-12', price=(1.25, 10), efforts=NLMH),
	V('GPT-5.1 Instant', ['GPT-5.1 Chat', 'gpt-5.1-chat-latest'], GPT51, '2025-11-12', '2025-11-12', price=(1.25, 10)),
	V('GPT-5.1 Codex', ['GPT-5.1-Codex', 'gpt-5.1-codex'], GPT51_DEV, '2025-11-13', '2025-11-13', price=(1.25, 10), efforts=LMH),
	V('GPT-5.1 Codex Max', ['GPT-5.1-Codex-Max', 'gpt-5.1-codex-max'], 'https://openai.com/index/gpt-5-1-codex-max/', '2025-11-19', '2025-11-19', price=(1.25, 10), efforts=LMHX),
)
M('GPT-5.1 mini',
	V('GPT-5.1 Codex mini', ['GPT-5.1 Codex Mini', 'GPT-5.1-Codex-Mini', 'gpt-5.1-codex-mini'], GPT51_DEV, '2025-11-13', '2025-11-13', price=(0.25, 2), efforts=LMH),
)

# GPT-5.2
GPT52 = 'https://openai.com/index/introducing-gpt-5-2/'
M('GPT-5.2',
	V('GPT-5.2', ['GPT-5.2 Thinking', 'gpt-5.2', 'gpt-5.2-2025-12-11', 'robin-alpha-next-2025-11-24'], GPT52, '2025-12-11', '2025-12-11', price=(1.75, 14), efforts=NLMHX),
	V('GPT-5.2 Instant', ['GPT-5.2 Chat', 'gpt-5.2-chat-latest'], GPT52, '2025-12-11', '2025-12-11', price=(1.75, 14)),
	V('GPT-5.2 Pro', ['GPT-5.2 Pro (web)', 'GPT-5.2 Pro (Web App)', 'gpt-5.2-pro', 'gpt-5.2-pro-2025-12-11', 'gpt-5.2-pro-2025-12-11-webapp', 'robin-alpha-pro-2025-11-24'], GPT52, '2025-12-11', '2025-12-11', price=(21, 168), efforts=MHX),
	V('GPT-5.2 Codex', ['GPT-5.2-Codex', 'gpt-5.2-codex'], 'https://openai.com/index/introducing-gpt-5-2-codex/', '2025-12-18', '2025-12-18', price=(1.75, 14), efforts=LMHX),
)

# GPT-5.3
M('GPT-5.3',
	V('GPT-5.3 Codex', ['GPT-5.3-Codex', 'gpt-5.3-codex'], 'https://openai.com/index/introducing-gpt-5-3-codex/', '2026-02-05', '2026-02-05', price=(1.75, 14), efforts=LMHX),
	V('GPT-5.3 Instant', ['GPT-5.3 Chat', 'gpt-5.3-chat-latest'], DOCS + 'gpt-5.3-chat-latest', '2026-03-03', '2026-03-03', price=(1.75, 14)),
)
M('GPT-5.3 mini',
	V('GPT-5.3 Codex Spark', ['GPT-5.3-Codex-Spark', 'gpt-5.3-codex-spark'], 'https://openai.com/index/introducing-gpt-5-3-codex-spark/', '2026-02-12', '2026-02-12', efforts=LMHX),
)

# GPT-5.4
GPT54 = 'https://openai.com/index/introducing-gpt-5-4/'
GPT54_SMALL = 'https://openai.com/index/introducing-gpt-5-4-mini-and-nano/'
M('GPT-5.4',
	V('GPT-5.4', ['GPT-5.4 Thinking', 'gpt-5.4', 'gpt-5.4-2026-03-05'], GPT54, '2026-03-05', '2026-03-05', price=(2.5, 15), efforts=NLMHX),
	V('GPT-5.4 Pro', ['GPT-5.4 Pro (web)', 'GPT-5.4 Pro (Web App)', 'gpt-5.4-pro', 'gpt-5.4-pro-2026-03-05', 'gpt-5.4-pro-2026-03-05-web-app'], GPT54, '2026-03-05', '2026-03-05', price=(30, 180), efforts=MHX),
	V('GPT-5.4 Cyber', ['GPT-5.4-Cyber', 'gpt-5.4-cyber'], 'https://openai.com/index/scaling-trusted-access-for-cyber-defense/', '2026-04-14', None, efforts=NLMHX),
)
S('GPT-5.4 mini', ['GPT-5.4 Mini', 'gpt-5.4-mini', 'gpt-5.4-mini-2026-03-17'], GPT54_SMALL, '2026-03-17', '2026-03-17', price=(0.75, 4.5), efforts=NLMHX)
S('GPT-5.4 nano', ['GPT-5.4 Nano', 'gpt-5.4-nano', 'gpt-5.4-nano-2026-03-17'], GPT54_SMALL, '2026-03-17', '2026-03-17', price=(0.2, 1.25), efforts=NLMHX)
S('GPT-Rosalind', ['GPT Rosalind', 'gpt-rosalind-research'], 'https://openai.com/index/introducing-gpt-rosalind', '2026-04-16', None, price=(5, 25))

# GPT-5.5
GPT55 = 'https://openai.com/index/introducing-gpt-5-5/'
M('GPT-5.5',
	V('GPT-5.5', ['GPT-5.5 Thinking', 'gpt-5.5', 'gpt-5.5-2026-04-23', 'gpt-5.5-pre-release'], GPT55, '2026-04-23', '2026-04-23', price=(5, 30), efforts=NLMHX),
	V('GPT-5.5 Pro', ['GPT-5.5 Pro pre-release', 'gpt-5.5-pro', 'gpt-5.5-pro-2026-04-23', 'gpt-5.5-pro-pre-release'], GPT55, '2026-04-23', '2026-04-23', price=(30, 180), efforts=MHX),
	V('GPT-5.5 Instant (May 2026)', ['gpt-5-5-instant-05-26'], 'https://openai.com/index/gpt-5-5-instant/', '2026-05-05', '2026-05-05', price=(5, 30)),
	V('GPT-5.5 Cyber', ['GPT-5.5-Cyber', 'gpt-5.5-cyber'], 'https://openai.com/index/gpt-5-5-with-trusted-access-for-cyber/', '2026-05-07', None, price=(12.5, 75), efforts=NLMHX),
	V('GPT-5.5 Rosalind', ['GPT-Rosalind-5.5', 'GPT-Rosalind 5.5'], 'https://openai.com/index/introducing-new-capabilities-to-gpt-rosalind/', '2026-06-03', None, efforts=NLMHX),
	V('GPT-5.5 Instant (Jun 2026)', ['GPT-5.5 Instant (June 2026)', 'gpt-5-5-instant-06-26'], CHATGPT_NOTES, '2026-06-25', '2026-06-25', price=(5, 30)),
)

# GPT-5.6
GPT56 = 'https://openai.com/index/previewing-gpt-5-6-sol/'
M('GPT-5.6 Sol',
	V('GPT-5.6 Sol', ['GPT-5.6', 'gpt-5.6', 'gpt-5.6-sol', 'gpt-daybreak-blue-latest'], GPT56, '2026-06-26', '2026-07-09', price=(4, 20), efforts=NLMHXM),
	V('GPT-5.6 Sol Pro', ['GPT-5.6 Sol (pro)', 'GPT-5.6 Pro', 'gpt-5.6-sol-pro'], GPT56, '2026-06-26', '2026-07-09', efforts=['Xhigh', 'Max']),
	V('GPT-5.6 Cyber', ['GPT-5.6-Cyber', 'gpt-5.6-cyber', 'gpt-daybreak-red-latest'], 'https://openai.com/index/expanding-daybreak-as-the-cyber-defense-window-narrows/', '2026-08-10', None, price=(12.5, 75), efforts=NLMHXM),
)
S('GPT-5.6 Terra', ['gpt-5.6-terra'], GPT56, '2026-06-26', '2026-07-09', price=(2, 12), efforts=NLMHXM)
S('GPT-5.6 Luna', ['gpt-5.6-luna'], GPT56, '2026-06-26', '2026-07-09', price=(0.2, 1.2), efforts=NLMHXM)

# GPT-6
ASTRA = 'https://openai.com/index/gpt-6-astra/'
GPT6_SL = 'https://openai.com/index/introducing-gpt-6-sol-and-luna/'
M('GPT-6 Astra',
	V('GPT-6 Astra', ['GPT-6', 'gpt-6-astra'], ASTRA, '2026-08-01', '2026-09-04', price=(10, 50), efforts=LMHXM),
	V('GPT-6 Astra Pro', ['GPT-6 Astra (pro)', 'GPT-6 Pro', 'gpt-6-astra-pro'], ASTRA, '2026-08-01', '2026-09-04', efforts=['Max']),
	V('GPT-6 Astra Law', ['Astra for Law', 'gpt-6-astra-law'], 'https://openai.com/index/astra-for-law/', '2026-09-17', None, price=(10, 50), efforts=LMHXM),
	V('GPT-6 Rosalind', ['GPT-Rosalind-6', 'GPT-6 Rosalind Research', 'gpt-6-rosalind'], 'https://deploymentsafety.openai.com/gpt-rosalind-6', '2026-10-30', None, price=(15, 75), efforts=LMHXM),
)
S('GPT-6 Sol', ['gpt-6-sol'], GPT6_SL, '2026-09-22', '2026-09-22', price=(2, 10), efforts=NLMHXM)
S('GPT-6 Luna', ['gpt-6-luna'], GPT6_SL, '2026-09-22', '2026-09-22', price=(0.1, 0.5), efforts=NLMHXM)
S('GPT-6.1 Sol', ['gpt-6.1-sol'], 'https://openai.com/index/introducing-gpt-6-1-sol/', '2026-09-29', '2026-09-29', price=(2, 10), efforts=LMHXM)
S('GPT-6.1 Terra', ['gpt-6.1-terra'], DOCS + 'gpt-6.1-terra', '2026-10-26', '2026-10-26', price=(1, 5), efforts=NLMHXM)

write('openai')
