"""Generates data/models/google/*.json (one file per model).

Covers every Google lab: Google Research, Google Brain, DeepMind and Google DeepMind.
All the data lives in this script: edit it and run `python scripts/models/gen_google.py`.
Existing JSON files are deleted and rewritten, so manual edits to them are lost.
"""

from common import B, M, N, S, V, write

M_ = 1_000_000
T = ['', 'Thinking']							# Thinking on/off switch (thinking_budget = 0 or open-weights <|think|> token)
LH = ['Low', 'High']							# Gemini 3 Pro
LMH = ['Low', 'Medium', 'High']					# thinking_level without minimal
MLMH = ['Minimal', 'Low', 'Medium', 'High']		# thinking_level with minimal

ARXIV = 'https://arxiv.org/abs/'
DOCS = 'https://ai.google.dev/gemini-api/docs/models/'
CHANGELOG = 'https://ai.google.dev/gemini-api/docs/changelog'
GMODELS = 'https://blog.google/innovation-and-ai/models-and-research/gemini-models/'
HF = 'https://huggingface.co/google/'

# T5 (all sizes are versions: same exception as GPT-2 / GPT-3 / PaLM / PaLM 2, Flan-T5 included)
T5 = 'https://research.google/blog/exploring-transfer-learning-with-t5-the-text-to-text-transfer-transformer/'
FLAN = ARXIV + '2210.11416'
T5_SIZES = [
	('Small', 'Small', 60 * M_, 77 * M_),
	('Base', 'Base', 220 * M_, 248 * M_),
	('Large', 'Large', 770 * M_, 783 * M_),
	('3B', 'XL', 2_800 * M_, 2_850 * M_),
	('11B', 'XXL', 11 * B, 11_300 * M_),
]
M('T5',
	*[V(f'T5-{size}', [f'T5 {size}', f't5-{size.lower()}', f'google-t5/t5-{size.lower()}'] + (['T5'] if size == '11B' else []), T5, '2019-10-23', '2019-10-23', weights='2019-10-23', params=(p, p)) for size, _, p, _ in T5_SIZES],
	*[V(f'Flan-T5-{flan}', [f'Flan-T5 {flan}', f'FLAN-T5 {flan}', f'flan-t5-{flan.lower()}', f'google/flan-t5-{flan.lower()}'] + (['Flan-T5 11B', 'Flan-T5'] if flan == 'XXL' else []), FLAN, '2022-10-20', '2022-10-20', weights='2022-10-20', params=(fp, fp)) for _, flan, _, fp in T5_SIZES],
)

# Google Brain / Google Research language models (2020-2022)
S('Meena', ['Meena 2.6B'], ARXIV + '2001.09977', '2020-01-28', None, params=(2_600 * M_, 2_600 * M_))
S('mT5-XXL', ['mT5 XXL', 'mt5-xxl', 'google/mt5-xxl'], ARXIV + '2010.11934', '2020-10-22', '2020-10-22', weights='2020-10-22', params=(13 * B, 13 * B))
S('Switch-C', ['Switch Transformer', 'Switch-C 2048', 'switch-c-2048', 'google/switch-c-2048'], ARXIV + '2101.03961', '2021-01-11', '2022-11-04', weights='2022-11-04', params=(1_571 * B, None))
S('ByT5-XXL', ['ByT5 XXL', 'byt5-xxl', 'google/byt5-xxl'], ARXIV + '2105.13626', '2021-05-28', '2021-05-28', weights='2021-05-28', params=(12_900 * M_, 12_900 * M_))
M('LaMDA',	# Bard launched on a lightweight LaMDA, size not disclosed
	V('LaMDA', ['LaMDA 137B', 'LaMDA-PT'], 'https://blog.google/technology/ai/lamda/', '2021-05-18', None, params=(137 * B, 137 * B)),
	V('FLAN 137B', ['FLAN', 'FLAN (LaMDA-PT)'], ARXIV + '2109.01652', '2021-09-03', None, params=(137 * B, 137 * B)),
	V('Bard (LaMDA)', ['Bard (Mar 2023)'], 'https://blog.google/technology/ai/bard-google-ai-search-updates/', '2023-02-06', '2023-03-21'),
)
S('GLaM', ['GLaM (MoE)', 'GLaM 1.2T', 'GLaM 64B/64E'], ARXIV + '2112.06905', '2021-12-13', None, params=(1_200 * B, 96_600 * M_))
UL2 = ARXIV + '2205.05131'
M('UL2',
	V('UL2', ['UL2 20B', 'google/ul2'], UL2, '2022-05-10', '2022-06-16', weights='2022-06-16', params=(20 * B, 20 * B)),
	V('Flan-UL2', ['Flan-UL2 20B', 'FLAN-UL2', 'google/flan-ul2'], HF + 'flan-ul2', '2023-03-03', '2023-03-03', weights='2023-03-03', params=(19_500 * M_, 19_500 * M_)),
)

# PaLM (all sizes are versions, same exception as GPT-2 / GPT-3; Minerva, Flan-PaLM, U-PaLM and Med-PaLM are fine-tunes)
PALM = ARXIV + '2204.02311'
MINERVA = ARXIV + '2206.14858'
UPALM = ARXIV + '2210.11399'
M('PaLM',
	V('PaLM 8B', ['PaLM (8B)', 'PaLM-8B'], PALM, '2022-04-04', None, params=(8_600 * M_, 8_600 * M_)),
	V('PaLM 62B', ['PaLM (62B)', 'PaLM-62B'], PALM, '2022-04-04', None, params=(62_500 * M_, 62_500 * M_)),
	V('PaLM 540B', ['PaLM', 'PaLM (540B)', 'PaLM-540B'], PALM, '2022-04-04', None, params=(540_350 * M_, 540_350 * M_)),
	V('Minerva 8B', ['Minerva (8B)'], MINERVA, '2022-06-29', None, params=(8_600 * M_, 8_600 * M_)),
	V('Minerva 62B', ['Minerva (62B)'], MINERVA, '2022-06-29', None, params=(62_500 * M_, 62_500 * M_)),
	V('Minerva 540B', ['Minerva', 'Minerva (540B)'], MINERVA, '2022-06-29', None, params=(540_350 * M_, 540_350 * M_)),
	V('Flan-PaLM 8B', ['Flan-PaLM (8B)'], FLAN, '2022-10-20', None, params=(8_600 * M_, 8_600 * M_)),
	V('Flan-PaLM 62B', ['Flan-PaLM (62B)'], FLAN, '2022-10-20', None, params=(62_500 * M_, 62_500 * M_)),
	V('Flan-PaLM 540B', ['Flan-PaLM', 'Flan-PaLM (540B)'], FLAN, '2022-10-20', None, params=(540_350 * M_, 540_350 * M_)),
	V('U-PaLM 540B', ['U-PaLM', 'U-PaLM (540B)'], UPALM, '2022-10-20', None, params=(540_350 * M_, 540_350 * M_)),
	V('Flan-U-PaLM 540B', ['Flan-U-PaLM'], FLAN, '2022-10-20', None, params=(540_350 * M_, 540_350 * M_)),
	V('Med-PaLM', ['Med-PaLM 540B'], ARXIV + '2212.13138', '2022-12-26', None, params=(540_350 * M_, 540_350 * M_)),
)

# DeepMind language models (2021-2022)
GOPHER = ARXIV + '2112.11446'
S('Gopher 280B', ['Gopher', 'Gopher (280B)'], GOPHER, '2021-12-08', None, params=(280 * B, 280 * B))
S('Gopher 7.1B', ['Gopher (7.1B)'], GOPHER, '2021-12-08', None, params=(7_100 * M_, 7_100 * M_))
S('RETRO', ['RETRO 7.5B', 'RETRO-7B', 'Retro'], ARXIV + '2112.04426', '2021-12-08', None, params=(7_500 * M_, 7_500 * M_))
S('AlphaCode', ['AlphaCode 41B'], ARXIV + '2203.07814', '2022-02-02', None, params=(41_100 * M_, 41_100 * M_))
M('Chinchilla',
	V('Chinchilla', ['Chinchilla (70B)', 'Chinchilla 70B'], ARXIV + '2203.15556', '2022-03-29', None, params=(70 * B, 70 * B)),
	V('Sparrow', ['Sparrow 70B'], ARXIV + '2209.14375', '2022-09-22', None, params=(70 * B, 70 * B)),
)

# PaLM 2 (all sizes are versions, same exception as GPT-2 / GPT-3: paper sizes S/M/L, API sizes Gecko/Otter/Bison/Unicorn;
# PaLM API prices were per character, so none here)
PALM2 = 'https://blog.google/technology/ai/google-palm-2-ai-large-language-model/'
PALM2_PAPER = ARXIV + '2305.10403'
PALM2_SIZES = 'https://ai.google.dev/palm_docs/palm#model-sizes'
M('PaLM 2',
	V('Med-PaLM 2', ['Med-PaLM-2'], ARXIV + '2305.09617', '2023-03-14', None),
	V('PaLM 2-S', ['PaLM 2 S', 'PaLM-2-S', 'PaLM 2 Small'], PALM2_PAPER, '2023-05-10', None),
	V('PaLM 2-M', ['PaLM 2 M', 'PaLM-2-M', 'PaLM 2 Medium'], PALM2_PAPER, '2023-05-10', None),
	V('PaLM 2-L', ['PaLM 2', 'PaLM 2 L', 'PaLM-2-L', 'PaLM 2 Large'], PALM2_PAPER, '2023-05-10', None),
	V('Bard (PaLM 2)', ['Bard (May 2023)'], 'https://blog.google/technology/ai/google-bard-updates-io-2023/', '2023-05-10', '2023-05-10'),
	V('PaLM 2 Gecko', ['PaLM-2 Gecko'], PALM2_SIZES, '2023-05-10', None),
	V('PaLM 2 Otter', ['PaLM-2 Otter'], PALM2_SIZES, '2023-05-10', None),
	V('PaLM 2 Bison (text-bison@001)', ['PaLM 2 for Text', 'PaLM-2 Bison', 'PaLM 2 Bison', 'text-bison', 'text-bison@001', 'text-bison-001'], PALM2_SIZES, '2023-05-10', '2023-05-10', id='text-bison-001'),
	V('PaLM 2 Bison Chat (chat-bison@001)', ['PaLM 2 for Chat', 'PaLM 2 Chat', 'PALM-2', 'chat-bison', 'chat-bison@001', 'chat-bison-001'], PALM2_SIZES, '2023-05-10', '2023-05-10', id='chat-bison-001'),
	V('Codey (code-bison@001)', ['Codey', 'Codey for Code Generation', 'code-bison', 'code-bison@001', 'codechat-bison', 'codechat-bison@001'], PALM2, '2023-05-10', '2023-05-10', id='code-bison-001'),
	V('Codey (code-gecko@001)', ['Codey for Code Completion', 'code-gecko', 'code-gecko@001'], PALM2, '2023-05-10', '2023-06-07', id='code-gecko-001'),
	V('PaLM 2 Unicorn', ['PaLM 2 for Text Unicorn', 'PaLM-2 Unicorn', 'text-unicorn', 'text-unicorn@001'], 'https://cloud.google.com/blog/products/ai-machine-learning/palm-2-unicorn-and-more-generally-available', '2023-05-10', '2023-11-30'),
	V('PaLM 2 Bison (text-bison@002)', ['text-bison@002', 'text-bison-32k'], 'https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions', '2023-12-06', '2023-12-06', id='text-bison-002'),
	V('PaLM 2 Bison Chat (chat-bison@002)', ['chat-bison@002', 'chat-bison-32k'], 'https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions', '2023-12-06', '2023-12-06', id='chat-bison-002'),
	V('MedLM', ['MedLM Large', 'medlm-large'], 'https://cloud.google.com/blog/topics/healthcare-life-sciences/introducing-medlm-for-the-healthcare-industry', '2023-12-13', None),
	V('AMIE', ['Articulate Medical Intelligence Explorer', 'AMIE (PaLM 2)'], ARXIV + '2401.05654', '2024-01-11', None),
)

# Gemini 1.0
GEMINI1 = 'https://blog.google/technology/ai/google-gemini-ai/'
GEMINI1_REPORT = ARXIV + '2312.11805'
M('Gemini 1.0 Ultra',
	V('Gemini 1.0 Ultra', ['Gemini Ultra', 'Gemini Ultra 1.0', 'gemini-ultra', 'gemini-1.0-ultra', 'Gemini Advanced (Ultra 1.0)'], GEMINI1, '2023-05-10', '2024-02-08'),
	V('Med-Gemini-L 1.0', ['Med-Gemini', 'Med-Gemini-L'], ARXIV + '2404.18416', '2024-04-29', None),
)
M('Gemini 1.0 Pro',
	V('Bard (Gemini Pro)', ['Bard (Jan 2024, Gemini Pro)', 'Bard (Gemini Pro, Jan 2024)', 'bard-jan-24-gemini-pro'], 'https://blog.google/products/bard/google-bard-try-gemini-ai/', '2023-12-06', '2023-12-06'),
	V('AlphaCode 2', [], 'https://storage.googleapis.com/deepmind-media/AlphaCode2/AlphaCode2_Tech_Report.pdf', '2023-12-06', None),
	V('Gemini 1.0 Pro', ['Gemini Pro', 'Gemini Pro 1.0', 'gemini-pro', 'gemini-1.0-pro', 'gemini-1.0-pro-001'], GEMINI1, '2023-12-06', '2023-12-13', price=(0.5, 1.5)),
	V('Gemini 1.0 Pro Vision', ['Gemini Pro Vision', 'gemini-pro-vision', 'gemini-1.0-pro-vision', 'gemini-1.0-pro-vision-001'], GEMINI1, '2023-12-06', '2023-12-13', price=(0.5, 1.5)),
	V('Gemini 1.0 Pro (Apr 2024)', ['gemini-1.0-pro-002'], 'https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions', '2024-04-09', '2024-04-09', price=(0.5, 1.5)),
)
S('Gemini Nano-1', ['Gemini 1.0 Nano-1', 'Nano-1'], GEMINI1_REPORT, '2023-12-06', None, params=(1_800 * M_, 1_800 * M_))
S('Gemini Nano-2', ['Gemini 1.0 Nano-2', 'Nano-2'], GEMINI1_REPORT, '2023-12-06', None, params=(3_250 * M_, 3_250 * M_))

# Gemini 1.5
GEMINI15 = 'https://blog.google/technology/ai/google-gemini-next-generation-model-february-2024/'
IO24 = 'https://blog.google/technology/ai/google-gemini-update-flash-ai-assistant-io-2024/'
SEP24 = 'https://developers.googleblog.com/en/updated-production-ready-gemini-models-reduced-15-pro-pricing-increased-rate-limits-and-more/'
M('Gemini 1.5 Pro',
	V('Gemini 1.5 Pro (Feb 2024)', ["Gemini 1.5 Pro (Feb '24)", 'gemini-1.5-pro-preview-0215', 'gemini-1.5-pro-001-feb24'], GEMINI15, '2024-02-15', None),
	V('Gemini 1.5 Pro (Apr 2024)', ["Gemini 1.5 Pro (Apr '24)", 'gemini-1.5-pro-preview-0409', 'gemini-1.5-pro-0409', 'gemini-1.5-pro-api-0409-preview'], 'https://developers.googleblog.com/en/gemini-15-our-next-generation-model-now-available-for-private-preview-in-google-ai-studio/', '2024-04-09', '2024-04-09', price=(7, 21)),
	V('Gemini 1.5 Pro (May 2024)', ["Gemini 1.5 Pro (May '24)", 'Gemini 1.5 Pro (May 2024)', 'gemini-1.5-pro-001', 'gemini-1.5-pro-preview-0514', 'gemini-1.5-pro-api-0514'], IO24, '2024-05-14', '2024-05-14', price=(3.5, 10.5)),
	V('Gemini 1.5 Pro (Aug 2024)', ["Gemini 1.5 Pro (Aug '24)", 'Gemini 1.5 Pro Experimental 0801', 'gemini-1.5-pro-exp-0801', 'gemini-1.5-pro-exp-0827'], CHANGELOG, '2024-08-01', '2024-08-01'),
	V('Gemini 1.5 Pro (Sep 2024)', ["Gemini 1.5 Pro (Sep '24)", 'Gemini 1.5 Pro (Sept 2024)', 'gemini-1.5-pro-002', 'gemini-1.5-pro'], SEP24, '2024-09-24', '2024-09-24', price=(1.25, 5)),
	V('LearnLM 1.5 Pro Experimental', ['LearnLM 1.5 Pro', 'learnlm-1.5-pro-experimental'], CHANGELOG, '2024-11-19', '2024-11-19'),
)
M('Gemini 1.5 Flash',
	V('Gemini 1.5 Flash (May 2024)', ["Gemini 1.5 Flash (May '24)", 'gemini-1.5-flash-001', 'gemini-1.5-flash-preview-0514', 'gemini-1.5-flash-0514'], IO24, '2024-05-14', '2024-05-14', price=(0.35, 1.05)),
	V('Gemini 1.5 Flash (Aug 2024)', ["Gemini 1.5 Flash (Aug '24)", 'gemini-1.5-flash-exp-0827'], CHANGELOG, '2024-08-27', '2024-08-27'),
	V('Gemini 1.5 Flash (Sep 2024)', ["Gemini 1.5 Flash (Sep '24)", 'gemini-1.5-flash-002', 'gemini-1.5-flash'], SEP24, '2024-09-24', '2024-09-24', price=(0.075, 0.3)),
)
M('Gemini 1.5 Flash-8B',
	V('Gemini 1.5 Flash-8B (Aug 2024)', ['gemini-1.5-flash-8b-exp-0827'], CHANGELOG, '2024-08-27', '2024-08-27', params=(8 * B, 8 * B)),
	V('Gemini 1.5 Flash-8B (Sep 2024)', ['gemini-1.5-flash-8b-exp-0924'], CHANGELOG, '2024-09-24', '2024-09-24', params=(8 * B, 8 * B)),
	V('Gemini 1.5 Flash-8B', ['Gemini 1.5 Flash 8B', 'gemini-1.5-flash-8b', 'gemini-1.5-flash-8b-001'], 'https://developers.googleblog.com/en/gemini-15-flash-8b-is-now-generally-available-for-use/', '2024-10-03', '2024-10-03', params=(8 * B, 8 * B), price=(0.0375, 0.15)),
)
S('Gemini-Exp-1114', ['Gemini Exp 1114', 'Gemini (Exp 1114)', 'gemini-exp-1114'], CHANGELOG, '2024-11-14', '2024-11-14')
S('Gemini-Exp-1121', ['Gemini Exp 1121', 'Gemini (Exp 1121)', 'gemini-exp-1121'], CHANGELOG, '2024-11-21', '2024-11-21')

# Gemini 2.0
GEMINI20 = 'https://blog.google/technology/google-deepmind/google-gemini-ai-update-december-2024/'
FEB25 = 'https://blog.google/technology/google-deepmind/gemini-model-updates-february-2025/'
M('Gemini 2.0 Flash',
	V('Gemini 2.0 Flash Experimental', ['Gemini 2.0 Flash (experimental)', 'Gemini 2.0 Flash Exp', 'gemini-2.0-flash-exp'], GEMINI20, '2024-12-11', '2024-12-11'),
	V('Gemini 2.0 Flash Thinking Experimental (Dec 2024)', ["Gemini 2.0 Flash Thinking Experimental (Dec '24)", 'Gemini 2.0 Flash Thinking (Dec 2024)', 'gemini-2.0-flash-thinking-exp-1219'], 'https://deepmind.google/technologies/gemini/flash-thinking/', '2024-12-19', '2024-12-19', id='gemini-2-0-flash-thinking-dec-2024'),
	V('Gemini 2.0 Flash Thinking Experimental (Jan 2025)', ["Gemini 2.0 Flash Thinking Experimental (Jan '25)", 'Gemini 2.0 Flash Thinking (Jan 2025)', 'gemini-2.0-flash-thinking-exp-01-21', 'gemini-2.0-flash-thinking-exp'], CHANGELOG, '2025-01-21', '2025-01-21', id='gemini-2-0-flash-thinking-jan-2025'),
	V('Gemini 2.0 Flash', ["Gemini 2.0 Flash (Feb '25)", 'Gemini 2.0 Flash (Feb 2025)', 'gemini-2.0-flash', 'gemini-2.0-flash-001', 'gemini-2.0-flash-02-05'], FEB25, '2025-02-05', '2025-02-05', price=(0.1, 0.4)),
)
M('Gemini 2.0 Flash-Lite',
	V('Gemini 2.0 Flash-Lite Preview', ['Gemini 2.0 Flash-Lite (Preview)', 'gemini-2.0-flash-lite-preview', 'gemini-2.0-flash-lite-preview-02-05'], FEB25, '2025-02-05', '2025-02-05', price=(0.075, 0.3)),
	V('Gemini 2.0 Flash-Lite', ["Gemini 2.0 Flash-Lite (Feb '25)", 'Gemini 2.0 Flash-Lite (Feb 2025)', 'gemini-2.0-flash-lite', 'gemini-2.0-flash-lite-001'], CHANGELOG, '2025-02-05', '2025-02-25', price=(0.075, 0.3)),
)
M('Gemini 2.0 Pro',
	V('Gemini-Exp-1206', ['Gemini Exp 1206', 'Gemini (Exp 1206)', 'gemini-exp-1206'], CHANGELOG, '2024-12-06', '2024-12-06'),
	V('Gemini 2.0 Pro Experimental', ["Gemini 2.0 Pro Experimental (Feb '25)", 'Gemini 2.0 Pro', 'Gemini 2.0 Pro (Feb 2025)', 'gemini-2.0-pro-exp', 'gemini-2.0-pro-exp-02-05'], FEB25, '2025-02-05', '2025-02-05'),
)

# Gemini 2.5
GEMINI25 = 'https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/'
GEMINI25_GA = 'https://blog.google/products/gemini/gemini-2-5-model-family-expands/'
M('Gemini 2.5 Pro',
	V('Gemini 2.5 Pro (Mar 2025)', ["Gemini 2.5 Pro Preview (Mar' 25)", "Gemini 2.5 Pro Preview (Mar '25)", 'Gemini 2.5 Pro Experimental', 'gemini-2.5-pro-exp-03-25', 'gemini-2.5-pro-preview-03-25'], GEMINI25, '2025-03-25', '2025-03-25', price=(1.25, 10)),
	V('Gemini 2.5 Pro (May 2025)', ["Gemini 2.5 Pro Preview (May' 25)", "Gemini 2.5 Pro Preview (May '25)", 'Gemini 2.5 Pro I/O Edition', 'gemini-2.5-pro-preview-05-06'], CHANGELOG, '2025-05-06', '2025-05-06', price=(1.25, 10)),
	V('Gemini 2.5 Deep Think', ['gemini-2.5-deep-think', 'gemini-2.5-deep-think-2025-08-01-webapp'], 'https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-2-5-Deep-Think-Model-Card.pdf', '2025-05-20', '2025-08-01'),
	V('Gemini 2.5 Pro', ['Gemini 2.5 Pro (Jun 2025)', "Gemini 2.5 Pro (Jun '25)", 'gemini-2.5-pro', 'gemini-2.5-pro-preview-06-05'], GEMINI25_GA, '2025-06-05', '2025-06-05', price=(1.25, 10)),
	V('Gemini 2.5 Deep Think (IMO 2025)', ['Gemini Deep Think (IMO)', 'Gemini Deep Think IMO'], 'https://deepmind.google/discover/blog/advanced-version-of-gemini-with-deep-think-officially-achieves-gold-medal-standard-at-the-international-mathematical-olympiad/', '2025-07-21', None),
)
M('Gemini 2.5 Flash',
	V('Gemini 2.5 Flash (Apr 2025)', ['Gemini 2.5 Flash Preview', 'Gemini 2.5 Flash Preview (Apr 2025)', "Gemini 2.5 Flash Preview (Apr '25)", 'gemini-2.5-flash-preview-04-17'], 'https://developers.googleblog.com/en/start-building-with-gemini-25-flash/', '2025-04-17', '2025-04-17', price=(0.15, 3.5), efforts=T),
	V('Gemini 2.5 Flash (May 2025)', ['Gemini 2.5 Flash Preview (May 2025)', "Gemini 2.5 Flash Preview (May '25)", 'gemini-2.5-flash-preview-05-20'], CHANGELOG, '2025-05-20', '2025-05-20', price=(0.15, 3.5), efforts=T),
	V('Gemini 2.5 Flash', ['Gemini 2.5 Flash (Jun 2025)', "Gemini 2.5 Flash (Jun '25)", 'gemini-2.5-flash'], GEMINI25_GA, '2025-06-17', '2025-06-17', price=(0.3, 2.5), efforts=T),
	V('Gemini 2.5 Flash (Sep 2025)', ["Gemini 2.5 Flash Preview (Sep '25)", 'Gemini 2.5 Flash Preview (Sep 2025)', 'gemini-2.5-flash-preview-09-2025', 'gemini-2.5-flash-preview-09-25'], CHANGELOG, '2025-09-25', '2025-09-25', price=(0.3, 2.5), efforts=T),
)
M('Gemini 2.5 Flash-Lite',
	V('Gemini 2.5 Flash-Lite Preview', ['Gemini 2.5 Flash-Lite (Jun 2025)', 'Gemini 2.5 Flash-Lite Preview (Jun 2025)', 'gemini-2.5-flash-lite-preview-06-17'], GEMINI25_GA, '2025-06-17', '2025-06-17', price=(0.1, 0.4), efforts=T),
	V('Gemini 2.5 Flash-Lite', ['Gemini 2.5 Flash-Lite (Jul 2025)', 'gemini-2.5-flash-lite'], CHANGELOG, '2025-06-17', '2025-07-22', price=(0.1, 0.4), efforts=T),
	V('Gemini 2.5 Flash-Lite (Sep 2025)', ["Gemini 2.5 Flash-Lite Preview (Sep '25)", 'Gemini 2.5 Flash-Lite Preview (Sep 2025)', 'gemini-2.5-flash-lite-preview-09-2025'], CHANGELOG, '2025-09-25', '2025-09-25', price=(0.1, 0.4), efforts=T),
)
S('Gemini Diffusion', ['Gemini Diffusion (Experimental)'], 'https://deepmind.google/models/gemini-diffusion/', '2025-05-20', None)
S('Sec-Gemini v1', ['Sec-Gemini', 'Sec-Gemini 1'], 'https://security.googleblog.com/2025/04/google-launches-sec-gemini-v1-new.html', '2025-04-04', None)

# Gemini 3
GEMINI3 = 'https://blog.google/products/gemini/gemini-3/'
M('Gemini 3 Pro',
	V('Gemini 3 Pro', ['Gemini 3 Pro Preview', 'Gemini 3.0 Pro', 'gemini-3-pro-preview', 'gemini-3-pro'], GEMINI3, '2025-10-15', '2025-11-18', price=(2, 12), efforts=LH),
	V('Gemini 3 Deep Think (Dec 2025)', ['Gemini 3 Deep Think (Nov 2025)'], GMODELS + 'gemini-3-deep-think/', '2025-11-18', '2025-12-04'),
	V('Gemini 3 Deep Think (Feb 2026)', ['gemini-3-deep-think-preview'], 'https://gemini.google/release-notes/', '2026-02-12', '2026-02-12'),
)
S('Gemini 3 Flash', ['Gemini 3 Flash Preview', 'Gemini 3.0 Flash', 'gemini-3-flash-preview', 'gemini-3-flash'], DOCS + 'gemini-3-flash-preview', '2025-12-17', '2025-12-17', price=(0.5, 3), efforts=MLMH)
S('Gemini 3.1 Pro', ['Gemini 3.1 Pro Preview', 'gemini-3.1-pro-preview', 'gemini-3.1-pro-preview-customtools', 'gemini-3.1-pro'], GMODELS + 'gemini-3-1-pro/', '2026-02-19', '2026-02-19', price=(2, 12), efforts=LMH)
M('Gemini 3.1 Flash-Lite',
	V('Gemini 3.1 Flash-Lite Preview', ['gemini-3.1-flash-lite-preview'], GMODELS + 'gemini-3-1-flash-lite/', '2026-03-03', '2026-03-03', price=(0.25, 1.5), efforts=MLMH),
	V('Gemini 3.1 Flash-Lite', ['gemini-3.1-flash-lite'], DOCS + 'gemini-3.1-flash-lite', '2026-03-03', '2026-05-07', price=(0.25, 1.5), efforts=MLMH),
)
FLASH36 = GMODELS + 'gemini-3-6-flash-3-5-flash-lite-3-5-flash-cyber/'
M('Gemini 3.5 Flash',
	V('Gemini 3.5 Flash', ['gemini-3.5-flash'], 'https://deepmind.google/models/model-cards/gemini-3-5-flash/', '2026-05-19', '2026-05-19', price=(1.5, 9), efforts=MLMH),
	V('Gemini 3.5 Flash Cyber', ['Gemini 3.5 Flash-Cyber'], FLASH36, '2026-07-21', None, efforts=MLMH),
)
S('Gemini 3.5 Flash-Lite', ['gemini-3.5-flash-lite'], FLASH36, '2026-07-21', '2026-07-21', price=(0.3, 2.5), efforts=MLMH)
S('Gemini 3.6 Flash', ['gemini-3.6-flash'], FLASH36, '2026-07-21', '2026-07-21', price=(1.5, 7.5), efforts=MLMH)
S('Gemini 3.7 Flash', ['gemini-3.7-flash'], GMODELS + 'introducing-gemini-3-7-flash/', '2026-08-13', '2026-08-13', price=(1.5, 7.5), efforts=LMH)
FLASH38 = GMODELS + '3-8-flash-and-3-8-flash-cyber/'
M('Gemini 3.8 Flash',
	V('Gemini 3.8 Flash', ['gemini-3.8-flash'], FLASH38, '2026-09-02', '2026-09-02', price=(1.5, 7.5), efforts=LMH),
	V('Gemini 3.8 Flash Cyber', ['Gemini 3.8 Flash-Cyber'], FLASH38, '2026-09-02', None, efforts=LMH),
)

# Gemini 4 (prices are the list prices that apply after the introductory period)
S('Gemini 4 Argon', ['Gemini 4', 'gemini-4-argon'], GMODELS + 'introducing-gemini-4-argon/', '2026-07-21', '2026-10-13', price=(4, 20), efforts=LMH)
S('Gemini 4 Carbon', ['gemini-4-carbon'], GMODELS + 'introducing-gemini-4-carbon/', '2026-10-28', None, price=(4, 20), efforts=LMH)

# Gemma 1 (CodeGemma is built on the matching Gemma size)
GEMMA = 'https://blog.google/technology/developers/gemma-open-models/'
CODEGEMMA = 'https://ai.google.dev/gemma/docs/codegemma'
P2B, P7B = (2_510 * M_, 2_510 * M_), (8_540 * M_, 8_540 * M_)
M('Gemma 2B',
	V('Gemma 2B', ['Gemma 1 2B', 'Gemma 1.0 2B', 'gemma-2b', 'gemma-2b-it', 'google/gemma-2b', 'google/gemma-2b-it'], GEMMA, '2024-02-21', '2024-02-21', weights='2024-02-21', params=P2B),
	V('Gemma 1.1 2B', ['gemma-1.1-2b-it', 'google/gemma-1.1-2b-it'], HF + 'gemma-1.1-2b-it', '2024-04-05', '2024-04-05', weights='2024-04-05', params=P2B),
	V('CodeGemma 2B', ['codegemma-2b', 'google/codegemma-2b'], CODEGEMMA, '2024-04-09', '2024-04-09', weights='2024-04-09', params=P2B),
	V('CodeGemma 1.1 2B', ['codegemma-1.1-2b', 'google/codegemma-1.1-2b'], CODEGEMMA, '2024-05-03', '2024-05-03', weights='2024-05-03', params=P2B),
)
M('Gemma 7B',
	V('Gemma 7B', ['Gemma 1 7B', 'Gemma 1.0 7B', 'gemma-7b', 'gemma-7b-it', 'google/gemma-7b', 'google/gemma-7b-it'], GEMMA, '2024-02-21', '2024-02-21', weights='2024-02-21', params=P7B),
	V('Gemma 1.1 7B', ['Gemma 1.1 7B Instruct', 'gemma-1.1-7b-it', 'google/gemma-1.1-7b-it'], HF + 'gemma-1.1-7b-it', '2024-04-05', '2024-04-05', weights='2024-04-05', params=P7B),
	V('CodeGemma 7B', ['codegemma-7b', 'codegemma-7b-it', 'google/codegemma-7b', 'google/codegemma-7b-it'], CODEGEMMA, '2024-04-09', '2024-04-09', weights='2024-04-09', params=P7B),
	V('CodeGemma 1.1 7B', ['codegemma-1.1-7b-it', 'google/codegemma-1.1-7b-it'], CODEGEMMA, '2024-05-03', '2024-05-03', weights='2024-05-03', params=P7B),
)
RGEMMA = 'https://ai.google.dev/gemma/docs/recurrentgemma'
S('RecurrentGemma 2B', ['recurrentgemma-2b', 'recurrentgemma-2b-it', 'google/recurrentgemma-2b', 'google/recurrentgemma-2b-it'], RGEMMA, '2024-04-09', '2024-04-09', weights='2024-04-09')
S('RecurrentGemma 9B', ['recurrentgemma-9b', 'recurrentgemma-9b-it', 'google/recurrentgemma-9b', 'google/recurrentgemma-9b-it'], RGEMMA, '2024-06-11', '2024-06-11', weights='2024-06-11')

# Gemma 2 (DataGemma, TxGemma, Gemma 2 JPN and C2S-Scale are fine-tunes of the matching size)
GEMMA2 = 'https://blog.google/technology/developers/google-gemma-2/'
TXGEMMA = ARXIV + '2504.06196'
G2_2B, G2_9B, G2_27B = (2_610 * M_, 2_610 * M_), (9_240 * M_, 9_240 * M_), (27_230 * M_, 27_230 * M_)
M('Gemma 2 2B',
	V('Gemma 2 2B', ['gemma-2-2b', 'gemma-2-2b-it', 'google/gemma-2-2b', 'google/gemma-2-2b-it'], 'https://developers.googleblog.com/en/smaller-safer-more-transparent-advancing-responsible-ai-with-gemma/', '2024-07-31', '2024-07-31', weights='2024-07-31', params=G2_2B),
	V('Gemma 2 JPN', ['Gemma 2 2B JPN', 'gemma-2-2b-jpn-it', 'google/gemma-2-2b-jpn-it'], HF + 'gemma-2-2b-jpn-it', '2024-10-03', '2024-10-03', weights='2024-10-03', params=G2_2B),
	V('TxGemma 2B', ['txgemma-2b-predict', 'google/txgemma-2b-predict'], TXGEMMA, '2025-03-25', '2025-03-25', weights='2025-03-25', params=G2_2B),
)
M('Gemma 2 9B',
	V('Gemma 2 9B', ['gemma-2-9b', 'gemma-2-9b-it', 'google/gemma-2-9b', 'google/gemma-2-9b-it'], GEMMA2, '2024-06-27', '2024-06-27', weights='2024-06-27', params=G2_9B),
	V('TxGemma 9B', ['txgemma-9b-chat', 'txgemma-9b-predict', 'google/txgemma-9b-chat', 'google/txgemma-9b-predict'], TXGEMMA, '2025-03-25', '2025-03-25', weights='2025-03-25', params=G2_9B),
)
M('Gemma 2 27B',
	V('Gemma 2 27B', ['gemma-2-27b', 'gemma-2-27b-it', 'google/gemma-2-27b', 'google/gemma-2-27b-it'], GEMMA2, '2024-05-14', '2024-06-27', weights='2024-06-27', params=G2_27B),
	V('DataGemma', ['DataGemma 27B', 'DataGemma RIG', 'DataGemma RAG', 'datagemma-rig-27b-it', 'datagemma-rag-27b-it', 'google/datagemma-rig-27b-it', 'google/datagemma-rag-27b-it'], ARXIV + '2409.13741', '2024-09-12', '2024-09-12', weights='2024-09-12', params=G2_27B),
	V('TxGemma 27B', ['txgemma-27b-chat', 'txgemma-27b-predict', 'google/txgemma-27b-chat', 'google/txgemma-27b-predict'], TXGEMMA, '2025-03-25', '2025-03-25', weights='2025-03-25', params=G2_27B),
	V('C2S-Scale 27B', ['C2S-Scale', 'C2S-Scale-Gemma-2-27B', 'vandijklab/C2S-Scale-Gemma-2-27B'], 'https://huggingface.co/vandijklab/C2S-Scale-Gemma-2-27B', '2025-10-15', '2025-10-15', weights='2025-10-15', params=G2_27B),
)

# Gemma 3 (MedGemma, TranslateGemma and FunctionGemma are fine-tunes of the matching size)
GEMMA3 = 'https://blog.google/technology/developers/gemma-3/'
MEDGEMMA = ARXIV + '2507.05201'
TRANSLATEGEMMA = HF + 'translategemma-27b-it'
G3_4B, G3_12B, G3_27B = (4_300 * M_, 4_300 * M_), (12_200 * M_, 12_200 * M_), (27_400 * M_, 27_400 * M_)
M('Gemma 3 270M',
	V('Gemma 3 270M', ['gemma-3-270m', 'gemma-3-270m-it', 'google/gemma-3-270m', 'google/gemma-3-270m-it'], 'https://developers.googleblog.com/en/introducing-gemma-3-270m/', '2025-08-14', '2025-08-14', weights='2025-08-14', params=(268 * M_, 268 * M_)),
	V('FunctionGemma', ['FunctionGemma 270M', 'functiongemma-270m-it', 'google/functiongemma-270m-it'], 'https://ai.google.dev/gemma/docs/functiongemma', '2025-12-18', '2025-12-18', weights='2025-12-18', params=(268 * M_, 268 * M_)),
)
S('Gemma 3 1B', ['Gemma 3 1B Instruct', 'gemma-3-1b-it', 'gemma-3-1b-pt', 'google/gemma-3-1b-it', 'google/gemma-3-1b-pt'], GEMMA3, '2025-03-12', '2025-03-12', weights='2025-03-12', params=(1 * B, 1 * B))
M('Gemma 3 4B',
	V('Gemma 3 4B', ['Gemma 3 4B Instruct', 'gemma-3-4b-it', 'gemma-3-4b-pt', 'google/gemma-3-4b-it', 'google/gemma-3-4b-pt'], GEMMA3, '2025-03-12', '2025-03-12', weights='2025-03-12', params=G3_4B, price=(0.05, 0.1)),
	V('MedGemma 4B', ['medgemma-4b-it', 'medgemma-4b-pt', 'google/medgemma-4b-it', 'google/medgemma-4b-pt'], MEDGEMMA, '2025-05-20', '2025-05-20', weights='2025-05-20', params=G3_4B),
	V('MedGemma 1.5 4B', ['MedGemma 1.5', 'medgemma-1.5-4b-it', 'google/medgemma-1.5-4b-it'], HF + 'medgemma-1.5-4b-it', '2026-01-13', '2026-01-13', weights='2026-01-13', params=G3_4B),
	V('TranslateGemma 4B', ['translategemma-4b-it', 'google/translategemma-4b-it'], TRANSLATEGEMMA, '2026-01-15', '2026-01-15', weights='2026-01-15', params=G3_4B),
)
M('Gemma 3 12B',
	V('Gemma 3 12B', ['Gemma 3 12B Instruct', 'gemma-3-12b-it', 'gemma-3-12b-pt', 'google/gemma-3-12b-it', 'google/gemma-3-12b-pt'], GEMMA3, '2025-03-12', '2025-03-12', weights='2025-03-12', params=G3_12B, price=(0.07, 0.22)),
	V('TranslateGemma 12B', ['translategemma-12b-it', 'google/translategemma-12b-it'], TRANSLATEGEMMA, '2026-01-15', '2026-01-15', weights='2026-01-15', params=G3_12B),
)
M('Gemma 3 27B',
	V('Gemma 3 27B', ['Gemma 3 27B Instruct', 'gemma-3-27b-it', 'gemma-3-27b-pt', 'google/gemma-3-27b-it', 'google/gemma-3-27b-pt'], GEMMA3, '2025-03-12', '2025-03-12', weights='2025-03-12', params=G3_27B, price=(0.1, 0.3)),
	V('MedGemma 27B Text', ['MedGemma 27B', 'medgemma-27b-text-it', 'google/medgemma-27b-text-it'], MEDGEMMA, '2025-05-20', '2025-05-20', weights='2025-05-20', params=G3_27B),
	V('MedGemma 27B Multimodal', ['medgemma-27b-it', 'google/medgemma-27b-it'], MEDGEMMA, '2025-07-09', '2025-07-09', weights='2025-07-09', params=G3_27B),
	V('TranslateGemma 27B', ['translategemma-27b-it', 'google/translategemma-27b-it'], TRANSLATEGEMMA, '2026-01-15', '2026-01-15', weights='2026-01-15', params=G3_27B),
)
GEMMA3N = 'https://developers.googleblog.com/en/introducing-gemma-3n/'
S('Gemma 3n E2B', ['Gemma 3n E2B Instruct', 'gemma-3n-e2b-it', 'gemma-3n-E2B-it', 'google/gemma-3n-E2B-it', 'google/gemma-3n-E2B'], GEMMA3N, '2025-06-26', '2025-06-26', weights='2025-06-26', params=(5_440 * M_, 2 * B))
M('Gemma 3n E4B',
	V('Gemma 3n E4B Preview', ["Gemma 3n E4B Instruct Preview (May '25)", 'Gemma 3n Preview', 'gemma-3n-e4b-it-preview'], GEMMA3N, '2025-05-20', '2025-05-20', params=(7_850 * M_, 4 * B)),
	V('Gemma 3n E4B', ['Gemma 3n E4B Instruct', 'gemma-3n-e4b-it', 'gemma-3n-E4B-it', 'google/gemma-3n-E4B-it', 'google/gemma-3n-E4B'], GEMMA3N, '2025-05-20', '2025-06-26', weights='2025-06-26', params=(7_850 * M_, 4 * B)),
)
S('VaultGemma 1B', ['VaultGemma', 'vaultgemma-1b', 'google/vaultgemma-1b'], HF + 'vaultgemma-1b', '2025-09-12', '2025-09-12', weights='2025-09-12', params=(1 * B, 1 * B))

# Gemma 4 (prices: median of third-party providers on Artificial Analysis)
GEMMA4 = 'https://ai.google.dev/gemma/docs/core/model_card_4'
S('Gemma 4 E2B', ['gemma-4-e2b', 'gemma-4-E2B-it', 'google/gemma-4-E2B', 'google/gemma-4-E2B-it'], GEMMA4, '2026-04-02', '2026-04-02', weights='2026-04-02', params=(5_100 * M_, 2_300 * M_), efforts=T)
S('Gemma 4 E4B', ['gemma-4-e4b', 'gemma-4-E4B-it', 'google/gemma-4-E4B', 'google/gemma-4-E4B-it'], GEMMA4, '2026-04-02', '2026-04-02', weights='2026-04-02', params=(8 * B, 4_500 * M_), price=(0.02, 0.1), efforts=T)
S('Gemma 4 12B', ['Gemma 4 12B Unified', 'gemma-4-12b', 'gemma-4-12B-it', 'google/gemma-4-12B', 'google/gemma-4-12B-it'], GEMMA4, '2026-06-03', '2026-06-03', weights='2026-06-03', params=(11_950 * M_, 11_950 * M_), price=(0.1, 0.3), efforts=T)
S('Gemma 4 26B A4B', ['Gemma 4 26B', 'gemma-4-26b-a4b', 'gemma-4-26b-a4b-it', 'gemma-4-26B-A4B-it', 'google/gemma-4-26B-A4B', 'google/gemma-4-26B-A4B-it'], GEMMA4, '2026-04-02', '2026-04-02', weights='2026-04-02', params=(25_200 * M_, 3_800 * M_), price=(0.1, 0.34), efforts=T)
S('Gemma 4 31B', ['Gemma 4 31B IT', 'gemma-4-31b', 'gemma-4-31b-it', 'gemma-4-31B-it', 'google/gemma-4-31B', 'google/gemma-4-31B-it'], GEMMA4, '2026-04-02', '2026-04-02', weights='2026-04-02', params=(30_700 * M_, 30_700 * M_), price=(0.19, 0.45), efforts=T)
S('DiffusionGemma 26B A4B', ['DiffusionGemma', 'DiffusionGemma 26B-A4B', 'diffusiongemma-26b-a4b', 'diffusiongemma-26B-A4B-it', 'google/diffusiongemma-26B-A4B-it'], 'https://blog.google/innovation-and-ai/technology/developers-tools/diffusion-gemma-faster-text-generation/', '2026-06-10', '2026-06-10', weights='2026-06-10', params=(25_200 * M_, 3_800 * M_))

write('google')
