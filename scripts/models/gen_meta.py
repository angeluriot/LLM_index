"""Generates data/models/meta/*.json (one file per model).

Covers every Meta lab: Facebook AI Research (FAIR), Meta AI, the Llama team and Meta Superintelligence Labs (Muse).
All the data lives in this script: edit it and run `python scripts/models/gen_meta.py`.
Existing JSON files are deleted and rewritten, so manual edits to them are lost.
"""

from common import B, M, N, S, V, write

M_ = 1_000_000
LMHX = ['Low', 'Medium', 'High', 'Xhigh']						# Open-weights Muse models (reasoning strength, cannot be disabled)
MLMHX = ['Minimal', 'Low', 'Medium', 'High', 'Xhigh']			# Meta Model API, Muse Spark 1.1 / 1.2 ("none" returns an error)
MLMHXM = ['Minimal', 'Low', 'Medium', 'High', 'Xhigh', 'Max']	# Meta Model API, Muse Spark 1.3 / Muse Wonder

ARXIV = 'https://arxiv.org/abs/'
HF = 'https://huggingface.co/'
META_BLOG = 'https://ai.meta.com/blog/'
RESEARCH = 'https://research.meta.ai/blog/'


def hf_names(*repos):
	"""Repo ids, both bare and with their organisation."""
	return list(dict.fromkeys([r.split('/')[-1] for r in repos] + list(repos)))


# BlenderBot (open-domain chatbots; BlenderBot 3 is in OPT-30B and OPT-175B)
S('BlenderBot 9.4B', ['BlenderBot', 'BlenderBot 1', 'BlenderBot 1.0', 'Blender 9.4B', 'Generative BST', 'Generative BST 9.4B', 'blender_9B'], ARXIV + '2004.13637', '2020-04-28', '2020-04-28', weights='2020-04-28', params=(9_400 * M_, 9_400 * M_))
S('BlenderBot 2.0 2.7B', ['BlenderBot 2', 'BlenderBot 2.0', 'BlenderBot 2 2.7B', 'blenderbot2_3B'], 'https://parl.ai/projects/blenderbot2/', '2021-07-16', '2021-07-16', weights='2021-07-16', params=(2_700 * M_, 2_700 * M_))

# FAIR multilingual and MoE language models (2021)
S('XGLM 7.5B', ['XGLM', 'XGLM-7.5B', 'XGLM 7.5B', *hf_names('facebook/xglm-7.5B')], ARXIV + '2112.10668', '2021-12-20', '2021-12-20', weights='2021-12-20', params=(7_500 * M_, 7_500 * M_))
FAIRSEQ = ARXIV + '2112.10684'
S('Fairseq Dense 13B', ['Fairseq-dense 13B', 'fairseq-dense-13B', 'KoboldAI/fairseq-dense-13B'], FAIRSEQ, '2021-12-20', '2021-12-20', weights='2021-12-20', params=(13 * B, 13 * B))
S('Fairseq MoE 1.1T', ['MoE-1.1T', 'Fairseq MoE 1.1T', 'fairseq-moe-1.1T'], FAIRSEQ, '2021-12-20', '2021-12-20', weights='2021-12-20', params=(1_100 * B, None))

# OPT (all sizes are versions: same exception as GPT-2 / GPT-3 / T5 / PaLM, OPT-IML and BlenderBot 3 fine-tunes included)
OPT = ARXIV + '2205.01068'
OPT_IML = ARXIV + '2212.12017'
BB3 = ARXIV + '2208.03188'
OPT_SIZES = [('125M', 125 * M_), ('350M', 350 * M_), ('1.3B', 1_300 * M_), ('2.7B', 2_700 * M_), ('6.7B', 6_700 * M_), ('13B', 13 * B), ('30B', 30 * B), ('66B', 66 * B), ('175B', 175 * B)]
OPT_WEIGHTS = {'66B': '2022-06-23'}	# The other sizes were published (175B on request) the day after the paper
M('OPT',
	*[V(f'OPT-{size}', [f'OPT {size}', f'OPT ({size})', *hf_names(f'facebook/opt-{size.lower()}')] + (['OPT'] if size == '175B' else []), OPT, '2022-05-02', OPT_WEIGHTS.get(size, '2022-05-03'), weights=OPT_WEIGHTS.get(size, '2022-05-03'), params=(p, p)) for size, p in OPT_SIZES],
	V('BlenderBot 3 30B', ['BB3 30B', 'BB3-30B', 'blenderbot3_30B'], BB3, '2022-08-05', '2022-08-05', weights='2022-08-05', params=(30 * B, 30 * B)),
	V('BlenderBot 3', ['BlenderBot 3 175B', 'BB3', 'BB3-175B', 'blenderbot3_175B'], BB3, '2022-08-05', '2022-08-05', weights='2022-08-05', params=(175 * B, 175 * B)),
	V('OPT-IML 30B', ['OPT-IML (30B)', 'OPT-IML-Max 30B', 'OPT-IML-Max (30B)', *hf_names('facebook/opt-iml-30b', 'facebook/opt-iml-max-30b')], OPT_IML, '2022-12-22', '2023-01-20', weights='2023-01-20', params=(30 * B, 30 * B)),
	V('OPT-IML 175B', ['OPT-IML', 'OPT-IML (175B)', 'OPT-IML-Max 175B', 'OPT-IML-Max (175B)'], OPT_IML, '2022-12-22', None, params=(175 * B, 175 * B)),
)

# Galactica (all sizes are versions, same exception as OPT)
GALACTICA = ARXIV + '2211.09085'
M('Galactica',
	*[V(f'Galactica {size}', [f'GAL {size}', f'Galactica-{size}', *hf_names(f'facebook/galactica-{size.lower()}')] + (['Galactica'] if size == '120B' else []), GALACTICA, '2022-11-15', '2022-11-15', weights='2022-11-15', params=(p, p))
		for size, p in [('125M', 125 * M_), ('1.3B', 1_300 * M_), ('6.7B', 6_700 * M_), ('30B', 30 * B), ('120B', 120 * B)]],
)

# LLaMA (LIMA is a fine-tune of LLaMA 65B)
LLAMA = META_BLOG + 'large-language-model-llama-meta-ai/'
for size, alt, p in [('7B', None, 6_700 * M_), ('13B', None, 13 * B), ('33B', '30B', 32_500 * M_), ('65B', None, 65_200 * M_)]:
	names = [f'LLaMA-{size}', f'Llama {size}', f'LLaMA 1 {size}', f'Llama 1 {size}', f'llama-{size.lower()}']
	if alt:
		names += [f'LLaMA {alt}', f'LLaMA-{alt}', f'llama-{alt.lower()}']
	v = V(f'LLaMA {size}', names, LLAMA, '2023-02-24', '2023-02-24', weights='2023-02-24', params=(p, p))
	if size == '65B':
		M('LLaMA 65B', v, V('LIMA', ['LIMA 65B'], ARXIV + '2305.11206', '2023-05-18', None, params=(p, p)))
	else:
		M(f'LLaMA {size}', v)

# Llama 2 (Code Llama, Llama 2 Long and LLM Compiler are built on the matching Llama 2 size)
LLAMA2 = 'https://about.fb.com/news/2023/07/llama-2/'
LLAMA2_LONG = ARXIV + '2309.16039'
CODE_LLAMA = META_BLOG + 'code-llama-large-language-model-coding/'
LLM_COMPILER = ARXIV + '2407.02524'


def llama2(size, p, released=True, code_weights='2023-08-24', compiler=False, unnatural=False):
	s = size.lower()
	date = '2023-07-18' if released else None
	versions = [
		V(f'Llama 2 {size}', [f'Llama-2-{s}', f'Llama 2-{size}', f'Llama 2 Chat {size}', f'Llama-2-{s}-chat', f'llama-2-{s}-chat', f'Llama 2 {size} Chat'] + (hf_names(f'meta-llama/Llama-2-{s}-hf', f'meta-llama/Llama-2-{s}-chat-hf') if released else []), LLAMA2, '2023-07-18', date, weights=date, params=p),
		V(f'Code Llama {size}', [f'CodeLlama {size}', f'Code Llama-{size}', f'Code Llama - Instruct {size}', f'Code Llama Instruct {size}', *hf_names(f'codellama/CodeLlama-{s}-hf', f'codellama/CodeLlama-{s}-Instruct-hf', f'meta-llama/CodeLlama-{s}-hf', f'meta-llama/CodeLlama-{s}-Instruct-hf')], CODE_LLAMA, code_weights, code_weights, weights=code_weights, params=p),
		V(f'Code Llama - Python {size}', [f'Code Llama Python {size}', f'CodeLlama Python {size}', *hf_names(f'codellama/CodeLlama-{s}-Python-hf', f'meta-llama/CodeLlama-{s}-Python-hf')], CODE_LLAMA, code_weights, code_weights, weights=code_weights, params=p),
		V(f'Llama 2 Long {size}', [f'Llama 2 Long Chat {size}'], LLAMA2_LONG, '2023-09-27', None, params=p),
	]
	if unnatural:
		versions.append(V(f'Unnatural Code Llama {size}', ['Unnatural Code Llama'], CODE_LLAMA, '2023-08-24', None, params=p))
	if compiler:
		versions.append(V(f'LLM Compiler {size}', [f'LLM Compiler FTD {size}', *hf_names(f'facebook/llm-compiler-{s}', f'facebook/llm-compiler-{s}-ftd')], LLM_COMPILER, '2024-06-27', '2024-06-27', weights='2024-06-27', params=p))
	M(f'Llama 2 {size}', *versions)


llama2('7B', (6_740 * M_, 6_740 * M_), compiler=True)
llama2('13B', (13 * B, 13 * B), compiler=True)
llama2('34B', (34 * B, 34 * B), released=False, unnatural=True)
llama2('70B', (69 * B, 69 * B), code_weights='2024-01-29')

# Chameleon (early-fusion mixed-modal models)
CHAMELEON = ARXIV + '2405.09818'
S('Chameleon 7B', ['Chameleon-7B', *hf_names('facebook/chameleon-7b')], CHAMELEON, '2024-05-16', '2024-06-18', weights='2024-06-18', params=(7 * B, 7 * B))
S('Chameleon 34B', ['Chameleon-34B', *hf_names('facebook/chameleon-30b')], CHAMELEON, '2024-05-16', '2024-06-18', weights='2024-06-18', params=(34 * B, 34 * B))


# Llama 3 family (base and instruct in the same version; prices: median of third-party providers on Artificial Analysis)
def llama3(ver, size, extra=(), hf=None, meta_prefix=False):
	"""Usual aliases of a Llama 3.x size: AA ('Llama 3.1 Instruct 8B'), LLM Stats ('Llama 3.1 8B Instruct'), Epoch, HF repos."""
	s = size.lower()
	repo = hf or f'Llama-{ver}-{size}'
	repos = [f'meta-llama/{repo}', f'meta-llama/{repo}-Instruct']
	if meta_prefix:
		repos += [f'meta-llama/Meta-Llama-{ver}-{size}', f'meta-llama/Meta-Llama-{ver}-{size}-Instruct']
	return [f'Llama {ver} Instruct {size}', f'Llama {ver} {size} Instruct', f'Llama {ver}-{size}', f'llama-{ver}-{s}-instruct', f'llama{ver}:{s}', *hf_names(*repos), *extra]


LLAMA3 = META_BLOG + 'meta-llama-3/'
S('Llama 3 8B', llama3('3', '8B', ['LLaMA-3-Instruct-8B', 'llama3-8b'], hf='Meta-Llama-3-8B'), LLAMA3, '2024-04-18', '2024-04-18', weights='2024-04-18', params=(8 * B, 8 * B), price=(0.04, 0.14))
S('Llama 3 70B', llama3('3', '70B', ['LLaMA-3-Instruct-70B', 'llama3-70b'], hf='Meta-Llama-3-70B'), LLAMA3, '2024-04-18', '2024-04-18', weights='2024-04-18', params=(70 * B, 70 * B), price=(0.65, 2.75))

LLAMA31 = META_BLOG + 'meta-llama-3-1/'
S('Llama 3.1 8B', llama3('3.1', '8B', ['llama3.1:8b-instruct-q8_0'], meta_prefix=True), LLAMA31, '2024-07-23', '2024-07-23', weights='2024-07-23', params=(8 * B, 8 * B), price=(0.04, 0.07))
S('Llama 3.1 70B', llama3('3.1', '70B', meta_prefix=True), LLAMA31, '2024-07-23', '2024-07-23', weights='2024-07-23', params=(70 * B, 70 * B), price=(0.56, 0.56))
S('Llama 3.1 405B', llama3('3.1', '405B', ['llama3.1:405b-instruct-q4_K_M', 'meta-llama/Llama-3.1-405B-Instruct-FP8'], meta_prefix=True), LLAMA31, '2024-04-18', '2024-07-23', weights='2024-07-23', params=(405 * B, 405 * B))

LLAMA32 = META_BLOG + 'llama-3-2-connect-2024-vision-edge-mobile-devices/'
LLAMA32_Q = META_BLOG + 'meta-llama-quantized-lightweight-models/'
for size, p in [('1B', 1_230 * M_), ('3B', 3_210 * M_)]:
	M(f'Llama 3.2 {size}',
		V(f'Llama 3.2 {size}', llama3('3.2', size), LLAMA32, '2024-09-25', '2024-09-25', weights='2024-09-25', params=(p, p)),
		V(f'Llama 3.2 {size} Quantized', [f'Llama 3.2 {size} QLoRA', f'Llama 3.2 {size} SpinQuant', *hf_names(f'meta-llama/Llama-3.2-{size}-Instruct-QLORA_INT4_EO8', f'meta-llama/Llama-3.2-{size}-Instruct-SpinQuant_INT4_EO8')], LLAMA32_Q, '2024-10-24', '2024-10-24', weights='2024-10-24', params=(p, p)),
	)
for size, p, price in [('11B', 10_600 * M_, (0.34, 0.34)), ('90B', 88_600 * M_, (None, None))]:
	S(f'Llama 3.2 {size} Vision', [f'Llama 3.2 Instruct {size} (Vision)', f'Llama 3.2 {size} (Vision)', f'Llama 3.2 {size} Instruct', f'Llama 3.2 {size}', f'llama-3.2-{size.lower()}-instruct', *hf_names(f'meta-llama/Llama-3.2-{size}-Vision', f'meta-llama/Llama-3.2-{size}-Vision-Instruct')], LLAMA32, '2024-09-25', '2024-09-25', weights='2024-09-25', params=(p, p), price=price)

S('Llama 3.3 70B', llama3('3.3', '70B', ['llama3.3:70b-instruct-q8_0']), 'https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_3/', '2024-12-06', '2024-12-06', weights='2024-12-06', params=(70 * B, 70 * B), price=(0.6, 0.71))

# Llama 4
LLAMA4 = META_BLOG + 'llama-4-multimodal-intelligence/'
S('Llama 4 Scout', ['Llama-4-Scout', 'llama-4-scout', *hf_names('meta-llama/Llama-4-Scout-17B-16E', 'meta-llama/Llama-4-Scout-17B-16E-Instruct')], LLAMA4, '2025-04-05', '2025-04-05', weights='2025-04-05', params=(109 * B, 17 * B), price=(0.18, 0.66))
M('Llama 4 Maverick',
	V('Llama 4 Maverick', ['Llama-4-Maverick', 'llama-4-maverick', *hf_names('meta-llama/Llama-4-Maverick-17B-128E', 'meta-llama/Llama-4-Maverick-17B-128E-Instruct', 'meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8')], LLAMA4, '2025-04-05', '2025-04-05', weights='2025-04-05', params=(400 * B, 17 * B), price=(0.25, 0.92)),
	V('Llama 4 Maverick Experimental', ['Llama-4-Maverick-03-26-Experimental', 'llama-4-maverick-03-26-experimental', 'Llama 4 Maverick (Experimental)'], LLAMA4, '2025-04-05', None, params=(400 * B, 17 * B)),
)
S('Llama 4 Behemoth', ['Llama 4 Behemoth (preview)', 'Llama 4 Behemoth Preview', 'Llama-4-Behemoth'], LLAMA4, '2025-04-05', None, params=(2_000 * B, 288 * B))

# FAIR research models (2025)
S('CWM', ['Code World Model', 'CWM 32B', *hf_names('facebook/cwm', 'facebook/cwm-sft', 'facebook/cwm-pretrain')], HF + 'facebook/cwm', '2025-09-24', '2025-09-24', weights='2025-09-24', params=(32 * B, 32 * B))
S('MobileLLM-R1 950M', ['MobileLLM-R1', 'MobileLLM R1 950M', *hf_names('facebook/MobileLLM-R1-950M')], HF + 'facebook/MobileLLM-R1-950M', '2025-09-12', '2025-09-12', weights='2025-09-12', params=(950 * M_, 950 * M_))
S('MobileLLM-Pro', ['MobileLLM Pro', 'MobileLLM-Pro 1B', 'facebook/MobileLLM-Pro'], HF + 'facebook/MobileLLM-Pro', '2025-10-16', '2025-10-16', weights='2025-10-16', params=(1_080 * M_, 1_080 * M_))
S('MobileLLM-R1.5 950M', ['MobileLLM-R1.5', 'MobileLLM R1.5 950M', *hf_names('facebook/MobileLLM-R1.5-950M')], HF + 'facebook/MobileLLM-R1.5-950M', '2025-11-21', '2025-11-21', weights='2025-11-21', params=(950 * M_, 950 * M_))

# Muse (Meta Superintelligence Labs; API prices are the Meta Model API standard tier, the contributor tier ids are aliases)
SPARK = META_BLOG + 'introducing-muse-spark-msl/'
M('Muse Spark',
	V('Muse Spark', ['Muse Spark 1.0', 'muse-spark'], SPARK, '2026-04-08', '2026-04-08', efforts=['Instant', 'Thinking']),
	V('Muse Spark Contemplating', ['Muse Spark (Contemplating)', 'Muse Spark Contemplating mode'], SPARK, '2026-04-08', '2026-04-08'),
)
S('Muse Spark 1.1', ['muse-spark-1.1', 'meta/muse-spark-1.1'], META_BLOG + 'introducing-muse-spark-meta-model-api/', '2026-07-09', '2026-07-09', price=(1.25, 4.25), efforts=MLMHX)
S('Muse Spark 1.2', ['muse-spark-1.2', 'muse-spark-1.2-contributor'], RESEARCH + 'introducing-muse-code-and-muse-spark-1-2', '2026-08-05', '2026-08-05', price=(1.25, 4.25), efforts=MLMHX)
S('Muse Glimmer', ['Muse Glimmer 30B', 'Muse Glimmer-30B', 'muse-glimmer', 'muse-glimmer-30b', 'meta/muse-glimmer-30b', *hf_names('meta-models/Muse-Glimmer-30B')], RESEARCH + 'introducing-muse-glimmer-open-agentic-model', '2026-08-10', '2026-08-10', weights='2026-08-10', params=(29_600 * M_, 29_600 * M_), price=(0.3, 1.2), efforts=LMHX)
S('Muse Spark 1.3', ['muse-spark-1.3', 'muse-spark-1.3-contributor'], RESEARCH + 'introducing-muse-spark-1-3', '2026-09-02', '2026-09-02', price=(1.25, 4.25), efforts=MLMHXM)
S('Muse Spark Open', ['Muse Spark Open 816B', 'muse-spark-open', *hf_names('meta-models/Muse-Spark-Open-816B-A42B', 'meta-models/Muse-Spark-Open-816B-A42B-NVFP4')], RESEARCH + 'muse-spark-open', '2026-09-02', '2026-10-15', weights='2026-10-15', params=(816 * B, 42 * B), price=(1.25, 4.25), efforts=LMHX)
WONDER = RESEARCH + 'muse-wonder'
S('Muse Wonder', ['Muse Wonder 1.0', 'muse-wonder', 'muse-wonder-1.0'], WONDER, '2026-10-12', '2026-10-12', price=(4, 20), efforts=MLMHXM)
S('Muse Wonder Full', ['Muse Wonder Full 1.0', 'muse-wonder-full', 'muse-wonder-full-1.0'], WONDER, '2026-10-12', '2026-10-12', efforts=MLMHXM)

write('meta')
