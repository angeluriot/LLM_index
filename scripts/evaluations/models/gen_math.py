"""Generates data/results/models/math.jsonl and data/leftovers/math.jsonl.

Every number is written literally in this file (nothing is downloaded at run time), so it is the single source of
truth for the MATH results: edit it and run `python scripts/evaluations/models/gen_math.py`. The helpers (R, L, write) and
their conventions are in common.py.

Parts of the evaluation (data/evaluations/math.json):
- '': the full MATH test set (5,000 problems, Hendrycks et al. 2021)
- '-500': MATH-500, the 500-problem subset of "Let's Verify Step by Step" (PRM800K)
- 'Level 5': MATH Level 5, the 1,324 hardest problems (Open LLM Leaderboard v2, Epoch AI)

Sources:
- Epoch AI MATH Level 5 runs (https://epoch.ai/benchmarks/math-level-5, math_level_5.csv of
  https://epoch.ai/data/benchmark_data.zip): fractions with their standard error, 0-shot
- Artificial Analysis MATH-500 (https://artificialanalysis.ai/evaluations/math-500): the current page only embeds the
  top models, the others come from the Wayback Machine snapshots of the same page (Jun 2025 -> Jun 2026), latest value
  kept when a model was re-evaluated
- Vals AI MATH 500 (https://www.vals.ai/benchmarks/math500): accuracy, stderr and cost per test (cost type TASK)
- Kaggle Open Benchmarks MATH-500 (https://www.kaggle.com/benchmarks/open-benchmarks/math-500), read through the
  BenchLM mirror (https://benchlm.ai/benchmarks/math-500) because the Kaggle page is rendered client-side
- Hugging Face Open LLM Leaderboard v2 (MATH Lvl 5, 4-shot, dataset open-llm-leaderboard/contents), bf16 run
  preferred over fp16, plus the "Official Providers" models that are not in data/models as leftovers
- llm-stats (https://llm-stats.com/benchmarks/math, /math-500, /math-(cot)), using the source each entry cites
- Model cards, technical reports and announcements of the creators (and the third-party numbers they report)

Choices specific to MATH:
- OpenAI reports the o-series "MATH" scores on the MATH-500 subset (o1 / o1-mini / o3-mini posts): written as '-500'
- "MATH Lvl 5" / "MATH(4-shot)" columns of the Falcon cards are the Open LLM Leaderboard metric: written as 'Level 5'
- Base and instruct models share a version when the dataset does so; Qwen3 hybrid models compared by third parties
  without saying the mode get effort None
- llm-stats values not found in a plausible source are skipped (see the report in the commit / PR)
- cost is only given by Vals AI (real cost per test, not API prices)
"""

from common import A, HF, L, R, SE, write

P500, L5 = '-500', 'Level 5'
EPOCH = 'https://epoch.ai/benchmarks/math-level-5'
AA = 'https://artificialanalysis.ai/evaluations/math-500'
VALS = 'https://www.vals.ai/benchmarks/math500'
KAGGLE = 'https://www.kaggle.com/benchmarks/open-benchmarks/math-500'
OLB = 'https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard'
HF_LAB = ('smollm', 'zephyr', 'distilgpt2')	# models of Hugging Face, which runs the Open LLM Leaderboard


def EP(model, version, frac, stderr, effort='auto'):
	"""Epoch AI MATH Level 5 run (0-shot), score and standard error as fractions."""
	R(model, version, 100 * frac, 0, EPOCH, False, effort=effort, uncertainty=SE(100 * stderr), part=L5)


def AAR(model, version, frac, effort='auto'):
	"""Artificial Analysis MATH-500 run (0-shot), score as a fraction."""
	R(model, version, 100 * frac, 0, AA, False, effort=effort, part=P500)


def VA(model, version, acc, stderr, cost, effort='auto'):
	"""Vals AI MATH 500 run (0-shot): accuracy and stderr in %, cost per test in $."""
	R(model, version, acc, 0, VALS, False, effort=effort, uncertainty=SE(stderr), cost=None if cost is None else (cost, 'TASK'), part=P500)


def KG(model, version, frac, effort='auto'):
	"""Kaggle Open Benchmarks MATH-500 run (task version 8), score as a fraction."""
	R(model, version, 100 * frac, 0, KAGGLE, False, effort=effort, part=P500)


def OL(model, version, score):
	"""Open LLM Leaderboard v2 MATH Lvl 5 (4-shot), score in %."""
	R(model, version, score, 4, OLB, model.startswith(HF_LAB), part=L5)


# ---------------------------------------------------------------- Papers (full MATH unless stated)

# Measuring Mathematical Problem Solving With the MATH Dataset (GPT-2 fine-tuned on MATH, GPT-3 few-shot marked *)
MP = A + '2103.03874'
R('gpt-2', 'gpt-2-124m', 5.4, 0, MP, False)
R('gpt-2', 'gpt-2-355m', 6.2, 0, MP, False)
R('gpt-2', 'gpt-2-774m', 6.4, 0, MP, False)
R('gpt-2', 'gpt-2-1-5b', 6.9, 0, MP, False)
R('gpt-3', 'gpt-3-13b', 3.0, 0, MP, False)		# few-shot
R('gpt-3', 'gpt-3-13b', 5.6, 0, MP, False)		# fine-tuned
R('gpt-3', 'gpt-3-175b', 5.2, 0, MP, False)		# few-shot

# Solving Quantitative Reasoning Problems with Language Models (Minerva, 4-shot; maj1@k with k = 256, 64 for 540B)
MIN = A + '2206.14858'
R('palm', 'palm-8b', 1.5, 4, MIN, True)
R('palm', 'palm-62b', 4.4, 4, MIN, True)
R('palm', 'palm-540b', 8.8, 4, MIN, True)
R('palm', 'minerva-8b', 14.1, 4, MIN, True)
R('palm', 'minerva-62b', 27.6, 4, MIN, True)
R('palm', 'minerva-540b', 33.6, 4, MIN, True)
R('palm', 'minerva-8b', 25.4, 4, MIN, True, maj_k=256)
R('palm', 'minerva-62b', 43.4, 4, MIN, True, maj_k=256)
R('palm', 'minerva-540b', 50.3, 4, MIN, True, maj_k=64)
R('gpt-3-5', 'text-davinci-002', 19.1, 4, MIN, False)

# LLaMA: Open and Efficient Foundation Language Models (4-shot, maj1@256)
LLA = A + '2302.13971'
for m, s, maj in [('llama-7b', 2.9, 6.9), ('llama-13b', 3.9, 8.8), ('llama-33b', 7.1, 15.2), ('llama-65b', 10.6, 20.5)]:
	R(m, m, s, 4, LLA, True)
	R(m, m, maj, 4, LLA, True, maj_k=256)

# Llama 2: Open Foundation and Fine-Tuned Chat Models (Table 25, 4-shot)
LL2 = A + '2307.09288'
R('mpt-7b', 'mpt-7b', 3.0, 4, LL2, False)
R('mpt-30b', 'mpt-30b', 3.1, 4, LL2, False)
R('falcon-7b', 'falcon-7b', 2.3, 4, LL2, False)
R('falcon-40b', 'falcon-40b', 5.5, 4, LL2, False)
R('llama-7b', 'llama-7b', 2.9, 4, LL2, True)
R('llama-13b', 'llama-13b', 3.9, 4, LL2, True)
R('llama-33b', 'llama-33b', 7.1, 4, LL2, True)
R('llama-65b', 'llama-65b', 10.6, 4, LL2, True)
R('llama-2-7b', 'llama-2-7b', 2.5, 4, LL2, True)
R('llama-2-13b', 'llama-2-13b', 3.9, 4, LL2, True)
R('llama-2-34b', 'llama-2-34b', 6.24, 4, LL2, True)
R('llama-2-70b', 'llama-2-70b', 13.5, 4, LL2, True)

# Mistral 7B and Mixtral of Experts (4-shot, maj@4)
MIS = A + '2310.06825'
R('llama-2-7b', 'llama-2-7b', 3.9, 4, MIS, False, maj_k=4)
R('llama-2-13b', 'llama-2-13b', 6.0, 4, MIS, False, maj_k=4)
R('llama-2-7b', 'code-llama-7b', 5.2, 4, MIS, False, maj_k=4)
R('mistral-7b', 'mistral-7b-v0-1', 13.1, 4, MIS, True, maj_k=4)
MIX = A + '2401.04088'
R('llama-2-7b', 'llama-2-7b', 3.9, 4, MIX, False, maj_k=4)
R('llama-2-13b', 'llama-2-13b', 6.0, 4, MIX, False, maj_k=4)
R('llama-33b', 'llama-33b', 8.4, 4, MIX, False, maj_k=4)
R('llama-2-70b', 'llama-2-70b', 13.8, 4, MIX, False, maj_k=4)
R('mistral-7b', 'mistral-7b-v0-1', 12.7, 4, MIX, True, maj_k=4)
R('mixtral-8x7b', 'mixtral-8x7b', 28.4, 4, MIX, True, maj_k=4)

# PaLM 2 Technical Report (Table 7, 4-shot chain-of-thought)
PALM2 = A + '2305.10403'
R('palm-2', 'palm-2-l', 34.3, 4, PALM2, True)
R('gpt-4', None, 42.5, 0, PALM2, False)
L('Flan-PaLM 2', 33.2, PALM2)

# Gemini: A Family of Highly Capable Multimodal Models (4-shot)
GEM1 = A + '2312.11805'
R('gemini-1-0-ultra', 'gemini-1-0-ultra', 53.2, 4, GEM1, True)
R('gemini-1-0-pro', 'gemini-1-0-pro', 32.6, 4, GEM1, True)
R('gpt-4', None, 52.9, 4, GEM1, False)
R('gpt-3-5-turbo', None, 34.1, 4, GEM1, False)
R('palm-2', 'palm-2-l', 34.4, 4, GEM1, False)
L('Inflection-2', 34.8, GEM1)
R('grok-1', 'grok-1', 23.9, 4, GEM1, False)
R('llama-2-70b', 'llama-2-70b', 13.5, 4, GEM1, False)

# Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context (4-shot Minerva prompt)
G15 = A + '2403.05530'
R('gemini-1-5-pro', 'gemini-1-5-pro-may-2024', 67.7, 4, G15, True)
R('gemini-1-5-pro', 'gemini-1-5-pro-may-2024', 77.9, 4, G15, True, maj_k=64)
R('gemini-1-5-pro', 'gemini-1-5-pro-feb-2024', 58.5, 4, G15, True)
R('gemini-1-5-flash', 'gemini-1-5-flash-may-2024', 54.9, 4, G15, True)
R('gemini-1-5-flash-8b', None, 35.9, 4, G15, True)
R('gemini-1-0-ultra', 'gemini-1-0-ultra', 53.2, 4, G15, True)
R('gemini-1-0-pro', 'gemini-1-0-pro', 32.6, 4, G15, True)
R('claude-3-opus', 'claude-3-opus', 60.1, 0, G15, False)
R('gpt-4-turbo', None, 73.4, 0, G15, False)

# The Llama 3 Herd of Models (Table 2, 0-shot CoT; "Llama 3" there is Llama 3.1)
L3 = A + '2407.21783'
R('llama-3-1-8b', 'llama-3-1-8b', 51.9, 0, L3, True)
R('gemma-2-9b', 'gemma-2-9b', 44.3, 0, L3, False)
R('mistral-7b', None, 13.0, 0, L3, False)
R('llama-3-1-70b', 'llama-3-1-70b', 68.0, 0, L3, True)
R('mixtral-8x22b', 'mixtral-8x22b', 54.1, 0, L3, False)
R('gpt-3-5-turbo', None, 43.1, 0, L3, False)
R('llama-3-1-405b', 'llama-3-1-405b', 73.8, 0, L3, True)
R('nemotron-4-340b', 'nemotron-4-340b', 41.1, 0, L3, False)
R('gpt-4-turbo', 'gpt-4-turbo-jan-2024', 64.5, 0, L3, False)
R('gpt-4o', None, 76.6, 0, L3, False)
R('claude-3-5-sonnet', 'claude-3-5-sonnet-jun-2024', 71.1, 0, L3, False)

# Gemma 3 Technical Report (Table 6, instruction-tuned models, 0-shot)
G3R = A + '2503.19786'
R('gemini-1-5-flash', 'gemini-1-5-flash-sep-2024', 77.9, 0, G3R, True)
R('gemini-1-5-pro', 'gemini-1-5-pro-sep-2024', 86.5, 0, G3R, True)
R('gemini-2-0-flash', 'gemini-2-0-flash', 90.9, 0, G3R, True)
R('gemini-2-0-pro', 'gemini-2-0-pro-experimental', 91.8, 0, G3R, True)
R('gemma-2-2b', 'gemma-2-2b', 27.2, 0, G3R, True)
R('gemma-2-9b', 'gemma-2-9b', 49.4, 0, G3R, True)
R('gemma-2-27b', 'gemma-2-27b', 55.6, 0, G3R, True)
R('gemma-3-1b', 'gemma-3-1b', 48.0, 0, G3R, True)
R('gemma-3-4b', 'gemma-3-4b', 75.6, 0, G3R, True)
R('gemma-3-12b', 'gemma-3-12b', 83.8, 0, G3R, True)
R('gemma-3-27b', 'gemma-3-27b', 89.0, 0, G3R, True)

# Qwen3 Technical Report (base models: 4-shot CoT; post-trained flagship: MATH-500)
Q3 = A + '2505.09388'
R('qwen2-5-72b', 'qwen2-5-72b', 62.12, 4, Q3, True)
R('llama-4-maverick', 'llama-4-maverick', 63.32, 4, Q3, False)
R('deepseek-v3', 'deepseek-v3-dec-2024', 62.62, 4, Q3, False)
R('qwen3-235b-a22b', 'qwen3-235b-a22b', 71.84, 4, Q3, True, effort='')
R('qwen3-235b-a22b', 'qwen3-235b-a22b', 98.0, 0, Q3, True, effort='Thinking', part=P500)
R('qwen3-235b-a22b', 'qwen3-235b-a22b', 91.2, 0, Q3, True, effort='', part=P500)
R('qwen3-14b', 'qwen3-14b', 96.8, 0, Q3, True, effort='Thinking', part=P500)		# via llm-stats

# Qwen2.5-Coder Technical Report (instruct models, via llm-stats)
QC = A + '2409.12186'
R('qwen2-5-32b', 'qwen2-5-coder-32b', 57.2, 0, QC, True)
R('qwen2-5-7b', 'qwen2-5-coder-7b', 46.6, 0, QC, True)

# Kimi k1.5: Scaling Reinforcement Learning with LLMs (MATH-500)
K15 = A + '2501.12599'
R('kimi-k1-5', 'kimi-k1-5', 96.2, 0, K15, True, part=P500)
R('kimi-k1-5', 'kimi-k1-5-short-cot', 94.6, 0, K15, True, part=P500)

# DeepSeek-R1 (MATH-500, pass@1): R1-Zero, via llm-stats
R('deepseek-r1', 'deepseek-r1-zero', 95.9, 0, A + '2501.12948', True, part=P500)

# ---------------------------------------------------------------- Announcements of the creators

# OpenAI (the o-series "MATH" numbers are MATH-500)
R('gpt-4o', 'gpt-4o-may-2024', 76.6, 0, 'https://openai.com/index/hello-gpt-4o/', True)
R('gpt-4-turbo', 'gpt-4-turbo-apr-2024', 72.6, 0, 'https://openai.com/index/hello-gpt-4o/', True)
R('gpt-4o-mini', 'gpt-4o-mini', 70.2, 0, 'https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/', True)
O1M = 'https://openai.com/index/openai-o1-mini-advancing-cost-efficient-reasoning/'
R('o1-mini', 'o1-mini', 90.0, 0, O1M, True, part=P500)
R('o1', 'o1-preview', 85.5, 0, O1M, True, part=P500)
R('gpt-4o', 'gpt-4o-may-2024', 60.3, 0, O1M, True, part=P500)
R('o1', 'o1', 94.8, 0, 'https://openai.com/index/learning-to-reason-with-llms/', True, part=P500)
R('o3-mini', 'o3-mini', 97.9, 0, 'https://openai.com/index/openai-o3-mini/', True, effort='High', part=P500)

# Anthropic (0-shot CoT)
CL3 = 'https://www.anthropic.com/news/claude-3-family'
R('claude-3-opus', 'claude-3-opus', 60.1, 0, CL3, True)
R('claude-3-sonnet', 'claude-3-sonnet', 43.1, 0, CL3, True)
R('claude-3-haiku', 'claude-3-haiku', 38.9, 0, CL3, True)
R('claude-3-5-sonnet', 'claude-3-5-sonnet-jun-2024', 71.1, 0, 'https://www.anthropic.com/news/claude-3-5-sonnet', True)
R('claude-3-5-sonnet', 'claude-3-5-sonnet-oct-2024', 78.3, 0, 'https://www.anthropic.com/news/3-5-models-and-computer-use', True)
R('claude-3-5-haiku', 'claude-3-5-haiku', 69.4, 0, 'https://www.anthropic.com/claude/haiku', True)		# via llm-stats
CL37 = 'https://www.anthropic.com/news/claude-3-7-sonnet'
R('claude-3-7-sonnet', 'claude-3-7-sonnet', 82.2, 0, CL37, True, effort='', part=P500)
R('claude-3-7-sonnet', 'claude-3-7-sonnet', 96.2, 0, CL37, True, effort='Thinking', part=P500)

# Google
GDEC = 'https://blog.google/technology/google-deepmind/google-gemini-ai-update-december-2024/'
R('gemini-2-0-flash', 'gemini-2-0-flash-experimental', 89.7, 0, GDEC, True)
R('gemini-1-5-pro', 'gemini-1-5-pro-sep-2024', 86.5, 0, GDEC, True)
R('gemini-1-5-flash', 'gemini-1-5-flash-sep-2024', 77.9, 0, GDEC, True)
R('gemini-2-0-flash-lite', 'gemini-2-0-flash-lite', 86.8, 0, 'https://ai.google.dev/gemini-api/docs/models', True)		# via llm-stats
R('gemini-1-5-flash-8b', 'gemini-1-5-flash-8b', 58.7, 0, 'https://deepmind.google/technologies/gemini/flash/', True)	# via llm-stats

# xAI
R('grok-1-5', 'grok-1-5', 50.6, 4, 'https://x.ai/news/grok-1.5', True)
R('grok-2', 'grok-2-aug-2024', 76.1, 0, 'https://x.ai/news/grok-2', True)
R('grok-2-mini', 'grok-2-mini', 73.0, 0, 'https://x.ai/news/grok-2', True)

# Mistral
R('ministral-8b', 'ministral-8b', 54.5, 0, HF + 'mistralai/Ministral-8B-Instruct-2410', True)		# via llm-stats
R('mistral-nemo', 'pixtral-12b', 48.1, 0, 'https://mistral.ai/news/pixtral-12b/', True)
R('mistral-small-3', 'mistral-small-3', 70.6, 0, HF + 'mistralai/Mistral-Small-24B-Instruct-2501', True)

# DeepSeek / Zhipu / NVIDIA (MATH-500, via llm-stats)
R('deepseek-v3', 'deepseek-v3-mar-2025', 94.0, 0, 'https://api-docs.deepseek.com/news/news250325', True, part=P500)
R('glm-4-5', 'glm-4-5', 98.2, 0, 'https://z.ai/blog/glm-4.5', True, effort='Thinking', part=P500)
R('glm-4-5-air', 'glm-4-5-air', 98.1, 0, 'https://z.ai/blog/glm-4.5', True, effort='Thinking', part=P500)
R('llama-nemotron-ultra-253b', 'llama-nemotron-ultra-253b', 97.0, 0, 'https://build.nvidia.com/nvidia/llama-3_1-nemotron-ultra-253b-v1/modelcard', True, effort='Reasoning', part=P500)
R('llama-nemotron-super-49b', 'llama-nemotron-super-49b', 96.6, 0, 'https://build.nvidia.com/nvidia/llama-3_3-nemotron-super-49b-v1/modelcard', True, effort='Reasoning', part=P500)
R('llama-nemotron-nano-8b', 'llama-nemotron-nano-8b', 95.4, 0, 'https://build.nvidia.com/nvidia/llama-3_1-nemotron-nano-8b-v1/modelcard', True, effort='Reasoning', part=P500)

# Other models not in the list (via llm-stats)
NOVA = 'https://www.amazon.science/publications/the-amazon-nova-family-of-models-technical-report-and-model-card'
L('Nova Pro', 76.6, NOVA)
L('Nova Lite', 73.3, NOVA)
L('Nova Micro', 69.3, NOVA)
L('LongCat-Flash-Thinking', 99.2, 'https://github.com/meituan-longcat/LongCat-Flash-Thinking', part=P500)
L('LongCat-Flash-Lite', 96.8, HF + 'meituan-longcat/LongCat-Flash-Lite', part=P500)
L('LongCat-Flash-Chat', 96.4, HF + 'meituan-longcat/LongCat-Flash-Chat', part=P500)
L('Sarvam-105B', 98.6, 'https://www.sarvam.ai/blogs/sarvam-30b-105b', part=P500)
L('Sarvam-30B', 97.0, 'https://www.sarvam.ai/blogs/sarvam-30b-105b', part=P500)
L('Granite 3.3 8B Instruct', 69.02, HF + 'ibm-granite/granite-3.3-8b-instruct', part=P500)

# ---------------------------------------------------------------- Model cards (Hugging Face / GitHub / Google AI)

# Meta (GitHub model cards; MATH CoT, 0-shot for 3.1+ instruct, 4-shot for Llama 3 instruct and Llama 4 base)
LC3 = 'https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md'
R('llama-3-8b', 'llama-3-8b', 30.0, 4, LC3, True)
R('llama-2-7b', 'llama-2-7b', 3.8, 4, LC3, True)
R('llama-2-13b', 'llama-2-13b', 6.7, 4, LC3, True)
R('llama-3-70b', 'llama-3-70b', 50.4, 4, LC3, True)
R('llama-2-70b', 'llama-2-70b', 11.6, 4, LC3, True)
LC31 = 'https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md'
R('llama-3-8b', 'llama-3-8b', 29.1, 0, LC31, True)
R('llama-3-1-8b', 'llama-3-1-8b', 51.9, 0, LC31, True)
R('llama-3-70b', 'llama-3-70b', 51.0, 0, LC31, True)
R('llama-3-1-70b', 'llama-3-1-70b', 68.0, 0, LC31, True)
R('llama-3-1-405b', 'llama-3-1-405b', 73.8, 0, LC31, True)
LC32 = 'https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md'
R('llama-3-2-1b', 'llama-3-2-1b', 30.6, 0, LC32, True)
R('llama-3-2-1b', 'llama-3-2-1b-quantized', 25.3, 0, LC32, True)		# SpinQuant
R('llama-3-2-1b', 'llama-3-2-1b-quantized', 31.0, 0, LC32, True)		# QLoRA
R('llama-3-2-3b', 'llama-3-2-3b', 48.0, 0, LC32, True)
R('llama-3-2-3b', 'llama-3-2-3b-quantized', 45.3, 0, LC32, True)		# SpinQuant
R('llama-3-2-3b', 'llama-3-2-3b-quantized', 49.2, 0, LC32, True)		# QLoRA
LC32V = 'https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD_VISION.md'
R('llama-3-2-11b-vision', 'llama-3-2-11b-vision', 51.9, 0, LC32V, True)
R('llama-3-2-90b-vision', 'llama-3-2-90b-vision', 68.0, 0, LC32V, True)
R('llama-3-3-70b', 'llama-3-3-70b', 77.0, 0, 'https://github.com/meta-llama/llama-models/blob/main/models/llama3_3/MODEL_CARD.md', True)
LC4 = 'https://github.com/meta-llama/llama-models/blob/main/models/llama4/MODEL_CARD.md'	# base models, em_maj1@1
R('llama-3-1-70b', 'llama-3-1-70b', 41.6, 4, LC4, True)
R('llama-3-1-405b', 'llama-3-1-405b', 53.5, 4, LC4, True)
R('llama-4-scout', 'llama-4-scout', 50.3, 4, LC4, True)
R('llama-4-maverick', 'llama-4-maverick', 61.2, 4, LC4, True)

# Google (Gemma model cards, pre-trained models, 4-shot)
R('gemma-2b', 'gemma-2b', 11.8, 4, 'https://ai.google.dev/gemma/docs/model_card', True)
R('gemma-7b', 'gemma-7b', 24.3, 4, 'https://ai.google.dev/gemma/docs/model_card', True)
GC2 = 'https://ai.google.dev/gemma/docs/model_card_2'
R('gemma-2-2b', 'gemma-2-2b', 15.0, 4, GC2, True)
R('gemma-2-9b', 'gemma-2-9b', 36.6, 4, GC2, True)
R('gemma-2-27b', 'gemma-2-27b', 42.3, 4, GC2, True)
GC3 = 'https://ai.google.dev/gemma/docs/core/model_card_3'
R('gemma-3-4b', 'gemma-3-4b', 24.2, 4, GC3, True)
R('gemma-3-12b', 'gemma-3-12b', 43.3, 4, GC3, True)
R('gemma-3-27b', 'gemma-3-27b', 50.0, 4, GC3, True)

# Alibaba / Qwen
QW = HF + 'Qwen/Qwen-72B'	# 4-shot
R('llama-2-7b', 'llama-2-7b', 3.3, 4, QW, False)
R('llama-2-13b', 'llama-2-13b', 5.0, 4, QW, False)
R('llama-2-34b', 'llama-2-34b', 6.2, 4, HF + 'Qwen/Qwen-7B', False)
R('llama-2-70b', 'llama-2-70b', 12.0, 4, QW, False)
R('chatglm2-6b', 'chatglm2-6b', 6.5, 4, HF + 'Qwen/Qwen-7B', False)
L('InternLM-7B', 6.3, HF + 'Qwen/Qwen-7B')
L('InternLM-20B', 7.9, QW)
L('Baichuan2-7B', 5.6, HF + 'Qwen/Qwen-7B')
L('Baichuan2-13B', 10.1, HF + 'Qwen/Qwen-7B')
L('Yi-34B', 15.9, QW)
R('qwen-7b', 'qwen-7b', 11.6, 4, QW, True)
R('qwen-14b', 'qwen-14b', 24.8, 4, QW, True)
R('qwen-72b', 'qwen-72b', 35.2, 4, QW, True)
Q2 = HF + 'Qwen/'	# Qwen2 base models 4-shot, instruct models 0-shot
for card, rows in [('Qwen2-7B', [('mistral-7b', 'mistral-7b-v0-1', 13.1), ('gemma-7b', 'gemma-7b', 24.3), ('llama-3-8b', 'llama-3-8b', 20.5), ('qwen1-5-7b', 'qwen1-5-7b', 20.3), ('qwen2-7b', 'qwen2-7b', 44.2)]),
	('Qwen2-72B', [('deepseek-v2', 'deepseek-v2-may-2024', 43.6), ('mixtral-8x22b', 'mixtral-8x22b', 41.7), ('llama-3-70b', 'llama-3-70b', 42.5), ('qwen1-5-72b', 'qwen1-5-72b', 34.1), ('qwen1-5-110b', 'qwen1-5-110b', 49.6), ('qwen2-72b', 'qwen2-72b', 51.1)]),
	('Qwen2-57B-A14B', [('mixtral-8x7b', 'mixtral-8x7b', 30.8), ('qwen1-5-32b', 'qwen1-5-32b', 36.1), ('qwen2-57b-a14b', 'qwen2-57b-a14b', 43.0)]),
	('Qwen2-1.5B', [('phi-2', 'phi-2', 3.5), ('gemma-2b', 'gemma-2b', 11.8), ('qwen1-5-1-8b', 'qwen1-5-1-8b', 10.1), ('qwen2-0-5b', 'qwen2-0-5b', 10.7), ('qwen2-1-5b', 'qwen2-1-5b', 21.7)])]:
	for m, v, s in rows:
		R(m, v, s, 4, Q2 + card, m.startswith('qwen'))
L('Yi-1.5-34B', 41.7, Q2 + 'Qwen2-57B-A14B')
L('MiniCPM', 10.2, Q2 + 'Qwen2-1.5B')
for card, rows in [('Qwen2-7B-Instruct', [('llama-3-8b', 'llama-3-8b', 30.0), ('glm-4-9b', 'glm-4-9b-chat', 50.6), ('qwen1-5-7b', 'qwen1-5-7b', 23.2), ('qwen2-7b', 'qwen2-7b', 49.6)]),
	('Qwen2-72B-Instruct', [('llama-3-70b', 'llama-3-70b', 50.4), ('qwen1-5-72b', 'qwen1-5-72b', 42.5), ('qwen2-72b', 'qwen2-72b', 59.7)]),
	('Qwen2-57B-A14B-Instruct', [('mixtral-8x7b', 'mixtral-8x7b', 30.7), ('qwen1-5-32b', 'qwen1-5-32b', 42.4), ('qwen2-57b-a14b', 'qwen2-57b-a14b', 49.1)])]:
	for m, v, s in rows:
		R(m, v, s, 0, Q2 + card, m.startswith('qwen'))
L('Yi-1.5-9B-Chat', 47.7, Q2 + 'Qwen2-7B-Instruct')
L('Yi-1.5-34B-Chat', 50.1, Q2 + 'Qwen2-57B-A14B-Instruct')
Q25 = 'https://qwenlm.github.io/blog/qwen2.5-llm/'	# base models 4-shot, instruct models 0-shot
for m, v, s in [('llama-3-70b', 'llama-3-70b', 42.5), ('mixtral-8x22b', 'mixtral-8x22b', 41.7), ('llama-3-1-405b', 'llama-3-1-405b', 53.8), ('qwen2-72b', 'qwen2-72b', 50.9),
	('qwen2-5-72b', 'qwen2-5-72b', 62.1), ('qwen1-5-32b', 'qwen1-5-32b', 36.1), ('gemma-2-27b', 'gemma-2-27b', 42.7), ('qwen2-57b-a14b', 'qwen2-57b-a14b', 43.0),
	('qwen2-5-14b', 'qwen2-5-14b', 55.6), ('qwen2-5-32b', 'qwen2-5-32b', 57.7), ('mistral-7b', 'mistral-7b-v0-1', 10.2), ('llama-3-8b', 'llama-3-8b', 20.5),
	('gemma-2-9b', 'gemma-2-9b', 37.7), ('qwen2-7b', 'qwen2-7b', 43.5), ('qwen2-5-7b', 'qwen2-5-7b', 49.8), ('qwen2-0-5b', 'qwen2-0-5b', 11.2), ('qwen2-5-0-5b', 'qwen2-5-0-5b', 19.5),
	('qwen2-1-5b', 'qwen2-1-5b', 21.6), ('qwen2-5-1-5b', 'qwen2-5-1-5b', 35.0), ('gemma-2-2b', 'gemma-2-2b', 18.3), ('qwen2-5-3b', 'qwen2-5-3b', 42.6)]:
	R(m, v, s, 4, Q25, m.startswith('qwen'))
L('Yi-1.5-34B', 41.7, Q25)
for m, v, s in [('mistral-large-2', 'mistral-large-2', 69.9), ('llama-3-1-70b', 'llama-3-1-70b', 68.0), ('llama-3-1-405b', 'llama-3-1-405b', 73.8), ('qwen2-72b', 'qwen2-72b', 69.0),
	('qwen2-5-72b', 'qwen2-5-72b', 83.1), ('qwen2-57b-a14b', 'qwen2-57b-a14b', 49.1), ('gemma-2-27b', 'gemma-2-27b', 54.4), ('gpt-4o-mini', 'gpt-4o-mini', 70.2),
	('qwen-turbo', 'qwen-turbo-sep-2024', 81.0), ('qwen2-5-14b', 'qwen2-5-14b', 80.0), ('qwen2-5-32b', 'qwen2-5-32b', 83.1), ('gemma-2-9b', 'gemma-2-9b', 44.3),
	('llama-3-1-8b', 'llama-3-1-8b', 51.9), ('qwen2-7b', 'qwen2-7b', 52.9), ('qwen2-5-7b', 'qwen2-5-7b', 75.5), ('gemma-2-2b', 'gemma-2-2b', 26.6),
	('phi-3-5-mini', 'phi-3-5-mini-instruct', 48.5), ('qwen2-5-3b', 'qwen2-5-3b', 65.9), ('qwen2-0-5b', 'qwen2-0-5b', 13.9), ('qwen2-5-0-5b', 'qwen2-5-0-5b', 34.4),
	('qwen2-1-5b', 'qwen2-1-5b', 25.3), ('qwen2-5-1-5b', 'qwen2-5-1-5b', 55.2)]:
	R(m, v, s, 0, Q25, m.startswith('qwen'))
L('MiniCPM3-4B', 46.6, Q25)
QVL = HF + 'Qwen/Qwen2.5-VL-32B-Instruct'
R('qwen2-5-32b', 'qwen2-5-vl-32b', 82.2, 0, QVL, True)
R('mistral-small-3-1', 'mistral-small-3-1', 69.3, 0, QVL, False)
R('gemma-3-27b', 'gemma-3-27b', 89.0, 0, QVL, False)
R('gpt-4o-mini', 'gpt-4o-mini', 70.2, 0, QVL, False)
R('claude-3-5-haiku', 'claude-3-5-haiku', 69.2, 0, QVL, False)
QO = HF + 'Qwen/Qwen2.5-Omni-7B'
for m, v, s in [('qwen2-5-7b', 'qwen2-5-omni-7b', 71.5), ('qwen2-5-3b', 'qwen2-5-omni-3b', 63.6), ('qwen2-5-7b', 'qwen2-5-7b', 75.5), ('qwen2-5-3b', 'qwen2-5-3b', 65.9),
	('qwen2-7b', 'qwen2-7b', 52.9), ('llama-3-1-8b', 'llama-3-1-8b', 51.9), ('gemma-2-9b', 'gemma-2-9b', 44.3)]:
	R(m, v, s, 0, QO, m.startswith('qwen'))

# DeepSeek
DV3 = HF + 'deepseek-ai/DeepSeek-V3'
R('deepseek-v2', 'deepseek-v2-may-2024', 43.4, 4, DV3, True)		# base models
R('qwen2-5-72b', 'qwen2-5-72b', 54.4, 4, DV3, False)
R('llama-3-1-405b', 'llama-3-1-405b', 49.0, 4, DV3, False)
R('deepseek-v3', 'deepseek-v3-dec-2024', 61.6, 4, DV3, True)
for m, v, s, c in [('deepseek-v2', 'deepseek-v2-may-2024', 56.3, True), ('deepseek-v2-5', 'deepseek-v2-5-sep-2024', 74.7, True), ('qwen2-5-72b', 'qwen2-5-72b', 80.0, False),
	('llama-3-1-405b', 'llama-3-1-405b', 73.8, False), ('claude-3-5-sonnet', 'claude-3-5-sonnet-oct-2024', 78.3, False), ('gpt-4o', 'gpt-4o-may-2024', 74.6, False),
	('deepseek-v3', 'deepseek-v3-dec-2024', 90.2, True)]:
	R(m, v, s, 0, DV3, c, part=P500)
DR1 = HF + 'deepseek-ai/DeepSeek-R1'	# MATH-500 pass@1
for m, v, s, c in [('claude-3-5-sonnet', 'claude-3-5-sonnet-oct-2024', 78.3, False), ('gpt-4o', 'gpt-4o-may-2024', 74.6, False), ('deepseek-v3', 'deepseek-v3-dec-2024', 90.2, True),
	('o1-mini', 'o1-mini', 90.0, False), ('o1', 'o1', 96.4, False), ('deepseek-r1', 'deepseek-r1-jan-2025', 97.3, True), ('qwen2-5-32b', 'qwq-32b-preview', 90.6, False),
	('deepseek-r1-distill-qwen-1-5b', 'deepseek-r1-distill-qwen-1-5b', 83.9, True), ('deepseek-r1-distill-qwen-7b', 'deepseek-r1-distill-qwen-7b', 92.8, True),
	('deepseek-r1-distill-qwen-14b', 'deepseek-r1-distill-qwen-14b', 93.9, True), ('deepseek-r1-distill-qwen-32b', 'deepseek-r1-distill-qwen-32b', 94.3, True),
	('deepseek-r1-distill-llama-8b', 'deepseek-r1-distill-llama-8b', 89.1, True), ('deepseek-r1-distill-llama-70b', 'deepseek-r1-distill-llama-70b', 94.5, True)]:
	R(m, v, s, 0, DR1, c, part=P500)
R('deepseek-v2', 'deepseek-v2-may-2024', 53.9, 0, HF + 'deepseek-ai/DeepSeek-V2-Chat-0628', True)
R('deepseek-v2', 'deepseek-v2-jun-2024', 71.0, 0, HF + 'deepseek-ai/DeepSeek-V2-Chat-0628', True)
DV4 = HF + 'deepseek-ai/DeepSeek-V4-Pro'	# base models, 4-shot
R('deepseek-v3-2', 'deepseek-v3-2', 60.5, 4, DV4, True, effort='')
R('deepseek-v4-flash', 'deepseek-v4-flash-apr-2026', 57.4, 4, DV4, True)
R('deepseek-v4-pro', 'deepseek-v4-pro-apr-2026', 64.5, 4, DV4, True)
R('deepseek-v4-1-flash', None, 61.1, 4, HF + 'deepseek-ai/DeepSeek-V4.1-Flash', True)

# Microsoft
PH4 = HF + 'microsoft/phi-4'	# simple-evals
for m, v, s, c in [('phi-4', 'phi-4', 80.4, True), ('phi-3-medium', None, 44.6, True), ('qwen2-5-14b', 'qwen2-5-14b', 75.6, False), ('gpt-4o-mini', 'gpt-4o-mini', 73.0, False),
	('llama-3-3-70b', 'llama-3-3-70b', 66.3, False), ('qwen2-5-72b', 'qwen2-5-72b', 80.0, False), ('gpt-4o', None, 74.6, False)]:
	R(m, v, s, 0, PH4, c)
R('phi-4-mini', 'phi-4-mini', 64.0, 0, HF + 'microsoft/Phi-4-mini-instruct', True)
for card, first in [('Phi-3.5-mini-instruct', ('phi-3-5-mini', 'phi-3-5-mini-instruct', 48.5)), ('Phi-3.5-MoE-instruct', ('phi-3-5-moe', 'phi-3-5-moe', 59.5))]:
	R(*first, 0, HF + 'microsoft/' + card, True)
	for m, v, s in [('mistral-7b', 'mistral-7b-v0-3', 19.0), ('mistral-nemo', 'mistral-nemo', 31.2), ('llama-3-1-8b', 'llama-3-1-8b', 47.6), ('gemma-2-9b', 'gemma-2-9b', 50.9),
		('gemini-1-5-flash', 'gemini-1-5-flash-may-2024', 38.0), ('gpt-4o-mini', 'gpt-4o-mini', 70.2)]:
		R(m, v, s, 0, HF + 'microsoft/' + card, False)
PMR = HF + 'microsoft/Phi-4-mini-reasoning'	# MATH-500
for m, v, s, c in [('o1-mini', 'o1-mini', 90.0, False), ('deepseek-r1-distill-qwen-7b', 'deepseek-r1-distill-qwen-7b', 91.4, False),
	('deepseek-r1-distill-llama-8b', 'deepseek-r1-distill-llama-8b', 86.9, False), ('llama-3-2-3b', 'llama-3-2-3b', 44.4, False), ('phi-4-mini', 'phi-4-mini', 71.8, True),
	('phi-4-mini', 'phi-4-mini-reasoning', 94.6, True)]:
	R(m, v, s, 0, PMR, c, part=P500)
L('Bespoke-Stratos-7B', 82.0, PMR, part=P500)
L('OpenThinker-7B', 83.0, PMR, part=P500)
BIT = HF + 'microsoft/bitnet-b1.58-2B-4T'	# MATH-500
for m, v, s, c in [('llama-3-2-1b', 'llama-3-2-1b', 23.0, False), ('gemma-3-1b', 'gemma-3-1b', 42.0, False), ('qwen2-5-1-5b', 'qwen2-5-1-5b', 53.0, False),
	('smollm2-1-7b', 'smollm2-1-7b-instruct', 17.6, False), ('bitnet-b1-58-2b4t', 'bitnet-b1-58-2b4t', 43.4, True)]:
	R(m, v, s, 0, BIT, c, part=P500)
L('MiniCPM 2B', 14.8, BIT, part=P500)
WM = HF + 'WizardLMTeam/WizardMath-7B-V1.1'	# pass@1
for m, v, s, c in [('wizardmath-7b', 'wizardmath-7b-v1-1', 33.0, True), ('wizardmath-70b', 'wizardmath-70b', 22.7, True), ('wizardmath-13b', 'wizardmath-13b', 14.0, True),
	('wizardmath-7b', 'wizardmath-7b-v1-0', 10.7, True), ('mpt-7b', 'mpt-7b', 3.0, False), ('llama-7b', 'llama-7b', 2.9, False), ('llama-2-7b', 'llama-2-7b', 2.8, False),
	('mistral-7b', 'mistral-7b-v0-1', 9.1, False), ('qwen-7b', 'qwen-7b', 9.3, False), ('llemma-34b', 'llemma-34b', 25.0, False), ('palm', 'minerva-62b', 27.6, False),
	('llama-2-70b', 'llama-2-70b', 13.5, False), ('mixtral-8x7b', 'mixtral-8x7b', 28.4, False)]:
	R(m, v, s, 0, WM, c)
for n, s in [('Yi-6B', 5.8), ('MAmmoTH-7B (CoT)', 10.4), ('Abel-7B-001', 13.0), ('MetaMath-7B', 19.8), ('Arithmo-Mistral-7B', 25.3), ('MetaMath-Mistral-7B', 28.2),
	('Abel-7B-002', 29.5), ('MAmmoTH-70B', 21.1), ('Yi-34B', 15.9), ('MetaMath-70B', 26.6)]:
	L(n, s, WM)

# NVIDIA
for card, (m, v), vals in [('Nemotron-H-8B-Base-8K', ('nemotron-h-8b', 'nemotron-h-8b'), (46.52, 22.93, 44.43)),
	('Nemotron-H-56B-Base-8K', ('nemotron-h-56b', 'nemotron-h-56b'), (59.42, 35.19, 57.37)),
	('Nemotron-H-47B-Base-8K', ('nemotron-h-47b', 'nemotron-h-47b'), (57.4, 34.2, 57.9))]:	# 4-shot CoT
	for s, p in zip(vals, ('', L5, P500)):
		R(m, v, s, 4, HF + 'nvidia/' + card, True, part=p)
R('nemotron-h-4b', 'nemotron-h-4b', 76.4, 0, HF + 'nvidia/Nemotron-H-4B-Instruct-128K', True, part=P500)
NH8 = HF + 'nvidia/Nemotron-H-8B-Reasoning-128K'
R('llama-nemotron-nano-8b', 'llama-nemotron-nano-8b', 94.8, 0, NH8, True, effort='Reasoning', part=P500)
R('nemotron-h-8b', 'nemotron-h-8b-reasoning', 94.0, 0, NH8, True, effort='Reasoning', part=P500)
NH47 = HF + 'nvidia/Nemotron-H-47B-Reasoning-128K'
R('llama-nemotron-super-49b', 'llama-nemotron-super-49b', 96.6, 0, NH47, True, effort='Reasoning', part=P500)
R('nemotron-h-47b', 'nemotron-h-47b-reasoning', 96.2, 0, NH47, True, effort='Reasoning', part=P500)
R('qwen3-8b', 'qwen3-8b', 96.3, 0, HF + 'nvidia/NVIDIA-Nemotron-Nano-9B-v2', False, effort='Thinking', part=P500)
R('nvidia-nemotron-nano-9b-v2', 'nvidia-nemotron-nano-9b-v2', 97.8, 0, HF + 'nvidia/NVIDIA-Nemotron-Nano-9B-v2', True, effort='Reasoning', part=P500)
R('nvidia-nemotron-nano-12b-v2', 'nvidia-nemotron-nano-12b-v2', 97.75, 0, HF + 'nvidia/NVIDIA-Nemotron-Nano-12B-v2', True, effort='Reasoning', part=P500)
NNB = HF + 'nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base'	# base models, MATH Level 5
R('nvidia-nemotron-nano-12b-v2', 'nvidia-nemotron-nano-12b-v2', 67.61, 4, NNB, True, effort='', part=L5)
R('nvidia-nemotron-nano-9b-v2', 'nvidia-nemotron-nano-9b-v2', 63.64, 4, NNB, True, effort='', part=L5)
R('qwen3-8b', 'qwen3-8b', 29.91, 4, NNB, False, effort='', part=L5)
R('gemma-3-12b', 'gemma-3-12b', 17.71, 4, NNB, False, part=L5)
R('nemotron-3-nano-4b', 'nemotron-3-nano-4b', 95.4, 0, HF + 'nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16', True, effort='Reasoning', part=P500)
AMR = HF + 'nvidia/AceMath-RL-Nemotron-7B'	# MATH-500 avg@4
R('deepseek-r1-distill-qwen-7b', 'deepseek-r1-distill-qwen-7b', 92.8, 0, AMR, False, part=P500)
R('acemath-rl-nemotron-7b', 'acemath-rl-nemotron-7b', 94.1, 0, AMR, True, part=P500)
ACI = HF + 'nvidia/AceInstruct-7B'
for m, v, s, c in [('qwen2-5-1-5b', 'qwen2-5-1-5b', 55.2, False), ('aceinstruct-1-5b', 'aceinstruct-1-5b', 60.34, True), ('qwen2-5-7b', 'qwen2-5-7b', 75.5, False),
	('aceinstruct-7b', 'aceinstruct-7b', 76.4, True), ('qwen2-5-72b', 'qwen2-5-72b', 83.1, False), ('aceinstruct-72b', 'aceinstruct-72b', 84.5, True)]:
	R(m, v, s, 0, ACI, c)
OM2 = HF + 'nvidia/OpenMath2-Llama3.1-8B'
R('llama-3-1-8b', 'llama-3-1-8b', 51.9, 0, OM2, False)
R('openmath2-llama3-1-8b', 'openmath2-llama3-1-8b', 67.8, 0, OM2, True)
R('openmath2-llama3-1-8b', 'openmath2-llama3-1-8b', 76.1, 0, OM2, True, maj_k=256)
R('llama-3-1-70b', 'llama-3-1-70b', 67.9, 0, OM2, False)
R('openmath2-llama3-1-70b', 'openmath2-llama3-1-70b', 71.9, 0, OM2, True)
R('openmath2-llama3-1-70b', 'openmath2-llama3-1-70b', 79.6, 0, OM2, True, maj_k=256)
OM1 = HF + 'nvidia/OpenMath-Mistral-7B-v0.1-hf'	# greedy / majority@50
for m, greedy, maj in [('openmath-codellama-7b', 43.6, 55.6), ('openmath-mistral-7b', 44.5, 57.2), ('openmath-codellama-13b', 45.5, 57.6),
	('openmath-codellama-34b', 48.3, 60.2), ('openmath-llama-2-70b', 46.3, 58.3), ('openmath-codellama-70b', 50.7, 60.4)]:
	R(m, m, greedy, 0, OM1, True)
	R(m, m, maj, 0, OM1, True, maj_k=50)
NVLM = HF + 'nvidia/NVLM-D-72B'
R('nvlm-1-0', 'nvlm-d-72b', 73.1, 0, NVLM, True)
R('gemini-1-5-pro', 'gemini-1-5-pro-aug-2024', 67.7, 0, NVLM, False)
R('claude-3-5-sonnet', 'claude-3-5-sonnet-jun-2024', 71.1, 0, NVLM, False)
R('qwen2-72b', 'qwen2-72b', 59.7, 0, NVLM, False)
R('llama-3-70b', 'llama-3-70b', 51.0, 0, NVLM, False)
R('llama-3-1-70b', 'llama-3-1-70b', 68.0, 0, NVLM, False)
R('llama-3-1-405b', 'llama-3-1-405b', 73.8, 0, NVLM, False)
N3U = HF + 'nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-Base-BF16'	# base models, 4-shot EM
R('nemotron-3-ultra', 'nemotron-3-ultra', 82.0, 4, N3U, True, effort='')
R('deepseek-v3-2', 'deepseek-v3-2-exp', 60.12, 4, N3U, False, effort='')
R('mistral-large-3', 'mistral-large-3', 62.88, 4, N3U, False)
R('kimi-k2', 'kimi-k2-jul-2025', 68.4, 4, N3U, False)
R('glm-4-5', 'glm-4-5', 57.58, 4, N3U, False, effort='')
N3S = HF + 'nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-Base-BF16'
R('nemotron-3-super', 'nemotron-3-super', 84.84, 4, N3S, True, effort='')
R('nemotron-3-super', 'nemotron-3-super', 70.0, 4, N3S, True, effort='', part=L5)
L('Ling-flash-base-2.0', 63.8, N3S)
L('Ling-flash-base-2.0', 39.8, N3S, part=L5)
R('glm-4-5-air', 'glm-4-5-air', 50.36, 4, N3S, False, effort='')
R('glm-4-5-air', 'glm-4-5-air', 26.3, 4, N3S, False, effort='', part=L5)
N3N = HF + 'nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16'
R('qwen3-30b-a3b', 'qwen3-30b-a3b', 61.14, 4, N3N, False, effort='')
R('nemotron-3-nano', 'nemotron-3-nano', 82.88, 4, N3N, True, effort='')
R('qwen3-30b-a3b', 'qwen3-30b-a3b', 55.08, 4, N3N, False, effort='', part=P500)		# avg@32
R('nemotron-3-nano', 'nemotron-3-nano', 78.63, 4, N3N, True, effort='', part=P500)		# avg@32
R('nemotron-3-5-super', 'nemotron-3-5-super-vl', 81.56, 4, HF + 'nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Base-BF16', True, effort='', part=P500)	# n=32

# Mistral
MI3 = HF + 'mistralai/Ministral-3-14B-Instruct-2512'	# instruct models: MATH maj@1, base models: MATH CoT 2-shot (fractions)
for m, v, s, c, e in [('ministral-3-14b', 'ministral-3-14b', 0.904, True, 'auto'), ('qwen3-14b', 'qwen3-14b', 0.870, False, ''), ('gemma-3-12b', 'gemma-3-12b', 0.854, False, 'auto'),
	('ministral-3-8b', 'ministral-3-8b', 0.876, True, 'auto'), ('qwen3-8b', 'qwen3-vl-8b-instruct', 0.946, False, 'auto'), ('ministral-3-3b', 'ministral-3-3b', 0.830, True, 'auto'),
	('qwen3-4b', 'qwen3-vl-4b-instruct', 0.900, False, 'auto'), ('qwen3-1-7b', 'qwen3-vl-2b-instruct', 0.786, False, 'auto'), ('gemma-3-4b', 'gemma-3-4b', 0.759, False, 'auto')]:
	R(m, v, 100 * s, 0, MI3, c, effort=e)
for m, v, s, c in [('ministral-3-14b', 'ministral-3-14b', 0.676, True), ('qwen3-14b', 'qwen3-14b', 0.620, False), ('gemma-3-12b', 'gemma-3-12b', 0.487, False),
	('ministral-3-8b', 'ministral-3-8b', 0.626, True), ('qwen3-8b', 'qwen3-8b', 0.576, False), ('ministral-3-3b', 'ministral-3-3b', 0.601, True),
	('qwen3-4b', 'qwen3-4b', 0.405, False), ('gemma-3-4b', 'gemma-3-4b', 0.294, False)]:
	R(m, v, 100 * s, 2, MI3, c, effort='' if m.startswith('qwen3') else 'auto')
R('mistral-small-3-2', 'mistral-small-3-2', 69.42, 0, HF + 'mistralai/Mistral-Small-3.2-24B-Instruct-2506', True)
R('mistral-small-3-1', 'mistral-small-3-1', 69.3, 0, HF + 'mistralai/Mistral-Small-3.2-24B-Instruct-2506', True)
MS31 = HF + 'mistralai/Mistral-Small-3.1-24B-Instruct-2503'
R('mistral-small-3-1', 'mistral-small-3-1', 69.3, 0, MS31, True)
R('gemma-3-27b', 'gemma-3-27b', 89.0, 0, MS31, False)
R('gpt-4o-mini', 'gpt-4o-mini', 70.2, 0, MS31, False)
R('claude-3-5-haiku', 'claude-3-5-haiku', 69.2, 0, MS31, False)
L('Aya Vision 32B', 41.98, MS31)
R('mistral-small-3', 'mistral-small-3', 45.98, 4, HF + 'mistralai/Mistral-Small-24B-Base-2501', True)		# base model, "4-shot, MaJ" (k not given)
MATHS = HF + 'mistralai/Mathstral-7B-v0.1'
for m, v, s, c in [('mistral-7b', 'mathstral-7b', 56.6, True), ('deepseek-llm-7b', 'deepseekmath-7b', 44.4, False), ('llama-3-8b', 'llama-3-8b', 28.4, False),
	('glm-4-9b', 'glm-4-9b-chat', 50.2, False), ('qwen2-7b', 'qwen2-7b', 56.8, False), ('gemma-2-9b', 'gemma-2-9b', 48.3, False)]:
	R(m, v, s, 0, MATHS, c)

# Moonshot
KK2 = HF + 'moonshotai/Kimi-K2-Instruct'
for m, v, s, c, e in [('kimi-k2', 'kimi-k2-jul-2025', 97.4, True, 'auto'), ('deepseek-v3', 'deepseek-v3-mar-2025', 94.0, False, 'auto'),
	('qwen3-235b-a22b', 'qwen3-235b-a22b', 91.2, False, ''), ('claude-sonnet-4', 'claude-sonnet-4', 94.0, False, ''), ('claude-opus-4', 'claude-opus-4', 94.4, False, ''),
	('gpt-4-1', 'gpt-4-1', 92.4, False, 'auto'), ('gemini-2-5-flash', 'gemini-2-5-flash-may-2025', 95.4, False, None)]:
	R(m, v, s, 0, KK2, c, effort=e, part=P500)
for m, v, s, c in [('kimi-k2', 'kimi-k2-jul-2025', 70.2, True), ('deepseek-v3', 'deepseek-v3-dec-2024', 60.1, False), ('qwen2-5-72b', 'qwen2-5-72b', 61.0, False),
	('llama-4-maverick', 'llama-4-maverick', 63.0, False)]:	# base models, 4-shot EM
	R(m, v, s, 4, KK2, c)
MOON = HF + 'moonshotai/Moonlight-16B-A3B'	# base models
for m, v, s, c in [('llama-3-2-3b', 'llama-3-2-3b', 8.5, False), ('qwen2-5-3b', 'qwen2-5-3b', 42.6, False), ('deepseek-v2-lite', 'deepseek-v2-lite', 17.1, False),
	('moonlight-16b-a3b', 'moonlight-16b-a3b', 45.3, True)]:
	R(m, v, s, 4, MOON, c)

# MiniMax
MMT = HF + 'MiniMaxAI/MiniMax-Text-01'
for m, v, s, c in [('gpt-4o', 'gpt-4o-nov-2024', 76.6, False), ('claude-3-5-sonnet', 'claude-3-5-sonnet-oct-2024', 74.1, False), ('gemini-1-5-pro', 'gemini-1-5-pro-sep-2024', 84.6, False),
	('gemini-2-0-flash', 'gemini-2-0-flash-experimental', 83.9, False), ('qwen2-5-72b', 'qwen2-5-72b', 81.8, False), ('deepseek-v3', 'deepseek-v3-dec-2024', 84.6, False),
	('llama-3-1-405b', 'llama-3-1-405b', 73.8, False), ('minimax-text-01', 'minimax-text-01', 77.4, True)]:
	R(m, v, s, 0, MMT, c)
MM1 = HF + 'MiniMaxAI/MiniMax-M1-80k'	# MATH-500
for m, v, s, c, e in [('minimax-m1', 'minimax-m1-80k', 96.8, True, 'auto'), ('minimax-m1', 'minimax-m1-40k', 96.0, True, 'auto'),
	('qwen3-235b-a22b', 'qwen3-235b-a22b', 96.2, False, 'Thinking'), ('deepseek-r1', 'deepseek-r1-may-2025', 98.0, False, 'auto'),
	('deepseek-r1', 'deepseek-r1-jan-2025', 97.3, False, 'auto'), ('claude-opus-4', 'claude-opus-4', 98.2, False, 'Thinking'),
	('gemini-2-5-pro', 'gemini-2-5-pro', 98.8, False, 'auto'), ('o3', 'o3', 98.1, False, 'auto')]:
	R(m, v, s, 0, MM1, c, effort=e, part=P500)
L('Seed-Thinking-v1.5', 96.7, MM1, part=P500)
SYN = HF + 'MiniMaxAI/SynLogic-7B'	# MATH 500
R('qwen2-5-7b', 'qwen2-5-7b', 64.6, 0, SYN, False)		# base
R('qwen2-5-7b', 'qwen2-5-7b', 76.4, 0, SYN, False)		# instruct
R('synlogic-7b', 'synlogic-7b', 71.8, 0, SYN, True)

# Xiaomi
MIMO = HF + 'XiaomiMiMo/MiMo-7B-RL'	# MATH-500 pass@1
for m, v, s, c in [('gpt-4o', 'gpt-4o-may-2024', 74.6, False), ('claude-3-5-sonnet', 'claude-3-5-sonnet-oct-2024', 78.3, False), ('o1-mini', 'o1-mini', 90.0, False),
	('qwen2-5-32b', 'qwq-32b-preview', 90.6, False), ('deepseek-r1-distill-qwen-14b', 'deepseek-r1-distill-qwen-14b', 93.9, False),
	('deepseek-r1-distill-qwen-7b', 'deepseek-r1-distill-qwen-7b', 92.8, False), ('mimo-7b', 'mimo-7b-rl', 95.8, True), ('mimo-7b', 'mimo-7b-base', 37.4, True),
	('mimo-7b', 'mimo-7b-rl-zero', 93.6, True), ('mimo-7b', 'mimo-7b-sft', 93.0, True), ('mimo-7b', 'mimo-7b-rl-0530', 97.2, True)]:
	R(m, v, s, 0, MIMO, c, part=P500)
MV25 = HF + 'XiaomiMiMo/MiMo-V2.5-Pro'	# base models, 4-shot
R('mimo-v2-5-pro', 'mimo-v2-5-pro', 86.2, 4, MV25, True, effort='')
R('mimo-v2-5', 'mimo-v2-5', 67.7, 4, MV25, True, effort='')
R('deepseek-v4-pro', 'deepseek-v4-pro-apr-2026', 64.5, 4, MV25, False)
R('deepseek-v4-flash', 'deepseek-v4-flash-apr-2026', 57.4, 4, MV25, False)
R('kimi-k2', 'kimi-k2-jul-2025', 70.2, 4, MV25, False)
MV2F = HF + 'XiaomiMiMo/MiMo-V2-Flash'	# base models, 4-shot
R('mimo-v2-flash', 'mimo-v2-flash-dec-2025', 71.0, 4, MV2F, True, effort='')
R('kimi-k2', 'kimi-k2-jul-2025', 70.2, 4, MV2F, False)
R('deepseek-v3-1', 'deepseek-v3-1', 62.6, 4, MV2F, False, effort='')
R('deepseek-v3-2', 'deepseek-v3-2-exp', 62.5, 4, MV2F, False, effort='')

# Zhipu
for card, c in [('zai-org/glm-4-9b', True)]:	# base model (and Llama-3-8B-Instruct / ChatGLM3-6B-Base)
	R('llama-3-8b', 'llama-3-8b', 30.0, 0, HF + card, False)
	R('chatglm3-6b', 'chatglm3-6b', 25.7, 0, HF + card, False)
	R('glm-4-9b', 'glm-4-9b-chat', 30.4, 0, HF + card, c)
R('glm-4-9b', 'glm-4-9b-chat', 50.6, 0, HF + 'zai-org/glm-4-9b-chat', True)

# Hugging Face
SM3 = HF + 'HuggingFaceTB/SmolLM3-3B-Base'	# base models, 4-shot
for m, v, s, c in [('smollm3-3b', 'smollm3-3b-base', 46.1, True), ('qwen2-5-3b', 'qwen2-5-3b', 40.1, False), ('llama-3-2-3b', 'llama-3-2-3b', 7.44, False),
	('qwen3-1-7b', 'qwen3-1-7b', 41.64, False), ('qwen3-4b', 'qwen3-4b', 51.2, False)]:
	R(m, v, s, 4, SM3, c, effort='' if m.startswith('qwen3') else 'auto')
R('smollm2-1-7b', 'smollm2-1-7b-instruct', 20.12, 0, HF + 'HuggingFaceTB/SmolLM2-1.7B-Instruct-16k', True)
R('smollm2-1-7b', 'smollm2-1-7b-instruct-16k', 18.22, 0, HF + 'HuggingFaceTB/SmolLM2-1.7B-Instruct-16k', True)
OR1 = HF + 'open-r1/OpenR1-Distill-7B'	# MATH-500
R('open-r1-7b', 'openr1-distill-7b', 89.0, 0, OR1, True, part=P500)
R('deepseek-r1-distill-qwen-7b', 'deepseek-r1-distill-qwen-7b', 93.5, 0, OR1, False, part=P500)
R('open-r1-7b', 'openr1-qwen-7b', 90.6, 0, HF + 'open-r1/OpenR1-Qwen-7B', True, part=P500)
L('OpenThinker-7B', 86.4, HF + 'open-r1/OpenR1-Qwen-7B', part=P500)

# EleutherAI (Llemma, 4-shot; majority voting over 256 samples, 64 for Minerva 540B)
LLE = HF + 'EleutherAI/llemma_7b'
for m, v, s, c in [('llama-2-7b', 'llama-2-7b', 3.2, False), ('llama-2-7b', 'code-llama-7b', 4.5, False), ('llemma-7b', 'llemma-7b', 18.0, True), ('palm', 'minerva-8b', 14.1, False),
	('llama-2-34b', 'code-llama-34b', 12.2, False), ('llemma-34b', 'llemma-34b', 25.0, True), ('palm', 'minerva-62b', 27.6, False), ('palm', 'minerva-540b', 33.6, False)]:
	R(m, v, s, 4, LLE, c)
for m, v, s, c, k in [('llemma-7b', 'llemma-7b', 33.5, True, 256), ('palm', 'minerva-8b', 25.4, False, 256), ('llemma-34b', 'llemma-34b', 43.1, True, 256),
	('palm', 'minerva-62b', 43.4, False, 256), ('palm', 'minerva-540b', 50.3, False, 64)]:
	R(m, v, s, 4, LLE, c, maj_k=k)

# SeaLLMs (0-shot CoT, English MATH)
SEA = HF + 'SeaLLMs/SeaLLM-7B-v2.5'
R('gpt-3-5-turbo', None, 34.1, 0, SEA, False)
R('qwen-14b', 'qwen-14b', 18.4, 0, SEA, False)
R('qwen1-5-7b', 'qwen1-5-7b', 15.3, 0, SEA, False)
R('seallm-7b-v2', 'seallm-7b-v2', 27.5, 0, SEA, True)
R('seallm-7b-v2-5', 'seallm-7b-v2-5', 34.9, 0, SEA, True)
L('Vistral-7B-Chat', 12.5, SEA)

# Cerebras (REAP-pruned models, MATH-500)
RM2 = HF + 'cerebras/MiniMax-M2-REAP-172B-A10B'
R('minimax-m2', 'minimax-m2', 91.6, 0, RM2, False, part=P500)
R('minimax-m2-reap-172b-a10b', 'minimax-m2-reap-172b-a10b', 89.4, 0, RM2, True, part=P500)
R('minimax-m2-reap-162b-a10b', 'minimax-m2-reap-162b-a10b', 89.4, 0, RM2, True, part=P500)
R('minimax-m2-reap-139b-a10b', 'minimax-m2-reap-139b-a10b', 93.8, 0, RM2, True, part=P500)
RKL = HF + 'cerebras/Kimi-Linear-REAP-35B-A3B-Instruct'
R('kimi-linear-48b-a3b', 'kimi-linear-48b-a3b', 81.8, 0, RKL, False, part=P500)
R('kimi-linear-reap-35b-a3b-instruct', 'kimi-linear-reap-35b-a3b-instruct', 80.8, 0, RKL, True, part=P500)
RG46 = HF + 'cerebras/GLM-4.6-REAP-268B-A32B-FP8'
R('glm-4-6', 'glm-4-6', 95.5, 0, RG46, False, effort='Thinking', part=P500)
R('glm-4-6-reap-268b-a32b', 'glm-4-6-reap-268b-a32b', 97.0, 0, RG46, True, effort='Thinking', part=P500)
R('glm-4-6-reap-252b-a32b', 'glm-4-6-reap-252b-a32b', 94.8, 0, RG46, True, effort='Thinking', part=P500)
R('glm-4-6-reap-218b-a32b', 'glm-4-6-reap-218b-a32b', 93.3, 0, RG46, True, effort='Thinking', part=P500)
RG45 = HF + 'cerebras/GLM-4.5-Air-REAP-82B-A12B'
R('glm-4-5-air', 'glm-4-5-air', 94.8, 0, RG45, False, effort='Thinking', part=P500)
R('glm-4-5-air-reap-82b-a12b', 'glm-4-5-air-reap-82b-a12b', 94.8, 0, RG45, True, effort='Thinking', part=P500)

# TII (Falcon cards; the "MATH Lvl-5 (4-shot)" / "MATH (4-shot)" rows are the Open LLM Leaderboard v2 metric)
FH = HF + 'tiiuae/'
for card, rows in [
	('Falcon3-10B-Instruct', [('mistral-nemo', 'mistral-nemo', 6.5), ('gemma-2-9b', 'gemma-2-9b', 0.23), ('falcon3-10b', 'falcon3-10b-instruct', 25.91),
		('mistral-nemo', 'mistral-nemo', 5.9), ('falcon3-10b', 'falcon3-10b-instruct', 22.1)]),
	('Falcon3-10B-Base', [('gemma-2-9b', 'gemma-2-9b', 10.5), ('mistral-nemo', 'mistral-nemo', 4.9), ('falcon3-10b', 'falcon3-10b-base', 22.9), ('falcon3-10b', 'falcon3-10b-base', 24.77)]),
	('Falcon3-7B-Instruct', [('llama-3-1-8b', 'llama-3-1-8b', 19.34), ('qwen2-5-7b', 'qwen2-5-7b', 0.0), ('falcon3-7b', 'falcon3-7b-instruct', 31.87),
		('llama-3-1-8b', 'llama-3-1-8b', 15.4), ('falcon3-7b', 'falcon3-7b-instruct', 29.4)]),
	('Falcon3-7B-Base', [('llama-3-1-8b', 'llama-3-1-8b', 4.1), ('qwen2-7b', 'qwen2-7b', 17.5), ('qwen2-5-7b', 'qwen2-5-7b', 15.5), ('falcon3-7b', 'falcon3-7b-base', 18.0)]),
	('Falcon3-3B-Instruct', [('llama-3-2-3b', 'llama-3-2-3b', 14.6), ('qwen2-5-3b', 'qwen2-5-3b', 0.0), ('minitron-4b', 'nemotron-mini-4b-instruct', 0.0), ('falcon3-3b', 'falcon3-3b-instruct', 19.9)]),
	('Falcon3-3B-Base', [('llama-3-2-3b', 'llama-3-2-3b', 1.4), ('qwen2-5-3b', 'qwen2-5-3b', 8.4), ('minitron-4b', 'minitron-4b', 1.7), ('falcon3-3b', 'falcon3-3b-base', 9.4)]),
	('Falcon3-1B-Instruct', [('llama-3-2-1b', 'llama-3-2-1b', 5.2), ('qwen2-5-1-5b', 'qwen2-5-1-5b', 1.1), ('smollm2-1-7b', 'smollm2-1-7b-instruct', 3.1), ('falcon3-1b', 'falcon3-1b-instruct', 1.0)]),
	('Falcon3-1B-Base', [('llama-3-2-1b', 'llama-3-2-1b', 0.2), ('qwen2-5-1-5b', 'qwen2-5-1-5b', 6.7), ('smollm2-1-7b', 'smollm2-1-7b', 1.4), ('falcon3-1b', 'falcon3-1b-base', 2.2)]),
	('Falcon3-Mamba-7B-Instruct', [('llama-3-1-8b', 'llama-3-1-8b', 10.4), ('falcon3-mamba-7b', 'falcon3-mamba-7b-instruct', 27.3)]),
	('Falcon3-Mamba-7B-Base', [('llama-3-1-8b', 'llama-3-1-8b', 5.1), ('falcon-mamba-7b', 'falcon-mamba-7b', 3.6), ('falcon3-mamba-7b', 'falcon3-mamba-7b-base', 15.6)]),
	('Falcon3-7B-Instruct-1.58bit', [('falcon3-7b', 'falcon3-7b-instruct-1-58bit', 2.93)]), ('Falcon3-7B-Base-1.58bit', [('falcon3-7b', 'falcon3-7b-base-1-58bit', 0.88)]),
	('Falcon3-3B-Instruct-1.58bit', [('falcon3-3b', 'falcon3-3b-instruct-1-58bit', 0.77)]), ('Falcon3-3B-Base-1.58bit', [('falcon3-3b', 'falcon3-3b-base-1-58bit', 0.43)]),
	('Falcon3-1B-Instruct-1.58bit', [('falcon3-1b', 'falcon3-1b-instruct-1-58bit', 0.17)]), ('Falcon3-10B-Instruct-1.58bit', [('falcon3-10b', 'falcon3-10b-instruct-1-58bit', 2.44)]),
	('Falcon3-10B-Base-1.58bit', [('falcon3-10b', 'falcon3-10b-base-1-58bit', 0.48)]),
	('falcon-mamba-7b', [('falcon-mamba-7b', 'falcon-mamba-7b', 3.63), ('recurrentgemma-9b', 'recurrentgemma-9b', 4.83), ('falcon2-11b', 'falcon2-11b', 2.34),
		('llama-3-8b', 'llama-3-8b', 3.25), ('llama-3-1-8b', 'llama-3-1-8b', 4.61), ('mistral-7b', 'mistral-7b-v0-1', 2.49), ('mistral-nemo', 'mistral-nemo', 4.98), ('gemma-7b', 'gemma-7b', 6.42)])]:
	for m, v, s in rows:
		R(m, v, s, 4, FH + card, m.startswith('falcon'), part=L5)
for n, s, card in [('Yi-1.5-9B-Chat', 12.76, 'Falcon3-10B-Instruct'), ('Yi-1.5-9B-Chat', 3.3, 'Falcon3-10B-Instruct'), ('Yi-1.5-9B', 9.2, 'Falcon3-10B-Base'),
	('Zamba2-7B-Instruct', 13.6, 'Falcon3-Mamba-7B-Instruct'), ('Jamba-1.5-Mini', 6.9, 'Falcon3-Mamba-7B-Instruct'), ('Zamba2-7B', 10.3, 'Falcon3-Mamba-7B-Base'),
	('Llama3-8B-1.58-100B-tokens', 0.26, 'Falcon3-7B-Base-1.58bit'), ('mamba-7b-rw', 0.45, 'falcon-mamba-7b'), ('Zamba-7B-v1', 3.32, 'falcon-mamba-7b'),
	('RWKV-v6-Finch-7B', 1.11, 'falcon-mamba-7b'), ('RWKV-v6-Finch-14B', 1.13, 'falcon-mamba-7b')]:
	L(n, s, FH + card, part=L5)
FHI = {	# Falcon-H1 instruct cards: MATH-500 (Qwen3 mode not given)
	'Falcon-H1-7B-Instruct': [('falcon-h1-7b', 'falcon-h1-7b-instruct', 73.4), ('qwen3-8b', 'qwen3-8b', 83.8), ('qwen2-5-7b', 'qwen2-5-7b', 75.8), ('gemma-3-12b', 'gemma-3-12b', 86.2),
		('llama-3-1-8b', 'llama-3-1-8b', 45.8), ('falcon3-7b', 'falcon3-7b-instruct', 69.0), ('falcon3-10b', 'falcon3-10b-instruct', 68.6)],
	'Falcon-H1-3B-Instruct': [('falcon-h1-3b', 'falcon-h1-3b-instruct', 74.2), ('qwen3-4b', 'qwen3-4b', 85.0), ('qwen2-5-3b', 'qwen2-5-3b', 64.2), ('gemma-3-4b', 'gemma-3-4b', 76.4),
		('llama-3-2-3b', 'llama-3-2-3b', 41.2), ('falcon3-3b', 'falcon3-3b-instruct', 54.2)],
	'Falcon-H1-34B-Instruct': [('falcon-h1-34b', 'falcon-h1-34b-instruct', 83.8), ('qwen3-32b', 'qwen3-32b', 82.0), ('qwen2-5-72b', 'qwen2-5-72b', 83.6), ('qwen2-5-32b', 'qwen2-5-32b', 82.2),
		('gemma-3-27b', 'gemma-3-27b', 90.0), ('llama-3-3-70b', 'llama-3-3-70b', 70.6), ('llama-4-scout', 'llama-4-scout', 83.2)],
	'Falcon-H1-1.5B-Instruct': [('falcon-h1-1-5b', 'falcon-h1-1-5b-instruct', 74.0), ('qwen3-1-7b', 'qwen3-1-7b', 73.0), ('qwen2-5-1-5b', 'qwen2-5-1-5b', 48.4), ('gemma-3-1b', 'gemma-3-1b', 45.4),
		('llama-3-2-1b', 'llama-3-2-1b', 13.2), ('falcon3-1b', 'falcon3-1b-instruct', 19.8)],
	'Falcon-H1-1.5B-Deep-Instruct': [('falcon-h1-1-5b-deep', 'falcon-h1-1-5b-deep-instruct', 77.8)],
	'Falcon-H1-0.5B-Instruct': [('falcon-h1-0-5b', 'falcon-h1-0-5b-instruct', 58.4), ('qwen3-0-6b', 'qwen3-0-6b', 46.0), ('qwen2-5-0-5b', 'qwen2-5-0-5b', 27.8)],
}
for card, rows in FHI.items():
	for m, v, s in rows:
		R(m, v, s, 0, FH + card, m.startswith('falcon'), effort=None if m.startswith('qwen3') else 'auto', part=P500)
FHB = {	# Falcon-H1 base cards: MATH Level 5
	'Falcon-H1-7B-Base': [('falcon-h1-7b', 'falcon-h1-7b-base', 34.67), ('qwen3-8b', 'qwen3-8b', 28.85), ('qwen2-5-7b', 'qwen2-5-7b', 22.58), ('gemma-3-12b', 'gemma-3-12b', 17.22),
		('llama-3-1-8b', 'llama-3-1-8b', 6.57), ('falcon3-7b', 'falcon3-7b-base', 20.09), ('falcon3-10b', 'falcon3-10b-base', 25.38)],
	'Falcon-H1-3B-Base': [('falcon-h1-3b', 'falcon-h1-3b-base', 25.83), ('qwen3-4b', 'qwen3-4b', 24.47), ('qwen2-5-3b', 'qwen2-5-3b', 16.09), ('gemma-3-4b', 'gemma-3-4b', 6.95),
		('llama-3-2-3b', 'llama-3-2-3b', 1.74), ('falcon3-3b', 'falcon3-3b-base', 11.56)],
	'Falcon-H1-34B-Base': [('falcon-h1-34b', 'falcon-h1-34b-base', 40.71), ('qwen2-5-72b', 'qwen2-5-72b', 38.14), ('qwen2-5-32b', 'qwen2-5-32b', 36.4), ('gemma-3-27b', 'gemma-3-27b', 25.38),
		('llama-3-1-70b', 'llama-3-1-70b', 18.81), ('llama-4-scout', 'llama-4-scout', 27.19)],
	'Falcon-H1-1.5B-Base': [('falcon-h1-1-5b', 'falcon-h1-1-5b-base', 20.39), ('qwen3-1-7b', 'qwen3-1-7b', 16.39), ('qwen2-5-1-5b', 'qwen2-5-1-5b', 8.84), ('gemma-3-1b', 'gemma-3-1b', 1.21),
		('llama-3-2-1b', 'llama-3-2-1b', 0.98), ('falcon3-1b', 'falcon3-1b-base', 3.4)],
	'Falcon-H1-1.5B-Deep-Base': [('falcon-h1-1-5b-deep', 'falcon-h1-1-5b-deep-base', 24.77)],
	'Falcon-H1-0.5B-Base': [('falcon-h1-0-5b', 'falcon-h1-0-5b-base', 15.18), ('qwen3-0-6b', 'qwen3-0-6b', 9.29), ('qwen2-5-0-5b', 'qwen2-5-0-5b', 4.23)],
}
for card, rows in FHB.items():
	for m, v, s in rows:
		R(m, v, s, 4, FH + card, m.startswith('falcon'), effort='' if m.startswith('qwen3') else 'auto', part=L5)
FHR = FH + 'Falcon-H1R-7B'	# MATH500
for m, v, s, c, e in [('falcon-h1-7b', 'falcon-h1r-7b', 97.4, True, 'auto'), ('qwen3-8b', 'qwen3-8b', 97.4, False, 'Thinking'),
	('deepseek-r1-0528-qwen3-8b', 'deepseek-r1-0528-qwen3-8b', 96.8, False, 'auto'), ('phi-4', 'phi-4-reasoning-plus', 95.4, False, 'auto'),
	('gpt-oss-20b', 'gpt-oss-20b', 94.8, False, 'auto'), ('qwen3-32b', 'qwen3-32b', 96.8, False, 'Thinking'), ('nemotron-h-47b', 'nemotron-h-47b-reasoning', 91.4, False, 'Reasoning')]:
	R(m, v, s, 0, FHR, c, effort=e, part=P500)
L('Apriel-1.5-15b-Thinker', 97.2, FHR, part=P500)

# ---------------------------------------------------------------- Kaggle Open Benchmarks (MATH-500, task version 8)
KG('claude-sonnet-5-5', 'claude-sonnet-5-5', 0.994125)
KG('gemini-3-8-flash', 'gemini-3-8-flash', 0.991125)
KG('grok-4-6', 'grok-4-6', 0.99075)
KG('gpt-6-1-sol', 'gpt-6-1-sol', 0.990125)
KG('gemini-3-7-flash', 'gemini-3-7-flash', 0.99)
KG('gemini-4-argon', 'gemini-4-argon', 0.99)
KG('claude-haiku-5-5', 'claude-haiku-5-5', 0.987125)
KG('gpt-6-sol', 'gpt-6-sol', 0.986375)
KG('claude-opus-5', 'claude-opus-5', 0.976125)
KG('gpt-5-6-sol', 'gpt-5-6-sol', 0.9665)
KG('gemini-3-5-flash', 'gemini-3-5-flash', 0.963875)
KG('gpt-5-6-terra', 'gpt-5-6-terra', 0.959)
KG('gemini-3-6-flash', 'gemini-3-6-flash', 0.958625)
KG('gemini-3-flash', 'gemini-3-flash', 0.941125)
KG('gpt-5-6-luna', 'gpt-5-6-luna', 0.936875)
KG('claude-opus-4-8', 'claude-opus-4-8', 0.935125)
KG('grok-4-5', 'grok-4-5', 0.928625)
KG('gemini-3-5-flash-lite', 'gemini-3-5-flash-lite', 0.92825)
KG('claude-sonnet-5', 'claude-sonnet-5', 0.926736)
KG('gpt-5-5', 'gpt-5-5', 0.9265)
KG('claude-sonnet-4-6', 'claude-sonnet-4-6', 0.90375)
KG('gpt-oss-120b', 'gpt-oss-120b', 0.76525)

# ---------------------------------------------------------------- Epoch AI (MATH Level 5)
EP('claude-2', 'claude-2', 0.117258, 0.006596)	# claude-2.0
EP('claude-3-5-haiku', 'claude-3-5-haiku', 0.463557, 0.011612)	# claude-3-5-haiku-20241022
EP('claude-3-5-sonnet', 'claude-3-5-sonnet-jun-2024', 0.516805, 0.011625)	# claude-3-5-sonnet-20240620
EP('claude-3-5-sonnet', 'claude-3-5-sonnet-oct-2024', 0.569486, 0.011091)	# claude-3-5-sonnet-20241022
EP('claude-3-7-sonnet', 'claude-3-7-sonnet', 0.681835, 0.01067, effort='')	# claude-3-7-sonnet-20250219
EP('claude-3-7-sonnet', 'claude-3-7-sonnet', 0.862538, 0.009467, effort='Thinking')	# claude-3-7-sonnet-20250219_16K
EP('claude-3-7-sonnet', 'claude-3-7-sonnet', 0.900302, 0.008237, effort='Thinking')	# claude-3-7-sonnet-20250219_32K
EP('claude-3-7-sonnet', 'claude-3-7-sonnet', 0.911631, 0.007803, effort='Thinking')	# claude-3-7-sonnet-20250219_64K
EP('claude-3-haiku', 'claude-3-haiku', 0.148792, 0.00723)	# claude-3-haiku-20240307
EP('claude-3-opus', 'claude-3-opus', 0.374811, 0.010587)	# claude-3-opus-20240229
EP('claude-3-sonnet', 'claude-3-sonnet', 0.181741, 0.007982)	# claude-3-sonnet-20240229
EP('claude-haiku-4-5', 'claude-haiku-4-5', 0.869147, 0.007748, effort='')	# claude-haiku-4-5-20251001
EP('claude-haiku-4-5', 'claude-haiku-4-5', 0.963609, 0.005158, effort='Thinking')	# claude-haiku-4-5-20251001_32K
EP('claude-opus-4', 'claude-opus-4', 0.850453, 0.009805, effort='')	# claude-opus-4-20250514
EP('claude-sonnet-4', 'claude-sonnet-4', 0.843656, 0.009985, effort='')	# claude-sonnet-4-20250514
EP('claude-sonnet-4-5', 'claude-sonnet-4-5', 0.977341, 0.004091, effort='Thinking')	# claude-sonnet-4-5-20250929_32K
EP('dbrx', 'dbrx-instruct', 0.116503, 0.007356)	# dbrx-instruct
EP('deepseek-llm-67b', 'deepseek-llm-67b', 0.063916, 0.004017)	# deepseek-llm-67b-chat
EP('deepseek-r1', 'deepseek-r1-jan-2025', 0.930514, 0.006991)	# DeepSeek-R1
EP('deepseek-r1', 'deepseek-r1-may-2025', 0.96639, 0.004216)	# DeepSeek-R1-0528
EP('deepseek-r1-distill-llama-70b', 'deepseek-r1-distill-llama-70b', 0.89898, 0.006818)	# DeepSeek-R1-Distill-Llama-70B
EP('deepseek-r1-distill-qwen-14b', 'deepseek-r1-distill-qwen-14b', 0.871224, 0.007486)	# DeepSeek-R1-Distill-Qwen-14B
EP('deepseek-v3', 'deepseek-v3-dec-2024', 0.648508, 0.010486)	# DeepSeek-V3
EP('deepseek-v3', 'deepseek-v3-mar-2025', 0.755476, 0.008603)	# DeepSeek-V3-0324
EP('gemini-1-0-pro', 'gemini-1-0-pro', 0.112443, 0.005991)	# gemini-1.0-pro-001
EP('gemini-1-5-flash', 'gemini-1-5-flash-may-2024', 0.251227, 0.009456)	# gemini-1.5-flash-001
EP('gemini-1-5-flash', 'gemini-1-5-flash-sep-2024', 0.618674, 0.011201)	# gemini-1.5-flash-002
EP('gemini-1-5-pro', 'gemini-1-5-pro-may-2024', 0.407477, 0.010575)	# gemini-1.5-pro-001
EP('gemini-1-5-pro', 'gemini-1-5-pro-sep-2024', 0.703927, 0.010175)	# gemini-1.5-pro-002
EP('gemini-2-0-flash', 'gemini-2-0-flash', 0.821658, 0.008734)	# gemini-2.0-flash-001
EP('gemini-2-0-pro', 'gemini-2-0-pro-experimental', 0.834592, 0.010215)	# gemini-2.0-pro-exp-02-05
EP('gemini-2-5-pro', 'gemini-2-5-pro-mar-2025', 0.955627, 0.004475)	# gemini-2.5-pro-preview-03-25
EP('gemini-2-5-pro', 'gemini-2-5-pro-may-2025', 0.959026, 0.004221)	# gemini-2.5-pro-preview-05-06
EP('gemma-2-27b', 'gemma-2-27b', 0.27889, 0.009939)	# gemma-2-27b-it
EP('gemma-2-9b', 'gemma-2-9b', 0.210064, 0.008991)	# gemma-2-9b-it
EP('gemma-3-27b', 'gemma-3-27b', 0.74037, 0.011196)	# gemma-3-27b-it
EP('gpt-3-5-turbo', 'gpt-3-5-turbo-jan-2024', 0.116314, 0.005544)	# gpt-3.5-turbo-0125
EP('gpt-3-5-turbo', 'gpt-3-5-turbo-nov-2023', 0.158894, 0.006959)	# gpt-3.5-turbo-1106
EP('gpt-4-turbo', 'gpt-4-turbo-jan-2024', 0.354135, 0.010855)	# gpt-4-0125-preview
EP('gpt-4', 'gpt-4-jun-2023', 0.229702, 0.008466)	# gpt-4-0613
EP('gpt-4-turbo', 'gpt-4-turbo-nov-2023', 0.400208, 0.010539)	# gpt-4-1106-preview
EP('gpt-4-turbo', 'gpt-4-turbo-apr-2024', 0.467334, 0.011428)	# gpt-4-turbo-2024-04-09
EP('gpt-4-1', 'gpt-4-1', 0.83006, 0.00849)	# gpt-4.1-2025-04-14
EP('gpt-4-1-mini', 'gpt-4-1-mini', 0.872923, 0.007298)	# gpt-4.1-mini-2025-04-14
EP('gpt-4-1-nano', 'gpt-4-1-nano', 0.699962, 0.010232)	# gpt-4.1-nano-2025-04-14
EP('gpt-4-5', 'gpt-4-5', 0.786254, 0.011271)	# gpt-4.5-preview-2025-02-27
EP('gpt-4o', 'gpt-4o-may-2024', 0.51048, 0.011208)	# gpt-4o-2024-05-13
EP('gpt-4o', 'gpt-4o-aug-2024', 0.532761, 0.011221)	# gpt-4o-2024-08-06
EP('gpt-4o', 'gpt-4o-nov-2024', 0.497734, 0.011103)	# gpt-4o-2024-11-20
EP('gpt-4o-mini', 'gpt-4o-mini', 0.526341, 0.011275)	# gpt-4o-mini-2024-07-18
EP('gpt-5', 'gpt-5', 0.981307, 0.003192, effort='High')	# gpt-5-2025-08-07_high
EP('gpt-5', 'gpt-5', 0.97923, 0.003446, effort='Medium')	# gpt-5-2025-08-07_medium
EP('gpt-5-mini', 'gpt-5-mini', 0.978474, 0.003369, effort='High')	# gpt-5-mini-2025-08-07_high
EP('gpt-5-mini', 'gpt-5-mini', 0.9679, 0.003853, effort='Medium')	# gpt-5-mini-2025-08-07_medium
EP('gpt-5-nano', 'gpt-5-nano', 0.949018, 0.004849, effort='High')	# gpt-5-nano-2025-08-07_high
EP('gpt-5-nano', 'gpt-5-nano', 0.952417, 0.004627, effort='Medium')	# gpt-5-nano-2025-08-07_medium
EP('grok-2', 'grok-2-dec-2024', 0.635196, 0.013234)	# grok-2-1212
EP('grok-3', 'grok-3', 0.887462, 0.008688)	# grok-3-beta
EP('grok-3-mini', 'grok-3-mini', 0.880665, 0.008913, effort='High')	# grok-3-mini-beta_high
EP('grok-3-mini', 'grok-3-mini', 0.909366, 0.007893, effort='Low')	# grok-3-mini-beta_low
L('Hermes-2-Theta-Llama-3-70B', 100 * 0.226869, EPOCH, part=L5)
EP('llama-2-70b', 'llama-2-70b', 0.032855, 0.002542)	# Llama-2-70b-chat-hf
EP('llama-3-1-405b', 'llama-3-1-405b', 0.497734, 0.012185)	# Llama-3.1-405B-Instruct
EP('llama-3-1-70b', 'llama-3-1-70b', 0.366786, 0.01023)	# Llama-3.1-70B-Instruct
EP('llama-3-1-8b', 'llama-3-1-8b', 0.228758, 0.008519)	# Llama-3.1-8B-Instruct
L('Llama-3.1-Tulu-3-70B-DPO', 100 * 0.426643, EPOCH, part=L5)
EP('llama-3-2-90b-vision', 'llama-3-2-90b-vision', 0.394354, 0.010293)	# Llama-3.2-90B-Vision-Instruct
EP('llama-3-3-70b', 'llama-3-3-70b', 0.415974, 0.011817)	# Llama-3.3-70B-Instruct
EP('llama-4-maverick', 'llama-4-maverick', 0.730174, 0.010836)	# Llama-4-Maverick-17B-128E-Instruct-FP8
EP('llama-4-scout', 'llama-4-scout', 0.622734, 0.012022)	# Llama-4-Scout-17B-16E-Instruct
EP('llama-3-70b', 'llama-3-70b', 0.225548, 0.008258)	# Meta-Llama-3-70B-Instruct
EP('llama-3-8b', 'llama-3-8b', 0.061273, 0.003743)	# Meta-Llama-3-8B-Instruct
EP('ministral-3b', 'ministral-3b', 0.144449, 0.007446)	# ministral-3b-2410
EP('ministral-8b', 'ministral-8b', 0.149358, 0.007455)	# ministral-8b-2410
EP('mistral-7b', 'mistral-7b-v0-3', 0.035971, 0.002667)	# Mistral-7B-Instruct-v0.3
EP('mistral-large', 'mistral-large-1', 0.244619, 0.008923)	# mistral-large-2402
EP('mistral-large-2', 'mistral-large-2', 0.448168, 0.010703)	# mistral-large-2407
EP('mistral-large-2-1', 'mistral-large-2-1', 0.502832, 0.011469)	# mistral-large-2411
EP('mistral-medium-3', 'mistral-medium-3', 0.816276, 0.008996)	# mistral-medium-2505
EP('mistral-small-3', 'mistral-small-3', 0.448168, 0.010855)	# mistral-small-2501
EP('mistral-small-3-1', 'mistral-small-3-1', 0.467711, 0.011996)	# mistral-small-2503
EP('mixtral-8x7b', 'mixtral-8x7b', 0.0929, 0.005102)	# Mixtral-8x7B-Instruct-v0.1
EP('o1', 'o1', 0.94713, 0.006152, effort='High')	# o1-2024-12-17_high
EP('o1', 'o1', 0.944109, 0.006315, effort='Medium')	# o1-2024-12-17_medium
EP('o1-mini', 'o1-mini', 0.891805, 0.006803)	# o1-mini-2024-09-12_high
EP('o1-mini', 'o1-mini', 0.842523, 0.008257)	# o1-mini-2024-09-12_medium
EP('o1', 'o1-preview', 0.816465, 0.010643)	# o1-preview-2024-09-12
EP('o3', 'o3', 0.977719, 0.003323, effort='High')	# o3-2025-04-16_high
EP('o3-mini', 'o3-mini', 0.964879, 0.004265, effort='High')	# o3-mini-2025-01-31_high
EP('o3-mini', 'o3-mini', 0.951662, 0.004652, effort='Medium')	# o3-mini-2025-01-31_medium
EP('o4-mini', 'o4-mini', 0.978285, 0.00332, effort='High')	# o4-mini-2025-04-16_high
EP('mistral-7b', None, 0.03682, 0.002772)	# open-mistral-7b
EP('mistral-nemo', 'mistral-nemo', 0.108289, 0.006093)	# open-mistral-nemo-2407
EP('mixtral-8x22b', 'mixtral-8x22b', 0.242447, 0.008297)	# open-mixtral-8x22b
EP('mixtral-8x7b', 'mixtral-8x7b', 0.099509, 0.005303)	# open-mixtral-8x7b
EP('phi-3-medium', 'phi-3-medium-128k-instruct', 0.175604, 0.007296)	# Phi-3-medium-128k-instruct
EP('phi-4', 'phi-4', 0.649358, 0.010657)	# phi-4
EP('qwen2-5-max', 'qwen2-5-max', 0.671828, 0.011144)	# qwen-max-2025-01-25
EP('qwen-plus', 'qwen-plus-jan-2025', 0.652757, 0.011335)	# qwen-plus-2025-01-25
EP('qwen2-5-turbo', 'qwen2-5-turbo', 0.562311, 0.011365)	# qwen-turbo-2024-11-01
EP('qwen2-72b', 'qwen2-72b', 0.390672, 0.010317)	# qwen2-72b-instruct
EP('qwen2-5-32b', 'qwen2-5-32b', 0.560706, 0.010767)	# qwen2.5-32b-instruct
EP('qwen2-5-72b', 'qwen2-5-72b', 0.631703, 0.010714)	# qwen2.5-72b-instruct
EP('qwen3-235b-a22b', 'qwen3-235b-a22b', 0.68857, 0.009487)	# qwen3-235b-a22b
EP('qwen3-max', 'qwen3-max', 0.971299, 0.004201)	# qwen3-max-2025-09-23
EP('wizardlm-2-8x22b', 'wizardlm-2-8x22b', 0.257364, 0.00872)	# WizardLM-2-8x22B
L('Yi-1.5-34B-Chat', 100 * 0.254815, EPOCH, part=L5)
L('Yi-34B-Chat', 100 * 0.051454, EPOCH, part=L5)

# ---------------------------------------------------------------- Artificial Analysis (MATH-500, latest value of each model)
AAR('qwen2-5-32b', 'qwq-32b-preview', 0.91)	# QwQ 32B-Preview
L('Aya Expanse 32B', 100 * 0.449, AA, part=P500)
L('Aya Expanse 8B', 100 * 0.321, AA, part=P500)
AAR('claude-2-1', 'claude-2-1', 0.374)	# Claude 2.1
AAR('claude-3-5-haiku', 'claude-3-5-haiku', 0.720667)	# Claude 3.5 Haiku
AAR('claude-3-7-sonnet', 'claude-3-7-sonnet', 0.85, effort='')	# Claude 3.7 Sonnet (Non-reasoning)
AAR('claude-3-7-sonnet', 'claude-3-7-sonnet', 0.946667, effort='Thinking')	# Claude 3.7 Sonnet (Reasoning)
AAR('claude-3-haiku', 'claude-3-haiku', 0.394)	# Claude 3 Haiku
AAR('claude-3-opus', 'claude-3-opus', 0.640667)	# Claude 3 Opus
AAR('claude-3-sonnet', 'claude-3-sonnet', 0.414)	# Claude 3 Sonnet
AAR('claude-3-5-sonnet', 'claude-3-5-sonnet-oct-2024', 0.771333)	# Claude 3.5 Sonnet (Oct '24)
AAR('claude-3-5-sonnet', 'claude-3-5-sonnet-jun-2024', 0.695333)	# Claude 3.5 Sonnet (June '24)
AAR('claude-opus-4', 'claude-opus-4', 0.940667, effort='')	# Claude 4 Opus (Non-reasoning)
AAR('claude-opus-4', 'claude-opus-4', 0.982, effort='Thinking')	# Claude 4 Opus (Reasoning)
AAR('claude-sonnet-4', 'claude-sonnet-4', 0.934, effort='')	# Claude 4 Sonnet (Non-reasoning)
AAR('claude-sonnet-4', 'claude-sonnet-4', 0.990667, effort='Thinking')	# Claude 4 Sonnet (Reasoning)
AAR('claude-instant', None, 0.264)	# Claude Instant
AAR('codestral', 'codestral-25-01', 0.607)	# Codestral (Jan '25)
AAR('codestral-22b', 'codestral-22b', 0.354)	# Codestral (May '24)
AAR('codestral-mamba', 'codestral-mamba', 0.24)	# Codestral-Mamba
L('Command A', 100 * 0.818667, AA, part=P500)
L("Command-R (Aug '24)", 100 * 0.149, AA, part=P500)
L("Command-R (Mar '24)", 100 * 0.164, AA, part=P500)
L("Command-R+ (Aug '24)", 100 * 0.402, AA, part=P500)
L("Command-R+ (Apr '24)", 100 * 0.278667, AA, part=P500)
AAR('dbrx', 'dbrx-instruct', 0.279333)	# DBRX Instruct
L('DeepHermes 3 - Llama-3.1 8B Preview', 100 * 0.218, AA, part=P500)
L('DeepHermes 3 - Mistral 24B Preview', 100 * 0.594667, AA, part=P500)
AAR('deepseek-coder-v2', 'deepseek-coder-v2-jun-2024', 0.742667)	# DeepSeek-Coder-V2
AAR('deepseek-r1', 'deepseek-r1-may-2025', 0.982667)	# DeepSeek R1 0528 (May '25)
AAR('deepseek-r1', 'deepseek-r1-jan-2025', 0.966)	# DeepSeek R1 (Jan '25)
AAR('deepseek-r1-distill-llama-70b', 'deepseek-r1-distill-llama-70b', 0.934667)	# DeepSeek R1 Distill Llama 70B
AAR('deepseek-r1-distill-llama-8b', 'deepseek-r1-distill-llama-8b', 0.852667)	# DeepSeek R1 Distill Llama 8B
AAR('deepseek-r1-distill-qwen-1-5b', 'deepseek-r1-distill-qwen-1-5b', 0.687333)	# DeepSeek R1 Distill Qwen 1.5B
AAR('deepseek-r1-distill-qwen-14b', 'deepseek-r1-distill-qwen-14b', 0.948667)	# DeepSeek R1 Distill Qwen 14B
AAR('deepseek-r1-distill-qwen-32b', 'deepseek-r1-distill-qwen-32b', 0.940667)	# DeepSeek R1 Distill Qwen 32B
AAR('deepseek-r1-0528-qwen3-8b', 'deepseek-r1-0528-qwen3-8b', 0.932)	# DeepSeek R1 0528 Qwen3 8B
AAR('deepseek-v2-5', 'deepseek-v2-5-dec-2024', 0.762667)	# DeepSeek-V2.5 (Dec '24)
AAR('deepseek-v3', 'deepseek-v3-dec-2024', 0.886667)	# DeepSeek V3 (Dec '24)
AAR('deepseek-v3', 'deepseek-v3-mar-2025', 0.942)	# DeepSeek V3 0324
AAR('devstral-medium', 'devstral-medium', 0.707333)	# Devstral Medium
AAR('devstral-small-1-1', 'devstral-small-1-1', 0.634667)	# Devstral Small (Jul '25)
AAR('devstral-small-1-0', 'devstral-small-1-0', 0.684)	# Devstral Small (May '25)
L('ERNIE 4.5 300B A47B', 100 * 0.930667, AA, part=P500)
L('EXAONE 4.0 32B (Non-reasoning)', 100 * 0.939333, AA, part=P500)
L('EXAONE 4.0 32B (Reasoning)', 100 * 0.976667, AA, part=P500)
AAR('gemini-1-0-pro', 'gemini-1-0-pro', 0.402667)	# Gemini 1.0 Pro
AAR('gemini-1-5-flash', 'gemini-1-5-flash-sep-2024', 0.827333)	# Gemini 1.5 Flash (Sep '24)
AAR('gemini-1-5-flash-8b', 'gemini-1-5-flash-8b', 0.689333)	# Gemini 1.5 Flash-8B
AAR('gemini-1-5-flash', 'gemini-1-5-flash-may-2024', 0.554)	# Gemini 1.5 Flash (May '24)
AAR('gemini-1-5-pro', 'gemini-1-5-pro-sep-2024', 0.876)	# Gemini 1.5 Pro (Sep '24)
AAR('gemini-1-5-pro', 'gemini-1-5-pro-may-2024', 0.673333)	# Gemini 1.5 Pro (May '24)
AAR('gemini-2-0-flash', 'gemini-2-0-flash', 0.93)	# Gemini 2.0 Flash (Feb '25)
AAR('gemini-2-0-flash', 'gemini-2-0-flash-experimental', 0.910667)	# Gemini 2.0 Flash (experimental)
AAR('gemini-2-0-flash-lite', 'gemini-2-0-flash-lite', 0.872667)	# Gemini 2.0 Flash-Lite (Feb '25)
AAR('gemini-2-0-flash-lite', 'gemini-2-0-flash-lite-preview', 0.873333)	# Gemini 2.0 Flash-Lite (Preview)
AAR('gemini-2-0-flash', 'gemini-2-0-flash-thinking-jan-2025', 0.944)	# Gemini 2.0 Flash Thinking Experimental (Jan '25)
AAR('gemini-2-0-flash', 'gemini-2-0-flash-thinking-dec-2024', 0.48)	# Gemini 2.0 Flash Thinking Experimental (Dec '24)
AAR('gemini-2-0-pro', 'gemini-2-0-pro-experimental', 0.922667)	# Gemini 2.0 Pro Experimental (Feb '25)
AAR('gemini-2-5-flash', 'gemini-2-5-flash-may-2025', 0.932, effort='')	# Gemini 2.5 Flash (Non-reasoning)
AAR('gemini-2-5-flash', 'gemini-2-5-flash-apr-2025', 0.926, effort='')	# Gemini 2.5 Flash Preview (Non-reasoning)
AAR('gemini-2-5-flash-lite', None, 0.926)	# Gemini 2.5 Flash-Lite (Non-reasoning)
AAR('gemini-2-5-flash-lite', None, 0.968667)	# Gemini 2.5 Flash-Lite (Reasoning)
AAR('gemini-2-5-flash', 'gemini-2-5-flash-may-2025', 0.981333, effort='Thinking')	# Gemini 2.5 Flash (Reasoning)
AAR('gemini-2-5-flash', 'gemini-2-5-flash-apr-2025', 0.980667, effort='Thinking')	# Gemini 2.5 Flash Preview (Reasoning)
AAR('gemini-2-5-pro', 'gemini-2-5-pro', 0.967333)	# Gemini 2.5 Pro
AAR('gemini-2-5-pro', 'gemini-2-5-pro-mar-2025', 0.98)	# Gemini 2.5 Pro Preview (Mar' 25)
AAR('gemini-2-5-pro', 'gemini-2-5-pro-may-2025', 0.986)	# Gemini 2.5 Pro Preview (May' 25)
AAR('gemma-2-27b', 'gemma-2-27b', 0.541)	# Gemma 2 27B
AAR('gemma-2-9b', 'gemma-2-9b', 0.517)	# Gemma 2 9B
AAR('gemma-3-12b', 'gemma-3-12b', 0.853333)	# Gemma 3 12B Instruct
AAR('gemma-3-1b', 'gemma-3-1b', 0.484)	# Gemma 3 1B Instruct
AAR('gemma-3-27b', 'gemma-3-27b', 0.882667)	# Gemma 3 27B Instruct
AAR('gemma-3-4b', 'gemma-3-4b', 0.766)	# Gemma 3 4B Instruct
AAR('gemma-3n-e2b', 'gemma-3n-e2b', 0.691333)	# Gemma 3n E2B Instruct
AAR('gemma-3n-e4b', 'gemma-3n-e4b', 0.770667)	# Gemma 3n E4B Instruct
AAR('gemma-3n-e4b', 'gemma-3n-e4b-preview', 0.748667)	# Gemma 3n E4B Instruct Preview (May '25)
AAR('glm-4-5-air', 'glm-4-5-air', 0.965333, effort=None)	# GLM-4.5-Air
AAR('glm-4-5', 'glm-4-5', 0.978667, effort='Thinking')	# GLM-4.5 (Reasoning)
AAR('gpt-3-5-turbo', None, 0.440667)	# GPT-3.5 Turbo
AAR('gpt-4-1', 'gpt-4-1', 0.912667)	# GPT-4.1
AAR('gpt-4-1-mini', 'gpt-4-1-mini', 0.925333)	# GPT-4.1 mini
AAR('gpt-4-1-nano', 'gpt-4-1-nano', 0.848)	# GPT-4.1 nano
AAR('gpt-4-turbo', None, 0.736667)	# GPT-4 Turbo
AAR('gpt-4o', 'gpt-4o-nov-2024', 0.759333)	# GPT-4o (Nov '24)
AAR('gpt-4o', 'gpt-4o-may-2024', 0.791333)	# GPT-4o (May '24)
AAR('gpt-4o', 'gpt-4o-aug-2024', 0.794667)	# GPT-4o (Aug '24)
AAR('gpt-4o', None, 0.797333)	# GPT-4o (ChatGPT)
AAR('gpt-4o', 'gpt-4o-mar-2025', 0.892667)	# GPT-4o (March 2025, chatgpt-4o-latest)
AAR('gpt-4o-mini', 'gpt-4o-mini', 0.788667)	# GPT-4o mini
AAR('gpt-5', 'gpt-5', 0.994, effort='High')	# GPT-5 (High)
AAR('gpt-5', 'gpt-5', 0.987333, effort='Low')	# GPT-5 (low)
AAR('gpt-5', 'gpt-5', 0.991333, effort='Medium')	# GPT-5 (medium)
AAR('gpt-5', 'gpt-5', 0.861333, effort='Minimal')	# GPT-5 (minimal)
L('Granite 3.3 8B Instruct', 100 * 0.664667, AA, part=P500)
AAR('grok-2', 'grok-2-dec-2024', 0.778)	# Grok 2 (Dec '24)
AAR('grok-3', 'grok-3', 0.87)	# Grok 3
AAR('grok-3-mini', 'grok-3-mini', 0.992, effort='High')	# Grok 3 mini Reasoning (High)
AAR('grok-4', 'grok-4', 0.99)	# Grok 4
AAR('grok-2', 'grok-beta', 0.736667)	# Grok Beta
L('Hermes 3 - Llama-3.1 70B', 100 * 0.538, AA, part=P500)
L('Jamba 1.5 Large', 100 * 0.606, AA, part=P500)
L('Jamba 1.5 Mini', 100 * 0.356667, AA, part=P500)
L('Jamba 1.6 Large', 100 * 0.58, AA, part=P500)
L('Jamba 1.6 Mini', 100 * 0.256667, AA, part=P500)
L('Jamba 1.7 Large', 100 * 0.6, AA, part=P500)
L('Jamba 1.7 Mini', 100 * 0.258, AA, part=P500)
L('Jamba Instruct', 100 * 0.243, AA, part=P500)
AAR('kimi-k2', 'kimi-k2-jul-2025', 0.971333)	# Kimi K2
L('LFM 40B', 100 * 0.48, AA, part=P500)
AAR('llama-2-13b', 'llama-2-13b', 0.328667)	# Llama 2 Chat 13B
AAR('llama-2-70b', 'llama-2-70b', 0.322667)	# Llama 2 Chat 70B
AAR('llama-2-7b', 'llama-2-7b', 0.058667)	# Llama 2 Chat 7B
AAR('llama-3-1-405b', 'llama-3-1-405b', 0.703333)	# Llama 3.1 Instruct 405B
AAR('llama-3-1-70b', 'llama-3-1-70b', 0.649333)	# Llama 3.1 Instruct 70B
AAR('llama-3-1-8b', 'llama-3-1-8b', 0.519333)	# Llama 3.1 Instruct 8B
AAR('llama-3-1-nemotron-70b-instruct', 'llama-3-1-nemotron-70b-instruct', 0.732667)	# Llama 3.1 Nemotron Instruct 70B
AAR('llama-nemotron-nano-4b', 'llama-nemotron-nano-4b', 0.946667, effort='Reasoning')	# Llama 3.1 Nemotron Nano 4B v1.1 (Reasoning)
AAR('llama-nemotron-ultra-253b', 'llama-nemotron-ultra-253b', 0.952, effort='Reasoning')	# Llama 3.1 Nemotron Ultra 253B v1 (Reasoning)
AAR('llama-3-2-11b-vision', 'llama-3-2-11b-vision', 0.516)	# Llama 3.2 Instruct 11B (Vision)
AAR('llama-3-2-1b', 'llama-3-2-1b', 0.14)	# Llama 3.2 Instruct 1B
AAR('llama-3-2-3b', 'llama-3-2-3b', 0.488667)	# Llama 3.2 Instruct 3B
AAR('llama-3-2-90b-vision', 'llama-3-2-90b-vision', 0.629333)	# Llama 3.2 Instruct 90B (Vision)
AAR('llama-3-3-70b', 'llama-3-3-70b', 0.772667)	# Llama 3.3 Instruct 70B
AAR('llama-nemotron-super-49b', 'llama-nemotron-super-49b', 0.775333, effort='')	# Llama 3.3 Nemotron Super 49B v1 (Non-reasoning)
AAR('llama-nemotron-super-49b', 'llama-nemotron-super-49b', 0.958667, effort='Reasoning')	# Llama 3.3 Nemotron Super 49B v1 (Reasoning)
AAR('llama-3-70b', 'llama-3-70b', 0.482667)	# Llama 3 Instruct 70B
AAR('llama-3-8b', 'llama-3-8b', 0.499333)	# Llama 3 Instruct 8B
AAR('llama-4-maverick', 'llama-4-maverick', 0.888667)	# Llama 4 Maverick
AAR('llama-4-scout', 'llama-4-scout', 0.844)	# Llama 4 Scout
AAR('llama-nemotron-super-49b-v1-5', 'llama-nemotron-super-49b-v1-5', 0.77, effort='')	# Llama Nemotron Super 49B v1.5 (Non-reasoning)
AAR('llama-nemotron-super-49b-v1-5', 'llama-nemotron-super-49b-v1-5', 0.983333, effort='Reasoning')	# Llama Nemotron Super 49B v1.5 (Reasoning)
AAR('magistral-medium-1-0', 'magistral-medium-1-0', 0.917333)	# Magistral Medium 1
AAR('magistral-small-1-0', 'magistral-small-1-0', 0.962667)	# Magistral Small 1
AAR('minimax-m1', 'minimax-m1-40k', 0.972)	# MiniMax M1 40k
AAR('minimax-m1', 'minimax-m1-80k', 0.98)	# MiniMax M1 80k
AAR('minimax-text-01', 'minimax-text-01', 0.753)	# MiniMax-Text-01
AAR('ministral-3b', 'ministral-3b', 0.537)	# Ministral 3B
AAR('ministral-8b', 'ministral-8b', 0.571)	# Ministral 8B
AAR('mistral-7b', None, 0.121333)	# Mistral 7B Instruct
AAR('mixtral-8x22b', 'mixtral-8x22b', 0.544667)	# Mixtral 8x22B Instruct
AAR('mistral-large', 'mistral-large-1', 0.526667)	# Mistral Large (Feb '24)
AAR('mistral-large-2-1', 'mistral-large-2-1', 0.736)	# Mistral Large 2 (Nov '24)
AAR('mistral-large-2', 'mistral-large-2', 0.714)	# Mistral Large 2 (Jul '24)
AAR('mistral-medium', 'mistral-medium', 0.404667)	# Mistral Medium
AAR('mistral-medium-3', 'mistral-medium-3', 0.906667)	# Mistral Medium 3
AAR('mistral-nemo', 'mistral-nemo', 0.395)	# Mistral NeMo
AAR('mistral-small-3', 'mistral-saba', 0.676667)	# Mistral Saba
AAR('mistral-small-2', 'mistral-small-v24-09', 0.562667)	# Mistral Small (Sep '24)
AAR('mistral-small', 'mistral-small-1', 0.562)	# Mistral Small (Feb '24)
AAR('mistral-small-3', 'mistral-small-3', 0.715333)	# Mistral Small 3
AAR('mistral-small-3-1', 'mistral-small-3-1', 0.706667)	# Mistral Small 3.1
AAR('mistral-small-3-2', 'mistral-small-3-2', 0.882667)	# Mistral Small 3.2
AAR('mixtral-8x7b', 'mixtral-8x7b', 0.299333)	# Mixtral 8x7B Instruct
L('Nova Lite', 100 * 0.764667, AA, part=P500)
L('Nova Micro', 100 * 0.703333, AA, part=P500)
L('Nova Premier', 100 * 0.839333, AA, part=P500)
L('Nova Pro', 100 * 0.786, AA, part=P500)
AAR('o1', 'o1', 0.97)	# o1
AAR('o1-mini', 'o1-mini', 0.944)	# o1-mini
AAR('o1', 'o1-preview', 0.924)	# o1-preview
AAR('o3', 'o3', 0.992)	# o3
AAR('o3-mini', 'o3-mini', 0.973333, effort='Medium')	# o3-mini
AAR('o3-mini', 'o3-mini', 0.984667, effort='High')	# o3-mini (High)
AAR('o4-mini', 'o4-mini', 0.988667, effort='High')	# o4-mini (High)
L('OpenChat 3.5 (1210)', 100 * 0.307333, AA, part=P500)
AAR('phi-3-medium', None, 0.463)	# Phi-3 Medium Instruct 14B
AAR('phi-3-mini', None, 0.456667)	# Phi-3 Mini Instruct 3.8B
AAR('phi-4', 'phi-4', 0.81)	# Phi-4
AAR('phi-4-mini', 'phi-4-mini', 0.696)	# Phi-4 Mini Instruct
AAR('phi-4-mini', 'phi-4-multimodal', 0.692667)	# Phi-4 Multimodal Instruct
AAR('mistral-nemo', 'pixtral-12b', 0.458)	# Pixtral 12B (2409)
AAR('mistral-large-2', 'pixtral-large', 0.714)	# Pixtral Large
AAR('qwen2-5-max', 'qwen2-5-max', 0.834667)	# Qwen2.5 Max
AAR('qwen2-5-turbo', 'qwen2-5-turbo', 0.805333)	# Qwen2.5 Turbo
AAR('qwen2-5-72b', 'qwen2-5-72b', 0.858)	# Qwen2.5 Instruct 72B
AAR('qwen2-5-32b', 'qwen2-5-coder-32b', 0.766667)	# Qwen2.5 Coder Instruct 32B
AAR('qwen2-5-7b', 'qwen2-5-coder-7b', 0.66)	# Qwen2.5 Coder Instruct 7B 
AAR('qwen2-72b', 'qwen2-72b', 0.701333)	# Qwen2 Instruct 72B
AAR('qwen2-5-32b', 'qwen2-5-32b', 0.805333)	# Qwen2.5 Instruct 32B
AAR('qwen3-0-6b', 'qwen3-0-6b', 0.520667, effort='')	# Qwen3 0.6B (Non-reasoning)
AAR('qwen3-0-6b', 'qwen3-0-6b', 0.75, effort='Thinking')	# Qwen3 0.6B (Reasoning)
AAR('qwen3-1-7b', 'qwen3-1-7b', 0.717333, effort='')	# Qwen3 1.7B (Non-reasoning)
AAR('qwen3-1-7b', 'qwen3-1-7b', 0.894, effort='Thinking')	# Qwen3 1.7B (Reasoning)
AAR('qwen3-14b', 'qwen3-14b', 0.871333, effort='')	# Qwen3 14B (Non-reasoning)
AAR('qwen3-14b', 'qwen3-14b', 0.961333, effort='Thinking')	# Qwen3 14B (Reasoning)
AAR('qwen3-235b-a22b', 'qwen3-235b-a22b', 0.902, effort='')	# Qwen3 235B A22B (Non-reasoning)
AAR('qwen3-235b-a22b', 'qwen3-235b-a22b-instruct-2507', 0.98)	# Qwen3 235B A22B 2507 Instruct
AAR('qwen3-235b-a22b', 'qwen3-235b-a22b-thinking-2507', 0.984)	# Qwen3 235B A22B 2507 (Reasoning)
AAR('qwen3-235b-a22b', 'qwen3-235b-a22b', 0.93, effort='Thinking')	# Qwen3 235B A22B (Reasoning)
AAR('qwen3-30b-a3b', 'qwen3-30b-a3b-instruct-2507', 0.975333)	# Qwen3 30B A3B 2507 Instruct
AAR('qwen3-30b-a3b', 'qwen3-30b-a3b-thinking-2507', 0.976)	# Qwen3 30B A3B 2507 (Reasoning)
AAR('qwen3-30b-a3b', 'qwen3-30b-a3b', 0.862667, effort='')	# Qwen3 30B A3B (Non-reasoning)
AAR('qwen3-30b-a3b', 'qwen3-30b-a3b', 0.959333, effort='Thinking')	# Qwen3 30B A3B (Reasoning)
AAR('qwen3-32b', 'qwen3-32b', 0.868667, effort='')	# Qwen3 32B (Non-reasoning)
AAR('qwen3-32b', 'qwen3-32b', 0.960667, effort='Thinking')	# Qwen3 32B (Reasoning)
AAR('qwen3-4b', 'qwen3-4b', 0.842667, effort='')	# Qwen3 4B (Non-reasoning)
AAR('qwen3-4b', 'qwen3-4b', 0.933333, effort='Thinking')	# Qwen3 4B (Reasoning)
AAR('qwen3-8b', 'qwen3-8b', 0.828, effort='')	# Qwen3 8B (Non-reasoning)
AAR('qwen3-8b', 'qwen3-8b', 0.904, effort='Thinking')	# Qwen3 8B (Reasoning)
AAR('qwen3-30b-a3b', 'qwen3-coder-30b-a3b', 0.893333)	# Qwen3 Coder 30B A3B Instruct
AAR('qwen3-coder-480b-a35b', 'qwen3-coder-480b-a35b', 0.942)	# Qwen3 Coder 480B A35B Instruct
AAR('qwen2-5-32b', 'qwq-32b', 0.957333)	# QwQ 32B
L('R1 1776', 100 * 0.954, AA, part=P500)
L('Reka Core', 100 * 0.558, AA, part=P500)
L('Reka Edge', 100 * 0.216, AA, part=P500)
L("Reka Flash (Sep '24)", 100 * 0.528667, AA, part=P500)
L("Reka Flash (Feb '24)", 100 * 0.326, AA, part=P500)
L('Reka Flash 3', 100 * 0.892667, AA, part=P500)
L('Sarvam M (Reasoning)', 100 * 0.847333, AA, part=P500)
L('Solar Mini', 100 * 0.331333, AA, part=P500)
L('Solar Pro 2 (Non-reasoning)', 100 * 0.888667, AA, part=P500)
L('Solar Pro 2 (Preview) (Non-reasoning)', 100 * 0.870667, AA, part=P500)
L('Solar Pro 2 (Preview) (Reasoning)', 100 * 0.9, AA, part=P500)
L('Solar Pro 2 (Reasoning)', 100 * 0.966667, AA, part=P500)
L('Sonar', 100 * 0.816667, AA, part=P500)
L('Sonar Pro', 100 * 0.744667, AA, part=P500)
L('Sonar Reasoning', 100 * 0.921333, AA, part=P500)
L('Sonar Reasoning Pro', 100 * 0.957333, AA, part=P500)
L('Llama 3.1 Tulu3 405B', 100 * 0.778, AA, part=P500)
L('Yi-Large', 100 * 0.562, AA, part=P500)

# ---------------------------------------------------------------- Vals AI (MATH 500, cost per test)
VA('gemini-3-pro', 'gemini-3-pro', 96.4, 0.833, 0.050514)	# google/gemini-3-pro-preview
VA('grok-4', 'grok-4', 96.2, 0.855, 0.037425)	# grok/grok-4-0709
VA('gpt-5', 'gpt-5', 96.0, 0.876, 0.020071)	# openai/gpt-5-2025-08-07
VA('claude-opus-4-1', 'claude-opus-4-1', 95.4, 0.937, 0.202262, effort='Thinking')	# anthropic/claude-opus-4-1-20250805-thinking
VA('gemini-2-5-pro', 'gemini-2-5-pro-mar-2025', 95.2, 0.956, 0.007941)	# google/gemini-2.5-pro-exp-03-25
VA('gpt-5-mini', 'gpt-5-mini', 94.8, 0.993, 0.003686)	# openai/gpt-5-mini-2025-08-07
VA('gpt-oss-120b', 'gpt-oss-120b', 94.8, 0.993, 0.001381)	# fireworks/gpt-oss-120b
VA('o3', 'o3', 94.6, 1.011, 0.010795)	# openai/o3-2025-04-16
VA('qwen3-235b-a22b', 'qwen3-235b-a22b', 94.6, 1.011, 0.003361)	# fireworks/qwen3-235b-a22b
VA('o4-mini', 'o4-mini', 94.2, 1.045, 0.004784)	# openai/o4-mini-2025-04-16
VA('grok-3-mini', 'grok-3-mini', 94.2, 1.045, 0.001879, effort='High')	# grok/grok-3-mini-fast-high-reasoning
VA('gpt-oss-20b', 'gpt-oss-20b', 94.2, 1.045, 0.000955)	# fireworks/gpt-oss-20b
VA('kimi-k2', 'kimi-k2-jul-2025', 94.2, 1.045, 0.002985)	# together/moonshotai/Kimi-K2-Instruct
VA('glm-4-5', 'glm-4-5', 94.0, 1.062, 0.012744)	# zai/glm-4.5
VA('gpt-5-nano', 'gpt-5-nano', 93.8, 1.078, 0.001664)	# openai/gpt-5-nano-2025-08-07
VA('claude-sonnet-4', 'claude-sonnet-4', 93.8, 1.078, 0.068378, effort='Thinking')	# anthropic/claude-sonnet-4-20250514-thinking
VA('claude-opus-4-1', 'claude-opus-4-1', 93.0, 1.141, 0.045983, effort='')	# anthropic/claude-opus-4-1-20250805
VA('deepseek-r1', 'deepseek-r1-jan-2025', 92.2, 2.364, 0.010886)	# fireworks/deepseek-r1
VA('o3-mini', 'o3-mini', 91.8, 2.417, 0.007661)	# openai/o3-mini-2025-01-31
VA('gemini-2-5-flash', 'gemini-2-5-flash-apr-2025', 91.8, 1.227, 0.011241, effort='Thinking')	# google/gemini-2.5-flash-preview-04-17-thinking
VA('gemini-2-5-flash', 'gemini-2-5-flash-apr-2025', 91.6, 1.24, 0.006461, effort='')	# google/gemini-2.5-flash-preview-04-17
VA('claude-3-7-sonnet', 'claude-3-7-sonnet', 91.6, 1.24, 0.125312, effort='Thinking')	# anthropic/claude-3-7-sonnet-20250219-thinking
VA('llama-nemotron-super-49b', 'llama-nemotron-super-49b', 91.4, 1.254, None, effort='Reasoning')	# together/langston/nim/nvidia/llama-3.3-nemotron-super-49b-v1-42e84561-thinking
VA('claude-opus-4', 'claude-opus-4', 90.4, 1.318, 0.045429, effort='')	# anthropic/claude-opus-4-20250514
VA('o1', 'o1', 90.4, 2.591, 0.095269)	# openai/o1-2024-12-17
VA('claude-sonnet-4', 'claude-sonnet-4', 90.323, 1.288, 0.010201, effort='')	# anthropic/claude-sonnet-4-20250514
VA('grok-3', 'grok-3', 89.8, 1.354, 0.012781)	# grok/grok-3
VA('gemini-2-0-flash', 'gemini-2-0-flash-experimental', 89.0, 2.748, 0.0002)	# google/gemini-2.0-flash-exp
VA('minimax-m2-1', 'minimax-m2-1', 89.0, 1.399, 0.004642)	# minimax/MiniMax-M2.1
VA('deepseek-v3', 'deepseek-v3-mar-2025', 88.6, 1.421, 0.00074)	# fireworks/deepseek-v3-0324
VA('gemini-2-0-flash', 'gemini-2-0-flash', 88.0, 2.852, 0.00026)	# google/gemini-2.0-flash-001
VA('gpt-4-1-mini', 'gpt-4-1-mini', 88.0, 1.453, 0.00122)	# openai/gpt-4.1-mini-2025-04-14
VA('gpt-4-1', 'gpt-4-1', 87.2, 1.494, 0.009426)	# openai/gpt-4.1-2025-04-14
VA('mistral-medium-3', 'mistral-medium-3', 87.0, 1.504, 0.001841)	# mistralai/mistral-medium-2505
VA('llama-4-maverick', 'llama-4-maverick', 85.2, 1.588, 0.00059)	# fireworks/llama4-maverick-instruct-basic
VA('gemini-2-0-flash', 'gemini-2-0-flash-thinking-jan-2025', 84.6, 3.163, 0.000387)	# google/gemini-2.0-flash-thinking-exp-01-21
VA('gemini-1-5-pro', 'gemini-1-5-pro-sep-2024', 82.8, 3.305, 0.002164)	# google/gemini-1.5-pro-002
VA('deepseek-v3', 'deepseek-v3-dec-2024', 80.4, 1.775, 0.000563)	# fireworks/deepseek-v3
VA('gpt-4-1-nano', 'gpt-4-1-nano', 80.2, 1.782, 0.000314)	# openai/gpt-4.1-nano-2025-04-14
VA('llama-4-scout', 'llama-4-scout', 79.2, 1.815, 0.00043)	# together/meta-llama/Llama-4-Scout-17B-16E-Instruct
VA('gemini-1-5-flash', 'gemini-1-5-flash-sep-2024', 78.8, 3.576, 0.00012)	# google/gemini-1.5-flash-002
VA('grok-2', 'grok-2-dec-2024', 78.4, 3.6, 0.006277)	# grok/grok-2-1212
VA('claude-3-7-sonnet', 'claude-3-7-sonnet', 76.8, 1.888, 0.006868, effort='')	# anthropic/claude-3-7-sonnet-20250219
L('Command A', 76.2, VALS, part=P500)
VA('gpt-4o', 'gpt-4o-aug-2024', 75.2, 3.776, 0.005641)	# openai/gpt-4o-2024-08-06
VA('mistral-large-2-1', 'mistral-large-2-1', 74.4, 3.815, 0.003323)	# mistralai/mistral-large-2411
VA('gpt-4o', 'gpt-4o-nov-2024', 74.0, 3.834, 0.008846)	# openai/gpt-4o-2024-11-20
VA('llama-3-3-70b', 'llama-3-3-70b', 73.4, 3.862, 0.000698)	# together/meta-llama/Llama-3.3-70B-Instruct-Turbo
VA('gpt-4o-mini', 'gpt-4o-mini', 72.6, 3.898, 0.00035)	# openai/gpt-4o-mini-2024-07-18
VA('claude-3-5-sonnet', 'claude-3-5-sonnet-oct-2024', 72.4, 3.907, 0.004988)	# anthropic/claude-3-5-sonnet-20241022
VA('llama-3-1-405b', 'llama-3-1-405b', 71.4, 3.949, 0.002967)	# together/meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo
VA('llama-nemotron-super-49b', 'llama-nemotron-super-49b', 71.2, 2.025, None, effort='')	# together/langston/nim/nvidia/llama-3.3-nemotron-super-49b-v1-42e84561
VA('mistral-small', 'mistral-small-1', 70.6, 3.981, 0.000327)	# mistralai/mistral-small-2402
VA('grok-3-mini', 'grok-3-mini', 70.2, 2.046, 0.000893, effort='Low')	# grok/grok-3-mini-fast-low-reasoning
VA('mistral-small-3-1', 'mistral-small-3-1', 68.4, 2.079, 0.000158)	# mistralai/mistral-small-2503
VA('llama-3-1-70b', 'llama-3-1-70b', 65.0, 4.166, 0.001035)	# together/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo
VA('claude-3-5-haiku', 'claude-3-5-haiku', 64.2, 4.188, 0.001498)	# anthropic/claude-3-5-haiku-20241022
L('Jamba 1.6 Large', 54.8, VALS, part=P500)
VA('llama-3-1-8b', 'llama-3-1-8b', 44.4, 4.339, 0.000351)	# together/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo
L('Jamba 1.6 Mini', 25.4, VALS, part=P500)

# ---------------------------------------------------------------- Open LLM Leaderboard v2 (MATH Lvl 5, 4-shot, bf16 preferred over fp16)
# Models of the list (found through the Hugging Face repo ids in their names), plus the "Official Providers" models as leftovers
OL('qwen2-7b', 'marco-o1', 37.46)	# AIDC-AI/Marco-o1 (float16)
OL('starcoder2-15b', 'starcoder2-15b', 5.97)	# bigcode/starcoder2-15b (bfloat16)
OL('starcoder2-3b', 'starcoder2-3b', 1.51)	# bigcode/starcoder2-3b (bfloat16)
OL('starcoder2-7b', 'starcoder2-7b', 3.1)	# bigcode/starcoder2-7b (bfloat16)
OL('bloom-1b1', 'bloom-1b1', 0.53)	# bigscience/bloom-1b1 (bfloat16)
OL('bloom-1b7', 'bloom-1b7', 0.53)	# bigscience/bloom-1b7 (bfloat16)
OL('bloom-3b', 'bloom-3b', 0.83)	# bigscience/bloom-3b (bfloat16)
OL('bloom-560m', 'bloom-560m', 0.38)	# bigscience/bloom-560m (bfloat16)
OL('bloom-7b1', 'bloom-7b1', 0.53)	# bigscience/bloom-7b1 (float16)
OL('dbrx', 'dbrx-base', 10.0)	# databricks/dbrx-base (float16)
OL('dbrx', 'dbrx-instruct', 6.87)	# databricks/dbrx-instruct (bfloat16)
OL('dolly-v1-6b', 'dolly-v1-6b', 1.89)	# databricks/dolly-v1-6b (bfloat16)
OL('dolly-v2-12b', 'dolly-v2-12b', 1.36)	# databricks/dolly-v2-12b (bfloat16)
OL('dolly-v2-3b', 'dolly-v2-3b', 1.51)	# databricks/dolly-v2-3b (bfloat16)
OL('dolly-v2-7b', 'dolly-v2-7b', 1.44)	# databricks/dolly-v2-7b (bfloat16)
OL('deepseek-llm-67b', 'deepseek-llm-67b', 9.29)	# deepseek-ai/deepseek-llm-67b-chat (bfloat16)
OL('deepseek-llm-7b', 'deepseek-llm-7b', 1.96)	# deepseek-ai/deepseek-llm-7b-base (bfloat16)
OL('deepseek-llm-7b', 'deepseek-llm-7b', 2.04)	# deepseek-ai/deepseek-llm-7b-chat (bfloat16)
OL('deepseekmoe-16b', 'deepseekmoe-16b', 2.42)	# deepseek-ai/deepseek-moe-16b-base (bfloat16)
OL('deepseekmoe-16b', 'deepseekmoe-16b', 2.57)	# deepseek-ai/deepseek-moe-16b-chat (bfloat16)
OL('deepseek-r1-distill-llama-70b', 'deepseek-r1-distill-llama-70b', 30.74)	# deepseek-ai/DeepSeek-R1-Distill-Llama-70B (bfloat16)
OL('deepseek-r1-distill-llama-8b', 'deepseek-r1-distill-llama-8b', 21.98)	# deepseek-ai/DeepSeek-R1-Distill-Llama-8B (bfloat16)
OL('deepseek-r1-distill-qwen-1-5b', 'deepseek-r1-distill-qwen-1-5b', 16.92)	# deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B (bfloat16)
OL('deepseek-r1-distill-qwen-14b', 'deepseek-r1-distill-qwen-14b', 57.02)	# deepseek-ai/DeepSeek-R1-Distill-Qwen-14B (bfloat16)
OL('deepseek-r1-distill-qwen-32b', 'deepseek-r1-distill-qwen-32b', 17.07)	# deepseek-ai/DeepSeek-R1-Distill-Qwen-32B (bfloat16)
OL('deepseek-r1-distill-qwen-7b', 'deepseek-r1-distill-qwen-7b', 19.56)	# deepseek-ai/DeepSeek-R1-Distill-Qwen-7B (bfloat16)
OL('distilgpt2', 'distilgpt2', 0.6)	# distilbert/distilgpt2 (bfloat16)
OL('gpt-j-6b', 'gpt-j-6b', 1.36)	# EleutherAI/gpt-j-6b (bfloat16)
OL('gpt-neo-1-3b', 'gpt-neo-1-3b', 1.06)	# EleutherAI/gpt-neo-1.3B (bfloat16)
OL('gpt-neo-125m', 'gpt-neo-125m', 0.6)	# EleutherAI/gpt-neo-125m (bfloat16)
OL('gpt-neo-2-7b', 'gpt-neo-2-7b', 1.06)	# EleutherAI/gpt-neo-2.7B (bfloat16)
OL('gpt-neox-20b', 'gpt-neox-20b', 1.36)	# EleutherAI/gpt-neox-20b (float16)
OL('pythia-1-4b', 'pythia-1-4b', 1.51)	# EleutherAI/pythia-1.4b (float16)
OL('pythia-12b', 'pythia-12b', 1.66)	# EleutherAI/pythia-12b (float16)
OL('pythia-160m', 'pythia-160m', 0.91)	# EleutherAI/pythia-160m (float16)
OL('pythia-1b', 'pythia-1b', 0.91)	# EleutherAI/pythia-1b (float16)
OL('pythia-2-8b', 'pythia-2-8b', 1.36)	# EleutherAI/pythia-2.8b (float16)
OL('pythia-410m', 'pythia-410m', 0.98)	# EleutherAI/pythia-410m (float16)
OL('pythia-6-9b', 'pythia-6-9b', 1.44)	# EleutherAI/pythia-6.9b (float16)
OL('opt', 'opt-1-3b', 0.91)	# facebook/opt-1.3b (float16)
OL('opt', 'opt-30b', 1.06)	# facebook/opt-30b (float16)
OL('gemma-2b', 'codegemma-1-1-2b', 1.28)	# google/codegemma-1.1-2b (bfloat16)
OL('t5', 'flan-t5-base', 1.06)	# google/flan-t5-base (float16)
OL('t5', 'flan-t5-large', 1.44)	# google/flan-t5-large (float16)
OL('t5', 'flan-t5-small', 0.76)	# google/flan-t5-small (float16)
OL('t5', 'flan-t5-xl', 0.08)	# google/flan-t5-xl (bfloat16)
OL('t5', 'flan-t5-xxl', 1.06)	# google/flan-t5-xxl (float16)
OL('ul2', 'flan-ul2', 0.91)	# google/flan-ul2 (bfloat16)
OL('gemma-2b', 'gemma-1-1-2b', 1.81)	# google/gemma-1.1-2b-it (bfloat16)
OL('gemma-7b', 'gemma-1-1-7b', 4.91)	# google/gemma-1.1-7b-it (bfloat16)
OL('gemma-2-27b', 'gemma-2-27b', 16.62)	# google/gemma-2-27b (bfloat16)
OL('gemma-2-27b', 'gemma-2-27b', 23.87)	# google/gemma-2-27b-it (bfloat16)
OL('gemma-2-2b', 'gemma-2-2b', 2.87)	# google/gemma-2-2b (bfloat16)
OL('gemma-2-2b', 'gemma-2-2b', 0.08)	# google/gemma-2-2b-it (bfloat16)
OL('gemma-2-2b', 'gemma-2-jpn', 4.76)	# google/gemma-2-2b-jpn-it (bfloat16)
OL('gemma-2-9b', 'gemma-2-9b', 13.44)	# google/gemma-2-9b (bfloat16)
OL('gemma-2-9b', 'gemma-2-9b', 19.49)	# google/gemma-2-9b-it (bfloat16)
OL('gemma-2b', 'gemma-2b', 3.02)	# google/gemma-2b (bfloat16)
OL('gemma-2b', 'gemma-2b', 2.04)	# google/gemma-2b-it (bfloat16)
OL('gemma-7b', 'gemma-7b', 7.4)	# google/gemma-7b (bfloat16)
OL('gemma-7b', 'gemma-7b', 2.95)	# google/gemma-7b-it (bfloat16)
OL('mt5-xxl', 'mt5-xxl', 0.0)	# google/mt5-xxl (float16)
OL('recurrentgemma-2b', 'recurrentgemma-2b', 2.04)	# google/recurrentgemma-2b (bfloat16)
OL('recurrentgemma-2b', 'recurrentgemma-2b', 1.96)	# google/recurrentgemma-2b-it (bfloat16)
OL('recurrentgemma-9b', 'recurrentgemma-9b', 6.65)	# google/recurrentgemma-9b (bfloat16)
OL('recurrentgemma-9b', 'recurrentgemma-9b', 6.65)	# google/recurrentgemma-9b-it (bfloat16)
OL('zephyr-7b', 'zephyr-7b-alpha', 1.96)	# HuggingFaceH4/zephyr-7b-alpha (bfloat16)
OL('zephyr-7b', 'zephyr-7b-beta', 2.87)	# HuggingFaceH4/zephyr-7b-beta (bfloat16)
OL('zephyr-7b-gemma', 'zephyr-7b-gemma', 8.16)	# HuggingFaceH4/zephyr-7b-gemma-v0.1 (bfloat16)
OL('zephyr-141b-a39b', 'zephyr-141b-a39b', 20.47)	# HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1 (float16)
OL('smollm-1-7b', 'smollm-1-7b', 1.66)	# HuggingFaceTB/SmolLM-1.7B (bfloat16)
OL('smollm-1-7b', 'smollm-1-7b-instruct', 2.11)	# HuggingFaceTB/SmolLM-1.7B-Instruct (bfloat16)
OL('smollm-135m', 'smollm-135m', 1.36)	# HuggingFaceTB/SmolLM-135M (bfloat16)
OL('smollm-135m', 'smollm-135m-instruct', 0.53)	# HuggingFaceTB/SmolLM-135M-Instruct (bfloat16)
OL('smollm-360m', 'smollm-360m', 1.13)	# HuggingFaceTB/SmolLM-360M (bfloat16)
OL('smollm-360m', 'smollm-360m-instruct', 1.81)	# HuggingFaceTB/SmolLM-360M-Instruct (bfloat16)
OL('smollm2-1-7b', 'smollm2-1-7b', 2.64)	# HuggingFaceTB/SmolLM2-1.7B (bfloat16)
OL('smollm2-1-7b', 'smollm2-1-7b-instruct', 5.82)	# HuggingFaceTB/SmolLM2-1.7B-Instruct (bfloat16)
OL('smollm2-135m', 'smollm2-135m', 1.21)	# HuggingFaceTB/SmolLM2-135M (bfloat16)
OL('smollm2-135m', 'smollm2-135m-instruct', 0.3)	# HuggingFaceTB/SmolLM2-135M-Instruct (bfloat16)
OL('smollm2-360m', 'smollm2-360m', 1.21)	# HuggingFaceTB/SmolLM2-360M (bfloat16)
OL('smollm2-360m', 'smollm2-360m-instruct', 1.51)	# HuggingFaceTB/SmolLM2-360M-Instruct (bfloat16)
OL('llama-2-13b', 'llama-2-13b', 1.36)	# meta-llama/Llama-2-13b-chat-hf (float16)
OL('llama-2-13b', 'llama-2-13b', 1.51)	# meta-llama/Llama-2-13b-hf (float16)
OL('llama-2-70b', 'llama-2-70b', 2.95)	# meta-llama/Llama-2-70b-chat-hf (float16)
OL('llama-2-70b', 'llama-2-70b', 3.25)	# meta-llama/Llama-2-70b-hf (float16)
OL('llama-2-7b', 'llama-2-7b', 1.96)	# meta-llama/Llama-2-7b-chat-hf (float16)
OL('llama-2-7b', 'llama-2-7b', 1.74)	# meta-llama/Llama-2-7b-hf (float16)
OL('llama-3-1-70b', 'llama-3-1-70b', 18.43)	# meta-llama/Llama-3.1-70B (bfloat16)
OL('llama-3-1-70b', 'llama-3-1-70b', 38.07)	# meta-llama/Llama-3.1-70B-Instruct (bfloat16)
OL('llama-3-1-8b', 'llama-3-1-8b', 6.57)	# meta-llama/Llama-3.1-8B (float16)
OL('llama-3-1-8b', 'llama-3-1-8b', 15.56)	# meta-llama/Llama-3.1-8B-Instruct (float16)
OL('llama-3-2-1b', 'llama-3-2-1b', 1.21)	# meta-llama/Llama-3.2-1B (bfloat16)
OL('llama-3-2-1b', 'llama-3-2-1b', 7.02)	# meta-llama/Llama-3.2-1B-Instruct (bfloat16)
OL('llama-3-2-3b', 'llama-3-2-3b', 1.89)	# meta-llama/Llama-3.2-3B (bfloat16)
OL('llama-3-2-3b', 'llama-3-2-3b', 17.67)	# meta-llama/Llama-3.2-3B-Instruct (bfloat16)
OL('llama-3-3-70b', 'llama-3-3-70b', 48.34)	# meta-llama/Llama-3.3-70B-Instruct (bfloat16)
OL('llama-3-70b', 'llama-3-70b', 18.58)	# meta-llama/Meta-Llama-3-70B (bfloat16)
OL('llama-3-70b', 'llama-3-70b', 24.47)	# meta-llama/Meta-Llama-3-70B-Instruct (bfloat16)
OL('llama-3-8b', 'llama-3-8b', 4.53)	# meta-llama/Meta-Llama-3-8B (bfloat16)
OL('llama-3-8b', 'llama-3-8b', 8.69)	# meta-llama/Meta-Llama-3-8B-Instruct (bfloat16)
OL('orca-2-13b', 'orca-2-13b', 3.17)	# microsoft/Orca-2-13b (bfloat16)
OL('orca-2-7b', 'orca-2-7b', 1.96)	# microsoft/Orca-2-7b (bfloat16)
OL('phi-1', 'phi-1', 0.98)	# microsoft/phi-1 (bfloat16)
OL('phi-1-5', 'phi-1-5', 1.81)	# microsoft/phi-1_5 (float16)
OL('phi-2', 'phi-2', 2.95)	# microsoft/phi-2 (float16)
OL('phi-3-medium', 'phi-3-medium-128k-instruct', 19.18)	# microsoft/Phi-3-medium-128k-instruct (bfloat16)
OL('phi-3-medium', 'phi-3-medium-4k-instruct', 19.56)	# microsoft/Phi-3-medium-4k-instruct (bfloat16)
OL('phi-3-mini', 'phi-3-mini-128k-instruct', 14.05)	# microsoft/Phi-3-mini-128k-instruct (bfloat16)
OL('phi-3-mini', 'phi-3-mini-4k-instruct', 11.63)	# microsoft/Phi-3-mini-4k-instruct (bfloat16)
OL('phi-3-small', 'phi-3-small-128k-instruct', 20.26)	# microsoft/Phi-3-small-128k-instruct (bfloat16)
OL('phi-3-small', 'phi-3-small-8k-instruct', 18.87)	# microsoft/Phi-3-small-8k-instruct (bfloat16)
OL('phi-3-5-mini', 'phi-3-5-mini-instruct', 19.64)	# microsoft/Phi-3.5-mini-instruct (bfloat16)
OL('phi-3-5-moe', 'phi-3-5-moe', 31.19)	# microsoft/Phi-3.5-MoE-instruct (bfloat16)
OL('phi-4', 'phi-4', 31.65)	# microsoft/phi-4 (bfloat16)
OL('phi-4-mini', 'phi-4-mini', 16.99)	# microsoft/Phi-4-mini-instruct (bfloat16)
OL('codestral-22b', 'codestral-22b', 10.05)	# mistralai/Codestral-22B-v0.1 (bfloat16)
OL('ministral-8b', 'ministral-8b', 19.56)	# mistralai/Ministral-8B-Instruct-2410 (bfloat16)
OL('mistral-7b', 'mistral-7b-v0-1', 2.27)	# mistralai/Mistral-7B-Instruct-v0.1 (bfloat16)
OL('mistral-7b', 'mistral-7b-v0-2', 3.02)	# mistralai/Mistral-7B-Instruct-v0.2 (bfloat16)
OL('mistral-7b', 'mistral-7b-v0-3', 3.85)	# mistralai/Mistral-7B-Instruct-v0.3 (bfloat16)
OL('mistral-7b', 'mistral-7b-v0-1', 2.95)	# mistralai/Mistral-7B-v0.1 (bfloat16)
OL('mistral-7b', 'mistral-7b-v0-3', 3.02)	# mistralai/Mistral-7B-v0.3 (bfloat16)
OL('mistral-large-2-1', 'mistral-large-2-1', 49.55)	# mistralai/Mistral-Large-Instruct-2411 (float16)
OL('mistral-nemo', 'mistral-nemo', 5.97)	# mistralai/Mistral-Nemo-Base-2407 (bfloat16)
OL('mistral-nemo', 'mistral-nemo', 12.69)	# mistralai/Mistral-Nemo-Instruct-2407 (bfloat16)
OL('mistral-small-3', 'mistral-small-3', 19.71)	# mistralai/Mistral-Small-24B-Base-2501 (bfloat16)
OL('mistral-small-2', 'mistral-small-v24-09', 14.35)	# mistralai/Mistral-Small-Instruct-2409 (bfloat16)
OL('mixtral-8x22b', 'mixtral-8x22b', 18.73)	# mistralai/Mixtral-8x22B-Instruct-v0.1 (bfloat16)
OL('mixtral-8x22b', 'mixtral-8x22b', 18.35)	# mistralai/Mixtral-8x22B-v0.1 (bfloat16)
OL('mixtral-8x7b', 'mixtral-8x7b', 9.14)	# mistralai/Mixtral-8x7B-Instruct-v0.1 (bfloat16)
OL('mixtral-8x7b', 'mixtral-8x7b', 10.2)	# mistralai/Mixtral-8x7B-v0.1 (bfloat16)
OL('mpt-7b', 'mpt-7b', 1.59)	# mosaicml/mpt-7b (bfloat16)
OL('aceinstruct-1-5b', 'aceinstruct-1-5b', 31.27)	# nvidia/AceInstruct-1.5B (bfloat16)
OL('aceinstruct-72b', 'aceinstruct-72b', 62.61)	# nvidia/AceInstruct-72B (bfloat16)
OL('aceinstruct-7b', 'aceinstruct-7b', 52.95)	# nvidia/AceInstruct-7B (bfloat16)
OL('acemath-1-5b-instruct', 'acemath-1-5b-instruct', 52.87)	# nvidia/AceMath-1.5B-Instruct (bfloat16)
OL('acemath-72b-instruct', 'acemath-72b-instruct', 71.45)	# nvidia/AceMath-72B-Instruct (bfloat16)
OL('acemath-7b-instruct', 'acemath-7b-instruct', 63.37)	# nvidia/AceMath-7B-Instruct (bfloat16)
OL('hymba-1-5b', 'hymba-1-5b', 1.36)	# nvidia/Hymba-1.5B-Base (bfloat16)
OL('hymba-1-5b', 'hymba-1-5b', 2.72)	# nvidia/Hymba-1.5B-Instruct (bfloat16)
OL('llama-3-1-minitron-4b', 'llama-3-1-minitron-4b-depth', 1.96)	# nvidia/Llama-3.1-Minitron-4B-Depth-Base (bfloat16)
OL('llama-3-1-nemotron-70b-instruct', 'llama-3-1-nemotron-70b-instruct', 42.67)	# nvidia/Llama-3.1-Nemotron-70B-Instruct-HF (bfloat16)
OL('minitron-4b', 'minitron-4b', 1.96)	# nvidia/Minitron-4B-Base (bfloat16)
OL('minitron-8b', 'minitron-8b', 2.57)	# nvidia/Minitron-8B-Base (bfloat16)
OL('mistral-nemo-minitron-8b', 'mistral-nemo-minitron-8b', 4.83)	# nvidia/Mistral-NeMo-Minitron-8B-Base (bfloat16)
OL('mistral-nemo-minitron-8b', 'mistral-nemo-minitron-8b', 11.63)	# nvidia/Mistral-NeMo-Minitron-8B-Instruct (bfloat16)
OL('minitron-4b', 'nemotron-mini-4b-instruct', 2.57)	# nvidia/Nemotron-Mini-4B-Instruct (bfloat16)
OL('openmath2-llama3-1-8b', 'openmath2-llama3-1-8b', 26.74)	# nvidia/OpenMath2-Llama3.1-8B (float16)
OL('qwen1-5-0-5b', 'qwen1-5-0-5b', 1.74)	# Qwen/Qwen1.5-0.5B (bfloat16)
OL('qwen1-5-0-5b', 'qwen1-5-0-5b', 0.68)	# Qwen/Qwen1.5-0.5B-Chat (bfloat16)
OL('qwen1-5-1-8b', 'qwen1-5-1-8b', 3.17)	# Qwen/Qwen1.5-1.8B (bfloat16)
OL('qwen1-5-1-8b', 'qwen1-5-1-8b', 1.96)	# Qwen/Qwen1.5-1.8B-Chat (bfloat16)
OL('qwen1-5-110b', 'qwen1-5-110b', 24.7)	# Qwen/Qwen1.5-110B (bfloat16)
OL('qwen1-5-110b', 'qwen1-5-110b', 23.41)	# Qwen/Qwen1.5-110B-Chat (bfloat16)
OL('qwen1-5-14b', 'qwen1-5-14b', 20.24)	# Qwen/Qwen1.5-14B (bfloat16)
OL('qwen1-5-14b', 'qwen1-5-14b', 15.26)	# Qwen/Qwen1.5-14B-Chat (bfloat16)
OL('qwen1-5-32b', 'qwen1-5-32b', 30.29)	# Qwen/Qwen1.5-32B (bfloat16)
OL('qwen1-5-32b', 'qwen1-5-32b', 19.56)	# Qwen/Qwen1.5-32B-Chat (bfloat16)
OL('qwen1-5-4b', 'qwen1-5-4b', 5.29)	# Qwen/Qwen1.5-4B (bfloat16)
OL('qwen1-5-4b', 'qwen1-5-4b', 2.79)	# Qwen/Qwen1.5-4B-Chat (bfloat16)
OL('qwen1-5-7b', 'qwen1-5-7b', 9.29)	# Qwen/Qwen1.5-7B (bfloat16)
OL('qwen1-5-7b', 'qwen1-5-7b', 6.27)	# Qwen/Qwen1.5-7B-Chat (bfloat16)
OL('qwen1-5-moe-a2-7b', 'qwen1-5-moe-a2-7b', 9.29)	# Qwen/Qwen1.5-MoE-A2.7B (bfloat16)
OL('qwen1-5-moe-a2-7b', 'qwen1-5-moe-a2-7b', 6.34)	# Qwen/Qwen1.5-MoE-A2.7B-Chat (bfloat16)
OL('qwen2-0-5b', 'qwen2-0-5b', 2.64)	# Qwen/Qwen2-0.5B (bfloat16)
OL('qwen2-0-5b', 'qwen2-0-5b', 2.87)	# Qwen/Qwen2-0.5B-Instruct (bfloat16)
OL('qwen2-1-5b', 'qwen2-1-5b', 7.02)	# Qwen/Qwen2-1.5B (bfloat16)
OL('qwen2-1-5b', 'qwen2-1-5b', 7.18)	# Qwen/Qwen2-1.5B-Instruct (bfloat16)
OL('qwen2-57b-a14b', 'qwen2-57b-a14b', 18.66)	# Qwen/Qwen2-57B-A14B (bfloat16)
OL('qwen2-57b-a14b', 'qwen2-57b-a14b', 28.17)	# Qwen/Qwen2-57B-A14B-Instruct (bfloat16)
OL('qwen2-72b', 'qwen2-72b', 31.12)	# Qwen/Qwen2-72B (bfloat16)
OL('qwen2-72b', 'qwen2-72b', 41.77)	# Qwen/Qwen2-72B-Instruct (bfloat16)
OL('qwen2-7b', 'qwen2-7b', 20.39)	# Qwen/Qwen2-7B (bfloat16)
OL('qwen2-7b', 'qwen2-7b', 27.64)	# Qwen/Qwen2-7B-Instruct (bfloat16)
OL('qwen2-72b', 'qwen2-math-72b', 55.36)	# Qwen/Qwen2-Math-72B-Instruct (bfloat16)
OL('qwen2-7b', 'qwen2-math-7b', 24.77)	# Qwen/Qwen2-Math-7B (bfloat16)
OL('qwen2-72b', 'qwen2-vl-72b', 34.44)	# Qwen/Qwen2-VL-72B-Instruct (bfloat16)
OL('qwen2-7b', 'qwen2-vl-7b', 19.86)	# Qwen/Qwen2-VL-7B-Instruct (bfloat16)
OL('qwen2-5-0-5b', 'qwen2-5-0-5b', 3.93)	# Qwen/Qwen2.5-0.5B (bfloat16)
OL('qwen2-5-0-5b', 'qwen2-5-0-5b', 0.0)	# Qwen/Qwen2.5-0.5B-Instruct (bfloat16)
OL('qwen2-5-1-5b', 'qwen2-5-1-5b', 9.14)	# Qwen/Qwen2.5-1.5B (bfloat16)
OL('qwen2-5-1-5b', 'qwen2-5-1-5b', 22.05)	# Qwen/Qwen2.5-1.5B-Instruct (bfloat16)
OL('qwen2-5-14b', 'qwen2-5-14b', 29.0)	# Qwen/Qwen2.5-14B (bfloat16)
OL('qwen2-5-14b', 'qwen2-5-14b', 54.76)	# Qwen/Qwen2.5-14B-Instruct (bfloat16)
OL('qwen2-5-14b', 'qwen2-5-14b-instruct-1m', 53.02)	# Qwen/Qwen2.5-14B-Instruct-1M (bfloat16)
OL('qwen2-5-32b', 'qwen2-5-32b', 35.65)	# Qwen/Qwen2.5-32B (bfloat16)
OL('qwen2-5-32b', 'qwen2-5-32b', 62.54)	# Qwen/Qwen2.5-32B-Instruct (bfloat16)
OL('qwen2-5-3b', 'qwen2-5-3b', 14.8)	# Qwen/Qwen2.5-3B (bfloat16)
OL('qwen2-5-3b', 'qwen2-5-3b', 36.78)	# Qwen/Qwen2.5-3B-Instruct (bfloat16)
OL('qwen2-5-72b', 'qwen2-5-72b', 39.12)	# Qwen/Qwen2.5-72B (bfloat16)
OL('qwen2-5-72b', 'qwen2-5-72b', 59.82)	# Qwen/Qwen2.5-72B-Instruct (bfloat16)
OL('qwen2-5-7b', 'qwen2-5-7b', 25.08)	# Qwen/Qwen2.5-7B (bfloat16)
OL('qwen2-5-7b', 'qwen2-5-7b', 50.0)	# Qwen/Qwen2.5-7B-Instruct (bfloat16)
OL('qwen2-5-7b', 'qwen2-5-7b-instruct-1m', 43.35)	# Qwen/Qwen2.5-7B-Instruct-1M (bfloat16)
OL('qwen2-5-14b', 'qwen2-5-coder-14b', 22.51)	# Qwen/Qwen2.5-Coder-14B (bfloat16)
OL('qwen2-5-14b', 'qwen2-5-coder-14b', 32.48)	# Qwen/Qwen2.5-Coder-14B-Instruct (bfloat16)
OL('qwen2-5-32b', 'qwen2-5-coder-32b', 30.89)	# Qwen/Qwen2.5-Coder-32B (float16)
OL('qwen2-5-32b', 'qwen2-5-coder-32b', 49.55)	# Qwen/Qwen2.5-Coder-32B-Instruct (float16)
OL('qwen2-5-7b', 'qwen2-5-coder-7b', 19.18)	# Qwen/Qwen2.5-Coder-7B (bfloat16)
OL('qwen2-5-7b', 'qwen2-5-coder-7b', 37.16)	# Qwen/Qwen2.5-Coder-7B-Instruct (bfloat16)
OL('qwen2-5-1-5b', 'qwen2-5-math-1-5b', 26.28)	# Qwen/Qwen2.5-Math-1.5B-Instruct (bfloat16)
OL('qwen2-5-72b', 'qwen2-5-math-72b', 62.39)	# Qwen/Qwen2.5-Math-72B-Instruct (bfloat16)
OL('qwen2-5-7b', 'qwen2-5-math-7b', 30.51)	# Qwen/Qwen2.5-Math-7B (bfloat16)
OL('qwen2-5-7b', 'qwen2-5-math-7b', 58.08)	# Qwen/Qwen2.5-Math-7B-Instruct (bfloat16)
OL('qwen2-5-32b', 'qwq-32b', 16.09)	# Qwen/QwQ-32B (bfloat16)
OL('qwen2-5-32b', 'qwq-32b-preview', 44.94)	# Qwen/QwQ-32B-Preview (bfloat16)
OL('seallm-7b-v2', 'seallm-7b-v2', 8.53)	# SeaLLMs/SeaLLM-7B-v2 (bfloat16)
OL('seallm-7b-v2-5', 'seallm-7b-v2-5', 10.88)	# SeaLLMs/SeaLLM-7B-v2.5 (bfloat16)
OL('glm-4-9b', 'glm-4-9b-chat', 0.0)	# THUDM/glm-4-9b (bfloat16)
OL('glm-4-9b', 'glm-4-9b-chat', 0.0)	# THUDM/glm-4-9b-chat (bfloat16)
OL('glm-4-9b', 'glm-4-9b-chat-1m', 0.0)	# THUDM/glm-4-9b-chat-1m (bfloat16)
OL('glm-4-9b', 'glm-4-9b-chat-1m', 4.83)	# THUDM/glm-4-9b-chat-1m-hf (bfloat16)
OL('glm-4-9b', 'glm-4-9b-chat', 8.46)	# THUDM/glm-4-9b-chat-hf (bfloat16)
OL('falcon2-11b', 'falcon2-11b', 2.79)	# tiiuae/falcon-11B (bfloat16)
OL('falcon-40b', 'falcon-40b', 1.81)	# tiiuae/falcon-40b (bfloat16)
OL('falcon-40b', 'falcon-40b-instruct', 1.96)	# tiiuae/falcon-40b-instruct (bfloat16)
OL('falcon-7b', 'falcon-7b', 0.98)	# tiiuae/falcon-7b (bfloat16)
OL('falcon-7b', 'falcon-7b-instruct', 1.21)	# tiiuae/falcon-7b-instruct (bfloat16)
OL('falcon-mamba-7b', 'falcon-mamba-7b', 4.46)	# tiiuae/falcon-mamba-7b (bfloat16)
OL('falcon3-10b', 'falcon3-10b-base', 24.92)	# tiiuae/Falcon3-10B-Base (bfloat16)
OL('falcon3-10b', 'falcon3-10b-instruct', 27.64)	# tiiuae/Falcon3-10B-Instruct (bfloat16)
OL('falcon3-1b', 'falcon3-1b-base', 3.32)	# tiiuae/Falcon3-1B-Base (bfloat16)
OL('falcon3-1b', 'falcon3-1b-instruct', 6.34)	# tiiuae/Falcon3-1B-Instruct (bfloat16)
OL('falcon3-3b', 'falcon3-3b-base', 11.78)	# tiiuae/Falcon3-3B-Base (bfloat16)
OL('falcon3-3b', 'falcon3-3b-instruct', 25.0)	# tiiuae/Falcon3-3B-Instruct (bfloat16)
OL('falcon3-7b', 'falcon3-7b-base', 19.41)	# tiiuae/Falcon3-7B-Base (bfloat16)
OL('falcon3-7b', 'falcon3-7b-instruct', 40.86)	# tiiuae/Falcon3-7B-Instruct (bfloat16)
OL('falcon3-mamba-7b', 'falcon3-mamba-7b-base', 19.41)	# tiiuae/Falcon3-Mamba-7B-Base (bfloat16)
OL('falcon3-mamba-7b', 'falcon3-mamba-7b-instruct', 30.06)	# tiiuae/Falcon3-Mamba-7B-Instruct (bfloat16)
OL('wizardlm-13b', 'wizardlm-13b-v1-0', 0.0)	# WizardLMTeam/WizardLM-13B-V1.0 (bfloat16)
OL('wizardlm-13b', 'wizardlm-13b-v1-2', 1.89)	# WizardLMTeam/WizardLM-13B-V1.2 (float16)
OL('wizardlm-70b', 'wizardlm-70b', 3.93)	# WizardLMTeam/WizardLM-70B-V1.0 (float16)
L('Yi-1.5-34B', 15.33, OLB, part=L5)
L('Yi-1.5-34B-32K', 15.41, OLB, part=L5)
L('Yi-1.5-34B-Chat', 27.72, OLB, part=L5)
L('Yi-1.5-34B-Chat-16K', 21.37, OLB, part=L5)
L('Yi-1.5-6B', 6.65, OLB, part=L5)
L('Yi-1.5-6B-Chat', 16.24, OLB, part=L5)
L('Yi-1.5-9B', 11.4, OLB, part=L5)
L('Yi-1.5-9B-32K', 10.8, OLB, part=L5)
L('Yi-1.5-9B-Chat', 22.58, OLB, part=L5)
L('Yi-1.5-9B-Chat-16K', 17.82, OLB, part=L5)
L('Yi-34B', 5.14, OLB, part=L5)
L('Yi-34B-200K', 5.74, OLB, part=L5)
L('Yi-34B-Chat', 6.27, OLB, part=L5)
L('Yi-6B', 1.59, OLB, part=L5)
L('Yi-6B-200K', 1.81, OLB, part=L5)
L('Yi-6B-Chat', 1.36, OLB, part=L5)
L('Yi-9B', 5.59, OLB, part=L5)
L('Yi-9B-200K', 6.65, OLB, part=L5)
L('Yi-Coder-9B-Chat', 4.0, OLB, part=L5)
L('bigstral-12b-32k', 1.51, OLB, part=L5)
L('bigyi-15b', 2.95, OLB, part=L5)
L('Dracarys-72B-Instruct', 39.65, OLB, part=L5)
L('Liberated-Qwen1.5-14B', 16.01, OLB, part=L5)
L('Llama-3-Smaug-8B', 8.53, OLB, part=L5)
L('Smaug-34B-v0.1', 7.18, OLB, part=L5)
L('Smaug-72B-v0.1', 19.11, OLB, part=L5)
L('Smaug-Llama-3-70B-Instruct-32K', 27.49, OLB, part=L5)
L('Smaug-Mixtral-v0.1', 9.52, OLB, part=L5)
L('Smaug-Qwen2-72B-Instruct', 41.31, OLB, part=L5)
L('Jamba-v0.1', 1.59, OLB, part=L5)
L('Llama-3.1-Tulu-3-70B', 45.02, OLB, part=L5)
L('Llama-3.1-Tulu-3-70B-DPO', 44.94, OLB, part=L5)
L('Llama-3.1-Tulu-3-70B-SFT', 33.16, OLB, part=L5)
L('Llama-3.1-Tulu-3-8B', 19.64, OLB, part=L5)
L('Llama-3.1-Tulu-3-8B-DPO', 23.64, OLB, part=L5)
L('Llama-3.1-Tulu-3-8B-SFT', 11.78, OLB, part=L5)
L('OLMo-1.7-7B-hf', 0.23, OLB, part=L5)
L('OLMo-1B-hf', 1.74, OLB, part=L5)
L('OLMo-2-1124-7B-Instruct', 14.88, OLB, part=L5)
L('OLMo-7B-hf', 1.21, OLB, part=L5)
L('OLMo-7B-Instruct-hf', 1.36, OLB, part=L5)
L('OLMoE-1B-7B-0125-Instruct', 8.99, OLB, part=L5)
L('OLMoE-1B-7B-0924', 1.66, OLB, part=L5)
L('OLMoE-1B-7B-0924-Instruct', 2.79, OLB, part=L5)
L('AMD-Llama-135m', 0.76, OLB, part=L5)
L('Llama-3.1-8B-MagPie-Ultra', 7.7, OLB, part=L5)
L('notus-7b-v1', 3.17, OLB, part=L5)
L('notux-8x7b-v1', 9.97, OLB, part=L5)
L('Starling-LM-7B-alpha', 8.38, OLB, part=L5)
L('14B', 7.55, OLB, part=L5)
L('34b-beta', 4.83, OLB, part=L5)
L('preview-1-hf', 3.02, OLB, part=L5)
L('dolphin-2.9-llama3-8b', 5.74, OLB, part=L5)
L('dolphin-2.9.1-llama-3-70b', 18.2, OLB, part=L5)
L('dolphin-2.9.1-yi-1.5-34b', 18.66, OLB, part=L5)
L('dolphin-2.9.1-yi-1.5-9b', 15.18, OLB, part=L5)
L('dolphin-2.9.2-Phi-3-Medium', 18.28, OLB, part=L5)
L('dolphin-2.9.2-Phi-3-Medium-abliterated', 18.2, OLB, part=L5)
L('dolphin-2.9.2-qwen2-72b', 28.02, OLB, part=L5)
L('dolphin-2.9.2-qwen2-7b', 13.44, OLB, part=L5)
L('dolphin-2.9.3-mistral-7B-32k', 5.06, OLB, part=L5)
L('dolphin-2.9.3-mistral-nemo-12b', 7.4, OLB, part=L5)
L('dolphin-2.9.3-Yi-1.5-34B-32k', 16.69, OLB, part=L5)
L('dolphin-2.9.4-gemma2-2b', 4.91, OLB, part=L5)
L('dolphin-2.9.4-llama3.1-8b', 1.21, OLB, part=L5)
L('Dolphin3.0-Llama3.1-8B', 12.31, OLB, part=L5)
L('Dolphin3.0-Llama3.2-1B', 2.79, OLB, part=L5)
L('Dolphin3.0-Qwen2.5-0.5B', 5.14, OLB, part=L5)
L('Dolphin3.0-R1-Mistral-24B', 31.19, OLB, part=L5)
L('aya-23-35B', 3.47, OLB, part=L5)
L('aya-23-8B', 1.66, OLB, part=L5)
L('aya-expanse-32b', 15.33, OLB, part=L5)
L('aya-expanse-8b', 8.61, OLB, part=L5)
L('c4ai-command-r-plus', 8.01, OLB, part=L5)
L('c4ai-command-r-plus-08-2024', 12.39, OLB, part=L5)
L('c4ai-command-r-v01', 3.47, OLB, part=L5)
L('c4ai-command-r7b-12-2024', 29.91, OLB, part=L5)
L('DeciLM-7B', 2.87, OLB, part=L5)
L('DeciLM-7B-instruct', 3.02, OLB, part=L5)
L('mt5-base', 0.91, OLB, part=L5)
L('mt5-small', 0.0, OLB, part=L5)
L('mt5-xl', 0.0, OLB, part=L5)
L('switch-base-8', 0.0, OLB, part=L5)
L('umt5-base', 0.45, OLB, part=L5)
OL('gpt-2', 'gpt-2-124m', 0.3)	# gpt2 (bfloat16)
L('Llama-3-8B-Instruct-Gradient-1048k', 5.36, OLB, part=L5)
L('granite-3.0-1b-a400m-base', 2.64, OLB, part=L5)
L('granite-3.0-1b-a400m-instruct', 2.79, OLB, part=L5)
L('granite-3.0-2b-base', 5.44, OLB, part=L5)
L('granite-3.0-2b-instruct', 9.21, OLB, part=L5)
L('granite-3.0-3b-a800m-base', 4.83, OLB, part=L5)
L('granite-3.0-3b-a800m-instruct', 7.02, OLB, part=L5)
L('granite-3.0-8b-base', 10.12, OLB, part=L5)
L('granite-3.0-8b-instruct', 14.2, OLB, part=L5)
L('granite-3.1-1b-a400m-base', 2.72, OLB, part=L5)
L('granite-3.1-1b-a400m-instruct', 4.53, OLB, part=L5)
L('granite-3.1-2b-base', 5.66, OLB, part=L5)
L('granite-3.1-2b-instruct', 15.26, OLB, part=L5)
L('granite-3.1-3b-a800m-base', 4.53, OLB, part=L5)
L('granite-3.1-3b-a800m-instruct', 11.4, OLB, part=L5)
L('granite-3.1-8b-base', 9.44, OLB, part=L5)
L('granite-3.1-8b-instruct', 21.98, OLB, part=L5)
L('granite-3.2-2b-instruct', 14.43, OLB, part=L5)
L('granite-3.2-8b-instruct', 23.79, OLB, part=L5)
L('granite-7b-base', 1.59, OLB, part=L5)
L('granite-7b-instruct', 2.04, OLB, part=L5)
L('merlinite-7b', 2.42, OLB, part=L5)
L('PowerLM-3b', 3.63, OLB, part=L5)
L('Ziya-LLaMA-13B-v1', 0.0, OLB, part=L5)
L('neural-chat-7b-v3', 2.95, OLB, part=L5)
L('neural-chat-7b-v3-1', 3.55, OLB, part=L5)
L('neural-chat-7b-v3-2', 4.76, OLB, part=L5)
L('neural-chat-7b-v3-3', 4.08, OLB, part=L5)
L('internlm2-1_8b', 2.11, OLB, part=L5)
L('internlm2-7b', 8.57, OLB, part=L5)
L('internlm2-chat-1_8b', 3.25, OLB, part=L5)
L('internlm2_5-1_8b-chat', 15.86, OLB, part=L5)
L('internlm2_5-20b-chat', 40.79, OLB, part=L5)
L('internlm2_5-7b-chat', 25.3, OLB, part=L5)
L('helium-1-preview-2b', 1.36, OLB, part=L5)
L('K2', 2.72, OLB, part=L5)
L('K2-Chat', 10.35, OLB, part=L5)
L('vicuna-13b-v1.3', 1.44, OLB, part=L5)
L('vicuna-7b-v1.3', 1.28, OLB, part=L5)
L('vicuna-7b-v1.5', 1.36, OLB, part=L5)
OL('mistral-7b', 'mistral-7b-v0-2', 3.02)	# mistral-community/Mistral-7B-v0.2 (bfloat16)
OL('mixtral-8x22b', 'mixtral-8x22b', 15.43)	# mistral-community/Mixtral-8x22B-v0.1 (float16)
L('mixtral-8x22B-v0.3', 18.35, OLB, part=L5)
L('AlphaMonarch-7B', 4.08, OLB, part=L5)
L('Beyonder-4x7B-v3', 5.36, OLB, part=L5)
L('BigQwen2.5-52B-Instruct', 54.76, OLB, part=L5)
L('BigQwen2.5-Echo-47B-Instruct', 43.81, OLB, part=L5)
L('ChimeraLlama-3-8B-v2', 9.06, OLB, part=L5)
L('ChimeraLlama-3-8B-v3', 8.84, OLB, part=L5)
L('Daredevil-8B', 10.65, OLB, part=L5)
L('Daredevil-8B-abliterated', 9.44, OLB, part=L5)
L('Hermes-3-Llama-3.1-70B-lorablated', 22.43, OLB, part=L5)
L('Meta-Llama-3.1-8B-Instruct-abliterated', 6.87, OLB, part=L5)
L('NeuralBeagle14-7B', 5.21, OLB, part=L5)
L('NeuralDaredevil-8B-abliterated', 8.53, OLB, part=L5)
L('OrpoLlama-3-8B', 5.59, OLB, part=L5)
L('phixtral-2x2_8', 3.55, OLB, part=L5)
L('NexusRaven-V2-13B', 2.95, OLB, part=L5)
L('DeepHermes-3-Mistral-24B-Preview', 25.76, OLB, part=L5)
L('Hermes-2-Pro-Llama-3-8B', 8.38, OLB, part=L5)
L('Hermes-2-Pro-Mistral-7B', 6.04, OLB, part=L5)
L('Hermes-2-Theta-Llama-3-8B', 9.67, OLB, part=L5)
L('Hermes-3-Llama-3.1-70B', 21.0, OLB, part=L5)
L('Hermes-3-Llama-3.1-8B', 4.76, OLB, part=L5)
L('Hermes-3-Llama-3.2-3B', 3.93, OLB, part=L5)
L('Nous-Hermes-2-Mistral-7B-DPO', 4.76, OLB, part=L5)
L('Nous-Hermes-2-Mixtral-8x7B-DPO', 12.24, OLB, part=L5)
L('Nous-Hermes-2-Mixtral-8x7B-SFT', 2.11, OLB, part=L5)
L('Nous-Hermes-2-SOLAR-10.7B', 5.74, OLB, part=L5)
L('Nous-Hermes-llama-2-7b', 0.91, OLB, part=L5)
L('Yarn-Llama-2-13b-128k', 1.74, OLB, part=L5)
L('Yarn-Llama-2-7b-128k', 1.51, OLB, part=L5)
L('Yarn-Llama-2-7b-64k', 1.59, OLB, part=L5)
L('Yarn-Mistral-7b-128k', 3.17, OLB, part=L5)
L('Yarn-Mistral-7b-64k', 3.7, OLB, part=L5)
L('Yarn-Solar-10b-32k', 3.02, OLB, part=L5)
L('Yarn-Solar-10b-64k', 2.87, OLB, part=L5)
L('Mistral-7B-OpenOrca', 3.55, OLB, part=L5)
OL('gpt-2', 'gpt-2-124m', 0.23)	# openai-community/gpt2 (bfloat16)
OL('gpt-2', 'gpt-2-774m', 1.21)	# openai-community/gpt2-large (bfloat16)
OL('gpt-2', 'gpt-2-355m', 0.76)	# openai-community/gpt2-medium (bfloat16)
OL('gpt-2', 'gpt-2-1-5b', 0.98)	# openai-community/gpt2-xl (bfloat16)
L('oasst-sft-1-pythia-12b', 1.51, OLB, part=L5)
L('openchat-3.5-0106', 7.63, OLB, part=L5)
L('openchat-3.5-1210', 7.85, OLB, part=L5)
L('openchat-3.6-8b-20240522', 9.97, OLB, part=L5)
L('openchat_3.5', 7.25, OLB, part=L5)
L('openchat_v3.2', 1.28, OLB, part=L5)
L('openchat_v3.2_super', 2.11, OLB, part=L5)
L('INTELLECT-1', 0.0, OLB, part=L5)
L('INTELLECT-1-Instruct', 2.27, OLB, part=L5)
L('pygmalion-6b', 0.83, OLB, part=L5)
L('Llama-3-Refueled', 6.65, OLB, part=L5)
L('Aria', 19.34, OLB, part=L5)
L('LLaMA-3-8B-SFR-Iterative-DPO-R', 9.14, OLB, part=L5)
L('Bielik-11B-v2', 7.85, OLB, part=L5)
L('Bielik-11B-v2.0-Instruct', 11.86, OLB, part=L5)
L('Bielik-11B-v2.1-Instruct', 26.66, OLB, part=L5)
L('Bielik-11B-v2.2-Instruct', 26.81, OLB, part=L5)
L('Bielik-11B-v2.3-Instruct', 20.85, OLB, part=L5)
L('StableBeluga2', 4.38, OLB, part=L5)
L('stablelm-2-12b', 4.31, OLB, part=L5)
L('stablelm-2-12b-chat', 5.36, OLB, part=L5)
L('stablelm-2-1_6b', 0.76, OLB, part=L5)
L('stablelm-2-1_6b-chat', 2.49, OLB, part=L5)
L('stablelm-2-zephyr-1_6b', 3.32, OLB, part=L5)
L('stablelm-3b-4e1t', 1.06, OLB, part=L5)
L('stablelm-zephyr-3b', 4.31, OLB, part=L5)
L('CollectiveCognition-v1.1-Mistral-7B', 3.1, OLB, part=L5)
L('OpenHermes-13B', 1.21, OLB, part=L5)
L('OpenHermes-2-Mistral-7B', 4.53, OLB, part=L5)
L('OpenHermes-2.5-Mistral-7B', 5.06, OLB, part=L5)
L('OpenHermes-7B', 1.59, OLB, part=L5)
L('LLaMA-Pro-8B', 1.89, OLB, part=L5)
L('LLaMA-Pro-8B-Instruct', 2.49, OLB, part=L5)
L('MetaMath-Mistral-Pro', 7.63, OLB, part=L5)
L('Mistral_Pro_8B_v0.1', 5.66, OLB, part=L5)
L('TinyLlama-1.1B-Chat-v0.1', 0.6, OLB, part=L5)
L('TinyLlama-1.1B-Chat-v0.5', 0.38, OLB, part=L5)
L('TinyLlama-1.1B-Chat-v0.6', 1.59, OLB, part=L5)
L('TinyLlama-1.1B-Chat-v1.0', 1.51, OLB, part=L5)
L('TinyLlama-1.1B-intermediate-step-1431k-3T', 1.21, OLB, part=L5)
L('TinyLlama_v1.1', 1.21, OLB, part=L5)
L('GPT-JT-6B-v1', 1.06, OLB, part=L5)
L('GPT-NeoXT-Chat-Base-20B', 2.34, OLB, part=L5)
L('LLaMA-2-7B-32K', 1.44, OLB, part=L5)
L('Llama-2-7B-32K-Instruct', 1.59, OLB, part=L5)
L('RedPajama-INCITE-7B-Base', 1.59, OLB, part=L5)
L('RedPajama-INCITE-7B-Chat', 0.68, OLB, part=L5)
L('RedPajama-INCITE-7B-Instruct', 2.11, OLB, part=L5)
L('RedPajama-INCITE-Base-3B-v1', 1.44, OLB, part=L5)
L('RedPajama-INCITE-Chat-3B-v1', 0.91, OLB, part=L5)
L('RedPajama-INCITE-Instruct-3B-v1', 1.28, OLB, part=L5)
L('SOLAR-10.7B-Instruct-v1.0', 5.66, OLB, part=L5)
L('SOLAR-10.7B-v1.0', 2.64, OLB, part=L5)
L('solar-pro-preview-instruct', 22.05, OLB, part=L5)

write('math')
