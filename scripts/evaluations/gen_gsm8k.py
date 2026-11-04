"""Generates data/results/gsm8k.jsonl and data/leftovers/gsm8k.jsonl.

Every number is written literally in this file (nothing is downloaded at run time), so it is the single source of
truth for the GSM8K results: edit it and run `python scripts/evaluations/gen_gsm8k.py`. The helpers (R, L, write) and
their conventions are in common.py.

Sources:
- Epoch AI "GSM8K (external)" table (https://epoch.ai/benchmarks/gsm8k, CSV in https://epoch.ai/data/benchmark_data.zip),
  each row pointing to the paper / HELM page it was taken from
- llm-stats (https://llm-stats.com/benchmarks/gsm8k), using the source it cites
- Hugging Face Open LLM Leaderboard v1 (5-shot, dataset open-llm-leaderboard-old/contents), best of the bf16 / fp16 runs
- Model cards, technical reports and announcements of the model creators (and the third-party numbers they report)

Choices specific to GSM8K:
- cost is always None: no source gives the real cost per task / per evaluation (API prices are not costs)
- Mixtral / Mistral papers, Gemini 1.0, LLaMA, Minerva and PaLM 2 report majority voting: written with maj_k
"""

from common import A, HF, L, R, write

HELM = 'https://crfm.stanford.edu/helm/classic/latest/#/groups/gsm'
CLI = 'https://www.anthropic.com/news/releasing-claude-instant-1-2'
OLB = 'https://huggingface.co/spaces/open-llm-leaderboard-old/open_llm_leaderboard'

# ---------------------------------------------------------------- Epoch AI external table
# HELM uses 5 shots. Skipped rows: the Mixtral report ones (maj@8, written in the maj@k section with maj_k), the phi-1.5
# report ones (copies of other papers) and Minerva 540B from the LLaMA report (Epoch's 68.5 is Minerva 62B maj1@100).

# Stanford HELM
L('Palmyra X (43B)', 63.3, HELM)
R('gpt-3-5', 'code-davinci-002', 56.8, 5, HELM, False)
R('gpt-3-5-turbo', 'gpt-3-5-turbo-mar-2023', 53.1, 5, HELM, False)
R('gpt-3-5', 'text-davinci-003', 50.6, 5, HELM, False)
R('llama-2-70b', 'llama-2-70b', 48.4, 5, HELM, False)
R('gpt-3-5-turbo', 'gpt-3-5-turbo-jun-2023', 46.9, 5, HELM, False)
R('llama-65b', 'llama-65b', 46.6, 5, HELM, False)
R('gpt-3-5', 'text-davinci-002', 41.5, 5, HELM, False)
R('mistral-7b', 'mistral-7b-v0-1', 38.1, 5, HELM, False)
R('mpt-30b', 'mpt-30b-instruct', 34.4, 5, HELM, False)
R('falcon-40b', 'falcon-40b-instruct', 33.8, 5, HELM, False)
R('llama-33b', 'llama-33b', 32.0, 5, HELM, False)
R('falcon-40b', 'falcon-40b', 25.0, 5, HELM, False)
R('llama-2-13b', 'llama-2-13b', 24.5, 5, HELM, False)
L('Vicuna v1.3 (13B)', 22.6, HELM)
L('Jurassic-2 Jumbo (178B)', 22.5, HELM)
R('anthropic-lm-52b', 'anthropic-lm-52b', 17.1, 5, HELM, False)
R('mpt-30b', 'mpt-30b', 16.4, 5, HELM, False)
R('llama-13b', 'llama-13b', 15.4, 5, HELM, False)
R('megatron-turing-nlg-530b', 'megatron-turing-nlg-530b', 14.6, 5, HELM, False)
L('Cohere Command beta (52.4B)', 13.8, HELM)
L('Vicuna v1.3 (7B)', 13.4, HELM)
R('llama-2-7b', 'llama-2-7b', 13.3, 5, HELM, False)
L('Jurassic-2 Grande (17B)', 13.3, HELM)
L('Luminous Supreme (70B)', 11.2, HELM)
L('Cohere xlarge v20221108 (52.4B)', 10.0, HELM)
L('J1-Grande v2 beta (17B)', 9.6, HELM)
R('bloom', 'bloom', 9.5, 5, HELM, False)
R('gpt-3', 'gpt-3-175b', 9.0, 5, HELM, False)
R('llama-7b', 'llama-7b', 8.0, 5, HELM, False)
L('Cohere xlarge v20220609 (52.4B)', 7.0, HELM)
L('Luminous Extended (30B)', 6.7, HELM)
L('InstructPalmyra (30B)', 6.3, HELM)
R('glm-130b', 'glm-130b', 6.1, 5, HELM, False)
L('J1-Jumbo v1 (178B)', 5.4, HELM)
L('J1-Grande v1 (17B)', 5.4, HELM)
R('gpt-neox-20b', 'gpt-neox-20b', 5.3, 5, HELM, False)
R('falcon-7b', 'falcon-7b-instruct', 5.2, 5, HELM, False)
R('gpt-3', 'codex-12b', 4.9, 5, HELM, False)
R('opt', 'opt-175b', 4.0, 5, HELM, False)
R('falcon-7b', 'falcon-7b', 4.0, 5, HELM, False)
L('Cohere Command beta (6.1B)', 3.6, HELM)
R('gpt-j-6b', 'gpt-j-6b', 3.6, 5, HELM, False)
R('pythia-12b', 'pythia-12b', 3.2, 5, HELM, False)
L('Jurassic-2 Large (7.5B)', 3.0, HELM)
L('Luminous Base (13B)', 2.6, HELM)
R('ul2', 'ul2', 2.4, 5, HELM, False)
R('t5', 't5-11b', 2.3, 5, HELM, False)
L('RedPajama-INCITE-Base (7B)', 2.1, HELM)
L('TNLG v2 (6.7B)', 1.8, HELM)
R('opt', 'opt-66b', 1.8, 5, HELM, False)
L('Cohere large v20220720 (13.1B)', 1.8, HELM)
L('Cohere medium v20221108 (6.1B)', 1.7, HELM)
L('RedPajama-INCITE-Instruct (7B)', 1.6, HELM)
R('gpt-3', 'gpt-3-6-7b', 1.6, 5, HELM, False)
L('Cohere medium v20220720 (6.1B)', 1.5, HELM)
R('pythia-6-9b', 'pythia-6-9b', 1.4, 5, HELM, False)
L('J1-Large v1 (7.5B)', 1.4, HELM)
L('Alpaca (7B)', 1.2, HELM)
L('RedPajama-INCITE-Instruct-v1 (3B)', 1.1, HELM)
L('RedPajama-INCITE-Base-v1 (3B)', 1.0, HELM)
R('gpt-3', 'gpt-3-1-3b', 0.7, 5, HELM, False)
R('gpt-3', 'gpt-3-350m', 0.6, 5, HELM, False)
R('gpt-3', 'instructgpt-6b', 0.6, 5, HELM, False)
L('Cohere small v20220720 (410M)', 0.4, HELM)
R('gpt-3', 'instructgpt-350m', 0.4, 5, HELM, False)
R('gpt-3', 'instructgpt-1-3b', 0.0, 5, HELM, False)
R('t0', 't0pp', 0.0, 5, HELM, False)
L('YaLM (100B)', 0.0, HELM)

# Gemma: Open Models Based on Gemini Research and Technology
R('llama-2-7b', 'llama-2-7b', 14.6, 5, A + '2403.08295', False)
R('llama-2-13b', 'llama-2-13b', 28.7, 5, A + '2403.08295', False)
R('mistral-7b', None, 35.4, 5, A + '2403.08295', False)
R('gemma-2b', 'gemma-2b', 17.7, 5, A + '2403.08295', True)
R('gemma-7b', 'gemma-7b', 46.4, 5, A + '2403.08295', True)

# Nemotron-4 15B Technical Report
R('qwen-14b', 'qwen-14b', 60.1, 8, A + '2402.16819', False)
R('nemotron-4-15b', 'nemotron-4-15b', 46.0, 8, A + '2402.16819', True)
R('llama-2-13b', 'llama-2-13b', 28.7, 8, A + '2402.16819', False)
R('llama-2-34b', 'llama-2-34b', 42.2, 8, A + '2402.16819', False)
L('Baichuan 2 13B', 52.8, A + '2402.16819')
R('mistral-7b', 'mistral-7b-v0-1', 35.4, 8, A + '2402.16819', False)
R('gemma-7b', 'gemma-7b', 46.4, 8, A + '2402.16819', False)

# Baichuan 2: Open Large-scale Language Models
R('gpt-4', 'gpt-4-jun-2023', 89.99, 4, A + '2309.10305', False)
R('gpt-3-5-turbo', 'gpt-3-5-turbo-jun-2023', 57.77, 4, A + '2309.10305', False)
R('llama-7b', 'llama-7b', 9.78, 4, A + '2309.10305', False)
R('llama-2-7b', 'llama-2-7b', 16.22, 4, A + '2309.10305', False)
R('mpt-7b', 'mpt-7b', 8.64, 4, A + '2309.10305', False)
R('falcon-7b', 'falcon-7b', 5.46, 4, A + '2309.10305', False)
R('chatglm2-6b', 'chatglm2-6b', 28.9, 4, A + '2309.10305', False)
L('Baichuan 1 7B', 9.2, A + '2309.10305')
L('Baichuan 2 7B Base', 24.5, A + '2309.10305')
R('llama-13b', 'llama-13b', 20.55, 4, A + '2309.10305', False)
R('llama-2-13b', 'llama-2-13b', 28.89, 4, A + '2309.10305', False)
L('Vicuna 13B', 28.13, A + '2309.10305')
L('Chinese-Alpaca-Plus-13B', 12.0, A + '2309.10305')
L('XVERSE-13B', 18.0, A + '2309.10305')
L('Baichuan 1 13B Base', 26.8, A + '2309.10305')
L('Baichuan 2 13B Base', 52.8, A + '2309.10305')

# Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone
R('phi-3-5-mini', 'phi-3-5-mini-instruct', 86.2, 8, A + '2404.14219', True)
R('phi-3-5-moe', 'phi-3-5-moe', 88.7, 8, A + '2404.14219', True)
R('mistral-7b', 'mistral-7b-v0-3', 54.4, 8, A + '2404.14219', False)
R('mistral-nemo', 'mistral-nemo', 84.2, 8, A + '2404.14219', False)
R('llama-3-1-8b', 'llama-3-1-8b', 82.4, 8, A + '2404.14219', False)
R('gemma-2-9b', 'gemma-2-9b', 84.9, 8, A + '2404.14219', False)
R('gemini-1-5-flash', 'gemini-1-5-flash-may-2024', 82.4, 8, A + '2404.14219', False)
R('gpt-4o-mini', 'gpt-4o-mini', 91.3, 8, A + '2404.14219', False)

# Yi: Open Foundation Models by 01.AI
R('gpt-3-5', 'text-davinci-003', 57.1, 0, A + '2403.04652', False)
R('gpt-4', 'gpt-4-mar-2023', 92.0, 0, A + '2403.04652', False)
R('falcon-180b', 'falcon-180b', 54.4, 0, A + '2403.04652', False)
R('qwen-7b', 'qwen-7b', 51.7, 0, A + '2403.04652', False)
R('qwen-14b', 'qwen-14b', 61.3, 0, A + '2403.04652', False)
L('Baichuan 2 7B', 24.5, A + '2403.04652')
L('Baichuan 2 13B', 22.1, A + '2403.04652')
R('llama-2-7b', 'llama-2-7b', 16.7, 0, A + '2403.04652', False)
R('llama-2-34b', 'llama-2-34b', 42.2, 0, A + '2403.04652', False)
R('llama-2-70b', 'llama-2-70b', 56.8, 0, A + '2403.04652', False)
R('mistral-7b', 'mistral-7b-v0-1', 47.5, 0, A + '2403.04652', False)
L('InternLM 20B', 62.9, A + '2403.04652')
L('Skywork 7B', 55.8, A + '2403.04652')
L('Yi 6B', 32.5, A + '2403.04652')
L('Yi 34B', 67.2, A + '2403.04652')

# Llama 2: Open Foundation and Fine-Tuned Chat Models
R('mpt-7b', 'mpt-7b', 6.8, 8, A + '2307.09288', False)
R('mpt-30b', 'mpt-30b', 15.2, 8, A + '2307.09288', False)
R('falcon-7b', 'falcon-7b', 6.8, 8, A + '2307.09288', False)
R('falcon-40b', 'falcon-40b', 19.6, 8, A + '2307.09288', False)
R('llama-7b', 'llama-7b', 11.0, 8, A + '2307.09288', True)
R('llama-13b', 'llama-13b', 17.8, 8, A + '2307.09288', True)
R('llama-33b', 'llama-33b', 35.6, 8, A + '2307.09288', True)
R('llama-65b', 'llama-65b', 50.9, 8, A + '2307.09288', True)
R('llama-2-7b', 'llama-2-7b', 14.6, 8, A + '2307.09288', True)
R('llama-2-13b', 'llama-2-13b', 28.7, 8, A + '2307.09288', True)
R('llama-2-34b', 'llama-2-34b', 42.2, 8, A + '2307.09288', True)
R('llama-2-70b', 'llama-2-70b', 56.8, 8, A + '2307.09288', True)

# LLaMA: Open and Efficient Foundation Language Models
R('palm', 'palm-8b', 4.1, 8, A + '2302.13971', False)
R('palm', 'palm-62b', 33.0, 8, A + '2302.13971', False)
R('palm', 'palm-540b', 56.5, 8, A + '2302.13971', False)
R('palm', 'minerva-8b', 16.2, 8, A + '2302.13971', False)
R('palm', 'minerva-62b', 52.4, 8, A + '2302.13971', False)
R('llama-7b', 'llama-7b', 11.0, 8, A + '2302.13971', True)
R('llama-13b', 'llama-13b', 17.8, 8, A + '2302.13971', True)
R('llama-33b', 'llama-33b', 35.6, 8, A + '2302.13971', True)
R('llama-65b', 'llama-65b', 50.9, 8, A + '2302.13971', True)

# Qwen Technical Report
R('gpt-3-5-turbo', None, 78.2, 8, A + '2309.16609', False)
R('gpt-4', None, 91.4, 8, A + '2309.16609', False)
R('chatglm2-6b', 'chatglm2-6b', 28.8, 8, A + '2309.16609', False)
L('InternLM-Chat 7B', 33.0, A + '2309.16609')
L('Baichuan 2 Chat 7B', 32.8, A + '2309.16609')
L('Baichuan 2 Chat 13B', 55.3, A + '2309.16609')
R('llama-2-7b', 'llama-2-7b', 26.3, 8, A + '2309.16609', False)
R('llama-2-13b', 'llama-2-13b', 37.1, 8, A + '2309.16609', False)
R('llama-2-70b', 'llama-2-70b', 59.3, 8, A + '2309.16609', False)
R('qwen-1-8b', 'qwen-1-8b', 27.8, 0, A + '2309.16609', True)
R('qwen-1-8b', 'qwen-1-8b', 19.5, 8, A + '2309.16609', True)
R('qwen-7b', 'qwen-7b', 50.3, 0, A + '2309.16609', True)
R('qwen-7b', 'qwen-7b', 54.1, 8, A + '2309.16609', True)
R('qwen-14b', 'qwen-14b', 60.1, 0, A + '2309.16609', True)
R('qwen-14b', 'qwen-14b', 59.3, 8, A + '2309.16609', True)
R('mpt-7b', 'mpt-7b', 9.1, 8, A + '2309.16609', False)
R('mpt-30b', 'mpt-30b', 15.2, 8, A + '2309.16609', False)
R('falcon-7b', 'falcon-7b', 6.8, 8, A + '2309.16609', False)
R('falcon-40b', 'falcon-40b', 19.6, 8, A + '2309.16609', False)
R('chatglm2-6b', 'chatglm2-6b', 32.4, 8, A + '2309.16609', False)
L('InternLM 7B', 31.2, A + '2309.16609')
L('InternLM 20B', 52.6, A + '2309.16609')
L('Baichuan 2 7B', 24.6, A + '2309.16609')
L('Baichuan 2 13B', 52.8, A + '2309.16609')
R('llama-7b', 'llama-7b', 11.0, 8, A + '2309.16609', False)
R('llama-13b', 'llama-13b', 20.3, 8, A + '2309.16609', False)
R('llama-33b', 'llama-33b', 42.3, 8, A + '2309.16609', False)
R('llama-65b', 'llama-65b', 54.4, 8, A + '2309.16609', False)
R('llama-2-7b', 'llama-2-7b', 16.7, 8, A + '2309.16609', False)
R('llama-2-13b', 'llama-2-13b', 29.6, 8, A + '2309.16609', False)
R('llama-2-34b', 'llama-2-34b', 42.2, 8, A + '2309.16609', False)
R('llama-2-70b', 'llama-2-70b', 63.3, 8, A + '2309.16609', False)
L('StableBeluga2 70B', 69.6, A + '2309.16609')
R('qwen-1-8b', 'qwen-1-8b', 21.2, 8, A + '2309.16609', True)
R('qwen-7b', 'qwen-7b', 51.7, 8, A + '2309.16609', True)
R('qwen-14b', 'qwen-14b', 61.3, 8, A + '2309.16609', True)

# Qwen2.5-Coder Technical Report
R('deepseek-coder-v2-lite', 'deepseek-coder-v2-lite', 87.6, 0, A + '2409.12186', False)
R('deepseek-coder-v2', 'deepseek-coder-v2-jun-2024', 94.5, 0, A + '2409.12186', False)
R('qwen2-5-3b', 'qwen2-5-coder-3b', 80.7, 0, A + '2409.12186', True)
R('qwen2-5-7b', 'qwen2-5-coder-7b', 86.7, 0, A + '2409.12186', True)
R('qwen2-5-14b', 'qwen2-5-coder-14b', 94.2, 0, A + '2409.12186', True)
R('qwen2-5-32b', 'qwen2-5-coder-32b', 93.0, 0, A + '2409.12186', True)
R('qwen2-5-0-5b', 'qwen2-5-coder-0-5b', 34.5, 4, A + '2409.12186', True)
R('deepseek-coder-1-3b', 'deepseek-coder-1-3b', 4.4, 4, A + '2409.12186', False)
R('qwen2-5-1-5b', 'qwen2-5-coder-1-5b', 65.8, 4, A + '2409.12186', True)
R('starcoder2-3b', 'starcoder2-3b', 21.6, 4, A + '2409.12186', False)
R('qwen2-5-3b', 'qwen2-5-coder-3b', 75.7, 4, A + '2409.12186', True)
R('starcoder2-7b', 'starcoder2-7b', 32.7, 4, A + '2409.12186', False)
R('deepseek-coder-6-7b', 'deepseek-coder-6-7b', 21.3, 4, A + '2409.12186', False)
R('deepseek-coder-v2-lite', 'deepseek-coder-v2-lite', 67.1, 4, A + '2409.12186', False)
R('codeqwen1-5-7b', 'codeqwen1-5-7b', 37.7, 4, A + '2409.12186', True)
R('qwen2-5-7b', 'qwen2-5-coder-7b', 83.9, 4, A + '2409.12186', True)
R('starcoder2-15b', 'starcoder2-15b', 57.7, 4, A + '2409.12186', False)
R('qwen2-5-14b', 'qwen2-5-coder-14b', 88.7, 4, A + '2409.12186', True)
R('deepseek-coder-33b', 'deepseek-coder-33b', 35.4, 4, A + '2409.12186', False)
R('deepseek-coder-v2', 'deepseek-coder-v2-jun-2024', 85.8, 4, A + '2409.12186', False)
R('qwen2-5-32b', 'qwen2-5-coder-32b', 91.1, 4, A + '2409.12186', True)

# Falcon2-11B Technical Report
R('falcon-7b', 'falcon-7b', 4.62, 5, A + '2407.14885', True)
R('falcon-40b', 'falcon-40b', 21.46, 5, A + '2407.14885', True)
R('falcon2-11b', 'falcon2-11b', 53.83, 5, A + '2407.14885', True)

# GPT-4 Technical Report
R('gpt-4', 'gpt-4-mar-2023', 92.0, 5, A + '2303.08774', True)
R('gpt-3-5', 'text-davinci-003', 57.1, 5, A + '2303.08774', True)
R('palm', 'minerva-540b', 58.8, 8, A + '2303.08774', False)

# Releasing Claude Instant 1.2
R('claude-instant', 'claude-instant-1-1', 80.9, 0, CLI, True)
R('claude-instant', 'claude-instant-1-2', 86.7, 0, CLI, True)

# Yi: Open Foundation Models by 01.AI
R('llama-2-13b', 'llama-2-13b', 36.9, 0, A + '2403.04652', False)
R('llama-2-70b', 'llama-2-70b', 47.1, 0, A + '2403.04652', False)
L('Baichuan 2 Chat 13B', 45.7, A + '2403.04652')
R('qwen-14b', 'qwen-14b', 59.5, 0, A + '2403.04652', False)
L('InternLM-Chat 20B', 15.7, A + '2403.04652')
L('AquilaChat2 34B', 11.5, A + '2403.04652')
L('Yi-6B-Chat', 38.4, A + '2403.04652')
L('Yi-34B-Chat', 71.7, A + '2403.04652')
R('llama-2-13b', 'llama-2-13b', 2.7, 4, A + '2403.04652', False)
R('llama-2-70b', 'llama-2-70b', 58.7, 4, A + '2403.04652', False)
L('Baichuan 2 Chat 13B', 23.3, A + '2403.04652')
R('qwen-14b', 'qwen-14b', 61.2, 4, A + '2403.04652', False)
L('InternLM-Chat 20B', 43.4, A + '2403.04652')
L('AquilaChat2 34B', 48.5, A + '2403.04652')
L('Yi-6B-Chat', 44.9, A + '2403.04652')
L('Yi-34B-Chat', 76.0, A + '2403.04652')

# INTELLECT-1 Technical Report
L('INTELLECT-1-Instruct', 38.58, A + '2412.01152')

# ---------------------------------------------------------------- Creator / paper / model card results

# OpenAI
R('gpt-4-5', 'gpt-4-5', 97.0, 0, 'https://openai.com/index/introducing-gpt-4-5/', True)		# via llm-stats
R('o1', 'o1', 97.1, 0, 'https://openai.com/index/learning-to-reason-with-llms/', True)			# via llm-stats

# Anthropic
CL3 = 'https://www.anthropic.com/news/claude-3-family'
R('claude-3-opus', 'claude-3-opus', 95.0, 0, CL3, True)
R('claude-3-sonnet', 'claude-3-sonnet', 92.3, 0, CL3, True)
R('claude-3-haiku', 'claude-3-haiku', 88.9, 0, CL3, True)
R('claude-3-5-sonnet', 'claude-3-5-sonnet-jun-2024', 96.4, 0, 'https://www.anthropic.com/news/claude-3-5-sonnet', True)
CL2 = 'https://www-cdn.anthropic.com/bd2a28d2535bfb0494cc8e2a3bf135d2e7523226/Model-Card-Claude-2.pdf'
R('claude-2', 'claude-2', 88.0, 0, CL2, True)
R('claude-1', 'claude-1-3', 85.2, 0, CL2, True)

# Google
G15 = A + '2403.05530'
R('gemini-1-5-pro', 'gemini-1-5-pro-may-2024', 90.8, 11, G15, True)
R('gemini-1-5-flash', 'gemini-1-5-flash-may-2024', 86.2, 11, G15, True)
R('gemini-1-0-pro', 'gemini-1-0-pro', 77.9, 11, G15, True)
R('gemini-1-0-ultra', 'gemini-1-0-ultra', 88.9, 11, G15, True)
R('palm-2', 'palm-2-l', 80.7, 8, A + '2305.10403', True)
MIN = A + '2206.14858'
R('palm', 'minerva-8b', 16.2, 8, MIN, True)
R('palm', 'minerva-62b', 52.4, 8, MIN, True)
R('palm', 'minerva-540b', 58.8, 8, MIN, True)
R('palm', 'palm-540b', 56.5, 8, A + '2204.02311', True)
G2 = A + '2408.00118'
R('gemma-2b', 'gemma-2b', 15.1, 5, G2, True)
R('gemma-2-2b', 'gemma-2-2b', 24.3, 5, G2, True)
R('mistral-7b', 'mistral-7b-v0-1', 39.6, 5, G2, False)
R('llama-3-8b', 'llama-3-8b', 45.7, 5, G2, False)
R('gemma-7b', 'gemma-7b', 51.8, 5, G2, True)
R('gemma-2-9b', 'gemma-2-9b', 68.6, 5, G2, True)
R('gemma-2-27b', 'gemma-2-27b', 74.0, 5, G2, True)
G3 = 'https://ai.google.dev/gemma/docs/core/model_card_3'
R('gemma-3-1b', 'gemma-3-1b', 62.8, 0, G3, True)
R('gemma-3-4b', 'gemma-3-4b', 89.2, 0, G3, True)
R('gemma-3-12b', 'gemma-3-12b', 94.4, 0, G3, True)
R('gemma-3-27b', 'gemma-3-27b', 95.9, 0, G3, True)
R('gemma-3-4b', 'gemma-3-4b', 38.4, 8, G3, True)
R('gemma-3-12b', 'gemma-3-12b', 71.0, 8, G3, True)
R('gemma-3-27b', 'gemma-3-27b', 82.6, 8, G3, True)

# Meta
L3 = 'https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md'
R('llama-3-8b', 'llama-3-8b', 79.6, 8, L3, True)
R('llama-3-70b', 'llama-3-70b', 93.0, 8, L3, True)
L31 = A + '2407.21783'
R('llama-3-1-8b', 'llama-3-1-8b', 84.5, 8, L31, True)
R('llama-3-1-70b', 'llama-3-1-70b', 95.1, 8, L31, True)
R('llama-3-1-405b', 'llama-3-1-405b', 96.8, 8, L31, True)
R('llama-3-1-8b', 'llama-3-1-8b', 57.2, 8, L31, True)
R('llama-3-1-70b', 'llama-3-1-70b', 83.7, 8, L31, True)
R('llama-3-1-405b', 'llama-3-1-405b', 89.0, 8, L31, True)
R('gemma-2-9b', 'gemma-2-9b', 76.7, 8, L31, False)
R('mistral-7b', None, 53.2, 8, L31, False)
R('mixtral-8x22b', 'mixtral-8x22b', 88.2, 8, L31, False)
R('gpt-3-5-turbo', None, 81.6, 8, L31, False)
R('nemotron-4-340b', 'nemotron-4-340b', 92.3, 8, L31, False)
R('gpt-4-turbo', 'gpt-4-turbo-jan-2024', 94.2, 8, L31, False)
R('gpt-4o', None, 96.1, 8, L31, False)
R('claude-3-5-sonnet', 'claude-3-5-sonnet-jun-2024', 96.4, 8, L31, False)
R('mistral-7b', None, 52.5, 8, L31, False)
R('gemma-7b', 'gemma-7b', 46.4, 8, L31, False)
R('mixtral-8x22b', 'mixtral-8x22b', 88.4, 8, L31, False)
R('gpt-4', None, 92.0, 8, L31, False)
R('gemini-1-0-ultra', 'gemini-1-0-ultra', 88.9, 11, L31, False)
L32 = HF + 'meta-llama/Llama-3.2-1B-Instruct'
R('llama-3-2-1b', 'llama-3-2-1b', 44.4, 8, L32, True)
R('llama-3-2-3b', 'llama-3-2-3b', 77.7, 8, L32, True)

# xAI
R('grok-0', 'grok-0', 56.8, 8, 'https://x.ai/news/grok', True)
R('grok-1', 'grok-1', 62.9, 8, 'https://x.ai/news/grok', True)
R('grok-1-5', 'grok-1-5', 90.0, 8, 'https://x.ai/news/grok-1.5', True)

# Mistral
R('mistral-small-3', 'mistral-small-3', 80.73, 5, HF + 'mistralai/Mistral-Small-24B-Base-2501', True)
R('mistral-7b', 'mathstral-7b', 77.1, 8, HF + 'mistralai/Mathstral-7B-v0.1', True)
R('mistral-large-2', 'mistral-large-2', 93.0, 0, HF + 'mistralai/Mistral-Large-Instruct-2407', True)	# via llm-stats

# Alibaba / Qwen
R('qwen-72b', 'qwen-72b', 78.9, 8, HF + 'Qwen/Qwen-72B', True)
R('qwen-7b', 'qwen-7b', 50.3, 0, HF + 'Qwen/Qwen-7B-Chat', True)
R('qwen-14b', 'qwen-14b', 60.1, 0, HF + 'Qwen/Qwen-14B-Chat', True)
R('qwen-72b', 'qwen-72b', 76.4, 0, HF + 'Qwen/Qwen-72B-Chat', True)
Q15 = 'https://qwenlm.github.io/blog/qwen1.5/'
for m, s in [('qwen1-5-0-5b', 22.0), ('qwen1-5-1-8b', 38.4), ('qwen1-5-4b', 57.0), ('qwen1-5-7b', 62.5), ('qwen1-5-14b', 70.1), ('qwen1-5-72b', 79.5)]:
	R(m, m, s, 4, Q15, True)
R('qwen1-5-110b', 'qwen1-5-110b', 85.4, 4, 'https://qwenlm.github.io/blog/qwen1.5-110b/', True)
R('qwen1-5-moe-a2-7b', 'qwen1-5-moe-a2-7b', 61.5, 4, 'https://qwenlm.github.io/blog/qwen-moe/', True)
Q2 = A + '2407.10671'
R('qwen1-5-32b', 'qwen1-5-32b', 76.8, 4, Q2, True)
for m, s in [('qwen2-72b', 89.5), ('qwen2-57b-a14b', 80.7), ('qwen2-7b', 79.9), ('qwen2-1-5b', 58.5), ('qwen2-0-5b', 36.5)]:
	R(m, m, s, 4, Q2, True)
for m, s in [('qwen2-72b', 93.2), ('qwen2-57b-a14b', 85.3), ('qwen2-7b', 85.7), ('qwen2-1-5b', 61.6), ('qwen2-0-5b', 40.1),
	('qwen1-5-72b', 82.7), ('qwen1-5-110b', 84.5), ('qwen1-5-32b', 83.6), ('qwen1-5-7b', 60.3), ('qwen1-5-1-8b', 35.3), ('qwen1-5-0-5b', 11.3)]:
	R(m, m, s, 0, Q2, True)
R('llama-3-70b', 'llama-3-70b', 83.0, 4, Q2, False)
R('mixtral-8x22b', 'mixtral-8x22b', 83.7, 4, Q2, False)
R('llama-3-8b', 'llama-3-8b', 56.0, 4, Q2, False)
Q25 = A + '2412.15115'
for m, s in [('qwen2-5-0-5b', 41.6), ('qwen2-5-1-5b', 68.5), ('qwen2-5-3b', 79.1), ('qwen2-5-7b', 85.4), ('qwen2-5-14b', 90.2), ('qwen2-5-32b', 92.9), ('qwen2-5-72b', 91.5)]:
	R(m, m, s, 4, Q25, True)
for m, s in [('qwen2-5-0-5b', 49.6), ('qwen2-5-1-5b', 73.2), ('qwen2-5-3b', 86.7), ('qwen2-5-7b', 91.6), ('qwen2-5-14b', 94.8), ('qwen2-5-32b', 95.9), ('qwen2-5-72b', 95.8)]:
	R(m, m, s, 0, Q25, True)
R('qwen2-5-turbo', 'qwen2-5-turbo', 93.8, 0, Q25, True)
R('qwen-plus', 'qwen-plus-sep-2024', 96.0, 0, Q25, True)
R('llama-3-1-70b', 'llama-3-1-70b', 95.1, 0, Q25, False)
R('gpt-4o-mini', 'gpt-4o-mini', 93.2, 0, Q25, False)
R('gemma-2-27b', 'gemma-2-27b', 90.4, 0, Q25, False)
R('phi-3-5-mini', 'phi-3-5-mini-instruct', 86.2, 0, Q25, False)
R('gemma-2-2b', 'gemma-2-2b', 63.2, 0, Q25, False)
R('qwen2-5-max', 'qwen2-5-max', 94.5, 8, 'https://qwenlm.github.io/blog/qwen2.5-max/', True)
QO = HF + 'Qwen/Qwen2.5-Omni-7B'
R('qwen2-5-7b', 'qwen2-5-omni-7b', 88.7, 0, QO, True)
R('qwen2-5-3b', 'qwen2-5-omni-3b', 82.6, 0, QO, True)
Q3 = A + '2505.09388'
for m, s in [('qwen3-235b-a22b', 94.39), ('qwen3-32b', 93.40), ('qwen3-30b-a3b', 91.81), ('qwen3-14b', 92.49), ('qwen3-8b', 89.84),
	('qwen3-4b', 87.79), ('qwen3-1-7b', 75.44), ('qwen3-0-6b', 59.59)]:
	R(m, m, s, 4, Q3, True)
R('llama-4-maverick', 'llama-4-maverick', 87.72, 4, Q3, False)
R('llama-4-scout', 'llama-4-scout', 85.37, 4, Q3, False)
R('deepseek-v3', 'deepseek-v3-dec-2024', 87.57, 4, Q3, False)
R('gemma-3-27b', 'gemma-3-27b', 81.20, 4, Q3, False)
R('gemma-3-12b', 'gemma-3-12b', 78.01, 4, Q3, False)
SEA = HF + 'SeaLLMs/SeaLLM-7B-v2.5'
R('seallm-7b-v2', 'seallm-7b-v2', 78.2, 0, SEA, True)
R('seallm-7b-v2-5', 'seallm-7b-v2-5', 78.5, 0, SEA, True)

# DeepSeek
DLLM = A + '2401.02954'
R('deepseek-llm-7b', 'deepseek-llm-7b', 17.4, 8, DLLM, True)
R('deepseek-llm-7b', 'deepseek-llm-7b', 63.0, 0, DLLM, True)
R('deepseek-llm-67b', 'deepseek-llm-67b', 63.4, 8, DLLM, True)
R('deepseek-llm-67b', 'deepseek-llm-67b', 84.1, 0, DLLM, True)
DMOE = A + '2401.06066'
R('deepseekmoe-16b', 'deepseekmoe-16b', 18.8, 8, DMOE, True)
DV2 = A + '2405.04434'
R('deepseek-v2', 'deepseek-v2-may-2024', 79.2, 8, DV2, True)
R('deepseek-v2', 'deepseek-v2-may-2024', 92.2, 8, DV2, True)
R('deepseek-v2-lite', 'deepseek-v2-lite', 41.1, 8, DV2, True)
R('deepseek-v2-lite', 'deepseek-v2-lite', 72.0, 8, DV2, True)
R('deepseekmoe-16b', 'deepseekmoe-16b', 62.2, 8, DV2, True)
R('mixtral-8x22b', 'mixtral-8x22b', 80.3, 8, DV2, False)
R('mixtral-8x22b', 'mixtral-8x22b', 87.9, 8, DV2, False)
R('llama-3-70b', 'llama-3-70b', 93.2, 8, DV2, False)
R('qwen1-5-72b', 'qwen1-5-72b', 81.9, 8, DV2, False)
R('deepseek-v2-5', 'deepseek-v2-5-sep-2024', 95.1, 0, 'https://www.deepseek.com/', True)	# via llm-stats
DCV2 = A + '2406.11931'
R('deepseek-coder-v2', 'deepseek-coder-v2-jun-2024', 94.9, 0, DCV2, True)
R('deepseek-coder-v2-lite', 'deepseek-coder-v2-lite', 86.4, 0, DCV2, True)
R('gpt-4o', 'gpt-4o-may-2024', 95.8, 0, DCV2, False)
DM = A + '2402.03300'
R('deepseek-llm-7b', 'deepseekmath-7b', 64.2, 8, DM, True)
R('deepseek-llm-7b', 'deepseekmath-7b', 82.9, 0, DM, True)
R('deepseek-llm-7b', 'deepseekmath-7b', 88.2, 0, DM, True)
DV3 = A + '2412.19437'
R('deepseek-v3', 'deepseek-v3-dec-2024', 89.3, 8, DV3, True)
R('deepseek-v2', 'deepseek-v2-may-2024', 81.6, 8, DV3, True)
R('qwen2-5-72b', 'qwen2-5-72b', 88.3, 8, DV3, False)
R('llama-3-1-405b', 'llama-3-1-405b', 83.5, 8, DV3, False)
DV4 = HF + 'deepseek-ai/DeepSeek-V4-Pro'
R('deepseek-v3-2', None, 91.1, 8, DV4, True)
R('deepseek-v4-flash', 'deepseek-v4-flash-apr-2026', 90.8, 8, DV4, True)
R('deepseek-v4-pro', 'deepseek-v4-pro-apr-2026', 92.6, 8, DV4, True)
R('deepseek-v4-1-flash', 'deepseek-v4-1-flash-sep-2026', 93.0, 8, HF + 'deepseek-ai/DeepSeek-V4.1-Flash', True)

# Microsoft
R('phi-1-5', 'phi-1-5', 40.2, 0, A + '2309.05463', True)
R('phi-1-5', 'phi-1-5-web', 44.6, 0, A + '2309.05463', True)
R('phi-2', 'phi-2', 61.1, 0, 'https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/', True)
P3 = A + '2404.14219'
R('phi-3-mini', 'phi-3-mini-4k-instruct', 82.5, 8, P3, True)
R('phi-3-small', 'phi-3-small-8k-instruct', 89.6, 8, P3, True)
R('phi-3-medium', 'phi-3-medium-4k-instruct', 91.0, 8, P3, True)
R('phi-3-mini', 'phi-3-mini-4k-instruct', 85.7, 8, HF + 'microsoft/Phi-3-mini-4k-instruct', True)
R('phi-3-mini', 'phi-3-mini-128k-instruct', 85.3, 8, HF + 'microsoft/Phi-3-mini-128k-instruct', True)
R('phi-3-small', 'phi-3-small-128k-instruct', 87.3, 8, HF + 'microsoft/Phi-3-small-128k-instruct', True)
R('phi-3-medium', 'phi-3-medium-128k-instruct', 87.5, 8, HF + 'microsoft/Phi-3-medium-128k-instruct', True)
P3M = HF + 'microsoft/Phi-3-medium-4k-instruct'
R('gpt-3-5-turbo', 'gpt-3-5-turbo-nov-2023', 78.1, 8, P3M, False)
R('gpt-4-turbo', 'gpt-4-turbo-nov-2023', 94.2, 8, P3M, False)
R('gemini-1-0-pro', 'gemini-1-0-pro', 80.4, 8, P3M, False)
R('llama-3-70b', 'llama-3-70b', 93.5, 8, P3M, False)
R('mixtral-8x22b', 'mixtral-8x22b', 83.8, 8, P3M, False)
P3S = HF + 'microsoft/Phi-3-mini-4k-instruct'
R('gemma-7b', 'gemma-7b', 59.8, 8, P3S, False)
R('mistral-7b', None, 46.4, 8, P3S, False)
R('mixtral-8x7b', 'mixtral-8x7b', 64.7, 8, P3S, False)
R('llama-3-8b', 'llama-3-8b', 77.4, 8, P3S, False)
R('phi-4-mini', 'phi-4-mini', 88.6, 8, HF + 'microsoft/Phi-4-mini-instruct', True)
R('grin-moe', 'grin-moe', 90.4, 8, HF + 'microsoft/GRIN-MoE', True)
R('gpt-4o', None, 93.8, 8, HF + 'microsoft/GRIN-MoE', False)
SLIM = HF + 'microsoft/Phi-mini-MoE-instruct'
R('phi-mini-moe', 'phi-mini-moe', 84.89, 0, SLIM, True)
R('phi-tiny-moe', 'phi-tiny-moe', 78.47, 0, SLIM, True)
R('bitnet-b1-58-2b4t', 'bitnet-b1-58-2b4t', 58.38, 0, HF + 'microsoft/bitnet-b1.58-2B-4T', True)
R('orca-math', 'orca-math', 86.81, 0, A + '2402.14830', True)
WM = A + '2308.09583'
R('wizardmath-7b', 'wizardmath-7b-v1-0', 54.9, 0, WM, True)
R('wizardmath-13b', 'wizardmath-13b', 63.9, 0, WM, True)
R('wizardmath-70b', 'wizardmath-70b', 81.6, 0, WM, True)
R('wizardmath-7b', 'wizardmath-7b-v1-1', 83.2, 0, HF + 'WizardLMTeam/WizardMath-7B-V1.1', True)
WL = HF + 'WizardLMTeam/WizardLM-70B-V1.0'
R('wizardlm-70b', 'wizardlm-70b', 77.6, 0, WL, True)
R('wizardlm-13b', 'wizardlm-13b-v1-2', 55.3, 0, WL, True)

# NVIDIA
R('nemotron-4-340b', 'nemotron-4-340b', 92.3, 0, A + '2406.11704', True)
R('llama-3-1-nemotron-51b-instruct', 'llama-3-1-nemotron-51b-instruct', 91.43, 5, HF + 'nvidia/Llama-3_1-Nemotron-51B-Instruct', True)
for m, v, s, sh, url in [
	('nemotron-h-8b', 'nemotron-h-8b', 87.11, 8, 'nvidia/Nemotron-H-8B-Base-8K'),
	('nemotron-h-47b', 'nemotron-h-47b', 93.3, 8, 'nvidia/Nemotron-H-47B-Base-8K'),
	('nemotron-h-56b', 'nemotron-h-56b', 93.71, 8, 'nvidia/Nemotron-H-56B-Base-8K'),
	('nemotron-h-4b', 'nemotron-h-4b', 69.6, 8, 'nvidia/Nemotron-H-4B-Base-8K'),
	('nemotron-h-4b', 'nemotron-h-4b', 88.93, 0, 'nvidia/Nemotron-H-4B-Instruct-128K'),
	('nvidia-nemotron-nano-12b-v2', 'nvidia-nemotron-nano-12b-v2', 91.66, 8, 'nvidia/NVIDIA-Nemotron-Nano-12B-v2-Base'),
	('nvidia-nemotron-nano-9b-v2', 'nvidia-nemotron-nano-9b-v2', 91.36, 8, 'nvidia/NVIDIA-Nemotron-Nano-12B-v2-Base'),
	('nemotron-3-nano', 'nemotron-3-nano', 92.34, 8, 'nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16'),
	('nemotron-3-super', 'nemotron-3-super', 90.67, 8, 'nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-Base-BF16'),
	('nemotron-3-ultra', 'nemotron-3-ultra', 88.10, 8, 'nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-Base-BF16'),
	('nemotron-3-5-lightning', 'nemotron-3-5-lightning', 91.28, 8, 'nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Base-BF16'),
	('nemotron-3-nano', 'nemotron-3-nano', 91.43, 8, 'nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Base-BF16'),
	('nemotron-3-5-super', 'nemotron-3-5-super-vl', 91.74, 8, 'nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Base-BF16'),
	('minitron-4b', 'minitron-4b', 24.1, 5, 'nvidia/Minitron-4B-Base'),
	('minitron-8b', 'minitron-8b', 54.2, 5, 'nvidia/Minitron-8B-Base'),
	('mistral-nemo-minitron-8b', 'mistral-nemo-minitron-8b', 58.5, 5, 'nvidia/Mistral-NeMo-Minitron-8B-Base'),
	('llama-3-1-minitron-4b', 'llama-3-1-minitron-4b-width', 41.2, 5, 'nvidia/Llama-3.1-Minitron-4B-Width-Base'),
	('llama-3-1-minitron-4b', 'llama-3-1-minitron-4b-depth', 16.8, 5, 'nvidia/Llama-3.1-Minitron-4B-Depth-Base'),
	('aceinstruct-1-5b', 'aceinstruct-1-5b', 80.44, 0, 'nvidia/AceInstruct-1.5B'),
	('aceinstruct-7b', 'aceinstruct-7b', 93.10, 0, 'nvidia/AceInstruct-7B'),
	('aceinstruct-72b', 'aceinstruct-72b', 96.36, 0, 'nvidia/AceInstruct-72B'),
	('acemath-rl-nemotron-7b', 'acemath-rl-nemotron-7b', 93.3, 0, 'nvidia/AceMath-RL-Nemotron-7B'),
	('openmath2-llama3-1-8b', 'openmath2-llama3-1-8b', 91.7, 0, 'nvidia/OpenMath2-Llama3.1-8B'),
	('openmath2-llama3-1-70b', 'openmath2-llama3-1-70b', 94.9, 0, 'nvidia/OpenMath2-Llama3.1-70B'),
	('nvlm-1-0', 'nvlm-d-72b', 93.2, 0, 'nvidia/NVLM-D-72B'),
]:
	R(m, v, s, sh, HF + url, True)
NB = HF + 'nvidia/'
R('qwen3-30b-a3b', 'qwen3-30b-a3b', 89.01, 8, NB + 'NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16', False)
R('glm-4-5-air', 'glm-4-5-air', 82.60, 8, NB + 'NVIDIA-Nemotron-3-Super-120B-A12B-Base-BF16', False)
NU = NB + 'NVIDIA-Nemotron-3-Ultra-550B-A55B-Base-BF16'
R('deepseek-v3-2', 'deepseek-v3-2-exp', 84.38, 8, NU, False)
R('mistral-large-3', 'mistral-large-3', 91.21, 8, NU, False)
R('kimi-k2', 'kimi-k2-jul-2025', 91.05, 8, NU, False)
R('glm-4-5', 'glm-4-5', 85.37, 8, NU, False)
NL = NB + 'NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Base-BF16'
R('qwen3-5-35b-a3b', 'qwen3-5-35b-a3b', 90.07, 8, NL, False)
R('gemma-4-26b-a4b', 'gemma-4-26b-a4b', 77.03, 8, NL, False)
R('qwen3-8b', 'qwen3-8b', 84.00, 8, NB + 'NVIDIA-Nemotron-Nano-12B-v2-Base', False)
R('llama-3-1-8b', 'llama-3-1-8b', 84.5, 0, NB + 'OpenMath2-Llama3.1-8B', False)
R('llama-3-1-70b', 'llama-3-1-70b', 95.8, 0, NB + 'OpenMath2-Llama3.1-70B', False)
R('deepseek-r1-distill-qwen-7b', 'deepseek-r1-distill-qwen-7b', 92.7, 0, NB + 'AceMath-RL-Nemotron-7B', False)
R('nemotron-4-15b', 'nemotron-4-15b', 46.0, 8, A + '2402.16819', True)

# Moonshot
R('kimi-k2', 'kimi-k2-jul-2025', 92.1, 8, HF + 'moonshotai/Kimi-K2-Base', True)
R('kimi-k2', 'kimi-k2-jul-2025', 97.3, 0, 'https://moonshotai.github.io/Kimi-K2/', True)	# via llm-stats
MLT = HF + 'moonshotai/Moonlight-16B-A3B'
R('moonlight-16b-a3b', 'moonlight-16b-a3b', 77.4, 4, MLT, True)
R('llama-3-2-3b', 'llama-3-2-3b', 34.0, 4, MLT, False)
R('qwen2-5-3b', 'qwen2-5-3b', 79.1, 4, MLT, False)
R('llama-3-1-8b', 'llama-3-1-8b', 57.2, 4, A + '2502.16982', False)
R('gemma-2-9b', 'gemma-2-9b', 70.7, 4, A + '2502.16982', False)

# Xiaomi
MV2 = HF + 'XiaomiMiMo/MiMo-V2-Flash-Base'
R('mimo-v2-flash', 'mimo-v2-flash-dec-2025', 92.3, 8, MV2, True)
R('deepseek-v3-1', 'deepseek-v3-1', 91.4, 8, MV2, False)
MV25 = HF + 'XiaomiMiMo/MiMo-V2.5-Pro-Base'
R('mimo-v2-5-pro', 'mimo-v2-5-pro', 99.6, 8, MV25, True)
R('mimo-v2-5', 'mimo-v2-5', 83.3, 8, MV25, True)

# Zhipu
GLM = A + '2406.12793'
R('chatglm-6b', 'chatglm-6b', 1.5, 0, GLM, True)
R('chatglm2-6b', 'chatglm2-6b', 25.9, 0, GLM, True)
R('chatglm3-6b', 'chatglm3-6b', 72.3, 0, GLM, True)
R('glm-4-9b', 'glm-4-9b-chat', 84.0, 0, GLM, True)
R('glm-4-9b', 'glm-4-9b-chat', 79.6, 0, GLM, True)
R('glm-4-air', 'glm-4-air-jun-2024', 90.9, 0, GLM, True)
R('glm-4', 'glm-4-jan-2024', 87.6, 0, GLM, True)
R('glm-4', 'glm-4-may-2024', 93.3, 0, GLM, True)
R('gpt-4-turbo', 'gpt-4-turbo-nov-2023', 95.7, 0, GLM, False)
R('gpt-4-turbo', 'gpt-4-turbo-apr-2024', 95.6, 0, GLM, False)

# MiniMax
MM = A + '2501.08313'
R('minimax-text-01', 'minimax-text-01', 94.8, 0, MM, True)
R('gpt-4o', 'gpt-4o-nov-2024', 95.6, 0, MM, False)
R('claude-3-5-sonnet', 'claude-3-5-sonnet-oct-2024', 96.9, 0, MM, False)
R('gemini-1-5-pro', 'gemini-1-5-pro-sep-2024', 95.2, 0, MM, False)
R('gemini-2-0-flash', 'gemini-2-0-flash-experimental', 95.4, 0, MM, False)
R('qwen2-5-72b', 'qwen2-5-72b', 95.8, 0, MM, False)
R('deepseek-v3', 'deepseek-v3-dec-2024', 96.7, 0, MM, False)
R('llama-3-1-405b', 'llama-3-1-405b', 96.7, 0, MM, False)

# EleutherAI
LLE = A + '2310.10631'
R('llemma-7b', 'llemma-7b', 36.4, 0, LLE, True)
R('llemma-34b', 'llemma-34b', 51.5, 0, LLE, True)
R('llama-2-7b', 'code-llama-7b', 10.5, 0, LLE, False)
R('llama-2-34b', 'code-llama-34b', 29.6, 0, LLE, False)
R('llama-2-7b', 'llama-2-7b', 11.8, 0, LLE, False)

# Hugging Face
SM2 = HF + 'HuggingFaceTB/SmolLM2-1.7B-Instruct'
for m, v, s, url in [
	('smollm2-1-7b', 'smollm2-1-7b', 31.0, SM2), ('smollm2-1-7b', 'smollm2-1-7b-instruct', 48.2, SM2),
	('smollm-1-7b', 'smollm-1-7b', 5.5, SM2), ('smollm-1-7b', 'smollm-1-7b-instruct', 4.62, SM2),
	('llama-3-2-1b', 'llama-3-2-1b', 7.2, SM2), ('llama-3-2-1b', 'llama-3-2-1b', 26.8, SM2),
	('qwen2-5-1-5b', 'qwen2-5-1-5b', 61.3, SM2), ('qwen2-5-1-5b', 'qwen2-5-1-5b', 42.8, SM2),
	('smollm2-1-7b', 'smollm2-1-7b-instruct-16k', 44.27, HF + 'HuggingFaceTB/SmolLM2-1.7B-Instruct-16k'),
	('smollm2-360m', 'smollm2-360m', 3.2, HF + 'HuggingFaceTB/SmolLM2-360M'), ('smollm2-360m', 'smollm2-360m-instruct', 7.43, HF + 'HuggingFaceTB/SmolLM2-360M'),
	('smollm-360m', 'smollm-360m', 1.6, HF + 'HuggingFaceTB/SmolLM2-360M'), ('smollm-360m', 'smollm-360m-instruct', 1.36, HF + 'HuggingFaceTB/SmolLM2-360M'),
	('qwen2-5-0-5b', 'qwen2-5-0-5b', 33.4, HF + 'HuggingFaceTB/SmolLM2-360M'), ('qwen2-5-0-5b', 'qwen2-5-0-5b', 26.8, HF + 'HuggingFaceTB/SmolLM2-360M'),
	('smollm2-135m', 'smollm2-135m', 1.4, HF + 'HuggingFaceTB/SmolLM2-135M'), ('smollm2-135m', 'smollm2-135m-instruct', 1.4, HF + 'HuggingFaceTB/SmolLM2-135M'),
	('smollm-135m', 'smollm-135m', 1.0, HF + 'HuggingFaceTB/SmolLM2-135M'), ('smollm-135m', 'smollm-135m-instruct', 1.4, HF + 'HuggingFaceTB/SmolLM2-135M'),
	('smollm3-3b', 'smollm3-3b-base', 67.63, HF + 'HuggingFaceTB/SmolLM3-3B-Base'),
	('qwen2-5-3b', 'qwen2-5-3b', 70.13, HF + 'HuggingFaceTB/SmolLM3-3B-Base'), ('llama-3-2-3b', 'llama-3-2-3b', 25.92, HF + 'HuggingFaceTB/SmolLM3-3B-Base'),
	('qwen3-1-7b', 'qwen3-1-7b', 65.88, HF + 'HuggingFaceTB/SmolLM3-3B-Base'), ('qwen3-4b', 'qwen3-4b', 74.14, HF + 'HuggingFaceTB/SmolLM3-3B-Base'),
]:
	R(m, v, s, 5, url, m.startswith('smollm'))

# Cerebras
CDC = HF + 'cerebras/Llama3-DocChat-1.0-8B'
R('llama3-docchat-1-0-8b', 'llama3-docchat-1-0-8b', 55.57, 0, CDC, True)
R('llama-3-8b', 'llama-3-8b', 76.12, 0, CDC, False)
R('llama3-chatqa-1-5-8b', 'llama3-chatqa-1-5-8b', 13.72, 0, CDC, False)
KLR = HF + 'cerebras/Kimi-Linear-REAP-35B-A3B-Instruct'
R('kimi-linear-reap-35b-a3b-instruct', 'kimi-linear-reap-35b-a3b-instruct', 85.8, 0, KLR, True)
R('kimi-linear-48b-a3b', 'kimi-linear-48b-a3b', 87.3, 0, KLR, False)

# TII
FH = HF + 'tiiuae/'
for m, v, s, sh, url in [
	('falcon-h1-0-5b', 'falcon-h1-0-5b-base', 60.2, 0, 'Falcon-H1-0.5B-Base'), ('falcon-h1-0-5b', 'falcon-h1-0-5b-instruct', 68.39, 0, 'Falcon-H1-0.5B-Instruct'),
	('falcon-h1-1-5b', 'falcon-h1-1-5b-base', 52.01, 0, 'Falcon-H1-1.5B-Base'), ('falcon-h1-1-5b', 'falcon-h1-1-5b-instruct', 74.98, 0, 'Falcon-H1-1.5B-Instruct'),
	('falcon-h1-1-5b-deep', 'falcon-h1-1-5b-deep-base', 68.69, 0, 'Falcon-H1-1.5B-Deep-Base'), ('falcon-h1-1-5b-deep', 'falcon-h1-1-5b-deep-instruct', 82.34, 0, 'Falcon-H1-1.5B-Deep-Instruct'),
	('falcon-h1-3b', 'falcon-h1-3b-base', 68.31, 0, 'Falcon-H1-3B-Base'), ('falcon-h1-3b', 'falcon-h1-3b-instruct', 84.76, 0, 'Falcon-H1-3B-Instruct'),
	('falcon-h1-7b', 'falcon-h1-7b-base', 73.46, 0, 'Falcon-H1-7B-Base'), ('falcon-h1-7b', 'falcon-h1-7b-instruct', 81.65, 0, 'Falcon-H1-7B-Instruct'),
	('falcon-h1-34b', 'falcon-h1-34b-base', 76.5, 0, 'Falcon-H1-34B-Base'), ('falcon-h1-34b', 'falcon-h1-34b-instruct', 83.62, 0, 'Falcon-H1-34B-Instruct'),
	('falcon3-1b', 'falcon3-1b-base', 34.3, 5, 'Falcon3-1B-Base'), ('falcon3-1b', 'falcon3-1b-instruct', 43.9, 5, 'Falcon3-1B-Instruct'),
	('falcon3-1b', 'falcon3-1b-instruct', 45.8, 8, 'Falcon3-1B-Instruct'),
	('falcon3-3b', 'falcon3-3b-base', 63.9, 5, 'Falcon3-3B-Base'), ('falcon3-3b', 'falcon3-3b-instruct', 74.8, 5, 'Falcon3-3B-Instruct'),
	('falcon3-3b', 'falcon3-3b-instruct', 78.0, 8, 'Falcon3-3B-Instruct'),
	('falcon3-7b', 'falcon3-7b-base', 76.2, 5, 'Falcon3-7B-Base'), ('falcon3-7b', 'falcon3-7b-instruct', 81.4, 5, 'Falcon3-7B-Instruct'),
	('falcon3-7b', 'falcon3-7b-instruct', 79.7, 8, 'Falcon3-7B-Instruct'),
	('falcon3-10b', 'falcon3-10b-base', 81.4, 5, 'Falcon3-10B-Base'), ('falcon3-10b', 'falcon3-10b-instruct', 84.9, 5, 'Falcon3-10B-Instruct'),
	('falcon3-10b', 'falcon3-10b-instruct', 84.6, 8, 'Falcon3-10B-Instruct'),
	('falcon3-mamba-7b', 'falcon3-mamba-7b-base', 65.9, 5, 'Falcon3-Mamba-7B-Base'), ('falcon3-mamba-7b', 'falcon3-mamba-7b-instruct', 65.2, 5, 'Falcon3-Mamba-7B-Instruct'),
	('falcon-mamba-7b', 'falcon-mamba-7b', 52.54, 5, 'falcon-mamba-7b'),
]:
	R(m, v, s, sh, FH + url, True)

# ---------------------------------------------------------------- Majority voting (maj@k) results
GEM1 = A + '2312.11805'
R('gemini-1-0-ultra', 'gemini-1-0-ultra', 94.4, 0, GEM1, True, maj_k=32)
R('gemini-1-0-pro', 'gemini-1-0-pro', 86.5, 0, GEM1, True, maj_k=32)
LLA = A + '2302.13971'	# k = 100
R('llama-7b', 'llama-7b', 18.1, 8, LLA, True, maj_k=100)
R('llama-13b', 'llama-13b', 29.3, 8, LLA, True, maj_k=100)
R('llama-33b', 'llama-33b', 53.1, 8, LLA, True, maj_k=100)
R('llama-65b', 'llama-65b', 69.7, 8, LLA, True, maj_k=100)
MIN = A + '2206.14858'	# k = 100 (40 for 540B)
R('palm', 'minerva-8b', 28.4, 8, MIN, True, maj_k=100)
R('palm', 'minerva-62b', 68.5, 8, MIN, True, maj_k=100)
R('palm', 'minerva-540b', 78.5, 8, MIN, True, maj_k=40)
PALM2 = A + '2305.10403'	# self-consistency with 40 sample paths
R('palm-2', 'palm-2-l', 91.0, 8, PALM2, True, maj_k=40)
R('palm', 'palm-540b', 74.4, 8, PALM2, True, maj_k=40)
L('Flan-PaLM 2', 84.7, PALM2)
MIS = A + '2310.06825'	# 8-shot maj@8
R('mistral-7b', 'mistral-7b-v0-1', 52.2, 8, MIS, True, maj_k=8)
R('llama-2-7b', 'llama-2-7b', 16.0, 8, MIS, False, maj_k=8)
R('llama-2-13b', 'llama-2-13b', 34.3, 8, MIS, False, maj_k=8)
R('llama-2-7b', 'code-llama-7b', 20.8, 8, MIS, False, maj_k=8)
MIX = A + '2401.04088'	# 8-shot maj@8
R('mixtral-8x7b', 'mixtral-8x7b', 74.4, 8, MIX, True, maj_k=8)
R('mistral-7b', 'mistral-7b-v0-1', 50.0, 8, MIX, True, maj_k=8)
R('llama-2-70b', 'llama-2-70b', 69.6, 8, MIX, False, maj_k=8)
R('llama-33b', 'llama-33b', 44.1, 8, MIX, False, maj_k=8)
R('mixtral-8x22b', 'mixtral-8x22b', 90.8, 8, 'https://mistral.ai/news/mixtral-8x22b/', True, maj_k=8)	# instruct
MIN8 = HF + 'mistralai/Ministral-8B-Instruct-2410'	# base models, maj@8
R('ministral-8b', 'ministral-8b', 64.5, 8, MIN8, True, maj_k=8)
R('ministral-3b', 'ministral-3b', 50.9, 8, MIN8, True, maj_k=8)
R('mistral-7b', None, 32.0, 8, MIN8, True, maj_k=8)
R('llama-3-1-8b', 'llama-3-1-8b', 42.2, 8, MIN8, False, maj_k=8)
R('gemma-2-2b', 'gemma-2-2b', 35.5, 8, MIN8, False, maj_k=8)
R('llama-3-2-3b', 'llama-3-2-3b', 33.5, 8, MIN8, False, maj_k=8)
LLE_MAJ = A + '2310.10631'
R('llemma-7b', 'llemma-7b', 54.0, 0, LLE_MAJ, True, maj_k=100)
R('llemma-34b', 'llemma-34b', 69.3, 0, LLE_MAJ, True, maj_k=100)
R('openmath2-llama3-1-8b', 'openmath2-llama3-1-8b', 94.1, 0, HF + 'nvidia/OpenMath2-Llama3.1-8B', True, maj_k=256)
R('openmath2-llama3-1-70b', 'openmath2-llama3-1-70b', 96.0, 0, HF + 'nvidia/OpenMath2-Llama3.1-70B', True, maj_k=256)
OM = HF + 'nvidia/OpenMath-Mistral-7B-v0.1-hf'	# greedy / majority@50
for m, greedy, maj in [('openmath-codellama-7b', 75.9, 84.8), ('openmath-mistral-7b', 80.2, 86.9), ('openmath-codellama-13b', 78.8, 86.8),
	('openmath-codellama-34b', 80.7, 88.0), ('openmath-llama-2-70b', 84.7, 90.1), ('openmath-codellama-70b', 84.6, 90.8)]:
	R(m, m, greedy, 0, OM, True)
	R(m, m, maj, 0, OM, True, maj_k=50)

# ---------------------------------------------------------------- Open LLM Leaderboard v1 (5-shot, best of bf16 / fp16)
# Models of the list (found through the Hugging Face repo ids in their names), plus the "Maintainer's choice" models as leftovers
L('Yi-1.5-34B', 73.24, OLB)
L('Yi-1.5-34B-32K', 0.0, OLB)
L('Yi-1.5-34B-Chat', 71.65, OLB)
L('Yi-1.5-34B-Chat-16K', 67.93, OLB)
L('Yi-1.5-6B', 50.34, OLB)
L('Yi-1.5-6B-Chat', 67.1, OLB)
L('Yi-1.5-9B', 62.7, OLB)
L('Yi-1.5-9B-32K', 0.0, OLB)
L('Yi-1.5-9B-Chat', 71.87, OLB)
L('Yi-1.5-9B-Chat-16K', 59.67, OLB)
L('Yi-34B', 50.64, OLB)
L('Yi-34B-200K', 34.87, OLB)
L('Yi-34B-Chat', 31.92, OLB)
L('Yi-6B', 12.66, OLB)
L('Yi-6B-200K', 31.39, OLB)
L('Yi-9B', 48.98, OLB)
L('Yi-9B-200K', 47.61, OLB)
R('santacoder', 'santacoder', 0.53, 5, OLB, True)
R('santacoder', 'santacoder', 0.0, 5, OLB, True)
R('starcoder', 'starcoder', 9.17, 5, OLB, True)
R('starcoder2-15b', 'starcoder2-15b', 52.24, 5, OLB, True)
R('starcoder2-3b', 'starcoder2-3b', 19.64, 5, OLB, True)
R('starcoder2-7b', 'starcoder2-7b', 25.09, 5, OLB, True)
R('starcoder', 'starcoderbase', 7.88, 5, OLB, True)
R('starcoderbase-1b', 'starcoderbase-1b', 0.91, 5, OLB, True)
R('starcoderbase-3b', 'starcoderbase-3b', 1.74, 5, OLB, True)
R('starcoderbase-7b', 'starcoderbase-7b', 5.46, 5, OLB, True)
R('starcoder', 'starcoderplus', 8.04, 5, OLB, True)
L('tiny_starcoder_py', 0.99, OLB)
R('bloom', 'bloom', 6.9, 5, OLB, True)
R('bloom-1b1', 'bloom-1b1', 0.23, 5, OLB, True)
R('bloom-1b7', 'bloom-1b7', 0.83, 5, OLB, True)
R('bloom-3b', 'bloom-3b', 1.52, 5, OLB, True)
R('bloom-560m', 'bloom-560m', 0.3, 5, OLB, True)
R('bloom-7b1', 'bloom-7b1', 1.36, 5, OLB, True)
R('bloom-3b', 'bloomz-3b', 0.0, 5, OLB, True)
R('bloom-560m', 'bloomz-560m', 0.0, 5, OLB, True)
R('bloom-7b1', 'bloomz-7b1', 0.08, 5, OLB, True)
R('bloom-7b1', 'bloomz-7b1-mt', 0.0, 5, OLB, True)
R('cerebras-gpt-1-3b', 'cerebras-gpt-1-3b', 0.23, 5, OLB, False)
R('cerebras-gpt-111m', 'cerebras-gpt-111m', 0.0, 5, OLB, False)
R('cerebras-gpt-13b', 'cerebras-gpt-13b', 1.29, 5, OLB, False)
R('cerebras-gpt-2-7b', 'cerebras-gpt-2-7b', 0.45, 5, OLB, False)
R('cerebras-gpt-256m', 'cerebras-gpt-256m', 0.0, 5, OLB, False)
R('cerebras-gpt-590m', 'cerebras-gpt-590m', 0.45, 5, OLB, False)
R('cerebras-gpt-6-7b', 'cerebras-gpt-6-7b', 0.53, 5, OLB, False)
R('llama-2-13b', 'code-llama-13b', 12.13, 5, OLB, False)
R('llama-2-13b', 'code-llama-13b', 12.66, 5, OLB, False)
R('llama-2-34b', 'code-llama-34b', 34.34, 5, OLB, False)
R('llama-2-34b', 'code-llama-34b', 37.98, 5, OLB, False)
R('llama-2-34b', 'code-llama-python-34b', 32.3, 5, OLB, False)
R('llama-2-70b', 'code-llama-70b', 43.97, 5, OLB, False)
R('llama-2-70b', 'code-llama-70b', 46.25, 5, OLB, False)
R('llama-2-70b', 'code-llama-python-70b', 43.44, 5, OLB, False)
R('llama-2-7b', 'code-llama-7b', 7.96, 5, OLB, False)
R('llama-2-7b', 'code-llama-python-7b', 5.16, 5, OLB, False)
L('aya-23-8B', 43.52, OLB)
L('c4ai-command-r-plus', 70.74, OLB)
L('c4ai-command-r-v01', 56.63, OLB)
R('dbrx', 'dbrx-base', 68.54, 5, OLB, False)
R('dbrx', 'dbrx-instruct', 67.32, 5, OLB, False)
R('dolly-v2-12b', 'dolly-v2-12b', 1.21, 5, OLB, False)
R('dolly-v2-3b', 'dolly-v2-3b', 1.06, 5, OLB, False)
R('dolly-v2-7b', 'dolly-v2-7b', 1.14, 5, OLB, False)
L('DeciCoder-1b', 1.74, OLB)
L('DeciLM-7B', 47.38, OLB)
L('DeciLM-7B-instruct', 46.02, OLB)
R('deepseek-coder-1-3b', 'deepseek-coder-1-3b', 1.06, 5, OLB, False)
R('deepseek-coder-6-7b', 'deepseek-coder-6-7b', 17.97, 5, OLB, False)
R('deepseek-coder-6-7b', 'deepseek-coder-6-7b', 26.76, 5, OLB, False)
R('deepseek-llm-7b', 'deepseek-coder-7b-v1-5', 20.39, 5, OLB, False)
R('deepseek-llm-67b', 'deepseek-llm-67b', 56.71, 5, OLB, False)
R('deepseek-llm-67b', 'deepseek-llm-67b', 62.32, 5, OLB, False)
R('deepseek-llm-7b', 'deepseek-llm-7b', 46.55, 5, OLB, False)
R('deepseek-llm-7b', 'deepseekmath-7b', 59.21, 5, OLB, False)
R('deepseek-llm-7b', 'deepseekmath-7b', 19.41, 5, OLB, False)
R('deepseek-llm-7b', 'deepseekmath-7b', 5.38, 5, OLB, False)
R('deepseekmoe-16b', 'deepseekmoe-16b', 17.29, 5, OLB, False)
R('distilgpt2', 'distilgpt2', 0.0, 5, OLB, True)
R('gpt-j-6b', 'gpt-j-6b', 2.96, 5, OLB, False)
R('gpt-neo-1-3b', 'gpt-neo-1-3b', 0.45, 5, OLB, False)
R('gpt-neo-125m', 'gpt-neo-125m', 0.3, 5, OLB, False)
R('gpt-neo-2-7b', 'gpt-neo-2-7b', 1.29, 5, OLB, False)
R('gpt-neox-20b', 'gpt-neox-20b', 5.46, 5, OLB, False)
R('llemma-34b', 'llemma-34b', 50.87, 5, OLB, False)
R('llemma-7b', 'llemma-7b', 33.28, 5, OLB, False)
R('polyglot-ko-12-8b', 'polyglot-ko-12-8b', 0.15, 5, OLB, False)
R('pythia-1-4b', 'pythia-1-4b', 1.52, 5, OLB, False)
R('pythia-1-4b', 'pythia-1-4b-deduped', 0.83, 5, OLB, False)
R('pythia-12b', 'pythia-12b', 1.74, 5, OLB, False)
R('pythia-12b', 'pythia-12b-deduped', 1.44, 5, OLB, False)
R('pythia-160m', 'pythia-160m', 0.23, 5, OLB, False)
R('pythia-160m', 'pythia-160m-deduped', 0.23, 5, OLB, False)
R('pythia-1b', 'pythia-1b-deduped', 1.14, 5, OLB, False)
R('pythia-2-8b', 'pythia-2-8b', 1.06, 5, OLB, False)
R('pythia-2-8b', 'pythia-2-8b-deduped', 0.83, 5, OLB, False)
R('pythia-410m', 'pythia-410m', 0.68, 5, OLB, False)
R('pythia-410m', 'pythia-410m-deduped', 0.3, 5, OLB, False)
R('pythia-6-9b', 'pythia-6-9b', 1.06, 5, OLB, False)
R('pythia-6-9b', 'pythia-6-9b-deduped', 1.67, 5, OLB, False)
R('pythia-70m', 'pythia-70m', 0.3, 5, OLB, False)
R('pythia-70m', 'pythia-70m-deduped', 0.0, 5, OLB, False)
R('opt', 'opt-1-3b', 0.15, 5, OLB, False)
R('opt', 'opt-125m', 0.23, 5, OLB, False)
R('opt', 'opt-13b', 1.74, 5, OLB, False)
R('opt', 'opt-2-7b', 0.23, 5, OLB, False)
R('opt', 'opt-30b', 2.2, 5, OLB, False)
R('opt', 'opt-350m', 0.3, 5, OLB, False)
R('opt', 'opt-6-7b', 0.99, 5, OLB, False)
R('opt', 'opt-66b', 1.67, 5, OLB, False)
L('opt-iml-max-1.3b', 0.53, OLB)
R('opt', 'opt-iml-30b', 2.5, 5, OLB, False)
L('xglm-1.7B', 0.76, OLB)
L('xglm-4.5B', 0.23, OLB)
L('xglm-564M', 0.23, OLB)
R('xglm-7-5b', 'xglm-7-5b', 0.23, 5, OLB, False)
L('AceGPT-7B', 11.14, OLB)
L('phoenix-inst-chat-7b', 1.29, OLB)
R('gemma-2b', 'codegemma-2b', 4.62, 5, OLB, False)
R('gemma-7b', 'codegemma-7b', 45.49, 5, OLB, False)
R('gemma-7b', 'codegemma-7b', 52.46, 5, OLB, False)
R('gemma-7b', 'gemma-1-1-7b', 42.99, 5, OLB, False)
R('gemma-2b', 'gemma-2b', 17.36, 5, OLB, False)
R('gemma-2b', 'gemma-2b', 5.46, 5, OLB, False)
R('gemma-7b', 'gemma-7b', 50.87, 5, OLB, False)
R('gemma-7b', 'gemma-7b', 29.19, 5, OLB, False)
R('recurrentgemma-2b', 'recurrentgemma-2b', 15.69, 5, OLB, False)
R('recurrentgemma-2b', 'recurrentgemma-2b', 10.08, 5, OLB, False)
R('gpt-2', 'gpt-2-124m', 0.68, 5, OLB, False)
R('gpt-2', 'gpt-2-1-5b', 1.29, 5, OLB, False)
L('mistral-7b-grok', 34.34, OLB)
L('mistral-7b-sft-beta', 36.47, OLB)
R('starcoder', 'starchat-alpha', 2.43, 5, OLB, True)
R('starcoder', 'starchat-beta', 5.16, 5, OLB, True)
R('zephyr-7b', 'zephyr-7b-alpha', 14.03, 5, OLB, True)
R('zephyr-7b', 'zephyr-7b-beta', 29.04, 5, OLB, True)
R('zephyr-7b-gemma', 'zephyr-7b-gemma', 44.05, 5, OLB, True)
R('cosmo-1b', 'cosmo-1b', 5.38, 5, OLB, True)
L('merlinite-7b', 41.09, OLB)
L('neural-chat-7b-v3', 1.21, OLB)
L('neural-chat-7b-v3-1', 19.41, OLB)
L('neural-chat-7b-v3-2', 55.12, OLB)
L('neural-chat-7b-v3-3', 61.11, OLB)
L('neural-chat-7b-v3-3-Slerp', 69.98, OLB)
L('internlm-20b', 23.5, OLB)
L('internlm-20b-chat', 18.73, OLB)
L('internlm2-20b', 67.93, OLB)
L('internlm2-7b', 63.0, OLB)
R('fairseq-dense-13b', 'fairseq-dense-13b', 0.0, 5, OLB, False)
L('Amber', 2.81, OLB)
L('AmberChat', 5.53, OLB)
L('K2', 48.67, OLB)
L('longchat-13b-16k', 4.17, OLB)
L('longchat-7b-v1.5-32k', 4.78, OLB)
L('vicuna-13b-delta-v1.1', 8.64, OLB)
L('vicuna-13b-v1.1', 8.64, OLB)
L('vicuna-13b-v1.3', 10.77, OLB)
L('vicuna-13b-v1.5', 11.3, OLB)
L('vicuna-13b-v1.5-16k', 13.12, OLB)
L('vicuna-33b-v1.3', 13.72, OLB)
L('vicuna-7b-delta-v1.1', 5.53, OLB)
L('vicuna-7b-v1.3', 5.69, OLB)
L('vicuna-7b-v1.5', 8.11, OLB)
L('vicuna-7b-v1.5-16k', 6.37, OLB)
R('llama-2-13b', 'llama-2-13b', 15.24, 5, OLB, False)
R('llama-2-13b', 'llama-2-13b', 22.82, 5, OLB, False)
R('llama-2-70b', 'llama-2-70b', 26.69, 5, OLB, False)
R('llama-2-70b', 'llama-2-70b', 54.06, 5, OLB, False)
R('llama-2-7b', 'llama-2-7b', 7.35, 5, OLB, False)
R('llama-2-7b', 'llama-2-7b', 14.48, 5, OLB, False)
R('llama-3-70b', 'llama-3-70b', 76.88, 5, OLB, False)
R('llama-3-70b', 'llama-3-70b', 85.44, 5, OLB, False)
R('llama-3-8b', 'llama-3-8b', 45.34, 5, OLB, False)
R('llama-3-8b', 'llama-3-8b', 68.69, 5, OLB, False)
L('MetaMath-13B-V1.0', 28.51, OLB)
L('MetaMath-70B-V1.0', 44.66, OLB)
L('MetaMath-Llemma-7B', 60.96, OLB)
L('MetaMath-Mistral-7B', 68.84, OLB)
L('CodeGPT-small-py', 0.0, OLB)
R('dialogpt-762m', 'dialogpt-762m', 0.0, 5, OLB, False)
L('DialoGPT-medium', 0.0, OLB)
L('DialoGPT-small', 0.0, OLB)
R('orca-2-13b', 'orca-2-13b', 37.83, 5, OLB, False)
R('orca-2-7b', 'orca-2-7b', 14.71, 5, OLB, False)
R('phi-1-5', 'phi-1-5', 12.43, 5, OLB, False)
R('phi-2', 'phi-2', 54.97, 5, OLB, False)
R('phi-3-medium', 'phi-3-medium-128k-instruct', 81.05, 5, OLB, False)
R('phi-3-medium', 'phi-3-medium-4k-instruct', 79.38, 5, OLB, False)
R('phi-3-mini', 'phi-3-mini-128k-instruct', 69.52, 5, OLB, False)
R('phi-3-mini', 'phi-3-mini-4k-instruct', 74.53, 5, OLB, False)
L('rho-math-1b-v0.1', 0.0, OLB)
R('wizardlm-2-7b', 'wizardlm-2-7b', 43.75, 5, OLB, False)
R('mistral-7b', 'mistral-7b-v0-2', 34.72, 5, OLB, False)
R('mistral-7b', 'mistral-7b-v0-1', 14.25, 5, OLB, False)
R('mistral-7b', 'mistral-7b-v0-2', 40.03, 5, OLB, False)
R('mistral-7b', 'mistral-7b-v0-1', 37.83, 5, OLB, False)
R('mistral-7b', 'mistral-7b-v0-3', 34.5, 5, OLB, False)
R('mixtral-8x22b', 'mixtral-8x22b', 82.03, 5, OLB, False)
R('mixtral-8x22b', 'mixtral-8x22b', 73.69, 5, OLB, False)
R('mixtral-8x7b', 'mixtral-8x7b', 60.73, 5, OLB, False)
R('mixtral-8x7b', 'mixtral-8x7b', 57.47, 5, OLB, False)
R('mpt-30b', 'mpt-30b', 16.91, 5, OLB, False)
R('mpt-30b', 'mpt-30b-chat', 12.13, 5, OLB, False)
R('mpt-30b', 'mpt-30b-instruct', 15.31, 5, OLB, False)
R('mpt-7b', 'mpt-7b', 4.02, 5, OLB, False)
R('mpt-7b-8k', 'mpt-7b-8k', 8.34, 5, OLB, False)
R('mpt-7b-8k', 'mpt-7b-8k-chat', 4.4, 5, OLB, False)
R('mpt-7b-8k', 'mpt-7b-8k-instruct', 20.55, 5, OLB, False)
R('mpt-7b', 'mpt-7b-chat', 4.09, 5, OLB, False)
R('mpt-7b', 'mpt-7b-instruct', 2.81, 5, OLB, False)
R('mpt-7b', 'mpt-7b-storywriter', 0.0, 5, OLB, False)
L('Capybara-7B', 6.9, OLB)
L('Hermes-2-Pro-Llama-3-8B', 67.93, OLB)
L('Hermes-2-Pro-Mistral-7B', 61.26, OLB)
L('Hermes-2-Theta-Llama-3-8B', 70.74, OLB)
L('Nous-Capybara-7B', 6.97, OLB)
L('Nous-Hermes-13b', 8.34, OLB)
L('Nous-Hermes-2-Mistral-7B-DPO', 60.5, OLB)
L('Nous-Hermes-2-Mixtral-8x7B-DPO', 70.66, OLB)
L('Nous-Hermes-2-Mixtral-8x7B-SFT', 69.6, OLB)
L('Nous-Hermes-2-SOLAR-10.7B', 69.45, OLB)
L('Nous-Hermes-2-Yi-34B', 70.05, OLB)
L('Nous-Hermes-llama-2-7b', 5.76, OLB)
L('Nous-Hermes-Llama2-13b', 10.08, OLB)
L('Nous-Puffin-70B', 34.27, OLB)
L('Redmond-Puffin-13B', 11.22, OLB)
L('Yarn-Mistral-7b-128k', 32.6, OLB)
L('Yarn-Mistral-7b-64k', 33.28, OLB)
R('llama3-chatqa-1-5-70b', 'llama3-chatqa-1-5-70b', 65.5, 5, OLB, False)
R('llama3-chatqa-1-5-8b', 'llama3-chatqa-1-5-8b', 13.27, 5, OLB, False)
R('openmath-mistral-7b', 'openmath-mistral-7b', 0.08, 5, OLB, False)
L('LlongOrca-13B-16k', 12.28, OLB)
L('LlongOrca-7B-16k', 7.51, OLB)
L('Mistral-7B-OpenOrca', 19.94, OLB)
L('Mistral-7B-SlimOrca', 21.38, OLB)
L('Mixtral-SlimOrca-8x7B', 45.56, OLB)
L('OpenOrca-Platypus2-13B', 9.02, OLB)
L('OpenOrca-Preview1-13B', 4.93, OLB)
L('OpenOrcaxOpenChat-Preview2-13B', 13.72, OLB)
R('gpt-2', 'gpt-2-774m', 0.83, 5, OLB, False)
L('openchat-3.5-0106', 68.16, OLB)
L('openchat-3.5-0106-gemma', 72.86, OLB)
L('openchat-3.5-1210', 65.96, OLB)
L('openchat-3.6-8b-20240522', 71.8, OLB)
L('openchat_3.5', 26.84, OLB)
L('openchat_8192', 7.35, OLB)
L('openchat_v2', 9.1, OLB)
L('openchat_v2_w', 8.42, OLB)
L('openchat_v3.1', 13.8, OLB)
L('openchat_v3.2', 13.42, OLB)
L('openchat_v3.2_super', 13.5, OLB)
L('opencoderplus', 4.62, OLB)
L('Phind-CodeLlama-34B-Python-v1', 21.53, OLB)
L('Phind-CodeLlama-34B-v1', 20.47, OLB)
L('Phind-CodeLlama-34B-v2', 23.2, OLB)
R('codeqwen1-5-7b', 'codeqwen1-5-7b', 27.9, 5, OLB, False)
R('qwen-14b', 'qwen-14b', 58.98, 5, OLB, False)
R('qwen-72b', 'qwen-72b', 70.43, 5, OLB, False)
R('qwen-7b', 'qwen-7b', 44.96, 5, OLB, False)
R('qwen1-5-0-5b', 'qwen1-5-0-5b', 16.3, 5, OLB, False)
R('qwen1-5-0-5b', 'qwen1-5-0-5b', 7.66, 5, OLB, False)
R('qwen1-5-1-8b', 'qwen1-5-1-8b', 33.59, 5, OLB, False)
R('qwen1-5-1-8b', 'qwen1-5-1-8b', 19.03, 5, OLB, False)
R('qwen1-5-110b', 'qwen1-5-110b', 81.05, 5, OLB, False)
R('qwen1-5-110b', 'qwen1-5-110b', 30.1, 5, OLB, False)
R('qwen1-5-14b', 'qwen1-5-14b', 67.63, 5, OLB, False)
R('qwen1-5-14b', 'qwen1-5-14b', 30.63, 5, OLB, False)
R('qwen1-5-32b', 'qwen1-5-32b', 60.88, 5, OLB, False)
R('qwen1-5-32b', 'qwen1-5-32b', 7.05, 5, OLB, False)
R('qwen1-5-4b', 'qwen1-5-4b', 52.24, 5, OLB, False)
R('qwen1-5-4b', 'qwen1-5-4b', 2.43, 5, OLB, False)
R('qwen1-5-72b', 'qwen1-5-72b', 65.73, 5, OLB, False)
R('qwen1-5-72b', 'qwen1-5-72b', 20.92, 5, OLB, False)
R('qwen1-5-7b', 'qwen1-5-7b', 53.53, 5, OLB, False)
R('qwen1-5-7b', 'qwen1-5-7b', 13.19, 5, OLB, False)
R('qwen1-5-moe-a2-7b', 'qwen1-5-moe-a2-7b', 16.98, 5, OLB, False)
R('qwen1-5-moe-a2-7b', 'qwen1-5-moe-a2-7b', 28.2, 5, OLB, False)
R('qwen2-0-5b', 'qwen2-0-5b', 35.71, 5, OLB, False)
R('qwen2-1-5b', 'qwen2-1-5b', 55.8, 5, OLB, False)
R('qwen2-72b', 'qwen2-72b', 85.29, 5, OLB, False)
R('qwen2-7b', 'qwen2-7b', 72.93, 5, OLB, False)
L('codegen-16B-nl', 2.65, OLB)
L('codegen-6B-multi', 0.99, OLB)
L('codegen-6B-nl', 2.2, OLB)
L('xLAM-v0.1-r', 58.38, OLB)
R('seallm-7b-v2', 'seallm-7b-v2', 68.99, 5, OLB, False)
R('seallm-7b-v2-5', 'seallm-7b-v2-5', 74.37, 5, OLB, False)
L('japanese-stablelm-base-gamma-7b', 17.82, OLB)
L('japanese-stablelm-instruct-gamma-7b', 19.26, OLB)
L('stable-code-3b', 19.56, OLB)
L('StableBeluga-13B', 13.8, OLB)
L('StableBeluga-7B', 7.81, OLB)
L('StableBeluga1-Delta', 0.0, OLB)
L('StableBeluga2', 35.86, OLB)
L('stablelm-2-12b', 56.03, OLB)
L('stablelm-2-12b-chat', 57.85, OLB)
L('stablelm-2-1_6b', 17.44, OLB)
L('stablelm-2-1_6b-chat', 38.82, OLB)
L('stablelm-2-zephyr-1_6b', 35.33, OLB)
L('stablelm-3b-4e1t', 3.34, OLB)
L('stablelm-base-alpha-3b', 0.45, OLB)
L('stablelm-base-alpha-7b', 0.61, OLB)
L('stablelm-base-alpha-7b-v2', 2.58, OLB)
L('stablelm-tuned-alpha-3b', 0.53, OLB)
L('stablelm-tuned-alpha-7b', 0.83, OLB)
L('stablelm-zephyr-3b', 42.15, OLB)
L('CollectiveCognition-v1-Mistral-7B', 17.89, OLB)
L('CollectiveCognition-v1.1-Mistral-7B', 35.86, OLB)
L('Mistral-Trismegistus-7B', 9.93, OLB)
L('OpenHermes-13B', 11.6, OLB)
L('OpenHermes-2.5-Mistral-7B', 26.08, OLB)
L('OpenHermes-7B', 5.0, OLB)
L('LLaMA-Pro-8B', 17.82, OLB)
L('LLaMA-Pro-8B-Instruct', 44.2, OLB)
L('Mistral_Pro_8B_v0.1', 34.19, OLB)
R('falcon2-11b', 'falcon2-11b', 53.83, 5, OLB, False)
R('falcon-40b', 'falcon-40b', 21.46, 5, OLB, False)
R('falcon-7b', 'falcon-7b', 4.62, 5, OLB, False)
R('falcon-7b', 'falcon-7b-instruct', 4.62, 5, OLB, False)
R('falcon-rw-1b', 'falcon-rw-1b', 0.53, 5, OLB, False)
L('TinyLlama-1.1B-Chat-v0.6', 2.12, OLB)
L('TinyLlama-1.1B-Chat-v1.0', 2.43, OLB)
L('TinyLlama-1.1B-intermediate-step-1195k-token-2.5T', 1.44, OLB)
L('TinyLlama-1.1B-intermediate-step-1431k-3T', 1.44, OLB)
L('TinyLlama-1.1B-intermediate-step-955k-token-2T', 1.36, OLB)
L('GPT-JT-6B-v0', 1.21, OLB)
L('GPT-JT-6B-v1', 1.21, OLB)
L('GPT-JT-Moderation-6B', 0.99, OLB)
L('GPT-NeoXT-Chat-Base-20B', 6.9, OLB)
L('LLaMA-2-7B-32K', 4.32, OLB)
L('Llama-2-7B-32K-Instruct', 5.69, OLB)
L('Pythia-Chat-Base-7B', 4.09, OLB)
L('RedPajama-INCITE-7B-Base', 3.03, OLB)
L('RedPajama-INCITE-7B-Chat', 0.45, OLB)
L('RedPajama-INCITE-7B-Instruct', 1.59, OLB)
L('RedPajama-INCITE-Base-3B-v1', 1.29, OLB)
L('RedPajama-INCITE-Base-7B-v0.1', 1.59, OLB)
L('RedPajama-INCITE-Chat-3B-v1', 0.53, OLB)
L('RedPajama-INCITE-Chat-7B-v0.1', 0.45, OLB)
L('RedPajama-INCITE-Instruct-3B-v1', 1.36, OLB)
L('RedPajama-INCITE-Instruct-7B-v0.1', 1.59, OLB)
L('Llama-2-70b-instruct', 32.22, OLB)
L('llama-30b-instruct', 12.13, OLB)
L('llama-30b-instruct-2048', 17.82, OLB)
L('llama-65b-instruct', 26.23, OLB)
L('SOLAR-0-70b-16bit', 45.26, OLB)
L('SOLAR-10.7B-Instruct-v1.0', 64.75, OLB)
L('SOLAR-10.7B-v1.0', 55.5, OLB)
R('wizardcoder-15b', 'wizardcoder-15b', 2.12, 5, OLB, False)
R('wizardlm-13b', 'wizardlm-13b-v1-1', 8.11, 5, OLB, False)
R('wizardlm-13b', 'wizardlm-13b-v1-2', 13.5, 5, OLB, False)
R('wizardlm-30b', 'wizardlm-30b', 0.0, 5, OLB, False)
R('wizardmath-13b', 'wizardmath-13b', 12.36, 5, OLB, False)
R('wizardmath-70b', 'wizardmath-70b', 3.94, 5, OLB, False)
R('wizardmath-7b', 'wizardmath-7b-v1-0', 2.73, 5, OLB, False)
R('wizardmath-7b', 'wizardmath-7b-v1-1', 67.4, 5, OLB, False)

# ---------------------------------------------------------------- Other models not in the list
LS = [
	('Nova Pro', 94.8, 'https://www.amazon.science/publications/the-amazon-nova-family-of-models-technical-report-and-model-card'),
	('Nova Lite', 94.5, 'https://www.amazon.science/publications/the-amazon-nova-family-of-models-technical-report-and-model-card'),
	('Nova Micro', 92.3, 'https://www.amazon.science/publications/the-amazon-nova-family-of-models-technical-report-and-model-card'),
	('Jamba 1.5 Large', 87.0, 'https://huggingface.co/ai21labs/AI21-Jamba-1.5-Large'),
	('Jamba 1.5 Mini', 75.8, 'https://huggingface.co/ai21labs/AI21-Jamba-1.5-Mini'),
	('Granite 3.3 8B Instruct', 80.89, 'https://huggingface.co/ibm-granite/granite-3.3-8b-instruct'),
	('Granite 3.3 8B Base', 59.0, 'https://huggingface.co/ibm-granite/granite-3.3-8b-base'),
	('Granite 4.0 Tiny Preview', 70.1, 'https://www.ibm.com/new/announcements/ibm-granite-4-0-tiny-preview-sneak-peek'),
	('Command R+', 70.7, 'https://huggingface.co/CohereForAI/c4ai-command-r-plus'),
	('ERNIE 4.5', 25.2, 'https://cloud.baidu.com/doc/WENXINWORKSHOP/s/clntwmv7t'),
	('Inflection-2', 81.4, A + '2312.11805'),
]
for n, s, u in LS:
	L(n, s, u)
L('Ling-flash-base-2.0', 90.75, HF + 'nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-Base-BF16')
L('Command R+ (04-2024)', 78.3, HF + 'microsoft/Phi-3-medium-4k-instruct')
L('Yi-1.5-34B-Chat', 90.2, HF + 'Qwen/Qwen2-57B-A14B-Instruct')
L('Yi-1.5-9B-Chat', 84.8, HF + 'Qwen/Qwen2-7B-Instruct')
L('MiniCPM-2B', 53.8, HF + 'Qwen/Qwen2-1.5B')
L('Jamba-v0.1', 59.9, HF + 'Qwen/Qwen2-57B-A14B')
L('Yi-1.5-34B', 82.7, HF + 'Qwen/Qwen2-57B-A14B')
L('MiniCPM3-4B', 81.1, Q25)

write('gsm8k')
