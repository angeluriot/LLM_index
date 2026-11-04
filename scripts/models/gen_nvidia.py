"""Generates data/models/nvidia/*.json (one file per model).

Covers NVIDIA's language models: Megatron-LM and NeMo Megatron, Nemotron (from Nemotron-3 8B to Nemotron 3.5), the Minitron
pruned models, the Llama Nemotron family, Nemotron-H, and the research fine-tunes (ChatQA, SteerLM, OpenMath, AceMath, AceReason,
OpenCodeReasoning, OpenReasoning, Cascade...). Megatron-Turing NLG 530B is in gen_microsoft.py and Mistral NeMo in gen_mistral.py.
All the data lives in this script: edit it and run `python scripts/models/gen_nvidia.py`.
Existing JSON files are deleted and rewritten, so manual edits to them are lost.

Base checkpoints are folded into the version of their post-trained model (names of both). Each size is a model; the reasoning,
domain, language, vision and omni variants post-trained from the same base are its versions. Compressed models (Minitron, Puzzle,
Elastic, Nemotron Nano 9B v2 from 12B v2...) are separate models. The fine-tunes of third-party bases are one model per product
line and size. NVIDIA has no paid first-party API (build.nvidia.com is a free trial): prices are the median of third-party
providers on Artificial Analysis, when it lists one.
"""

from common import B, M, N, S, V, write

M_ = 1_000_000
R = ['', 'Reasoning']					# Reasoning on/off switch (enable_thinking, /think, "detailed thinking on" system prompt)
RL = ['', 'Low', 'Reasoning']			# Nemotron 3 Super: reasoning off / low_effort / on
RM = ['', 'Medium', 'Reasoning']		# Nemotron 3 Ultra and the Nemotron 3.5 Super VL family: reasoning off / medium_effort / on

ARXIV = 'https://arxiv.org/abs/'
HF = 'https://huggingface.co/nvidia/'
DEV = 'https://developer.nvidia.com/blog/'
NEMOTRON = 'https://research.nvidia.com/labs/nemotron/files/'


def names(*items):
	"""Flattens names and lists of names, without duplicates."""
	out = []
	for i in items:
		out += [i] if isinstance(i, str) else list(i)
	return list(dict.fromkeys(out))


def hf(*repos):
	"""NVIDIA Hugging Face repo ids, both bare and with the organisation."""
	return [r for r in repos] + ['nvidia/' + r for r in repos]


def aa(name):
	"""Artificial Analysis labels of a hybrid model: plain, reasoning and non-reasoning."""
	return [name, f'{name} (Reasoning)', f'{name} (Non-reasoning)']


# Megatron-LM (GPT-2 style models of the model-parallelism paper; only the 345M checkpoint was published)
MEGATRON = ARXIV + '1909.08053'
S('Megatron-LM 355M', names('Megatron-LM (355M)', 'Megatron-GPT2 345M', 'Megatron-LM 345M', hf('megatron-gpt2-345m')), MEGATRON, '2019-09-17', '2022-03-02', weights='2022-03-02', params=(355 * M_, 355 * M_))
S('Megatron-LM 1.2B', ['Megatron-LM (1.2B)'], MEGATRON, '2019-09-17', None, params=(1_200 * M_, 1_200 * M_))
S('Megatron-LM 2.5B', ['Megatron-LM (2.5B)'], MEGATRON, '2019-09-17', None, params=(2_500 * M_, 2_500 * M_))
S('Megatron-LM 8.3B', ['Megatron-LM', 'Megatron-LM (8.3B)', 'Megatron-GPT2 8.3B', 'Megatron 8.3B'], MEGATRON, '2019-09-17', None, params=(8_300 * M_, 8_300 * M_))

# NeMo Megatron GPT (2022-2023)
S('NeMo Megatron GPT 1.3B', names('NeMo Megatron-GPT 1.3B', 'Megatron-GPT 1.3B', hf('nemo-megatron-gpt-1.3B')), HF + 'nemo-megatron-gpt-1.3B', '2022-09-10', '2022-09-10', weights='2022-09-10', params=(1_300 * M_, 1_300 * M_))
S('NeMo Megatron GPT 5B', names('NeMo Megatron-GPT 5B', 'Megatron-GPT 5B', hf('nemo-megatron-gpt-5B')), HF + 'nemo-megatron-gpt-5B', '2022-09-15', '2022-09-15', weights='2022-09-15', params=(5 * B, 5 * B))
S('NeMo Megatron GPT 20B', names('NeMO Megatron GPT 20B', 'NeMo Megatron-GPT 20B', 'Megatron-GPT 20B', hf('nemo-megatron-gpt-20B')), HF + 'nemo-megatron-gpt-20B', '2022-09-15', '2022-09-15', weights='2022-09-15', params=(20 * B, 20 * B))
S('GPT-2B-001', names('GPT 2B 001', 'NeMo GPT-2B-001', 'nvidia/GPT-2B-001'), HF + 'GPT-2B-001', '2023-04-10', '2023-04-10', weights='2023-04-10', params=(2 * B, 2 * B))

# Nemotron-3 8B (base model and its four aligned variants)
NEMOTRON3 = DEV + 'nvidia-ai-foundation-models-build-custom-enterprise-chatbots-and-co-pilots-with-production-ready-llms/'
P_N3 = (8 * B, 8 * B)
M('Nemotron-3 8B',
	V('Nemotron-3 8B Base', names('Nemotron-3-8B', 'Nemotron-3-8B-Base-4k', 'Nemotron 3 8B Base', hf('nemotron-3-8b-base-4k')), NEMOTRON3, '2023-11-15', '2023-11-15', weights='2023-11-15', params=P_N3),
	V('Nemotron-3 8B Chat SteerLM', names('Nemotron-3-8B-Chat-4k-SteerLM', 'Nemotron-3 8B Chat', hf('nemotron-3-8b-chat-4k-steerlm')), NEMOTRON3, '2023-11-15', '2023-11-15', weights='2023-11-15', params=P_N3),
	V('Nemotron-3 8B Chat SFT', names('Nemotron-3-8B-Chat-4k-SFT', hf('nemotron-3-8b-chat-4k-sft')), NEMOTRON3, '2023-11-15', '2023-11-15', weights='2023-11-15', params=P_N3),
	V('Nemotron-3 8B Chat RLHF', names('Nemotron-3-8B-Chat-4k-RLHF', hf('nemotron-3-8b-chat-4k-rlhf')), NEMOTRON3, '2023-11-15', '2023-11-15', weights='2023-11-15', params=P_N3),
	V('Nemotron-3 8B QA', names('Nemotron-3-8B-QA-4k', hf('nemotron-3-8b-qa-4k')), NEMOTRON3, '2023-11-15', '2023-11-15', weights='2023-11-15', params=P_N3),
)

# SteerLM / RLHF alignments of Llama 2 70B and Llama 3 70B (HelpSteer papers)
S('Llama2-70B-SteerLM-Chat', names('Llama 2 70B SteerLM Chat', 'Llama2 70B SteerLM Chat', hf('Llama2-70B-SteerLM-Chat')), ARXIV + '2311.09528', '2023-11-16', '2023-11-22', weights='2023-11-22', params=(69 * B, 69 * B))
S('NV-Llama2-70B-RLHF-Chat', names('NV Llama2 70B RLHF Chat', 'Llama 2 70B RLHF Chat (NVIDIA)', hf('NV-Llama2-70B-RLHF-Chat')), HF + 'NV-Llama2-70B-RLHF-Chat', '2023-12-05', '2023-12-05', weights='2023-12-05', params=(69 * B, 69 * B))
S('Llama3-70B-SteerLM-Chat', names('Llama 3 70B SteerLM Chat', 'Llama3 70B SteerLM Chat', hf('Llama3-70B-SteerLM-Chat')), ARXIV + '2406.08673', '2024-06-12', '2024-06-13', weights='2024-06-13', params=(70_600 * M_, 70_600 * M_))

# Retro (retrieval-augmented pretraining)
INSTRUCT_RETRO = ARXIV + '2310.07713'
M('Retro 48B',
	V('Retro 48B', ['Retro-48B', 'Retro 48B Base'], INSTRUCT_RETRO, '2023-10-11', None, params=(48 * B, 48 * B)),
	V('InstructRetro 48B', names('InstructRetro', 'InstructRetro-48B', 'Retro 48B Instruct', 'retro-48b-instruct-4k', 'nvidia/retro-48b-instruct-4k'), INSTRUCT_RETRO, '2023-10-11', '2023-12-20', weights='2023-12-20', params=(48 * B, 48 * B)),
)
S('InstructRetro 8B', names('InstructRetro-8B', 'Retro 8B Instruct', hf('retro-8b-instruct-4k')), HF + 'retro-8b-instruct-4k', '2023-12-29', '2023-12-29', weights='2023-12-29', params=(8 * B, 8 * B))

# ChatQA (conversational QA / RAG fine-tunes of Llama 2 and Llama 3)
S('ChatQA 70B', ['ChatQA-70B', 'ChatQA 1.0 70B', 'ChatQA-1.0-70B'], ARXIV + '2401.10225', '2024-01-18', None, params=(69 * B, 69 * B))
S('Llama3-ChatQA-1.5-8B', names('Llama3 ChatQA 1.5 8B', 'ChatQA-1.5-8B', 'Llama3-ChatQA-1.5 8B', 'nvidia/llama3-chatqa-1.5-8b', hf('Llama3-ChatQA-1.5-8B')), HF + 'Llama3-ChatQA-1.5-8B', '2024-04-28', '2024-04-28', weights='2024-04-28', params=(8_030 * M_, 8_030 * M_))
S('Llama3-ChatQA-1.5-70B', names('Llama3 ChatQA 1.5 70B', 'ChatQA-1.5-70B', 'Llama3-ChatQA-1.5 70B', 'nvidia/llama3-chatqa-1.5-70b', hf('Llama3-ChatQA-1.5-70B')), HF + 'Llama3-ChatQA-1.5-70B', '2024-04-28', '2024-04-28', weights='2024-04-28', params=(70_550 * M_, 70_550 * M_))
CHATQA2 = ARXIV + '2407.14482'
S('Llama3-ChatQA-2-8B', names('Llama3 ChatQA 2 8B', 'ChatQA-2-8B', 'ChatQA 2 8B', hf('Llama3-ChatQA-2-8B')), CHATQA2, '2024-07-19', '2024-08-28', weights='2024-08-28', params=(8_030 * M_, 8_030 * M_))
S('Llama3-ChatQA-2-70B', names('Llama3 ChatQA 2 70B', 'ChatQA-2-70B', 'ChatQA 2 70B', hf('Llama3-ChatQA-2-70B')), CHATQA2, '2024-07-19', '2024-08-26', weights='2024-08-26', params=(70_550 * M_, 70_550 * M_))

# OpenMath (OpenMathInstruct-1 fine-tunes, Feb 2024)
OPENMATH = ARXIV + '2402.10176'
for display, repo, params in [
	('OpenMath-Mistral-7B', 'OpenMath-Mistral-7B-v0.1', 7_240 * M_),
	('OpenMath-CodeLlama-7B', 'OpenMath-CodeLlama-7b-Python', 6_740 * M_),
	('OpenMath-CodeLlama-13B', 'OpenMath-CodeLlama-13b-Python', 13 * B),
	('OpenMath-CodeLlama-34B', 'OpenMath-CodeLlama-34b-Python', 33_700 * M_),
	('OpenMath-CodeLlama-70B', 'OpenMath-CodeLlama-70b-Python', 69 * B),
	('OpenMath-Llama-2-70B', 'OpenMath-Llama-2-70b', 69 * B),
]:
	S(display, names(display.replace('-', ' '), hf(repo, repo + '-hf')), OPENMATH, '2024-02-15', '2024-02-15', weights='2024-02-15', params=(params, params))

# Mamba2-Hybrid 8B (research model of the Mamba-based LLM study; the context extensions are versions)
MAMBA2 = ARXIV + '2406.07887'
P_MH = (8_660 * M_, 8_660 * M_)
M('Mamba2-Hybrid 8B',
	V('Mamba2-Hybrid 8B', names('Mamba2-Hybrid', 'Mamba-2-Hybrid 8B', 'Mamba2-Hybrid-8B-3T-4K', hf('mamba2-hybrid-8b-3t-4k')), MAMBA2, '2024-06-12', '2024-06-13', weights='2024-06-13', params=P_MH),
	V('Mamba2-Hybrid 8B 32K', names('Mamba2-Hybrid-8B-3T-32K', hf('mamba2-hybrid-8b-3t-32k')), MAMBA2, '2024-06-12', '2024-06-13', weights='2024-06-13', params=P_MH),
	V('Mamba2-Hybrid 8B 128K', names('Mamba2-Hybrid-8B-3T-128K', hf('mamba2-hybrid-8b-3t-128k')), MAMBA2, '2024-06-12', '2024-06-13', weights='2024-06-13', params=P_MH),
)

# Nemotron-4
S('Nemotron-4 15B', ['Nemotron-4-15B', 'Nemotron 4 15B'], ARXIV + '2402.16819', '2024-02-27', None, params=(15 * B, 15 * B))
S('Nemotron-4 340B', names('Nemotron-4-340B', 'Nemotron 4 340B', 'Nemotron-4 340B Instruct', 'Nemotron-4-340B-Instruct', 'Nemotron-4 340B Base', 'nemotron-4-340b-instruct', 'nemotron-4-340b-Instruct',
	'nvidia/nemotron-4-340b-instruct', hf('Nemotron-4-340B-Instruct', 'Nemotron-4-340B-Base')), 'https://blogs.nvidia.com/blog/nemotron-4-synthetic-data-generation-llm-training/', '2024-06-14', '2024-06-14', weights='2024-06-14', params=(340 * B, 340 * B))

# Minitron (pruned and distilled from Nemotron-4 15B; Nemotron-Mini-4B-Instruct and Nemotron-4-Mini-Hindi-4B are post-trained from Minitron-4B-Base)
MINITRON = ARXIV + '2407.14679'
S('Minitron 8B', names('Minitron-8B', 'Minitron-8B-Base', hf('Minitron-8B-Base')), MINITRON, '2024-07-19', '2024-07-19', weights='2024-07-19', params=(8_300 * M_, 8_300 * M_))
P_MINI4 = (4_200 * M_, 4_200 * M_)
M('Minitron 4B',
	V('Minitron 4B', names('Minitron-4B', 'Minitron-4B-Base', hf('Minitron-4B-Base')), MINITRON, '2024-07-19', '2024-07-19', weights='2024-07-19', params=P_MINI4),
	V('Nemotron-Mini-4B-Instruct', names('Nemotron Mini 4B Instruct', 'Nemotron-Mini-4B', 'Nemotron Mini 4B', 'Nemotron-4 4B Instruct', 'Nemotron-4-4B-Instruct', 'nemotron-mini', 'nvidia/nemotron-mini-4b-instruct', hf('Nemotron-Mini-4B-Instruct')),
		'https://blogs.nvidia.com/blog/ai-decoded-gamescom-ace-nemotron-instruct/', '2024-08-21', '2024-09-10', weights='2024-09-10', params=P_MINI4),
	V('Nemotron-4-Mini-Hindi-4B', names('Nemotron-4 Mini Hindi 4B', 'Nemotron-4-Mini-Hindi-4B-Instruct', 'Nemotron-4-Mini-Hindi-4B-Base', 'Nemotron-Mini-Hindi-4B', hf('Nemotron-4-Mini-Hindi-4B-Instruct', 'Nemotron-4-Mini-Hindi-4B-Base')),
		HF + 'Nemotron-4-Mini-Hindi-4B-Instruct', '2024-10-24', '2024-10-24', weights='2024-10-24', params=(4_190 * M_, 4_190 * M_)),
)

# Minitron of Llama 3.1 8B and Mistral NeMo 12B
PRUNE_DISTILL = ARXIV + '2408.11796'
M('Llama-3.1-Minitron 4B',
	V('Llama-3.1-Minitron 4B Width', names('Llama-3.1-Minitron-4B', 'Llama-3.1-Minitron-4B-Width-Base', 'Llama 3.1 Minitron 4B Width', hf('Llama-3.1-Minitron-4B-Width-Base')),
		DEV + 'how-to-prune-and-distill-llama-3-1-8b-to-an-nvidia-llama-3-1-minitron-4b-model/', '2024-08-14', '2024-08-14', weights='2024-08-14', params=(4_510 * M_, 4_510 * M_)),
	V('Llama-3.1-Minitron 4B Depth', names('Llama-3.1-Minitron-4B-Depth-Base', 'Llama 3.1 Minitron 4B Depth', hf('Llama-3.1-Minitron-4B-Depth-Base')),
		DEV + 'how-to-prune-and-distill-llama-3-1-8b-to-an-nvidia-llama-3-1-minitron-4b-model/', '2024-08-14', '2024-08-14', weights='2024-08-14', params=(4_540 * M_, 4_540 * M_)),
)
S('Mistral-NeMo-Minitron 8B', names('Mistral-NeMo-Minitron-8B', 'Mistral NeMo Minitron 8B', 'Mistral-NeMo-Minitron 8B Instruct', 'Mistral-NeMo-Minitron-8B-Base', 'Mistral-NeMo-Minitron-8B-Instruct',
	'nvidia/mistral-nemo-minitron-8b-8k-instruct', hf('Mistral-NeMo-Minitron-8B-Base', 'Mistral-NeMo-Minitron-8B-Instruct')), PRUNE_DISTILL, '2024-08-21', '2024-08-21', weights='2024-08-21', params=(8_410 * M_, 8_410 * M_))

# Llama 3.1 Nemotron (2024)
S('Llama-3.1-Nemotron-51B-Instruct', names('Llama 3.1 Nemotron 51B Instruct', 'Llama-3.1-Nemotron-51B', 'Llama 3.1 Nemotron 51B', 'llama-3.1-nemotron-51b-instruct', 'nvidia/llama-3.1-nemotron-51b-instruct', hf('Llama-3_1-Nemotron-51B-Instruct')),
	DEV + 'advancing-the-accuracy-efficiency-frontier-with-llama-3-1-nemotron-51b/', '2024-09-23', '2024-09-23', weights='2024-09-23', params=(51_500 * M_, 51_500 * M_))
S('Llama-3.1-Nemotron-70B-Instruct', names('Llama 3.1 Nemotron 70B Instruct', 'Llama 3.1 Nemotron Instruct 70B', 'Llama 3.1 Nemotron 70B', 'Llama-3.1-Nemotron-70B', 'llama-3.1-nemotron-70b-instruct',
	'nvidia/llama-3.1-nemotron-70b-instruct', hf('Llama-3.1-Nemotron-70B-Instruct', 'Llama-3.1-Nemotron-70B-Instruct-HF')), HF + 'Llama-3.1-Nemotron-70B-Instruct-HF', '2024-10-02', '2024-10-12', weights='2024-10-12', params=(70_550 * M_, 70_550 * M_))

# NVLM 1.0 (decoder-only, hybrid and cross-attention architectures on a Qwen2-72B-Instruct backbone; only NVLM-D was released)
NVLM = ARXIV + '2409.11402'
P_NVLM = (79_400 * M_, 79_400 * M_)
M('NVLM 1.0',
	V('NVLM-D 72B', names('NVLM-D-72B', 'NVLM-D 1.0 72B', 'NVLM 1.0 D 72B', 'NVLM-D', hf('NVLM-D-72B')), NVLM, '2024-09-17', '2024-10-01', weights='2024-10-01', params=P_NVLM),
	V('NVLM-H 72B', ['NVLM-H-72B', 'NVLM-H'], NVLM, '2024-09-17', None, params=P_NVLM),
	V('NVLM-X 72B', ['NVLM-X-72B', 'NVLM-X'], NVLM, '2024-09-17', None, params=P_NVLM),
)

# OpenMath2 (OpenMathInstruct-2 fine-tunes of Llama 3.1)
OPENMATH2 = ARXIV + '2410.01560'
S('OpenMath2-Llama3.1-8B', names('OpenMath2 Llama3.1 8B', 'OpenMath2-Llama3.1-8B-nemo', hf('OpenMath2-Llama3.1-8B')), OPENMATH2, '2024-10-02', '2024-10-02', weights='2024-10-02', params=(8_030 * M_, 8_030 * M_))
S('OpenMath2-Llama3.1-70B', names('OpenMath2 Llama3.1 70B', 'OpenMath2-Llama3.1-70B-nemo', hf('OpenMath2-Llama3.1-70B')), OPENMATH2, '2024-10-02', '2024-10-02', weights='2024-10-02', params=(70_550 * M_, 70_550 * M_))

# Hymba (hybrid-head small model)
S('Hymba-1.5B', names('Hymba', 'Hymba 1.5B', 'Hymba-1.5B-Instruct', 'Hymba-1.5B-Base', hf('Hymba-1.5B-Instruct', 'Hymba-1.5B-Base')), ARXIV + '2411.13676', '2024-11-20', '2024-11-20', weights='2024-11-20', params=(1_520 * M_, 1_520 * M_))

# AceMath / AceInstruct (Qwen2.5 fine-tunes; the AceMath reward models are left out)
ACEMATH = ARXIV + '2412.15084'
S('AceMath-1.5B-Instruct', names('AceMath 1.5B Instruct', 'AceMath-1.5B', hf('AceMath-1.5B-Instruct')), ACEMATH, '2024-12-19', '2025-01-14', weights='2025-01-14', params=(1_540 * M_, 1_540 * M_))
S('AceMath-7B-Instruct', names('AceMath 7B Instruct', 'AceMath-7B', hf('AceMath-7B-Instruct')), ACEMATH, '2024-12-19', '2025-01-14', weights='2025-01-14', params=(7_620 * M_, 7_620 * M_))
S('AceMath-72B-Instruct', names('AceMath 72B Instruct', 'AceMath-72B', hf('AceMath-72B-Instruct')), ACEMATH, '2024-12-19', '2025-01-14', weights='2025-01-14', params=(72_710 * M_, 72_710 * M_))
S('AceInstruct-1.5B', names('AceInstruct 1.5B', hf('AceInstruct-1.5B')), HF + 'AceInstruct-1.5B', '2025-01-15', '2025-01-15', weights='2025-01-15', params=(1_540 * M_, 1_540 * M_))
S('AceInstruct-7B', names('AceInstruct 7B', hf('AceInstruct-7B')), HF + 'AceInstruct-7B', '2025-01-15', '2025-01-15', weights='2025-01-15', params=(7_620 * M_, 7_620 * M_))
S('AceInstruct-72B', names('AceInstruct 72B', hf('AceInstruct-72B')), HF + 'AceInstruct-72B', '2025-01-15', '2025-01-15', weights='2025-01-15', params=(72_710 * M_, 72_710 * M_))

# Llama Nemotron (reasoning on/off via the "detailed thinking on/off" system prompt, /no_think for v1.5)
LLAMA_NEMOTRON = ARXIV + '2505.00949'
S('Llama-3.1-Nemotron-8B-UltraLong', names('Llama 3.1 Nemotron 8B UltraLong', 'UltraLong-8B', 'Llama-3.1-Nemotron-8B-UltraLong-1M-Instruct', 'Llama-3.1-Nemotron-8B-UltraLong-2M-Instruct', 'Llama-3.1-Nemotron-8B-UltraLong-4M-Instruct',
	hf('Llama-3.1-Nemotron-8B-UltraLong-1M-Instruct', 'Llama-3.1-Nemotron-8B-UltraLong-2M-Instruct', 'Llama-3.1-Nemotron-8B-UltraLong-4M-Instruct')), ARXIV + '2504.06214', '2025-04-08', '2025-04-08', weights='2025-04-08', params=(8_040 * M_, 8_040 * M_))
S('Llama Nemotron Nano 8B', names('Llama-3.1-Nemotron-Nano-8B-v1', 'Llama 3.1 Nemotron Nano 8B v1', 'Llama 3.1 Nemotron Nano 8B V1', 'Llama Nemotron Nano 8B v1', aa('Llama 3.1 Nemotron Nano 8B'),
	'llama-3.1-nemotron-nano-8b-v1', hf('Llama-3.1-Nemotron-Nano-8B-v1')), LLAMA_NEMOTRON, '2025-03-18', '2025-03-18', weights='2025-03-18', params=(8_030 * M_, 8_030 * M_), efforts=R)
S('Llama Nemotron Super 49B', names('Llama-3.3-Nemotron-Super-49B-v1', 'Llama 3.3 Nemotron Super 49B v1', 'Llama-3.3 Nemotron Super 49B v1', 'Llama Nemotron Super 49B v1', 'Llama 3.3 Nemotron Super 49B',
	aa('Llama 3.3 Nemotron Super 49B v1'), 'llama-3.3-nemotron-super-49b-v1', hf('Llama-3_3-Nemotron-Super-49B-v1')), LLAMA_NEMOTRON, '2025-03-18', '2025-03-18', weights='2025-03-18', params=(49_870 * M_, 49_870 * M_), efforts=R)
S('Llama Nemotron Ultra 253B', names('Llama-3.1-Nemotron-Ultra-253B-v1', 'Llama 3.1 Nemotron Ultra 253B v1', 'Llama Nemotron Ultra 253B v1', 'Llama Nemotron Ultra', aa('Llama 3.1 Nemotron Ultra 253B v1'),
	'llama-3.1-nemotron-ultra-253b-v1', hf('Llama-3_1-Nemotron-Ultra-253B-v1')), LLAMA_NEMOTRON, '2025-03-18', '2025-04-07', weights='2025-04-07', params=(253_400 * M_, 253_400 * M_), efforts=R)
S('Llama Nemotron Nano 4B', names('Llama-3.1-Nemotron-Nano-4B-v1.1', 'Llama 3.1 Nemotron Nano 4B v1.1', 'Llama Nemotron Nano 4B v1.1', aa('Llama 3.1 Nemotron Nano 4B v1.1'), 'llama-3.1-nemotron-nano-4b-v1.1', hf('Llama-3.1-Nemotron-Nano-4B-v1.1')),
	HF + 'Llama-3.1-Nemotron-Nano-4B-v1.1', '2025-05-20', '2025-05-20', weights='2025-05-20', params=(4_510 * M_, 4_510 * M_), efforts=R)
S('Llama Nemotron Super 49B v1.5', names('Llama-3.3-Nemotron-Super-49B-v1.5', 'Llama 3.3 Nemotron Super 49B v1.5', 'Llama Nemotron Super v1.5', aa('Llama Nemotron Super 49B v1.5'), 'llama-3.3-nemotron-super-49b-v1.5', hf('Llama-3_3-Nemotron-Super-49B-v1_5')),
	DEV + 'build-more-accurate-and-efficient-ai-agents-with-the-new-nvidia-llama-nemotron-super-v1-5/', '2025-07-25', '2025-07-25', weights='2025-07-25', params=(49_870 * M_, 49_870 * M_), efforts=R)

# Nemotron-H (hybrid Mamba-Transformer; 47B is compressed from 56B with MiniPuzzle; the reasoning models are post-trained from the bases)
NEMOTRON_H = 'https://research.nvidia.com/labs/adlr/nemotronh/'
NEMOTRON_H_R = DEV + 'nemotron-h-reasoning-enabling-throughput-gains-with-no-compromises/'
S('Nemotron-H 4B', names('Nemotron-H-4B', 'Nemotron-H-4B-Base-8K', 'Nemotron-H-4B-Instruct-128K', hf('Nemotron-H-4B-Base-8K', 'Nemotron-H-4B-Instruct-128K')), HF + 'Nemotron-H-4B-Instruct-128K', '2025-04-15', '2025-04-15', weights='2025-04-15', params=(4_490 * M_, 4_490 * M_))
M('Nemotron-H 8B',
	V('Nemotron-H 8B', names('Nemotron-H-8B', 'Nemotron-H 8B Base', 'Nemotron-H-8B-Base-8K', hf('Nemotron-H-8B-Base-8K')), NEMOTRON_H, '2025-03-21', '2025-04-14', weights='2025-04-14', params=(8_100 * M_, 8_100 * M_)),
	V('Nemotron-H 8B Reasoning', names('Nemotron-H-8B-Reasoning-128K', 'Nemotron-H 8B Reasoning 128K', hf('Nemotron-H-8B-Reasoning-128K')), NEMOTRON_H_R, '2025-06-06', '2025-06-06', weights='2025-06-06', params=(8_100 * M_, 8_100 * M_), efforts=R),
)
M('Nemotron-H 47B',
	V('Nemotron-H 47B', names('Nemotron-H-47B', 'Nemotron-H 47B Base', 'Nemotron-H-47B-Base-8K', hf('Nemotron-H-47B-Base-8K')), NEMOTRON_H, '2025-03-21', '2025-04-14', weights='2025-04-14', params=(46_790 * M_, 46_790 * M_)),
	V('Nemotron-H 47B Reasoning', names('Nemotron-H-47B-Reasoning-128K', 'Nemotron-H 47B Reasoning 128K', hf('Nemotron-H-47B-Reasoning-128K')), NEMOTRON_H_R, '2025-06-06', '2025-06-06', weights='2025-06-06', params=(46_790 * M_, 46_790 * M_), efforts=R),
)
S('Nemotron-H 56B', names('Nemotron-H-56B', 'Nemotron-H 56B Base', 'Nemotron-H-56B-Base-8K', hf('Nemotron-H-56B-Base-8K')), NEMOTRON_H, '2025-03-21', '2025-04-14', weights='2025-04-14', params=(56_320 * M_, 56_320 * M_))

# OpenCodeReasoning-Nemotron (Qwen2.5-Instruct fine-tunes; v1.1 and the IOI variant are versions of the same size)
OCR = ARXIV + '2504.01943'
M('OpenCodeReasoning-Nemotron 7B',
	V('OpenCodeReasoning-Nemotron-7B', names('OpenCodeReasoning Nemotron 7B', 'OCR-Nemotron-7B', hf('OpenCodeReasoning-Nemotron-7B')), OCR, '2025-04-02', '2025-04-15', weights='2025-04-15', params=(7_620 * M_, 7_620 * M_)),
	V('OpenCodeReasoning-Nemotron-1.1-7B', names('OpenCodeReasoning Nemotron 1.1 7B', hf('OpenCodeReasoning-Nemotron-1.1-7B')), HF + 'OpenCodeReasoning-Nemotron-1.1-7B', '2025-06-12', '2025-06-12', weights='2025-06-12', params=(7_620 * M_, 7_620 * M_)),
)
M('OpenCodeReasoning-Nemotron 14B',
	V('OpenCodeReasoning-Nemotron-14B', names('OpenCodeReasoning Nemotron 14B', 'OCR-Nemotron-14B', hf('OpenCodeReasoning-Nemotron-14B')), OCR, '2025-04-02', '2025-04-15', weights='2025-04-15', params=(14_770 * M_, 14_770 * M_)),
	V('OpenCodeReasoning-Nemotron-1.1-14B', names('OpenCodeReasoning Nemotron 1.1 14B', hf('OpenCodeReasoning-Nemotron-1.1-14B')), HF + 'OpenCodeReasoning-Nemotron-1.1-14B', '2025-06-12', '2025-06-12', weights='2025-06-12', params=(14_770 * M_, 14_770 * M_)),
)
M('OpenCodeReasoning-Nemotron 32B',
	V('OpenCodeReasoning-Nemotron-32B', names('OpenCodeReasoning Nemotron 32B', 'OCR-Nemotron-32B', hf('OpenCodeReasoning-Nemotron-32B')), OCR, '2025-04-02', '2025-04-15', weights='2025-04-15', params=(32_760 * M_, 32_760 * M_)),
	V('OpenCodeReasoning-Nemotron-32B-IOI', names('OpenCodeReasoning Nemotron 32B IOI', 'OCR-Nemotron-32B-IOI', hf('OpenCodeReasoning-Nemotron-32B-IOI')), HF + 'OpenCodeReasoning-Nemotron-32B-IOI', '2025-05-07', '2025-05-07', weights='2025-05-07', params=(32_760 * M_, 32_760 * M_)),
	V('OpenCodeReasoning-Nemotron-1.1-32B', names('OpenCodeReasoning Nemotron 1.1 32B', hf('OpenCodeReasoning-Nemotron-1.1-32B')), HF + 'OpenCodeReasoning-Nemotron-1.1-32B', '2025-06-12', '2025-06-12', weights='2025-06-12', params=(32_760 * M_, 32_760 * M_)),
)

# OpenMath-Nemotron (AIMO-2 winning models, Qwen2.5 fine-tunes)
OPENMATH_N = ARXIV + '2504.16891'
S('OpenMath-Nemotron-1.5B', names('OpenMath Nemotron 1.5B', hf('OpenMath-Nemotron-1.5B')), OPENMATH_N, '2025-04-23', '2025-04-23', weights='2025-04-23', params=(1_540 * M_, 1_540 * M_))
S('OpenMath-Nemotron-7B', names('OpenMath Nemotron 7B', hf('OpenMath-Nemotron-7B')), OPENMATH_N, '2025-04-23', '2025-04-23', weights='2025-04-23', params=(7_620 * M_, 7_620 * M_))
M('OpenMath-Nemotron-14B',
	V('OpenMath-Nemotron-14B', names('OpenMath Nemotron 14B', hf('OpenMath-Nemotron-14B')), OPENMATH_N, '2025-04-23', '2025-04-23', weights='2025-04-23', params=(14_770 * M_, 14_770 * M_)),
	V('OpenMath-Nemotron-14B-Kaggle', names('OpenMath Nemotron 14B Kaggle', hf('OpenMath-Nemotron-14B-Kaggle')), OPENMATH_N, '2025-04-23', '2025-04-23', weights='2025-04-23', params=(14_770 * M_, 14_770 * M_)),
)
S('OpenMath-Nemotron-32B', names('OpenMath Nemotron 32B', hf('OpenMath-Nemotron-32B')), OPENMATH_N, '2025-04-23', '2025-04-23', weights='2025-04-23', params=(32_760 * M_, 32_760 * M_))

# AceReason (RL on DeepSeek-R1-Distill-Qwen for v1.0 and AceMath-RL, on Qwen2.5-Math-7B for v1.1)
S('AceMath-RL-Nemotron-7B', names('AceMath RL Nemotron 7B', hf('AceMath-RL-Nemotron-7B')), HF + 'AceMath-RL-Nemotron-7B', '2025-04-22', '2025-04-22', weights='2025-04-22', params=(7_620 * M_, 7_620 * M_))
ACEREASON = ARXIV + '2505.16400'
S('AceReason-Nemotron-7B', names('AceReason Nemotron 7B', 'AceReason-Nemotron 7B', hf('AceReason-Nemotron-7B')), ACEREASON, '2025-05-22', '2025-05-22', weights='2025-05-22', params=(7_620 * M_, 7_620 * M_))
S('AceReason-Nemotron-14B', names('AceReason Nemotron 14B', 'AceReason-Nemotron 14B', hf('AceReason-Nemotron-14B')), ACEREASON, '2025-05-22', '2025-05-22', weights='2025-05-22', params=(14_770 * M_, 14_770 * M_))
S('AceReason-Nemotron-1.1-7B', names('AceReason Nemotron 1.1 7B', 'AceReason-Nemotron 1.1 7B', hf('AceReason-Nemotron-1.1-7B')), ARXIV + '2506.13284', '2025-06-16', '2025-06-16', weights='2025-06-16', params=(7_620 * M_, 7_620 * M_))

# ProRL (RL of DeepSeek-R1-Distill-Qwen-1.5B)
S('Nemotron-Research-Reasoning-Qwen-1.5B', names('Nemotron Research Reasoning Qwen 1.5B', 'ProRL 1.5B', hf('Nemotron-Research-Reasoning-Qwen-1.5B')), ARXIV + '2505.24864', '2025-05-30', '2025-05-30', weights='2025-05-30', params=(1_780 * M_, 1_780 * M_))

# OpenReasoning-Nemotron (distilled from DeepSeek-R1-0528 into Qwen2.5 bases)
OPENREASONING = HF + 'OpenReasoning-Nemotron-32B'
for size, params in [('1.5B', 1_540 * M_), ('7B', 7_620 * M_), ('14B', 14_770 * M_), ('32B', 32_760 * M_)]:
	S(f'OpenReasoning-Nemotron-{size}', names(f'OpenReasoning Nemotron {size}', hf(f'OpenReasoning-Nemotron-{size}')), OPENREASONING, '2025-07-16', '2025-07-16', weights='2025-07-16', params=(params, params))

# NVIDIA Nemotron Nano 2 (9B v2 is pruned from 12B v2; the Japanese and VL models are post-trained from the matching size)
NANO2 = ARXIV + '2508.14444'
M('NVIDIA Nemotron Nano 9B v2',
	V('NVIDIA Nemotron Nano 9B v2', names('NVIDIA-Nemotron-Nano-9B-v2', 'Nemotron Nano 9B v2', 'Nemotron Nano 9B V2', 'Nemotron-Nano-9B-v2', aa('NVIDIA Nemotron Nano 9B V2'), 'nvidia-nemotron-nano-9b-v2', 'nemotron-nano-9b-v2',
		'NVIDIA-Nemotron-Nano-9B-v2-Base', hf('NVIDIA-Nemotron-Nano-9B-v2', 'NVIDIA-Nemotron-Nano-9B-v2-Base')), NANO2, '2025-08-18', '2025-08-18', weights='2025-08-18', params=(8_890 * M_, 8_890 * M_), price=(0.04, 0.16), efforts=R),
	V('NVIDIA Nemotron Nano 9B v2 Japanese', names('NVIDIA-Nemotron-Nano-9B-v2-Japanese', 'Nemotron Nano 9B v2 Japanese', hf('NVIDIA-Nemotron-Nano-9B-v2-Japanese')), HF + 'NVIDIA-Nemotron-Nano-9B-v2-Japanese', '2026-02-17', '2026-02-17', weights='2026-02-17', params=(8_890 * M_, 8_890 * M_), efforts=R),
)
M('NVIDIA Nemotron Nano 12B v2',
	V('NVIDIA Nemotron Nano 12B v2', names('NVIDIA-Nemotron-Nano-12B-v2', 'Nemotron Nano 12B v2', 'Nemotron-Nano-12B-v2', 'NVIDIA-Nemotron-Nano-12B-v2-Base', hf('NVIDIA-Nemotron-Nano-12B-v2', 'NVIDIA-Nemotron-Nano-12B-v2-Base')),
		NANO2, '2025-08-18', '2025-08-18', weights='2025-08-18', params=(12_310 * M_, 12_310 * M_), efforts=R),
	V('NVIDIA Nemotron Nano 12B v2 VL', names('NVIDIA-Nemotron-Nano-12B-v2-VL', 'Nemotron Nano 12B v2 VL', 'Nemotron Nano 2 VL', aa('NVIDIA Nemotron Nano 12B v2 VL'), 'nvidia/nemotron-nano-12b-v2-vl', hf('NVIDIA-Nemotron-Nano-12B-v2-VL-BF16')),
		HF + 'NVIDIA-Nemotron-Nano-12B-v2-VL-BF16', '2025-10-28', '2025-10-28', weights='2025-10-28', params=(13_180 * M_, 13_180 * M_), price=(0.2, 0.6), efforts=R),
)
S('Nemotron-Elastic-12B', names('Nemotron Elastic 12B', hf('Nemotron-Elastic-12B')), ARXIV + '2511.16664', '2025-11-20', '2025-11-20', weights='2025-11-20', params=(12_310 * M_, 12_310 * M_), efforts=R)

# Nemotron-Flash (small hybrid models optimised for latency)
FLASH = ARXIV + '2511.18890'
S('Nemotron-Flash-1B', names('Nemotron Flash 1B', hf('Nemotron-Flash-1B')), FLASH, '2025-11-24', '2025-11-24', weights='2025-11-24', params=(965 * M_, 965 * M_))
S('Nemotron-Flash-3B', names('Nemotron Flash 3B', 'Nemotron-Flash-3B-Instruct', 'Nemotron Flash 3B Instruct', hf('Nemotron-Flash-3B', 'Nemotron-Flash-3B-Instruct')), FLASH, '2025-11-24', '2025-11-24', weights='2025-11-24', params=(2_750 * M_, 2_750 * M_))

# Agentic fine-tunes of Qwen3 (ToolOrchestra, Terminal-Bench)
S('Nemotron-Orchestrator-8B', names('Nemotron Orchestrator 8B', 'Orchestrator-8B', 'ToolOrchestra Orchestrator-8B', hf('Nemotron-Orchestrator-8B')), ARXIV + '2511.21689', '2025-11-26', '2025-11-26', weights='2025-11-26', params=(8_190 * M_, 8_190 * M_))
TERMINAL = ARXIV + '2602.21193'
S('Nemotron-Terminal-8B', names('Nemotron Terminal 8B', hf('Nemotron-Terminal-8B')), TERMINAL, '2026-02-24', '2026-02-24', weights='2026-02-24', params=(8_190 * M_, 8_190 * M_))
S('Nemotron-Terminal-14B', names('Nemotron Terminal 14B', hf('Nemotron-Terminal-14B')), TERMINAL, '2026-02-24', '2026-02-24', weights='2026-02-24', params=(14_770 * M_, 14_770 * M_))
S('Nemotron-Terminal-32B', names('Nemotron Terminal 32B', hf('Nemotron-Terminal-32B')), TERMINAL, '2026-02-24', '2026-02-24', weights='2026-02-24', params=(32_760 * M_, 32_760 * M_))

# Nemotron-Cascade (cascaded RL of Qwen3 bases; the unified 8B has thinking and instruct modes)
CASCADE = ARXIV + '2512.13607'
M('Nemotron-Cascade 8B',
	V('Nemotron-Cascade-8B-Thinking', names('Nemotron Cascade 8B Thinking', hf('Nemotron-Cascade-8B-Thinking')), CASCADE, '2025-12-08', '2025-12-08', weights='2025-12-08', params=(8_190 * M_, 8_190 * M_)),
	V('Nemotron-Cascade-8B', names('Nemotron Cascade 8B', hf('Nemotron-Cascade-8B')), CASCADE, '2025-12-08', '2025-12-16', weights='2025-12-16', params=(8_190 * M_, 8_190 * M_), efforts=R),
)
M('Nemotron-Cascade 14B',
	V('Nemotron-Cascade-14B-Thinking', names('Nemotron Cascade 14B Thinking', 'Nemotron-Cascade 14B Thinking', hf('Nemotron-Cascade-14B-Thinking')), CASCADE, '2025-12-08', '2025-12-08', weights='2025-12-08', params=(14_770 * M_, 14_770 * M_)),
)

# Nemotron 3 (announced together on 2025-12-15; Cascade 2 and Nano Omni are post-trained from the Nano 30B-A3B base,
# the Teacher and Math models from the post-trained Ultra)
NANO3 = NEMOTRON + 'NVIDIA-Nemotron-3-Nano-Technical-Report.pdf'
P_NANO3 = (31_600 * M_, 3_200 * M_)
M('Nemotron 3 Nano',
	V('Nemotron 3 Nano', names('Nemotron 3 Nano 30B-A3B', 'Nemotron 3 Nano 30B A3B', 'Nemotron 3 Nano (30B A3B)', 'Nemotron-3-Nano-30B-A3B', 'Nemotron 3-Nano-30B-A3B', 'NVIDIA-Nemotron-3-Nano-30B-A3B',
		aa('NVIDIA Nemotron 3 Nano 30B A3B'), 'Nemotron 3 Nano (non-reasoning)', 'nemotron-3-nano-30b-a3b', 'nvidia/nemotron-3-nano-30b-a3b', 'NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16',
		hf('NVIDIA-Nemotron-3-Nano-30B-A3B-BF16', 'NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16')), NANO3, '2025-12-15', '2025-12-15', weights='2025-12-15', params=P_NANO3, price=(0.05, 0.2), efforts=R),
	V('Nemotron-Cascade 2 30B-A3B', names('Nemotron-Cascade-2-30B-A3B', 'Nemotron Cascade 2 30B A3B', 'Nemotron-Cascade 2', 'Nemotron Cascade 2', 'nemotron-cascade-2', hf('Nemotron-Cascade-2-30B-A3B')),
		ARXIV + '2603.19220', '2026-03-19', '2026-03-19', weights='2026-03-19', params=P_NANO3, efforts=R),
	V('Nemotron 3 Nano Omni', names('Nemotron 3 Nano Omni 30B-A3B', 'Nemotron 3 Nano Omni 30B A3B', 'Nemotron 3 Nano Omni 30B A3B Reasoning', 'Nemotron-3-Nano-Omni-30B-A3B-Reasoning', 'nemotron-3-nano-omni-30b-a3b-reasoning',
		'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning', hf('Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16')), ARXIV + '2604.24954', '2026-03-16', '2026-04-28', weights='2026-04-28', params=(33 * B, None), price=(0.5, 0.9), efforts=R),
)
S('Nemotron 3 Nano 4B', names('NVIDIA Nemotron 3 Nano 4B', 'Nemotron-3-Nano-4B', 'NVIDIA-Nemotron-3-Nano-4B', 'nemotron-3-nano-4b', hf('NVIDIA-Nemotron-3-Nano-4B', 'NVIDIA-Nemotron-3-Nano-4B-BF16')),
	HF + 'NVIDIA-Nemotron-3-Nano-4B-BF16', '2026-03-16', '2026-03-16', weights='2026-03-16', params=(3_970 * M_, 3_970 * M_), efforts=R)
S('Nemotron Labs 3 Elastic 30B-A3B', names('Nemotron-Labs-3-Elastic-30B-A3B', 'NVIDIA-Nemotron-Labs-3-Elastic-30B-A3B', hf('NVIDIA-Nemotron-Labs-3-Elastic-30B-A3B-BF16')),
	HF + 'NVIDIA-Nemotron-Labs-3-Elastic-30B-A3B-BF16', '2026-05-07', '2026-05-07', weights='2026-05-07', params=P_NANO3, efforts=R)
SUPER3 = DEV + 'introducing-nemotron-3-super-an-open-hybrid-mamba-transformer-moe-for-agentic-reasoning/'
S('Nemotron 3 Super', names('Nemotron 3 Super 120B-A12B', 'Nemotron 3 Super 120B A12B', 'Nemotron 3 Super (120B A12B)', 'Nemotron-3-Super-120B-A12B', 'NVIDIA-Nemotron-3-Super-120B-A12B', aa('Nemotron 3 Super 120B A12B'),
	'nemotron-3-super', 'nemotron-3-super-120b-a12b', 'nvidia/nemotron-3-super-120b-a12b', 'NVIDIA-Nemotron-3-Super-120B-A12B-Base-BF16', hf('NVIDIA-Nemotron-3-Super-120B-A12B-BF16', 'NVIDIA-Nemotron-3-Super-120B-A12B-Base-BF16')),
	SUPER3, '2025-12-15', '2026-03-11', weights='2026-03-11', params=(120_700 * M_, 12_800 * M_), price=(0.3, 0.9), efforts=RL)
S('Nemotron Labs 3 Puzzle 75B-A9B', names('Nemotron-Labs-3-Puzzle-75B-A9B', 'NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B', 'Puzzle-75B-A9B', hf('NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-BF16')),
	ARXIV + '2607.04371', '2026-07-06', '2026-07-06', weights='2026-07-06', params=(75_300 * M_, 9_300 * M_), efforts=RL)
ULTRA3 = NEMOTRON + 'NVIDIA-Nemotron-3-Ultra-Technical-Report.pdf'
P_ULTRA3 = (550 * B, 55 * B)
TEACHER = 'https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Teacher-'
M('Nemotron 3 Ultra',
	V('Nemotron 3 Ultra', names('Nemotron 3 Ultra 550B-A55B', 'Nemotron 3 Ultra 550B A55B', 'Nemotron 3 Ultra (550B A55B)', 'Nemotron-3-Ultra-550B-A55B', 'NVIDIA-Nemotron-3-Ultra-550B-A55B', aa('Nemotron 3 Ultra 550B A55B'),
		'nemotron-3-ultra', 'nemotron-3-ultra-550b-a55b', 'nvidia/nemotron-3-ultra-550b-a55b', 'NVIDIA-Nemotron-3-Ultra-550B-A55B-Base-BF16', hf('NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16', 'NVIDIA-Nemotron-3-Ultra-550B-A55B-Base-BF16')),
		ULTRA3, '2025-12-15', '2026-06-04', weights='2026-06-04', params=P_ULTRA3, price=(0.6, 2.7), efforts=RM),
	*[V(f'Nemotron Labs Teacher {d}', names(f'NVIDIA-Nemotron-Labs-Teacher-{r}', f'Nemotron-Labs-Teacher-{r}', hf(f'NVIDIA-Nemotron-Labs-Teacher-{r}')), TEACHER + r, '2026-08-14', '2026-08-14', weights='2026-08-14', params=P_ULTRA3, efforts=R)
		for d, r in [('STEM', 'STEM'), ('General Reasoning', 'General-Reasoning'), ('Instruction Following', 'Instruction-Following'), ('Competition Coding', 'Competition-Coding'), ('Chat', 'Chat')]],
	V('Nemotron 3 Labs Ultra Math', names('Nemotron-3-Labs-Ultra-Math-RL', 'Nemotron 3 Labs Ultra Math RL', 'Nemotron-3-Labs-Ultra-Math-SFT', hf('Nemotron-3-Labs-Ultra-Math-RL', 'Nemotron-3-Labs-Ultra-Math-SFT')),
		ARXIV + '2609.10712', '2026-09-03', '2026-09-03', weights='2026-09-03', params=P_ULTRA3),
)

# Third-party models compressed with Puzzle
S('gpt-oss-puzzle-88B', names('GPT OSS Puzzle 88B', 'gpt-oss-puzzle 88B', hf('gpt-oss-puzzle-88B')), HF + 'gpt-oss-puzzle-88B', '2026-03-26', '2026-03-26', weights='2026-03-26', params=(88 * B, None))

# Nemotron Labs Diffusion (autoregressive + diffusion decoding in the same model)
for size, params, date in [('3B', 3_830 * M_, '2026-03-02'), ('8B', 8_490 * M_, '2026-03-18'), ('14B', 13_510 * M_, '2026-04-22')]:
	S(f'Nemotron-Labs-Diffusion-{size}', names(f'Nemotron Labs Diffusion {size}', f'Nemotron-Labs-Diffusion-{size}-Base', hf(f'Nemotron-Labs-Diffusion-{size}', f'Nemotron-Labs-Diffusion-{size}-Base')),
		HF + f'Nemotron-Labs-Diffusion-{size}', date, date, weights=date, params=(params, params))

# Nemotron 3.5 (Lightning is distilled from Ultra; Super VL's backbone continues Nemotron 3 Super Base; Bolt 74B-A9B is compressed
# from Super VL and Bolt E4B post-trained from Gemma 4 E4B; the 4GB / 24GB checkpoints are quantized deployments of the same models)
S('Nemotron 3.5 Lightning', names('Nemotron 3.5 Lightning 30B-A3B', 'Nemotron 3.5 Lightning (30B A3B)', 'NVIDIA-Nemotron-3.5-Lightning-30B-A3B', 'nemotron-3.5-lightning', 'nemotron-3.5-lightning-30b-a3b', 'nemotron-3-5-lightning',
	'nvidia/nemotron-3.5-lightning', 'NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Base-BF16', hf('NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16', 'NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Base-BF16')),
	DEV + 'nvidia-nemotron-3-5-lightning-delivers-fast-accurate-specialized-task-execution-for-long-running-agents/', '2026-08-11', '2026-08-11', weights='2026-08-11', params=P_NANO3, price=(0.06, 0.2), efforts=R)
M('Nemotron 3.5 Super',
	V('Nemotron 3.5 Super VL', names('Nemotron 3.5 Super VL 120B-A12B', 'Nemotron 3.5 Super VL (120B A12B)', 'NVIDIA-Nemotron-3.5-Super-VL-120B-A12B', 'nemotron-3.5-super-vl-120b-a12b', 'nemotron-3-5-super-vl',
		'NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Base-BF16', hf('NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16', 'NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Base-BF16')),
		HF + 'NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16', '2026-10-15', '2026-10-15', weights='2026-10-15', params=(120 * B, 12 * B), price=(0.2, 0.5), efforts=RM),
)
S('Nemotron 3.5 Bolt 74B-A9B', names('Nemotron 3.5 Bolt 74B A9B', 'Nemotron 3.5 Bolt 74B-A9B 24GB', 'Nemotron 3.5 Bolt (74B A9B 24GB)', 'NVIDIA-Nemotron-3.5-Bolt-74B-A9B', 'nemotron-3.5-bolt-74b-a9b-24gb', 'nemotron-3-5-bolt-74b-a9b',
	hf('NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16', 'NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB')), HF + 'NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16', '2026-10-15', '2026-10-15', weights='2026-10-15', params=(74_900 * M_, 9 * B), efforts=RM)
S('Nemotron 3.5 Bolt E4B', names('Nemotron 3.5 Bolt E4B 4GB', aa('Nemotron 3.5 Bolt E4B 4GB'), 'NVIDIA-Nemotron-3.5-Bolt-E4B', 'nemotron-3.5-bolt-e4b', 'nemotron-3-5-bolt-e4b-4gb', hf('NVIDIA-Nemotron-3.5-Bolt-E4B-BF16', 'NVIDIA-Nemotron-3.5-Bolt-E4B-4GB')),
	HF + 'NVIDIA-Nemotron-3.5-Bolt-E4B-BF16', '2026-10-15', '2026-10-23', weights='2026-10-23', params=(8 * B, 4_500 * M_), efforts=R)

# Nemotron Labs Dory (looped latent-recurrent model)
S('Nemotron Labs Dory 1.0 4B', names('Nemotron-Labs-Dory1.0-4B', 'NVIDIA-Nemotron-Labs-Dory1.0-4B', 'Nemotron Labs Dory 1.0 (4B)', 'nemotron-labs-dory-1.0-4b', hf('NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16')),
	HF + 'NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16', '2026-10-16', '2026-10-16', weights='2026-10-16', params=(3_980 * M_, 3_980 * M_))

write('nvidia')
