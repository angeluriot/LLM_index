"""Generates data/models/microsoft/*.json (one file per model).

Covers every Microsoft lab: Microsoft Research (Turing, Phi, Orca, BioGPT, BitNet...), the WizardLM team and Microsoft AI (MAI).
All the data lives in this script: edit it and run `python scripts/models/gen_microsoft.py`.
Existing JSON files are deleted and rewritten, so manual edits to them are lost.

Each Phi size is a model; the context-length variants, the vision / multimodal variants built on the same language model
and the reasoning fine-tunes are its versions (Phi-4-mini-flash-reasoning is a new SambaY architecture, so a separate model).
The WizardLM fine-tunes are one model per size and base family, their V1.x updates are versions.
MAI-Thinking-1 is post-trained on MAI-Base-1; MAI-Cyber-1-Flash and MAI-Cyber-1.1-Flash are fine-tunes of MAI-Code-1-Flash.
Prices are Microsoft's own: Azure AI Foundry serverless list prices for Phi and MAI-DS-R1 / MAI-Thinking-1, and GitHub Copilot
token prices for the MAI-Code models.
"""

from common import B, M, N, S, V, write

M_ = 1_000_000
P_T1 = (962 * B, 34_700 * M_)	# MAI-Base-1 architecture (MAI-Thinking-1)
P_C1 = (137 * B, 5 * B)			# MAI-Code-1-Flash architecture (and its Cyber fine-tunes)

ARXIV = 'https://arxiv.org/abs/'
HF = 'https://huggingface.co/'
MSR = 'https://www.microsoft.com/en-us/research/blog/'
AZURE = 'https://azure.microsoft.com/en-us/blog/'
MAI = 'https://microsoft.ai/'


def names(*items):
	"""Flattens names and lists of names, without duplicates."""
	out = []
	for i in items:
		out += [i] if isinstance(i, str) else list(i)
	return list(dict.fromkeys(out))


def hf(*repos):
	"""Repo ids, both bare and with their organisation."""
	return [r.split('/')[-1] for r in repos] + list(repos)


# Microsoft Research language models (2019-2022)
S('DialoGPT 762M', names('DialoGPT', 'DialoGPT Large', 'DialoGPT-large', hf('microsoft/DialoGPT-large')), ARXIV + '1911.00536', '2019-11-01', '2019-11-01', weights='2019-11-01', params=(762 * M_, 762 * M_))
S('Turing-NLG', ['Turing NLG', 'T-NLG', 'Turing-NLG 17B'], MSR + 'turing-nlg-a-17-billion-parameter-language-model-by-microsoft/', '2020-02-13', None, params=(17 * B, 17 * B))
S('Megatron-Turing NLG 530B', ['MT-NLG', 'MT-NLG 530B', 'Megatron-Turing NLG', 'Megatron-Turing-NLG', 'Megatron Turing NLG 530B'],
	MSR + 'using-deepspeed-and-megatron-to-train-megatron-turing-nlg-530b-the-worlds-largest-and-most-powerful-generative-language-model/', '2021-10-11', None, params=(530 * B, 530 * B))
BIOGPT = ARXIV + '2210.10341'
S('BioGPT', names('BioGPT 347M', 'BioGPT-Base', hf('microsoft/biogpt')), BIOGPT, '2022-10-19', '2022-10-19', weights='2022-10-19', params=(347 * M_, 347 * M_))
M('BioGPT-Large',
	V('BioGPT-Large', names('BioGPT Large', 'BioGPT-Large 1.5B', hf('microsoft/BioGPT-Large')), BIOGPT, '2022-10-19', '2023-02-03', weights='2023-02-03', params=(1_500 * M_, 1_500 * M_)),
	V('BioGPT-Large-PubMedQA', names('BioGPT-Large (PubMedQA)', hf('microsoft/BioGPT-Large-PubMedQA')), BIOGPT, '2022-10-19', '2023-02-03', weights='2023-02-03', params=(1_500 * M_, 1_500 * M_)),
)

# WizardLM (Evol-Instruct fine-tunes of LLaMA / Llama 2 / StarCoder / Code Llama / Mistral bases, one model per size)
WIZ = ARXIV + '2304.12244'
WCODER = ARXIV + '2306.08568'
WMATH = ARXIV + '2308.09583'


def wiz(repo, *extra):
	"""Names of a WizardLM repo: the repo id with spaces, then the repo under both of its Hugging Face organisations."""
	name = repo.split('/')[-1]
	return names(name.replace('-', ' '), *extra, hf(repo, 'WizardLM/' + name))


S('WizardLM 7B', wiz('WizardLMTeam/WizardLM-7B-V1.0', 'WizardLM-7B', 'WizardLM 7B V1.0'), WIZ, '2023-04-24', '2023-04-24', weights='2023-04-24', params=(6_700 * M_, 6_700 * M_))
M('WizardLM 13B',
	V('WizardLM 13B V1.0', wiz('WizardLMTeam/WizardLM-13B-V1.0'), WIZ, '2023-05-13', '2023-05-13', weights='2023-05-13', params=(13 * B, 13 * B)),
	V('WizardLM 13B V1.1', wiz('WizardLMTeam/WizardLM-13B-V1.1'), WIZ, '2023-07-07', '2023-07-07', weights='2023-07-07', params=(13 * B, 13 * B)),
	V('WizardLM 13B V1.2', wiz('WizardLMTeam/WizardLM-13B-V1.2', 'WizardLM 13B v1.2'), WIZ, '2023-07-25', '2023-07-25', weights='2023-07-25', params=(13 * B, 13 * B)),
)
S('WizardLM 30B', wiz('WizardLMTeam/WizardLM-30B-V1.0', 'WizardLM-30B', 'WizardLM 30B V1.0'), WIZ, '2023-06-06', '2023-06-06', weights='2023-06-06', params=(32_500 * M_, 32_500 * M_))
S('WizardLM 70B', wiz('WizardLMTeam/WizardLM-70B-V1.0', 'WizardLM-70B', 'WizardLM 70B V1.0'), WIZ, '2023-08-09', '2023-08-09', weights='2023-08-09', params=(69 * B, 69 * B))
S('WizardCoder 15B', wiz('WizardLMTeam/WizardCoder-15B-V1.0', 'WizardCoder-15B', 'WizardCoder-15.5B', 'WizardCoder', 'WizardCoder 15B V1.0'), WCODER, '2023-06-14', '2023-06-14', weights='2023-06-14', params=(15_500 * M_, 15_500 * M_))
S('WizardCoder Python 13B', wiz('WizardLMTeam/WizardCoder-Python-13B-V1.0', 'WizardCoder-Python-13B', 'WizardCoder Python 13B V1.0'), WCODER, '2023-08-26', '2023-08-26', weights='2023-08-26', params=(13 * B, 13 * B))
S('WizardCoder Python 34B', wiz('WizardLMTeam/WizardCoder-Python-34B-V1.0', 'WizardCoder-Python-34B', 'WizardCoder Python 34B V1.0', 'WizardCoder 34B'), WCODER, '2023-08-26', '2023-08-26', weights='2023-08-26', params=(33_700 * M_, 33_700 * M_))
S('WizardCoder 33B', wiz('WizardLMTeam/WizardCoder-33B-V1.1', 'WizardCoder-33B', 'WizardCoder 33B V1.1'), WCODER, '2024-01-04', '2024-01-04', weights='2024-01-04', params=(33 * B, 33 * B))
M('WizardMath 7B',
	V('WizardMath 7B V1.0', wiz('WizardLMTeam/WizardMath-7B-V1.0'), WMATH, '2023-08-11', '2023-08-11', weights='2023-08-11', params=(6_700 * M_, 6_700 * M_)),
	V('WizardMath 7B V1.1', wiz('WizardLMTeam/WizardMath-7B-V1.1'), WMATH, '2023-12-19', '2023-12-19', weights='2023-12-19', params=(7_240 * M_, 7_240 * M_)),
)
S('WizardMath 13B', wiz('WizardLMTeam/WizardMath-13B-V1.0', 'WizardMath-13B', 'WizardMath 13B V1.0'), WMATH, '2023-08-11', '2023-08-11', weights='2023-08-11', params=(13 * B, 13 * B))
S('WizardMath 70B', wiz('WizardLMTeam/WizardMath-70B-V1.0', 'WizardMath-70B', 'WizardMath 70B V1.0'), WMATH, '2023-08-11', '2023-08-11', weights='2023-08-11', params=(69 * B, 69 * B))

# WizardLM-2 (weights published then withdrawn the same day for a missing toxicity test; the 70B was never released)
WIZ2 = 'https://wizardlm.github.io/WizardLM2/'
S('WizardLM-2 8x22B', names('WizardLM-2-8x22B', 'WizardLM 2 8x22B', 'microsoft/wizardlm-2-8x22b', hf('microsoft/WizardLM-2-8x22B')), WIZ2, '2024-04-15', '2024-04-15', weights='2024-04-15', params=(140_600 * M_, 39 * B))
S('WizardLM-2 70B', ['WizardLM-2-70B', 'WizardLM 2 70B'], WIZ2, '2024-04-15', None, params=(70 * B, 70 * B))
S('WizardLM-2 7B', names('WizardLM-2-7B', 'WizardLM 2 7B', 'microsoft/wizardlm-2-7b', hf('microsoft/WizardLM-2-7B')), WIZ2, '2024-04-15', '2024-04-15', weights='2024-04-15', params=(7_240 * M_, 7_240 * M_))

# Orca (explanation-tuned LLaMA / Llama 2 / Mistral models)
S('Orca 13B', ['Orca', 'Orca 1', 'Orca-13B'], ARXIV + '2306.02707', '2023-06-05', None, params=(13 * B, 13 * B))
ORCA2 = MSR + 'orca-2-teaching-small-language-models-how-to-reason/'
S('Orca 2 7B', names('Orca-2-7B', 'Orca 2 (7B)', 'Orca 2-7B', hf('microsoft/Orca-2-7b')), ORCA2, '2023-11-18', '2023-11-20', weights='2023-11-20', params=(6_740 * M_, 6_740 * M_))
S('Orca 2 13B', names('Orca-2-13B', 'Orca 2 (13B)', 'Orca 2-13B', 'Orca 2', hf('microsoft/Orca-2-13b')), ORCA2, '2023-11-18', '2023-11-20', weights='2023-11-20', params=(13 * B, 13 * B))
S('Orca-Math', ['Orca-Math 7B', 'Orca Math', 'Orca-Math-7B'], ARXIV + '2402.14830', '2024-02-16', None, params=(7_240 * M_, 7_240 * M_))

# phi-1 to phi-2
S('phi-1', names('Phi-1', 'phi-1 1.3B', hf('microsoft/phi-1')), ARXIV + '2306.11644', '2023-06-20', '2023-09-11', weights='2023-09-11', params=(1_300 * M_, 1_300 * M_))
PHI15 = ARXIV + '2309.05463'
M('phi-1.5',
	V('phi-1.5', names('Phi-1.5', 'phi-1.5 (1.3B)', 'phi-1_5', 'microsoft/phi-1_5'), PHI15, '2023-09-11', '2023-09-11', weights='2023-09-11', params=(1_420 * M_, 1_420 * M_)),
	V('phi-1.5-web', ['phi-1.5-web (1.3B)', 'Phi-1.5-web'], PHI15, '2023-09-11', None, params=(1_300 * M_, 1_300 * M_)),
	V('phi-1.5-web-only', ['phi-1.5-web-only (1.3B)', 'Phi-1.5-web-only'], PHI15, '2023-09-11', None, params=(1_300 * M_, 1_300 * M_)),
)
S('phi-2', names('Phi-2', 'Phi-2 2.7B', 'microsoft/phi-2'), MSR + 'phi-2-the-surprising-power-of-small-language-models/', '2023-11-15', '2023-12-12', weights='2023-12-12', params=(2_780 * M_, 2_780 * M_))

# Phi-3 (Phi-3-vision is built on Phi-3-mini; the June 2024 updates reuse the same repo ids and are not separated)
PHI3 = AZURE + 'introducing-phi-3-redefining-whats-possible-with-slms/'
PHI3_PAPER = ARXIV + '2404.14219'
PHI3_BUILD = AZURE + 'new-models-added-to-the-phi-3-family-available-on-microsoft-azure/'
P_MINI3 = (3_820 * M_, 3_820 * M_)
M('Phi-3-mini',
	V('Phi-3-mini-4k-instruct', names('Phi-3-mini', 'Phi-3-mini 4k', 'Phi-3 Mini', 'Phi-3 Mini 4k', 'phi-3-mini 3.8B', 'Phi-3-mini 3.8b', 'phi-3-mini-4k-instruct', hf('microsoft/Phi-3-mini-4k-instruct')),
		PHI3, '2024-04-22', '2024-04-23', weights='2024-04-23', params=P_MINI3, price=(0.13, 0.52)),
	V('Phi-3-mini-128k-instruct', names('Phi-3-mini 128k', 'Phi-3 Mini 128k', 'phi-3-mini-128k-instruct', 'microsoft/phi-3-mini-128k-instruct', hf('microsoft/Phi-3-mini-128k-instruct')),
		PHI3, '2024-04-22', '2024-04-23', weights='2024-04-23', params=P_MINI3, price=(0.13, 0.52)),
	V('Phi-3-vision-128k-instruct', names('Phi-3-vision', 'Phi-3 Vision', hf('microsoft/Phi-3-vision-128k-instruct')), PHI3_BUILD, '2024-05-21', '2024-05-21', weights='2024-05-21', params=(4_150 * M_, 4_150 * M_)),
)
P_SMALL3 = (7_390 * M_, 7_390 * M_)
M('Phi-3-small',
	V('Phi-3-small-8k-instruct', names('Phi-3-small', 'Phi-3-small 8k', 'Phi-3 Small', 'phi-3-small 7.4B', 'Phi-3-small 7b', 'Phi-3 (7B)', 'phi-3-small-8k-instruct', hf('microsoft/Phi-3-small-8k-instruct')),
		PHI3_PAPER, '2024-04-22', '2024-05-21', weights='2024-05-21', params=P_SMALL3, price=(0.15, 0.6)),
	V('Phi-3-small-128k-instruct', names('Phi-3-small 128k', 'Phi-3 Small 128k', hf('microsoft/Phi-3-small-128k-instruct')), PHI3_PAPER, '2024-04-22', '2024-05-21', weights='2024-05-21', params=P_SMALL3, price=(0.15, 0.6)),
)
P_MEDIUM3 = (13_960 * M_, 13_960 * M_)
M('Phi-3-medium',
	V('Phi-3-medium-4k-instruct', names('Phi-3-medium', 'Phi-3-medium 4k', 'Phi-3 Medium', 'Phi-3 Medium 14B', 'phi-3-medium 14B', 'Phi-3-medium 14b', 'Phi-3 (14B)', 'Phi-3 14b',
			'microsoft/phi-3-medium-4k-instruct', hf('microsoft/Phi-3-medium-4k-instruct')), PHI3_PAPER, '2024-04-22', '2024-05-21', weights='2024-05-21', params=P_MEDIUM3, price=(0.17, 0.68)),
	V('Phi-3-medium-128k-instruct', names('Phi-3-medium 128k', 'Phi-3 Medium 128k', 'microsoft/phi-3-medium-128k-instruct', hf('microsoft/Phi-3-medium-128k-instruct')),
		PHI3_PAPER, '2024-04-22', '2024-05-21', weights='2024-05-21', params=P_MEDIUM3, price=(0.17, 0.68)),
)

# Phi-3.5 (Phi-3.5-vision is built on Phi-3.5-mini) and GRIN-MoE (same 16x3.8B layout as Phi-3.5-MoE, trained separately)
M('Phi-3.5-mini',
	V('Phi-3.5-mini-instruct', names('Phi-3.5-mini', 'Phi-3.5 Mini', 'Phi 3.5 Mini', 'phi-3.5-mini-instruct', 'microsoft/phi-3.5-mini-128k-instruct', hf('microsoft/Phi-3.5-mini-instruct')),
		HF + 'microsoft/Phi-3.5-mini-instruct', '2024-08-20', '2024-08-20', weights='2024-08-20', params=P_MINI3, price=(0.13, 0.52)),
	V('Phi-3.5-vision-instruct', names('Phi-3.5-vision', 'Phi-3.5 Vision', 'Phi-3.5-Vision 4.2B', 'phi-3.5-vision-instruct', hf('microsoft/Phi-3.5-vision-instruct')),
		HF + 'microsoft/Phi-3.5-vision-instruct', '2024-08-20', '2024-08-20', weights='2024-08-20', params=(4_150 * M_, 4_150 * M_), price=(0.13, 0.52)),
)
S('Phi-3.5-MoE', names('Phi-3.5-MoE-instruct', 'Phi-3.5 MoE', 'Phi 3.5 MoE', 'phi-3.5-moe-instruct', hf('microsoft/Phi-3.5-MoE-instruct')),
	HF + 'microsoft/Phi-3.5-MoE-instruct', '2024-08-20', '2024-08-20', weights='2024-08-20', params=(41_900 * M_, 6_600 * M_), price=(0.16, 0.64))
S('GRIN-MoE', names('GRIN MoE', 'GRIN', 'microsoft/GRIN-MoE'), ARXIV + '2409.12136', '2024-09-18', '2024-09-18', weights='2024-09-18', params=(41_870 * M_, 6_600 * M_))

# Phi-4 (reasoning and vision-reasoning fine-tunes of Phi-4 are versions; weights came a month after the Foundry release)
ONE_YEAR = AZURE + 'one-year-of-phi-small-language-models-making-big-leaps-in-ai/'
P_PHI4 = (14_700 * M_, 14_700 * M_)
M('Phi-4',
	V('Phi-4', names('phi-4', 'Phi 4', 'Phi-4 14B', 'Phi-4 14b', 'microsoft/phi-4'), ARXIV + '2412.08905', '2024-12-12', '2024-12-13', weights='2025-01-08', params=P_PHI4, price=(0.125, 0.5)),
	V('Phi-4-reasoning', names('Phi-4 Reasoning', 'Phi 4 Reasoning', 'Phi-4-Reasoning', 'phi-4-reasoning', 'microsoft/phi-4-reasoning', hf('microsoft/Phi-4-reasoning')),
		ONE_YEAR, '2025-04-30', '2025-04-30', weights='2025-04-30', params=P_PHI4, price=(0.125, 0.5)),
	V('Phi-4-reasoning-plus', names('Phi-4 Reasoning Plus', 'Phi 4 Reasoning Plus', 'Phi-4-Reasoning-plus', 'Phi-4-reasoning+', 'phi-4-reasoning-plus', 'microsoft/phi-4-reasoning-plus', hf('microsoft/Phi-4-reasoning-plus')),
		ONE_YEAR, '2025-04-30', '2025-04-30', weights='2025-04-30', params=P_PHI4, price=(0.125, 0.5)),
	V('Phi-4-reasoning-vision-15B', names('Phi-4-reasoning-vision', 'Phi-4 Reasoning Vision', 'Phi-4-Reasoning-Vision-15B', hf('microsoft/Phi-4-reasoning-vision-15B')),
		HF + 'microsoft/Phi-4-reasoning-vision-15B', '2026-03-04', '2026-03-04', weights='2026-03-04', params=(15_120 * M_, 15_120 * M_)),
)

# Phi-4-mini (Phi-4-multimodal adds vision and audio LoRA adapters to the frozen Phi-4-mini, Phi-4-mini-reasoning is a fine-tune)
PHI4_MINI = AZURE + 'empowering-innovation-the-next-generation-of-the-phi-family/'
P_MINI4 = (3_840 * M_, 3_840 * M_)
M('Phi-4-mini',
	V('Phi-4-mini', names('Phi-4-mini-instruct', 'Phi-4 Mini', 'Phi 4 Mini', 'Phi-4 Mini Instruct', 'phi-4-mini', 'microsoft/phi-4-mini-instruct', hf('microsoft/Phi-4-mini-instruct')),
		PHI4_MINI, '2025-02-26', '2025-02-26', weights='2025-02-26', params=P_MINI4, price=(0.075, 0.3)),
	V('Phi-4-multimodal', names('Phi-4-multimodal-instruct', 'Phi-4 Multimodal', 'Phi-4-Multimodal', 'microsoft/phi-4-multimodal-instruct', hf('microsoft/Phi-4-multimodal-instruct')),
		PHI4_MINI, '2025-02-26', '2025-02-26', weights='2025-02-26', params=(5_570 * M_, 5_570 * M_), price=(0.08, 0.32)),
	V('Phi-4-mini-reasoning', names('Phi-4 Mini Reasoning', 'Phi 4 Mini Reasoning', 'phi-4-mini-reasoning', hf('microsoft/Phi-4-mini-reasoning')),
		ONE_YEAR, '2025-04-30', '2025-04-30', weights='2025-04-30', params=P_MINI4),
)
M('Phi-4-mini-flash',	# SambaY hybrid architecture, trained separately from Phi-4-mini
	V('Phi-4-mini-flash-reasoning', names('Phi-4 Mini Flash Reasoning', 'Phi-4-mini-flash', hf('microsoft/Phi-4-mini-flash-reasoning')),
		HF + 'microsoft/Phi-4-mini-flash-reasoning', '2025-07-09', '2025-07-09', weights='2025-07-09', params=(3_850 * M_, 3_850 * M_)),
)

# SlimMoE (Phi-3.5-MoE compressed into two smaller MoE models)
SLIMMOE = ARXIV + '2506.18349'
S('Phi-mini-MoE', names('Phi-mini-MoE-instruct', 'Phi mini MoE', hf('microsoft/Phi-mini-MoE-instruct')), SLIMMOE, '2025-06-23', '2025-06-23', weights='2025-06-23', params=(7_600 * M_, 2_400 * M_))
S('Phi-tiny-MoE', names('Phi-tiny-MoE-instruct', 'Phi tiny MoE', hf('microsoft/Phi-tiny-MoE-instruct')), SLIMMOE, '2025-06-23', '2025-06-23', weights='2025-06-23', params=(3_800 * M_, 1_100 * M_))

# BitNet (native 1.58-bit model)
S('BitNet b1.58 2B4T', names('BitNet b1.58 2B', 'BitNet-b1.58-2B-4T', 'BitNet 2B4T', hf('microsoft/bitnet-b1.58-2B-4T')), ARXIV + '2504.12285', '2025-04-16', '2025-04-16', weights='2025-04-16', params=(2 * B, 2 * B))

# MAI (Microsoft AI)
S('MAI-DS-R1', names('MAI DS R1', 'mai-ds-r1', 'microsoft/mai-ds-r1', 'microsoft/MAI-DS-R1'), HF + 'microsoft/MAI-DS-R1', '2025-04-16', '2025-04-16', weights='2025-04-16', params=(671 * B, 37 * B), price=(1.35, 5.4))
S('MAI-1-preview', ['MAI-1 Preview', 'MAI 1 Preview', 'mai-1-preview', 'MAI-1'], MAI + 'news/two-new-in-house-models/', '2025-08-28', '2025-08-28')
M('MAI-Thinking-1',	# Private preview on Foundry at Build, public preview on 2026-08-12
	V('MAI-Thinking-1', ['MAI Thinking 1', 'mai-thinking-1', 'mai-thinking-1-latest', 'microsoft/mai-thinking-1'], MAI + 'models/mai-thinking-1/', '2026-06-02', '2026-08-12', params=P_T1, price=(2, 8)),
	V('MAI-Base-1', ['MAI Base 1'], MAI + 'wp-content/uploads/2026/06/main_20260602_2.pdf', '2026-06-02', None, params=P_T1),
)
M('MAI-Code-1-Flash',	# Starts from MAI-Thinking-1's mid-training checkpoint; prices are GitHub Copilot's
	V('MAI-Code-1-Flash', ['MAI Code 1 Flash', 'mai-code-1-flash'], MAI + 'news/introducingmai-code-1-flash/', '2026-06-02', '2026-06-02', params=P_C1, price=(0.75, 4.5)),
	V('MAI-Cyber-1-Flash', ['MAI Cyber 1 Flash', 'mai-cyber-1-flash'], MAI + 'news/introducing-mai-cyber-1-flash-inside-mdash/', '2026-08-13', None, params=P_C1),
	V('MAI-Cyber-1.1-Flash', ['MAI Cyber 1.1 Flash', 'mai-cyber-1.1-flash'], MAI + 'news/mai-cyber-1-1-flash/', '2026-10-27', None, params=P_C1),
)
S('MAI-Code-1.1-Flash', ['MAI Code 1.1 Flash', 'mai-code-1.1-flash'], MAI + 'news/mai-code-1-1-flash-br-better-faster-at-a-quarter-of-the-cost/', '2026-08-11', '2026-08-11', price=(0.2, 1.2))
S('MAI-Base-2', ['MAI Base 2'], MAI + 'pdf/Frontier-Pre-training-at-Microsoft-AI.pdf', '2026-10-15', None, params=(2_800 * B, 80 * B))

write('microsoft')
