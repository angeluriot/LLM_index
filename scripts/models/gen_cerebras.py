"""Generates data/models/cerebras/*.json (one file per model).

Covers the language models released by Cerebras Systems: Cerebras-GPT, BTLM, Cerebras-LLaVA, DocChat, the CBHybrid sparse
attention Llamas and the REAP expert-pruned MoE checkpoints.
All the data lives in this script: edit it and run `python scripts/models/gen_cerebras.py`.
Existing JSON files are deleted and rewritten, so manual edits to them are lost.

Each size is a model. BTLM-3B-8K-chat (DPO on the base) is a version of BTLM-3B-8K; the two CBHybrid variants (25 or 28 of
32 attention layers made sparse in Llama-3.1-8B-Instruct) are versions of one model. Every REAP checkpoint is its own model,
since pruning experts changes the size; its FP8 release is only a name and its effort modes are those of the pruned model.
Cerebras' inference API only serves third-party models and no provider lists these models, so prices are unknown.
Jais (Inception / G42 and MBZUAI, trained on Cerebras hardware) belongs to Inception, the Cerebras-ViT vision encoders and the
Dragon-DocChat retrievers are not LLMs, and unreleased research models (Sparse Wide GPT-3) are out of scope.
"""

from common import B, M, N, S, V, write

M_ = 1_000_000
T = ['', 'Thinking']	# Thinking switch of the hybrid GLM and DeepSeek bases

ARXIV = 'https://arxiv.org/abs/'
BLOG = 'https://www.cerebras.ai/blog/'
HF = 'https://huggingface.co/cerebras/'


def names(*items):
	"""Flattens names and lists of names, without duplicates."""
	out = []
	for i in items:
		out += [i] if isinstance(i, str) else list(i)
	return list(dict.fromkeys(out))


def hf(*repos):
	"""Cerebras Hugging Face repo ids, both bare and with the cerebras organisation."""
	return list(repos) + ['cerebras/' + r for r in repos]


# Cerebras-GPT (compute-optimal GPT-3 style models trained on the Pile)
GPT = 'https://www.cerebras.ai/press-release/cerebras-systems-releases-seven-new-gpt-models-trained-on-cs-2-wafer-scale-systems'
for size, p in (('111M', 111 * M_), ('256M', 256 * M_), ('590M', 590 * M_), ('1.3B', 1_300 * M_), ('2.7B', 2_700 * M_), ('6.7B', 6_700 * M_), ('13B', 13 * B)):
	repo = f'Cerebras-GPT-{size}'
	S(repo, names(f'Cerebras-GPT {size}', f'Cerebras GPT {size}', f'Cerebras-GPT ({size})', hf(repo)), GPT, '2023-03-28', '2023-03-28', weights='2023-03-28', params=(p, p))

# BTLM-3B-8K (with Opentensor) and its DPO chat version
P_BTLM = (2_600 * M_, 2_600 * M_)
M('BTLM-3B-8K',
	V('BTLM-3B-8K-base', names('BTLM-3B-8K', 'BTLM-3B-8k-base', 'BTLM 3B 8K', 'BTLM-3B', 'BTLM', hf('btlm-3b-8k-base')), BLOG + 'btlm-3b-8k-7b-performance-in-a-3-billion-parameter-model', '2023-07-24', '2023-07-24', weights='2023-07-24', params=P_BTLM),
	V('BTLM-3B-8K-chat', names('BTLM-Chat', 'BTLM-3B-8k-chat', 'BTLM 3B 8K Chat', 'BTLM-3B-8K-DPO', hf('btlm-3b-8k-chat', 'btlm-3b-8k-dpo')), BLOG + 'fine-tuning-language-models-using-direct-preference-optimization', '2023-12-29', '2023-12-29', weights='2023-12-29', params=P_BTLM),
)

# Cerebras-LLaVA (LLaVA recipe on Vicuna, trained on Cerebras systems)
S('Cerebras-LLaVA-7B', names('Cerebras-LLaVA 7B', 'Cerebras LLaVA 7B', hf('Cerebras-LLaVA-7B')), HF + 'Cerebras-LLaVA-7B', '2024-03-19', '2024-03-19', weights='2024-03-19', params=(7 * B, 7 * B))
S('Cerebras-LLaVA-13B', names('Cerebras-LLaVA 13B', 'Cerebras LLaVA 13B', hf('Cerebras-LLaVA-13B')), HF + 'Cerebras-LLaVA-13B', '2024-03-19', '2024-03-19', weights='2024-03-19', params=(13 * B, 13 * B))

# DocChat (document QA fine-tune of Llama 3 8B)
S('Llama3-DocChat-1.0-8B', names('Llama3-DocChat 1.0 8B', 'Cerebras Llama3-DocChat 1.0 8B', 'Llama3-DocChat', 'Cerebras DocChat', hf('Llama3-DocChat-1.0-8B')),
	BLOG + 'train-a-gpt-4-level-conversational-qa-in-a-few-hours', '2024-08-21', '2024-08-21', weights='2024-08-21', params=(8_030 * M_, 8_030 * M_))

# CBHybrid (hybrid dense / sparse attention versions of Llama-3.1-8B-Instruct)
HYBRID = BLOG + 'compressing-kv-cache-memory-by-half-with-sparse-attention'
P_HYBRID = (8_030_261_248, 8_030_261_248)
M('Llama-3-CBHybrid-8B',
	V('Llama-3-CBHybridL-8B', names('Llama3.1-CBHybridL-8B', 'Llama 3 CBHybridL 8B', hf('Llama-3-CBHybridL-8B')), HYBRID, '2025-03-24', '2025-03-24', weights='2025-03-24', params=P_HYBRID),
	V('Llama-3-CBHybridM-8B', names('Llama3.1-CBHybridM-8B', 'Llama 3 CBHybridM 8B', hf('Llama-3-CBHybridM-8B')), HYBRID, '2025-03-24', '2025-03-24', weights='2025-03-24', params=P_HYBRID),
)

# REAP (router-weighted expert activation pruning of third-party MoE models)
REAP = ARXIV + '2510.13999'


def reap(repo, date, total, active, efforts=N, fp8=True, bf16=True):
	"""A REAP checkpoint; its id is the slug of its Hugging Face repo name."""
	repos = ([repo] if bf16 else []) + ([repo + '-FP8'] if fp8 else [])
	S(repo, names(repo.replace('-', ' '), hf(*repos)), HF + repos[0], date, date, weights=date, params=(total, active), efforts=efforts)


reap('Qwen3-Coder-REAP-246B-A35B', '2025-10-15', 246_082_237_952, 35 * B)
reap('Qwen3-Coder-REAP-363B-A35B', '2025-10-15', 363_118_556_672, 35 * B)
reap('Qwen3-Coder-REAP-25B-A3B', '2025-10-20', 24_867_354_624, 3 * B, fp8=False)
reap('GLM-4.5-Air-REAP-82B-A12B', '2025-10-20', 81_932_181_504, 12 * B, efforts=T)
reap('GLM-4.6-REAP-218B-A32B', '2025-10-23', 218_383_151_104, 32 * B, efforts=T)
reap('GLM-4.6-REAP-252B-A32B', '2025-10-23', 251_986_817_024, 32 * B, efforts=T)
reap('GLM-4.6-REAP-268B-A32B', '2025-10-23', 268_788_649_984, 32 * B, efforts=T)
reap('Kimi-Linear-REAP-35B-A3B-Instruct', '2025-11-06', 35_132_220_360, 3 * B, fp8=False)
reap('MiniMax-M2-REAP-139B-A10B', '2025-11-17', 139_149_203_788, 10 * B, fp8=False)
reap('MiniMax-M2-REAP-162B-A10B', '2025-11-15', 161_973_268_376, 10 * B, fp8=False)
reap('MiniMax-M2-REAP-172B-A10B', '2025-11-15', 172_517_891_840, 10 * B, fp8=False)
reap('DeepSeek-V3.2-REAP-345B-A37B', '2025-12-09', 344_891_268_784, 37 * B, efforts=T, fp8=False)
reap('DeepSeek-V3.2-REAP-508B-A37B', '2025-12-09', 508_405_050_672, 37 * B, efforts=T, fp8=False)
reap('GLM-4.7-REAP-218B-A32B', '2026-01-10', 218_383_159_648, 32 * B, efforts=T)
reap('GLM-4.7-REAP-268B-A32B', '2026-01-10', 268_788_660_664, 32 * B, efforts=T)
reap('GLM-4.7-Flash-REAP-23B-A3B', '2026-01-23', 22_996_118_432, 3 * B, efforts=T, fp8=False)
reap('MiniMax-M2.1-REAP-139B-A10B', '2026-01-27', 139_157_619_916, 10 * B, fp8=False)
reap('MiniMax-M2.1-REAP-172B-A10B', '2026-01-27', 172_517_903_744, 10 * B, fp8=False)
reap('MiniMax-M2.5-REAP-139B-A10B', '2026-02-18', 139_149_203_788, 10 * B, fp8=False)
reap('MiniMax-M2.5-REAP-172B-A10B', '2026-02-18', 172_517_903_744, 10 * B, fp8=False)
reap('Step-3.5-Flash-REAP-121B-A11B', '2026-02-25', 120_967_010_658, 11 * B, fp8=False)
reap('Step-3.5-Flash-REAP-149B-A11B', '2026-02-25', 149_380_333_680, 11 * B, fp8=False)

write('cerebras')
