"""Generates data/models/databricks/*.json (one file per model).

Covers the language models of Databricks, including those of MosaicML (acquired in July 2023, now Databricks Mosaic Research):
Dolly, MPT and DBRX.
All the data lives in this script: edit it and run `python scripts/models/gen_databricks.py`.
Existing JSON files are deleted and rewritten, so manual edits to them are lost.

Each size is a model; its base checkpoint and the instruct / chat / long-context variants fine-tuned from it are its versions.
MPT-7B-8k (MPT-7B continued for 500B tokens at 8k context) is a separate model. Dolly v1 (on GPT-J 6B) and Dolly 2.0 (on
Pythia 2.8B / 6.9B / 12B) are fine-tunes of EleutherAI bases, one model per size.
Prices are the Databricks Foundation Model APIs pay-per-token prices of the models it served (the Instruct versions).
The Hugging Face repos of Dolly, MPT and DBRX are no longer public. MosaicBERT (encoder), the MosaicML diffusion models and
Shutterstock ImageAI (image generation) are out of scope.
"""

from common import B, M, N, S, V, write

M_ = 1_000_000

BLOG = 'https://www.databricks.com/blog/'


def names(*items):
	"""Flattens names and lists of names, without duplicates."""
	out = []
	for i in items:
		out += [i] if isinstance(i, str) else list(i)
	return list(dict.fromkeys(out))


def hf(org, *repos):
	"""Hugging Face repo ids, both bare and with their organisation."""
	return list(repos) + [f'{org}/{r}' for r in repos]


# Dolly (instruction fine-tunes of EleutherAI's GPT-J and Pythia)
S('Dolly v1 6B', names('Dolly 1.0', 'Dolly v1', 'Dolly-v1-6B', 'dolly-v1-6b', 'databricks/dolly-v1-6b'), BLOG + '2023/03/24/hello-dolly-democratizing-magic-chatgpt-open-models.html',
	'2023-03-24', '2023-03-24', weights='2023-03-24', params=(6_053 * M_, 6_053 * M_))
DOLLY2 = BLOG + '2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm'
for size, p, extra in (('3B', 2_775 * M_, []), ('7B', 6_857 * M_, []), ('12B', 11_846 * M_, ['Dolly 2.0', 'Dolly v2', 'Dolly 2.0-12b'])):
	repo = f'dolly-v2-{size.lower()}'
	S(f'Dolly v2 {size}', names(extra, f'Dolly 2.0 {size}', f'Dolly 2.0 ({size})', f'Dolly-v2-{size}', hf('databricks', repo)), DOLLY2, '2023-04-12', '2023-04-12', weights='2023-04-12', params=(p, p))

# MPT (MosaicML Pretrained Transformer)
M('MPT-1B-RedPajama-200B',
	V('MPT-1b-RedPajama-200b', names('MPT-1B-RedPajama-200B', 'MPT 1B RedPajama', 'MPT-1B', hf('mosaicml', 'mpt-1b-redpajama-200b')), 'https://huggingface.co/mosaicml/mpt-1b-redpajama-200b',
		'2023-04-20', '2023-04-20', id='mpt-1b-redpajama-200b', weights='2023-04-20', params=(1_300 * M_, 1_300 * M_)),
	V('MPT-1b-RedPajama-200b-Dolly', names('MPT-1B-RedPajama-200B-Dolly', 'MPT 1B RedPajama Dolly', hf('mosaicml', 'mpt-1b-redpajama-200b-dolly')), 'https://huggingface.co/mosaicml/mpt-1b-redpajama-200b-dolly',
		'2023-04-20', '2023-04-20', id='mpt-1b-redpajama-200b-dolly', weights='2023-04-20', params=(1_300 * M_, 1_300 * M_)),
)
MPT7 = BLOG + 'mpt-7b'
P_MPT7 = (6_649 * M_, 6_649 * M_)
M('MPT-7B',
	V('MPT-7B', names('MPT 7B', 'MPT-7B Base', 'MosaicML MPT-7B', hf('mosaicml', 'mpt-7b')), MPT7, '2023-05-05', '2023-05-05', weights='2023-05-05', params=P_MPT7),
	V('MPT-7B-Instruct', names('MPT 7B Instruct', 'MPT-7B Instruct', 'databricks-mpt-7b-instruct', hf('mosaicml', 'mpt-7b-instruct')), MPT7, '2023-05-05', '2023-05-05', weights='2023-05-05', params=P_MPT7, price=(0.5, 0.5)),
	V('MPT-7B-Chat', names('MPT 7B Chat', 'MPT-7B Chat', hf('mosaicml', 'mpt-7b-chat')), MPT7, '2023-05-05', '2023-05-05', weights='2023-05-05', params=P_MPT7),
	V('MPT-7B-StoryWriter-65k+', names('MPT-7B-StoryWriter', 'MPT 7B StoryWriter', 'MPT-7B StoryWriter-65k+', hf('mosaicml', 'mpt-7b-storywriter')), MPT7, '2023-05-05', '2023-05-05', id='mpt-7b-storywriter', weights='2023-05-05', params=P_MPT7),
)
MPT30 = BLOG + 'mpt-30b'
P_MPT30 = (29_950 * M_, 29_950 * M_)
M('MPT-30B',
	V('MPT-30B', names('MPT 30B', 'MPT-30B Base', hf('mosaicml', 'mpt-30b')), MPT30, '2023-06-22', '2023-06-22', weights='2023-06-22', params=P_MPT30),
	V('MPT-30B-Instruct', names('MPT 30B Instruct', 'MPT-30B Instruct', 'databricks-mpt-30b-instruct', hf('mosaicml', 'mpt-30b-instruct')), MPT30, '2023-06-22', '2023-06-22', weights='2023-06-22', params=P_MPT30, price=(1, 1)),
	V('MPT-30B-Chat', names('MPT 30B Chat', 'MPT-30B Chat', hf('mosaicml', 'mpt-30b-chat')), MPT30, '2023-06-22', '2023-06-22', weights='2023-06-22', params=P_MPT30),
)
MPT7_8K = BLOG + 'long-context-mpt-7b-8k'
M('MPT-7B-8k',
	V('MPT-7B-8k', names('MPT 7B 8k', 'MPT-7B-8K', hf('mosaicml', 'mpt-7b-8k')), MPT7_8K, '2023-07-18', '2023-07-18', weights='2023-07-18', params=P_MPT7),
	V('MPT-7B-8k-Instruct', names('MPT 7B 8k Instruct', 'MPT-7B-8K-Instruct', hf('mosaicml', 'mpt-7b-8k-instruct')), MPT7_8K, '2023-07-18', '2023-07-18', weights='2023-07-18', params=P_MPT7),
	V('MPT-7B-8k-Chat', names('MPT 7B 8k Chat', 'MPT-7B-8K-Chat', hf('mosaicml', 'mpt-7b-8k-chat')), MPT7_8K, '2023-07-18', '2023-07-18', weights='2023-07-18', params=P_MPT7),
)

# DBRX (132B fine-grained MoE)
DBRX = BLOG + 'introducing-dbrx-new-state-art-open-llm'
P_DBRX = (132 * B, 36 * B)
M('DBRX',
	V('DBRX Base', names('DBRX-Base', 'dbrx-base', hf('databricks', 'dbrx-base')), DBRX, '2024-03-27', '2024-03-27', weights='2024-03-27', params=P_DBRX),
	V('DBRX Instruct', names('DBRX', 'DBRX-Instruct', 'DBRX 132B Instruct', 'dbrx-instruct', 'databricks-dbrx-instruct', hf('databricks', 'dbrx-instruct')), DBRX, '2024-03-27', '2024-03-27', weights='2024-03-27', params=P_DBRX, price=(0.75, 2.25)),
)

write('databricks')
