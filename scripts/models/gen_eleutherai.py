"""Generates data/models/eleutherai/*.json (one file per model).

Covers the language models of EleutherAI: GPT-Neo, GPT-J, GPT-NeoX-20B, Polyglot-Ko, Pythia, Llemma, Pile-T5, Comma (Common Pile)
and the Deep Ignorance research suite.
All the data lives in this script: edit it and run `python scripts/models/gen_eleutherai.py`.
Existing JSON files are deleted and rewritten, so manual edits to them are lost.

Each size is a model; Pythia's standard and deduplicated-Pile runs are versions of the same size, as are the 1T and 2T token runs
of Comma v0.1 and the Deep Ignorance 6.9B runs (unfiltered, filtered pretraining / annealing, circuit breakers).
EleutherAI has no API and Artificial Analysis lists none of these models, so prices are unknown.
Out of scope: the original Pythia v0 runs (Oct 2022, superseded), the PolyPythias extra seeds, intermediate checkpoints and
ablations, sparse autoencoders / transcoders / probes, toy and test models, and models where EleutherAI was one contributor
among many (RWKV, YaRN, SantaCoder / StarCoder, which belong to their lead organisations).
"""

from common import B, M, N, S, V, write

M_ = 1_000_000

ARXIV = 'https://arxiv.org/abs/'
BLOG = 'https://blog.eleuther.ai/'
HF = 'https://huggingface.co/EleutherAI/'


def names(*items):
	"""Flattens names and lists of names, without duplicates."""
	out = []
	for i in items:
		out += [i] if isinstance(i, str) else list(i)
	return list(dict.fromkeys(out))


def hf(*repos):
	"""Repo ids, both bare and with the EleutherAI organisation (unless one is given)."""
	return list(dict.fromkeys([r.split('/')[-1] for r in repos] + [r if '/' in r else 'EleutherAI/' + r for r in repos]))


# GPT-Neo (Mesh TensorFlow replications of GPT-3 trained on the Pile)
NEO = 'https://github.com/EleutherAI/gpt-neo'
for size, p in (('125M', 125_198_592), ('1.3B', 1_315_575_808), ('2.7B', 2_651_307_520)):
	S(f'GPT-Neo {size}', names(f'GPT-Neo-{size}', f'GPT-Neo ({size})', f'GPT Neo {size}', hf(f'gpt-neo-{size.lower() if size == "125M" else size}')), NEO, '2021-03-21', '2021-03-21', weights='2021-03-21', params=(p, p))

# GPT-J-6B (Mesh Transformer JAX, trained on TPUs)
S('GPT-J-6B', names('GPT-J', 'GPT-J 6B', 'GPT-J (6B)', 'GPT-J-6b', hf('gpt-j-6b')), 'https://github.com/kingoflolz/mesh-transformer-jax', '2021-06-04', '2021-06-04', weights='2021-06-04', params=(6_053_381_344, 6_053_381_344))

# GPT-NeoX-20B (first served by GooseAI, weights a week later)
S('GPT-NeoX-20B', names('GPT-NeoX', 'GPT-NeoX 20B', 'GPT-NeoX (20B)', 'GPT NeoX 20B', hf('gpt-neox-20b')), BLOG + 'announcing-20b/', '2022-02-02', '2022-02-02', weights='2022-02-09', params=(20_554_568_208, 20_554_568_208))

# Polyglot-Ko (Korean GPT-NeoX models)
POLYGLOT = ARXIV + '2306.02254'
for size, date, p in (('1.3B', '2022-09-15', 1_331_810_304), ('3.8B', '2022-09-09', 3_809_974_272), ('5.8B', '2022-09-22', 5_885_059_072), ('12.8B', '2022-10-14', 12_893_603_840)):
	S(f'Polyglot-Ko {size}', names(f'Polyglot-Ko-{size}', f'Polyglot-Ko ({size})', f'polyglot-ko-{size.lower()}', f'EleutherAI/polyglot-ko-{size.lower()}'), HF + f'polyglot-ko-{size.lower()}', date, date, weights=date, params=(p, p))

# Pythia (suite trained on the Pile and on the deduplicated Pile)
PYTHIA = ARXIV + '2304.01373'
for size, p, date in (('14M', 14_067_712, '2023-07-19'), ('31M', 30_494_720, '2023-07-19'), ('70M', 70_426_672, '2023-02-13'), ('160M', 162_323_040, '2023-02-13'),
		('410M', 405_334_208, '2023-02-13'), ('1B', 1_011_782_144, '2023-02-13'), ('1.4B', 1_414_648_192, '2023-02-13'), ('2.8B', 2_775_209_280, '2023-02-13'),
		('6.9B', 6_857_302_528, '2023-02-13'), ('12B', 11_846_072_896, '2023-02-28')):
	repo = f'pythia-{size.lower()}'
	M(f'Pythia-{size}',
		V(f'Pythia-{size}', names(f'Pythia {size}', f'Pythia-{size.lower()}', f'Pythia ({size})', hf(repo)), PYTHIA, date, date, weights=date, params=(p, p)),
		V(f'Pythia-{size}-deduped', names(f'Pythia {size} deduped', f'Pythia-{size.lower()}-deduped', f'Pythia ({size}, deduped)', hf(repo + '-deduped')), PYTHIA, date, date, weights=date, params=(p, p)),
	)

# Llemma (math models continued from Code Llama on Proof-Pile-2)
LLEMMA = BLOG + 'llemma/'
S('Llemma 7B', names('Llemma-7B', 'Llemma (7B)', 'llemma_7b', 'EleutherAI/llemma_7b'), LLEMMA, '2023-10-16', '2023-10-16', weights='2023-10-16', params=(6_740 * M_, 6_740 * M_))
S('Llemma 34B', names('Llemma', 'Llemma-34B', 'Llemma (34B)', 'llemma_34b', 'EleutherAI/llemma_34b'), LLEMMA, '2023-10-16', '2023-10-16', weights='2023-10-16', params=(33_740 * M_, 33_740 * M_))

# Pile-T5 (T5 trained on the Pile with the Llama tokenizer)
PILE_T5 = BLOG + 'pile-t5/'
for size, p in (('Base', 247_586_304), ('Large', 783_173_632), ('XL', 2_849_804_288), ('XXL', 11_135_426_560)):
	repo = f'pile-t5-{size.lower()}'
	S(f'Pile-T5 {size}', names(f'Pile-T5-{size}', f'Pile T5 {size}', hf(repo, repo + '-t5x')), PILE_T5, '2024-04-15', '2024-04-15', weights='2024-04-15', params=(p, p))

# Comma v0.1 (7B models trained on the openly licensed Common Pile, with many partners)
COMMA = ARXIV + '2506.05209'
P_COMMA = (7_002_656_768, 7_002_656_768)
M('Comma v0.1',
	V('Comma v0.1-1T', names('Comma v0.1 1T', 'Comma-v0.1-1T', 'Comma v0.1', 'comma-v0.1-1t', 'common-pile/comma-v0.1-1t'), COMMA, '2025-06-05', '2025-06-05', weights='2025-06-05', params=P_COMMA),
	V('Comma v0.1-2T', names('Comma v0.1 2T', 'Comma-v0.1-2T', 'comma-v0.1-2t', 'common-pile/comma-v0.1-2t'), COMMA, '2025-06-05', '2025-06-05', weights='2025-06-05', params=P_COMMA),
)

# Deep Ignorance (6.9B models with biothreat data filtered from pretraining)
DEEP = ARXIV + '2508.06601'
P_DEEP = (6_856_253_440, 6_856_253_440)
M('Deep Ignorance', *[
	V(display, names(display.replace(' ', '-'), hf('deep-ignorance-' + repo)), DEEP, '2025-08-08', '2025-08-08', weights='2025-08-08', params=P_DEEP)
	for display, repo in (
		('Deep Ignorance Unfiltered', 'unfiltered'),
		('Deep Ignorance Strong Filter', 'e2e-strong-filter'),
		('Deep Ignorance Weak Filter', 'e2e-weak-filter'),
		('Deep Ignorance Strong Filter PT Weak Filter Anneal', 'strong-filter-pt-weak-filter-anneal'),
		('Deep Ignorance Weak Filter PT Strong Filter Anneal', 'weak-filter-pt-strong-filter-anneal'),
		('Deep Ignorance Unfiltered CB', 'unfiltered-cb'),
		('Deep Ignorance Unfiltered CB-LAT', 'unfiltered-cb-lat'),
	)
])

write('eleutherai')
