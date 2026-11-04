"""Generates data/models/tii/*.json (one file per model).

Covers the language models of the Technology Innovation Institute (Abu Dhabi): Noor and the Falcon family.
All the data lives in this script: edit it and run `python scripts/models/gen_tii.py`.
Existing JSON files are deleted and rewritten, so manual edits to them are lost.

Each size is a model; its base and instruct checkpoints, the 1.58-bit (BitNet) conversions, the VLM and the language, dialect
and reasoning variants post-trained from that size are its versions (Falcon-Arabic on Falcon3-7B, Falcon-H1-Arabic on
Falcon-H1, Falcon-Emirati on Falcon-H1-Arabic 7B, Falcon H1R on Falcon-H1-7B). Falcon3-Mamba-7B (continued from Falcon Mamba
7B) and Falcon-H1-1.5B-Deep are separate models. Quantized releases (FP8, GGUF, GPTQ, AWQ) are only names.
TII has no paid API and no provider lists the models on Artificial Analysis, so prices are unknown. Falcon-Arabic,
Falcon-H1-Arabic and Falcon-Emirati are only available in the Falcon Chat app (no weights).
Falcon Perception, Falcon OCR and Falcon ASR (vision / speech) are out of scope.
"""

from common import B, M, N, S, V, write

M_ = 1_000_000

ARXIV = 'https://arxiv.org/abs/'
SITE = 'https://falconllm.tii.ae/'
BLOG = 'https://falcon-lm.github.io/blog/'
HF = 'https://huggingface.co/tiiuae/'


def names(*items):
	"""Flattens names and lists of names, without duplicates."""
	out = []
	for i in items:
		out += [i] if isinstance(i, str) else list(i)
	return list(dict.fromkeys(out))


def hf(*repos):
	"""TII Hugging Face repo ids, both bare and with the tiiuae organisation."""
	return list(repos) + ['tiiuae/' + r for r in repos]


# Noor (Arabic model built with LightOn, never released)
S('Noor', ['NOOR', 'Noor 10B'], 'https://www.tii.ae/news/technology-innovation-institute-announces-launch-noor-worlds-largest-arabic-nlp-model', '2022-04-11', None, params=(10 * B, 10 * B))

# Falcon (RefinedWeb research models, 7B, 40B, 180B)
RW = ARXIV + '2306.01116'
S('Falcon-RW-1B', names('Falcon RW 1B', 'Falcon-rw-1.3B', 'Falcon-RW-1.3B', hf('falcon-rw-1b')), RW, '2023-06-01', '2023-06-01', weights='2023-06-01', params=(1_300 * M_, 1_300 * M_))
S('Falcon-RW-7B', names('Falcon RW 7B', 'Falcon-rw-7.5B', hf('falcon-rw-7b')), RW, '2023-06-01', '2023-06-01', weights='2023-06-01', params=(7_782 * M_, 7_782 * M_))
FALCON = 'https://huggingface.co/blog/falcon'
P_7 = (7_217 * M_, 7_217 * M_)
M('Falcon-7B',
	V('Falcon-7B', names('Falcon 7B', 'Falcon (7B)', hf('falcon-7b')), FALCON, '2023-05-25', '2023-05-25', weights='2023-05-25', params=P_7),
	V('Falcon-7B-Instruct', names('Falcon 7B Instruct', 'Falcon-Instruct (7B)', hf('falcon-7b-instruct')), FALCON, '2023-05-25', '2023-05-25', weights='2023-05-25', params=P_7),
)
P_40 = (41_836 * M_, 41_836 * M_)
M('Falcon-40B',
	V('Falcon-40B', names('Falcon 40B', 'Falcon (40B)', hf('falcon-40b')), SITE + 'falcon-40b.html', '2023-03-15', '2023-05-25', weights='2023-05-25', params=P_40),
	V('Falcon-40B-Instruct', names('Falcon 40B Instruct', 'Falcon-Instruct (40B)', hf('falcon-40b-instruct')), FALCON, '2023-05-25', '2023-05-25', weights='2023-05-25', params=P_40),
)
F180 = SITE + 'falcon-180b.html'
P_180 = (179_523 * M_, 179_523 * M_)
M('Falcon-180B',
	V('Falcon-180B', names('Falcon 180B', 'falcon-180b', hf('falcon-180B')), F180, '2023-09-06', '2023-09-06', weights='2023-09-06', params=P_180),
	V('Falcon-180B-Chat', names('Falcon 180B Chat', hf('falcon-180B-chat')), F180, '2023-09-06', '2023-09-06', weights='2023-09-06', params=P_180),
)

# Falcon 2 (11B LLM and its VLM)
F2 = SITE + 'falcon-2.html'
M('Falcon2-11B',
	V('Falcon2-11B', names('Falcon 2 11B', 'Falcon 2-11B', 'Falcon-11B', 'falcon-11b', hf('falcon-11B')), F2, '2024-05-13', '2024-05-13', weights='2024-05-13', params=(11_103 * M_, 11_103 * M_)),
	V('Falcon2-11B VLM', names('Falcon 2 11B VLM', 'Falcon2-11B-VLM', 'Falcon-11B-VLM', hf('falcon-11B-vlm', 'Falcon-11B-vlm')), BLOG + 'falcon2-11b/', '2024-05-13', '2024-05-24', weights='2024-05-24', params=(11_427 * M_, 11_427 * M_)),
)

# Falcon Mamba (attention-free SSM)
MAMBA = BLOG + 'falcon-mamba/'
P_MAMBA = (7_273 * M_, 7_273 * M_)
M('Falcon Mamba 7B',
	V('Falcon Mamba 7B', names('Falcon Mamba', 'Falcon-Mamba-7B', 'FalconMamba 7B', hf('falcon-mamba-7b')), MAMBA, '2024-08-12', '2024-08-12', weights='2024-08-12', params=P_MAMBA),
	V('Falcon Mamba 7B Instruct', names('Falcon-Mamba-7B-Instruct', 'FalconMamba 7B Instruct', hf('falcon-mamba-7b-instruct')), MAMBA, '2024-08-12', '2024-08-12', weights='2024-08-12', params=P_MAMBA),
)

# Falcon 3 (7B trained from scratch, 10B depth-upscaled from it, 1B / 3B pruned and distilled; 1.58-bit BitNet conversions)
F3 = BLOG + 'falcon-3/'


def falcon3(size, params, bitnet=('Base', 'Instruct')):
	"""The versions of a Falcon3 size: base, instruct and their 1.58-bit conversions."""
	out = []
	for kind in ('Base', 'Instruct'):
		repo = f'Falcon3-{size}-{kind}'
		out.append(V(repo, names(repo.replace('-', ' '), hf(repo)), F3, '2024-12-17', '2024-12-17', weights='2024-12-17', params=params))
	for kind in bitnet:
		repo = f'Falcon3-{size}-{kind}-1.58bit'
		out.append(V(repo, names(f'Falcon3 {size} {kind} 1.58bit', hf(repo)), F3, '2024-12-17', '2024-12-17', weights='2024-12-17', params=params))
	return out


M('Falcon3-1B', *falcon3('1B', (1_669 * M_, 1_669 * M_), bitnet=('Instruct',)))
M('Falcon3-3B', *falcon3('3B', (3_228 * M_, 3_228 * M_)))
P_F37 = (7_456 * M_, 7_456 * M_)
M('Falcon3-7B',
	*falcon3('7B', P_F37),
	V('Falcon-Arabic', ['Falcon Arabic', 'Falcon-Arabic-7B-Instruct', 'Falcon-Arabic-7B'], BLOG + 'falcon-arabic/', '2025-05-21', '2025-05-21', params=(7 * B, 7 * B)),
)
M('Falcon3-10B', *falcon3('10B', (10_306 * M_, 10_306 * M_)))
M('Falcon3-Mamba-7B',
	V('Falcon3-Mamba-7B-Base', names('Falcon3 Mamba 7B Base', hf('Falcon3-Mamba-7B-Base')), F3, '2024-12-17', '2024-12-17', weights='2024-12-17', params=P_MAMBA),
	V('Falcon3-Mamba-7B-Instruct', names('Falcon3 Mamba 7B Instruct', 'Falcon3-Mamba-7B', hf('Falcon3-Mamba-7B-Instruct')), F3, '2024-12-17', '2024-12-17', weights='2024-12-17', params=P_MAMBA),
)

# Falcon-Edge (native 1.58-bit BitNet models)
EDGE = BLOG + 'falcon-edge/'
for size, p in (('1B', 1_720 * M_), ('3B', 3_054 * M_)):
	M(f'Falcon-E-{size}',
		V(f'Falcon-E-{size}-Base', names(f'Falcon-Edge {size} Base', f'Falcon E {size} Base', hf(f'Falcon-E-{size}-Base')), EDGE, '2025-05-15', '2025-05-15', weights='2025-05-15', params=(p, p)),
		V(f'Falcon-E-{size}-Instruct', names(f'Falcon-Edge {size} Instruct', f'Falcon E {size} Instruct', f'Falcon-E-{size}', hf(f'Falcon-E-{size}-Instruct')), EDGE, '2025-05-15', '2025-05-15', weights='2025-05-15', params=(p, p)),
	)

# Falcon-H1 (hybrid attention / Mamba-2 heads) and the Arabic, Emirati and reasoning models built on it
H1 = BLOG + 'falcon-h1/'
H1_ARABIC = BLOG + 'falcon-h1-arabic/'
H1R = BLOG + 'falcon-h1r-7b/'


def falcon_h1(size, p, extra=()):
	"""The base and instruct versions of a Falcon-H1 size."""
	base, inst = f'Falcon-H1-{size}-Base', f'Falcon-H1-{size}-Instruct'
	return [
		V(base, names(base.replace('-', ' '), hf(base, base.lower())), H1, '2025-05-20', '2025-05-20', weights='2025-05-20', params=(p, p)),
		V(inst, names(inst.replace('-', ' '), list(extra), hf(inst, inst.lower())), H1, '2025-05-20', '2025-05-20', weights='2025-05-20', params=(p, p)),
	]


def h1_arabic(size, p):
	return V(f'Falcon-H1-Arabic-{size}', [f'Falcon-H1-Arabic {size}', f'Falcon H1 Arabic {size}'], H1_ARABIC, '2026-01-05', '2026-01-05', params=(p, p))


M('Falcon-H1-0.5B', *falcon_h1('0.5B', 521 * M_))
M('Falcon-H1-1.5B', *falcon_h1('1.5B', 1_555 * M_))
M('Falcon-H1-1.5B-Deep', *falcon_h1('1.5B-Deep', 1_555 * M_))
M('Falcon-H1-3B', *falcon_h1('3B', 3_149 * M_), h1_arabic('3B', 3 * B))
P_H17 = (7_586 * M_, 7_586 * M_)
M('Falcon-H1-7B',
	*falcon_h1('7B', P_H17[0]),
	V('Falcon-H1R-7B', names('Falcon H1R 7B', 'Falcon-H1R 7B', 'Falcon H1R', 'Falcon Reasoning', 'Falcon-H1R-7B-FP8', hf('Falcon-H1R-7B', 'Falcon-H1R-7B-FP8')), H1R, '2026-01-05', '2026-01-05', weights='2026-01-05', params=P_H17),
	h1_arabic('7B', 7 * B),
	V('Falcon-Emirati-7B', ['Falcon-Emirati', 'Falcon Emirati 7B'], BLOG + 'falcon-emirati/', '2026-10-06', '2026-10-06', params=(7 * B, 7 * B)),
)
M('Falcon-H1-34B', *falcon_h1('34B', 33_643 * M_, extra=['Falcon-H1']), h1_arabic('34B', 34 * B))

# Falcon-H1-Tiny (sub-1B Falcon-H1 models)
TINY = 'https://huggingface.co/spaces/tiiuae/tiny-h1-blogpost'
P_T90 = (91 * M_, 91 * M_)
M('Falcon-H1-Tiny-90M',
	V('Falcon-H1-Tiny-90M-Base', names('Falcon H1 Tiny 90M Base', hf('Falcon-H1-Tiny-90M-Base')), TINY, '2026-01-16', '2026-01-16', weights='2026-01-16', params=P_T90),
	V('Falcon-H1-Tiny-90M-Instruct', names('Falcon H1 Tiny 90M Instruct', 'Falcon-H1-Tiny-90M', hf('Falcon-H1-Tiny-90M-Instruct')), TINY, '2026-01-16', '2026-01-16', weights='2026-01-16', params=P_T90),
	V('Falcon-H1-Tiny-90M-Instruct-Curriculum', names('Falcon H1 Tiny 90M Instruct Curriculum', hf('Falcon-H1-Tiny-90M-Instruct-Curriculum')), TINY, '2026-01-16', '2026-01-16', weights='2026-01-16', params=P_T90),
	V('Falcon-H1-Tiny-Tool-Calling-90M', names('Falcon H1 Tiny Tool Calling 90M', hf('Falcon-H1-Tiny-Tool-Calling-90M')), TINY, '2026-01-16', '2026-01-16', weights='2026-01-16', params=P_T90),
	V('Falcon-H1-Tiny-Coder-90M', names('Falcon H1 Tiny Coder 90M', hf('Falcon-H1-Tiny-Coder-90M')), TINY, '2026-01-16', '2026-01-16', weights='2026-01-16', params=P_T90),
	V('Falcon-H1-Tiny-R-90M', names('Falcon H1 Tiny R 90M', hf('Falcon-H1-Tiny-R-90M')), TINY, '2026-01-16', '2026-01-16', weights='2026-01-16', params=P_T90),
)
P_T100 = (108 * M_, 108 * M_)
M('Falcon-H1-Tiny-Multilingual-100M',
	V('Falcon-H1-Tiny-Multilingual-100M-Base', names('Falcon H1 Tiny Multilingual 100M Base', hf('Falcon-H1-Tiny-Multilingual-100M-Base')), TINY, '2026-01-16', '2026-01-16', weights='2026-01-16', params=P_T100),
	V('Falcon-H1-Tiny-Multilingual-100M-Instruct', names('Falcon H1 Tiny Multilingual 100M Instruct', 'Falcon-H1-Tiny-Multilingual-100M', hf('Falcon-H1-Tiny-Multilingual-100M-Instruct')), TINY, '2026-01-16', '2026-01-16', weights='2026-01-16', params=P_T100),
)
S('Falcon-H1-Tiny-R-0.6B', names('Falcon H1 Tiny R 0.6B', 'Falcon-H1-Tiny-R 0.6B', hf('Falcon-H1-Tiny-R-0.6B')), TINY, '2026-01-16', '2026-01-16', weights='2026-01-16', params=(622 * M_, 622 * M_))

write('tii')
