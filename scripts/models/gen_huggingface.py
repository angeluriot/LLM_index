"""Generates data/models/huggingface/*.json (one file per model).

Covers Hugging Face's language models, including the open-science collaborations it led: BigScience (T0, BLOOM, BLOOMZ, mT0),
BigCode with ServiceNow (SantaCoder, StarCoder, StarCoder2), the H4 alignment team (Zephyr, StarChat), the M4 multimodal team
(IDEFICS), the SmolLM team (Cosmo, SmolLM, SmolVLM) and Open R1.
All the data lives in this script: edit it and run `python scripts/models/gen_huggingface.py`.
Existing JSON files are deleted and rewritten, so manual edits to them are lost.

Each size is a model; its base and instruct checkpoints and the variants post-trained from it are its versions (BLOOMZ on
BLOOM, StarChat and OctoCoder on StarCoder, SmolVLM on the SmolLM2 language model it is built on...). The fine-tunes of
third-party bases (Zephyr, IDEFICS, Open R1) are one model per product line and size. Each new version number is a separate
model (SmolLM / SmolLM2 / SmolLM3, StarCoder / StarCoder2, Idefics / Idefics2 / Idefics3).
Hugging Face has no first-party API for its models and Artificial Analysis lists none of them, so prices are unknown.
Encoders (DistilBERT, CodeBERTa...), classifiers, ablation checkpoints and toy models (nanowhale) are out of scope.
"""

from common import B, M, N, S, V, write

M_ = 1_000_000
T = ['', 'Thinking']	# SmolLM3 /think and /no_think

ARXIV = 'https://arxiv.org/abs/'
HF = 'https://huggingface.co/'
BLOG = 'https://huggingface.co/blog/'


def names(*items):
	"""Flattens names and lists of names, without duplicates."""
	out = []
	for i in items:
		out += [i] if isinstance(i, str) else list(i)
	return list(dict.fromkeys(out))


def hf(*repos):
	"""Repo ids, both bare and with their organisation."""
	return list(dict.fromkeys([r.split('/')[-1] for r in repos] + list(repos)))


# DistilGPT2 (distilled from GPT-2 124M)
S('DistilGPT2', names('DistilGPT-2', 'distilgpt2', 'distilbert/distilgpt2'), HF + 'distilbert/distilgpt2', '2019-10-02', '2019-10-02', weights='2019-10-02', params=(82 * M_, 82 * M_))

# BigScience T0 (multitask prompted fine-tunes of the LM-adapted T5)
T0 = ARXIV + '2110.08207'
M('T0',
	V('T0', names('T0-XXL', 'T0 11B', hf('bigscience/T0')), T0, '2021-10-15', '2021-10-15', weights='2021-10-15', params=(11 * B, 11 * B)),
	V('T0+', names('T0p', 'T0 Plus', hf('bigscience/T0p')), T0, '2021-10-15', '2021-10-15', id='t0p', weights='2021-10-15', params=(11 * B, 11 * B)),
	V('T0++', names('T0pp', 'T0 Plus Plus', 'T0pp (11B)', hf('bigscience/T0pp')), T0, '2021-10-15', '2021-10-15', id='t0pp', weights='2021-10-15', params=(11 * B, 11 * B)),
)
S('T0 3B', names('T0_3B', 'T0-3B', 'bigscience/T0_3B'), T0, '2021-10-15', '2021-10-15', weights='2021-10-15', params=(2_850 * M_, 2_850 * M_))

# BigScience BLOOM and its BLOOMZ multitask fine-tunes (xP3, xP3mt for -MT, P3 for -P3)
BLOOM = 'https://bigscience.huggingface.co/blog/bloom'
BLOOMZ = ARXIV + '2211.01786'


def bloom(size, display, p, aliases, variants=('',)):
	"""The BLOOM base model of a size and its BLOOMZ versions."""
	repo = 'bloom' + (f'-{size.lower()}' if size else '')
	out = [V(display, names(aliases, hf('bigscience/' + repo)), BLOOM if not size else HF + 'bigscience/' + repo, '2022-07-12', '2022-07-12', weights='2022-07-12', params=(p, p))]
	for var in variants:
		z = 'bloomz' + (f'-{size.lower()}' if size else '') + (f'-{var.lower()}' if var else '')
		zdisplay = display.replace('BLOOM', 'BLOOMZ') + (f'-{var}' if var else '')
		out.append(V(zdisplay, names(zdisplay.replace('-', ' '), hf('bigscience/' + z)), BLOOMZ, '2022-11-03', '2022-11-03', weights='2022-11-03', params=(p, p)))
	return out


M('BLOOM-560M', *bloom('560m', 'BLOOM-560M', 559 * M_, ['BLOOM 560M', 'BLOOM (560M)']))
M('BLOOM-1B1', *bloom('1b1', 'BLOOM-1B1', 1_065 * M_, ['BLOOM 1.1B', 'BLOOM-1B', 'BLOOM (1.1B)']))
M('BLOOM-1B7', *bloom('1b7', 'BLOOM-1B7', 1_722 * M_, ['BLOOM 1.7B', 'BLOOM-1.7B', 'BLOOM (1.7B)']))
M('BLOOM-3B', *bloom('3b', 'BLOOM-3B', 3_003 * M_, ['BLOOM 3B', 'BLOOM (3B)']))
M('BLOOM-7B1', *bloom('7b1', 'BLOOM-7B1', 7_069 * M_, ['BLOOM 7.1B', 'BLOOM-7.1B', 'BLOOM (7.1B)'], variants=('', 'MT', 'P3')))
P_BLOOM = (176_247 * M_, 176_247 * M_)
M('BLOOM',
	V('BLOOM', names('BLOOM 176B', 'BLOOM-176B', 'BLOOM (176B)', hf('bigscience/bloom')), BLOOM, '2022-03-11', '2022-07-12', weights='2022-07-12', params=P_BLOOM),
	V('BLOOMZ', names('BLOOMZ 176B', 'BLOOMZ-176B', 'BLOOMZ (176B)', hf('bigscience/bloomz')), BLOOMZ, '2022-11-03', '2022-11-03', weights='2022-11-03', params=P_BLOOM),
	V('BLOOMZ-MT', names('BLOOMZ MT', 'BLOOMZ-MT (176B)', hf('bigscience/bloomz-mt')), BLOOMZ, '2022-11-03', '2022-11-03', weights='2022-11-03', params=P_BLOOM),
	V('BLOOMZ-P3', names('BLOOMZ P3', 'BLOOMZ-P3 (176B)', hf('bigscience/bloomz-p3')), BLOOMZ, '2022-11-03', '2022-11-03', weights='2022-11-03', params=P_BLOOM),
)

# BigScience mT0 (xP3 fine-tunes of mT5)
for size, p, extra in (('small', 300 * M_, ['mT0 300M']), ('base', 582 * M_, ['mT0 580M']), ('large', 1_230 * M_, ['mT0 1.2B']), ('xl', 3_743 * M_, ['mT0 3.7B'])):
	S(f'mT0-{size}', names(f'mT0 {size}', f'mt0-{size}', extra, f'bigscience/mt0-{size}'), BLOOMZ, '2022-11-03', '2022-11-03', weights='2022-11-03', params=(p, p))
P_MT0 = (13_946 * M_, 13_946 * M_)
M('mT0-xxl',
	V('mT0-xxl', names('mT0 xxl', 'mT0-13B', 'mT0 13B', hf('bigscience/mt0-xxl')), BLOOMZ, '2022-11-03', '2022-11-03', weights='2022-11-03', params=P_MT0),
	V('mT0-xxl-mt', names('mT0 xxl MT', 'mT0-XXL-MT', hf('bigscience/mt0-xxl-mt')), BLOOMZ, '2022-11-03', '2022-11-03', weights='2022-11-03', params=P_MT0),
	V('mT0-xxl-p3', names('mT0 xxl P3', 'mT0-XXL-P3', hf('bigscience/mt0-xxl-p3')), BLOOMZ, '2022-11-03', '2022-11-03', weights='2022-11-03', params=P_MT0),
)

# BigCode SantaCoder and StarCoder (with ServiceNow), StarChat (H4) and OctoCoder (OctoPack) on StarCoder
S('SantaCoder', names('SantaCoder 1.1B', hf('bigcode/santacoder', 'bigcode/gpt_bigcode-santacoder')), ARXIV + '2301.03988', '2022-12-22', '2022-12-22', weights='2022-12-22', params=(1_125 * M_, 1_125 * M_))
STARCODER = BLOG + 'starcoder'
P_SC = (15_517 * M_, 15_517 * M_)
M('StarCoder',
	V('StarCoderBase', names('StarCoder Base', 'StarCoderBase 15B', 'StarCoderBase-15B', hf('bigcode/starcoderbase')), STARCODER, '2023-05-04', '2023-05-04', weights='2023-05-04', params=P_SC),
	V('StarCoder', names('StarCoder 15B', 'StarCoder-15B', 'StarCoder (15.5B)', hf('bigcode/starcoder')), STARCODER, '2023-05-04', '2023-05-04', weights='2023-05-04', params=P_SC),
	V('StarChat Alpha', names('StarChat-Alpha', 'StarChat α', hf('HuggingFaceH4/starchat-alpha')), BLOG + 'starchat-alpha', '2023-05-09', '2023-05-09', weights='2023-05-09', params=P_SC),
	V('StarCoderPlus', names('StarCoder Plus', 'StarCoder+', hf('bigcode/starcoderplus')), HF + 'bigcode/starcoderplus', '2023-06-07', '2023-06-07', weights='2023-06-07', params=P_SC),
	V('StarChat Beta', names('StarChat-Beta', 'StarChat β', hf('HuggingFaceH4/starchat-beta')), HF + 'HuggingFaceH4/starchat-beta', '2023-06-07', '2023-06-07', weights='2023-06-07', params=P_SC),
	V('OctoCoder', names('OctoCoder 15B', hf('bigcode/octocoder')), ARXIV + '2308.07124', '2023-08-14', '2023-08-14', weights='2023-08-14', params=P_SC),
)
S('StarCoderBase-1B', names('StarCoderBase 1B', 'StarCoder 1B', hf('bigcode/starcoderbase-1b')), HF + 'bigcode/starcoderbase-1b', '2023-07-03', '2023-07-03', weights='2023-07-03', params=(1_137 * M_, 1_137 * M_))
S('StarCoderBase-3B', names('StarCoderBase 3B', 'StarCoder 3B', hf('bigcode/starcoderbase-3b')), HF + 'bigcode/starcoderbase-3b', '2023-06-30', '2023-06-30', weights='2023-06-30', params=(3 * B, 3 * B))
S('StarCoderBase-7B', names('StarCoderBase 7B', 'StarCoder 7B', hf('bigcode/starcoderbase-7b')), HF + 'bigcode/starcoderbase-7b', '2023-07-26', '2023-07-26', weights='2023-07-26', params=(7_529 * M_, 7_529 * M_))

# BigCode StarCoder2 (3B by ServiceNow, 7B by Hugging Face, 15B by NVIDIA)
SC2 = BLOG + 'starcoder2'
S('StarCoder2-3B', names('StarCoder2 3B', 'StarCoder 2 3B', hf('bigcode/starcoder2-3b')), SC2, '2024-02-28', '2024-02-28', weights='2024-02-28', params=(3_030 * M_, 3_030 * M_))
S('StarCoder2-7B', names('StarCoder2 7B', 'StarCoder 2 7B', hf('bigcode/starcoder2-7b')), SC2, '2024-02-28', '2024-02-28', weights='2024-02-28', params=(7_174 * M_, 7_174 * M_))
P_SC2 = (15_958 * M_, 15_958 * M_)
M('StarCoder2-15B',
	V('StarCoder2-15B', names('StarCoder2 15B', 'StarCoder 2 15B', 'StarCoder2', hf('bigcode/starcoder2-15b')), SC2, '2024-02-28', '2024-02-28', weights='2024-02-28', params=P_SC2),
	V('StarChat2 15B', names('StarChat2-15B', 'starchat2-15b-v0.1', 'HuggingFaceH4/starchat2-15b-v0.1'), HF + 'HuggingFaceH4/starchat2-15b-v0.1', '2024-03-12', '2024-03-12', weights='2024-03-12', params=P_SC2),
	V('StarCoder2-15B-Instruct-v0.1', names('StarCoder2-15B-Instruct', 'StarCoder2 15B Instruct', 'StarCoder2-Instruct', hf('bigcode/starcoder2-15b-instruct-v0.1')), BLOG + 'sc2-instruct', '2024-04-29', '2024-04-29', weights='2024-04-29', params=P_SC2),
)

# IDEFICS (M4 vision-language models on Llama / Mistral bases)
IDEFICS = BLOG + 'idefics'
P_I9 = (8_930 * M_, 8_930 * M_)
P_I80 = (79_968 * M_, 79_968 * M_)
M('IDEFICS 9B',
	V('IDEFICS 9B', names('IDEFICS-9B', 'Idefics 9B', hf('HuggingFaceM4/idefics-9b')), IDEFICS, '2023-08-22', '2023-08-22', weights='2023-08-22', params=P_I9),
	V('IDEFICS 9B Instruct', names('IDEFICS-9B-Instruct', 'Idefics 9B Instruct', hf('HuggingFaceM4/idefics-9b-instruct')), IDEFICS, '2023-08-22', '2023-08-22', weights='2023-08-22', params=P_I9),
)
M('IDEFICS 80B',
	V('IDEFICS 80B', names('IDEFICS-80B', 'Idefics 80B', 'IDEFICS', hf('HuggingFaceM4/idefics-80b')), IDEFICS, '2023-08-22', '2023-08-22', weights='2023-08-22', params=P_I80),
	V('IDEFICS 80B Instruct', names('IDEFICS-80B-Instruct', 'Idefics 80B Instruct', hf('HuggingFaceM4/idefics-80b-instruct')), IDEFICS, '2023-08-22', '2023-08-22', weights='2023-08-22', params=P_I80),
)
IDEFICS2 = BLOG + 'idefics2'
P_I2 = (8_403 * M_, 8_403 * M_)
M('Idefics2-8B',
	V('Idefics2-8B-Base', names('Idefics2 8B Base', hf('HuggingFaceM4/idefics2-8b-base')), IDEFICS2, '2024-04-15', '2024-04-15', weights='2024-04-15', params=P_I2),
	V('Idefics2-8B', names('Idefics2', 'Idefics2 8B', 'IDEFICS2-8B', hf('HuggingFaceM4/idefics2-8b')), IDEFICS2, '2024-04-15', '2024-04-15', weights='2024-04-15', params=P_I2),
	V('Idefics2-8B-Chatty', names('Idefics2 8B Chatty', hf('HuggingFaceM4/idefics2-8b-chatty')), ARXIV + '2405.02246', '2024-05-02', '2024-05-02', weights='2024-05-02', params=P_I2),
)
S('Idefics3-8B-Llama3', names('Idefics3', 'Idefics3-8B', 'Idefics3 8B', hf('HuggingFaceM4/Idefics3-8B-Llama3')), ARXIV + '2408.12637', '2024-08-05', '2024-08-05', weights='2024-08-05', params=(8_462 * M_, 8_462 * M_))

# Zephyr (H4 alignment fine-tunes)
P_Z7 = (7_242 * M_, 7_242 * M_)
M('Zephyr 7B',
	V('Zephyr 7B Alpha', names('Zephyr-7B-Alpha', 'Zephyr 7B α', 'zephyr-7b-alpha', 'HuggingFaceH4/zephyr-7b-alpha'), HF + 'HuggingFaceH4/zephyr-7b-alpha', '2023-10-10', '2023-10-10', weights='2023-10-10', params=P_Z7),
	V('Zephyr 7B Beta', names('Zephyr 7B', 'Zephyr-7B', 'Zephyr-7B-Beta', 'Zephyr 7B β', 'zephyr-7b-beta', 'HuggingFaceH4/zephyr-7b-beta'), ARXIV + '2310.16944', '2023-10-25', '2023-10-25', weights='2023-10-25', params=P_Z7),
)
S('Zephyr 7B Gemma', names('Zephyr-7B-Gemma', 'zephyr-7b-gemma-v0.1', 'HuggingFaceH4/zephyr-7b-gemma-v0.1'), HF + 'HuggingFaceH4/zephyr-7b-gemma-v0.1', '2024-03-01', '2024-03-01', weights='2024-03-01', params=(8_538 * M_, 8_538 * M_))
S('Zephyr 141B-A39B', names('Zephyr 141B', 'Zephyr-ORPO-141B-A35B', 'zephyr-orpo-141b-A35b-v0.1', 'HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1'), HF + 'HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1', '2024-04-10', '2024-04-10', weights='2024-04-10', params=(140_621 * M_, 39 * B))

# Cosmo and SmolLM (SmolVLM / SmolVLM2 are versions of the SmolLM2 size they are built on)
S('Cosmo-1B', names('Cosmo 1B', hf('HuggingFaceTB/cosmo-1b')), HF + 'HuggingFaceTB/cosmo-1b', '2024-02-20', '2024-02-20', weights='2024-02-20', params=(1_742 * M_, 1_742 * M_))
SMOLLM = BLOG + 'smollm'
for size, p in (('135M', 134_515_008), ('360M', 361_821_120), ('1.7B', 1_711_376_384)):
	M(f'SmolLM-{size}',
		V(f'SmolLM-{size}', names(f'SmolLM {size}', hf(f'HuggingFaceTB/SmolLM-{size}')), SMOLLM, '2024-07-16', '2024-07-16', weights='2024-07-16', params=(p, p)),
		V(f'SmolLM-{size}-Instruct', names(f'SmolLM {size} Instruct', hf(f'HuggingFaceTB/SmolLM-{size}-Instruct')), SMOLLM, '2024-07-16', '2024-07-16', weights='2024-07-16', params=(p, p)),
	)
SMOLLM2 = ARXIV + '2502.02737'
SMOLVLM = BLOG + 'smolvlm'
SMOLERVLM = BLOG + 'smolervlm'
SMOLVLM2 = BLOG + 'smolvlm2'


def smollm2(size, p):
	return [
		V(f'SmolLM2-{size}', names(f'SmolLM2 {size}', hf(f'HuggingFaceTB/SmolLM2-{size}')), SMOLLM2, '2024-10-31', '2024-10-31', weights='2024-10-31', params=(p, p)),
		V(f'SmolLM2-{size}-Instruct', names(f'SmolLM2 {size} Instruct', hf(f'HuggingFaceTB/SmolLM2-{size}-Instruct')), SMOLLM2, '2024-10-31', '2024-10-31', weights='2024-10-31', params=(p, p)),
	]


def smolvlm(display, repo, link, date, p, extra=()):
	return V(display, names(list(extra), display.replace('-', ' '), hf('HuggingFaceTB/' + repo)), link, date, date, weights=date, params=(p, p))


M('SmolLM2-135M',
	*smollm2('135M', 134_515_008),
	smolvlm('SmolVLM-256M-Instruct', 'SmolVLM-256M-Instruct', SMOLERVLM, '2025-01-23', 256_484_928, ['SmolVLM-256M']),
	smolvlm('SmolVLM2-256M-Video-Instruct', 'SmolVLM2-256M-Video-Instruct', SMOLVLM2, '2025-02-20', 256_484_928, ['SmolVLM2-256M']),
)
M('SmolLM2-360M',
	*smollm2('360M', 361_821_120),
	smolvlm('SmolVLM-500M-Instruct', 'SmolVLM-500M-Instruct', SMOLERVLM, '2025-01-23', 507_482_304, ['SmolVLM-500M']),
	smolvlm('SmolVLM2-500M-Video-Instruct', 'SmolVLM2-500M-Video-Instruct', SMOLVLM2, '2025-02-20', 507_482_304, ['SmolVLM2-500M']),
)
M('SmolLM2-1.7B',
	*smollm2('1.7B', 1_711_376_384),
	smolvlm('SmolVLM-Instruct', 'SmolVLM-Instruct', SMOLVLM, '2024-11-26', 2_246_272_880, ['SmolVLM', 'SmolVLM 2B', 'SmolVLM-2B']),
	V('SmolLM2-1.7B-Instruct-16k', names('SmolLM2 1.7B Instruct 16k', hf('HuggingFaceTB/SmolLM2-1.7B-Instruct-16k')), SMOLLM2, '2025-02-21', '2025-02-21', weights='2025-02-21', params=(1_711_376_384, 1_711_376_384)),
	smolvlm('SmolVLM2-2.2B-Instruct', 'SmolVLM2-2.2B-Instruct', SMOLVLM2, '2025-02-20', 2_246_784_880, ['SmolVLM2', 'SmolVLM2-2.2B']),
)
SMOLLM3 = BLOG + 'smollm3'
P_S3 = (3_075 * M_, 3_075 * M_)
M('SmolLM3-3B',
	V('SmolLM3-3B-Base', names('SmolLM3 3B Base', hf('HuggingFaceTB/SmolLM3-3B-Base')), SMOLLM3, '2025-07-08', '2025-07-08', weights='2025-07-08', params=P_S3),
	V('SmolLM3-3B', names('SmolLM3', 'SmolLM3 3B', 'SmolLM 3', 'HuggingFaceTB/SmolLM3-3B'), SMOLLM3, '2025-07-08', '2025-07-08', weights='2025-07-08', params=P_S3, efforts=T),
)

# Open R1 (reasoning fine-tunes of Qwen2.5 bases)
P_Q7 = (7_616 * M_, 7_616 * M_)
M('Open R1 7B',
	V('OpenR1-Qwen-7B', names('OpenR1 Qwen 7B', 'Open-R1-Qwen-7B', 'open-r1/OpenR1-Qwen-7B'), BLOG + 'open-r1/update-2', '2025-02-10', '2025-02-10', weights='2025-02-10', params=P_Q7),
	V('OpenR1-Distill-7B', names('OpenR1 Distill 7B', 'Open-R1-Distill-7B', 'open-r1/OpenR1-Distill-7B'), HF + 'open-r1/OpenR1-Distill-7B', '2025-05-26', '2025-05-26', weights='2025-05-26', params=P_Q7),
)
OLYMPIC = BLOG + 'open-r1/update-3'
S('OlympicCoder-7B', names('OlympicCoder 7B', 'open-r1/OlympicCoder-7B'), OLYMPIC, '2025-03-11', '2025-03-11', weights='2025-03-11', params=P_Q7)
S('OlympicCoder-32B', names('OlympicCoder 32B', 'OlympicCoder', 'open-r1/OlympicCoder-32B'), OLYMPIC, '2025-03-11', '2025-03-11', weights='2025-03-11', params=(32_764 * M_, 32_764 * M_))

write('huggingface')
