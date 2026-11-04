"""Generates data/models/thinkingmachines/*.json (one file per model).

Covers the language models of Thinking Machines Lab (Mira Murati's lab): the Inkling open-weights family.
All the data lives in this script: edit it and run `python scripts/models/gen_thinkingmachines.py`.
Existing JSON files are deleted and rewritten, so manual edits to them are lost.

Each size is a model. Inkling-Small (preview), benchmarked in the Inkling announcement, is the earlier checkpoint the released
Inkling-Small was trained from (two more weeks of agentic coding RL), so both are versions of the same model; the preview was
never usable. Prices are the first-party Tinker serverless inference prices. Effort is a float in [0, 1) with named presets
(none 0.0, minimal 0.1, low 0.2, medium 0.7, high 0.9, xhigh 0.99). NVFP4 releases are only names.
TML-Interaction-Small (May 2026 research preview of a full-duplex audio / video interaction model) is out of scope as a
real-time voice model, and the Tinker fine-tuning service is not a model.
"""

from common import B, M, N, S, V, write

EFFORTS = ['None', 'Minimal', 'Low', 'Medium', 'High', 'Xhigh']
NEWS = 'https://thinkingmachines.ai/news/'

# Inkling (975B MoE, text / image / audio input)
S('Inkling', ['Inkling (Xhigh)', 'inkling', 'thinkingmachines/Inkling', 'thinkingmachines/inkling', 'thinkingmachines/Inkling:peft:262144',
		'Inkling-NVFP4', 'thinkingmachines/Inkling-NVFP4'],
	NEWS + 'introducing-inkling/', '2026-07-15', '2026-07-15', weights='2026-07-15', params=(975 * B, 41 * B), price=(1, 4.05), efforts=EFFORTS)

# Inkling-Small (276B MoE, preview shown on Jul 15, released on Jul 30)
P_SMALL = (276 * B, 12 * B)
M('Inkling-Small',
	V('Inkling-Small (Preview)', ['Inkling-Small Preview', 'Inkling Small (Preview)', 'Inkling-Small (preview)'], NEWS + 'introducing-inkling/', '2026-07-15', None, id='inkling-small-preview', params=P_SMALL, efforts=EFFORTS),
	V('Inkling-Small', ['Inkling Small', 'inkling-small', 'thinkingmachines/Inkling-Small', 'thinkingmachines/Inkling-Small:peft:262144',
			'Inkling-Small-NVFP4', 'thinkingmachines/Inkling-Small-NVFP4'],
		NEWS + 'inkling-small/', '2026-07-30', '2026-07-30', weights='2026-07-30', params=P_SMALL, price=(0.3, 1.2), efforts=EFFORTS),
)

write('thinkingmachines')
