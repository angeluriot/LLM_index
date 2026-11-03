"""Generates data/models/xai/*.json (one file per model).

All the data lives in this script: edit it and run `python scripts/models/gen_xai.py`.
Existing JSON files are deleted and rewritten, so manual edits to them are lost.
"""

from common import B, M, N, S, V, write

R = ['', 'Reasoning']	# Separate reasoning / non-reasoning API ids
T = ['', 'Thinking']	# Thinking on/off in the Grok app
LH = ['Low', 'High']
LMH = ['Low', 'Medium', 'High']
LMHX = ['Low', 'Medium', 'High', 'Xhigh']
NLMH = ['None', 'Low', 'Medium', 'High']

NEWS = 'https://x.ai/news/'
DOCS = 'https://docs.x.ai/developers/models/'
GROK_NOTES = 'https://grok.com/release-notes'

# Grok-0 / Grok-1
GROK = NEWS + 'grok'
S('Grok-0', [], GROK, '2023-11-04', None, params=(33 * B, 33 * B))
S('Grok-1', ['Grok 1', 'xai-org/grok-1'], GROK, '2023-11-04', '2023-12-07', weights='2024-03-17', params=(314 * B, 78_500_000_000))

# Grok-1.5
M('Grok-1.5',
	V('Grok-1.5', ['Grok 1.5'], NEWS + 'grok-1.5', '2024-03-28', '2024-05-15'),
	V('Grok-1.5V', ['Grok 1.5V', 'Grok-1.5 Vision'], NEWS + 'grok-1.5v', '2024-04-12', None),
)

# Grok-2 (grok-beta is the API beta of the Grok-2 generation; Grok 2.5 are the 2024 weights opened in 2025)
GROK2 = NEWS + 'grok-2'
GROK2_DEC = NEWS + 'grok-1212'
M('Grok-2',
	V('Grok-2 (Aug 2024)', ['Grok-2 Beta', 'Grok 2 (Aug 2024)', 'sus-column-r'], GROK2, '2024-04-08', '2024-08-13'),
	V('Grok Beta', ['grok-beta'], NEWS + 'api', '2024-10-21', '2024-10-21', price=(5, 15)),
	V('Grok Vision Beta', ['grok-vision-beta'], NEWS + 'api', '2024-11-04', '2024-11-04', price=(5, 15)),
	V('Grok-2 (Dec 2024)', ["Grok 2 (Dec '24)", 'Grok 2 (Dec 2024)', 'grok-2-1212', 'grok-2-latest'], GROK2_DEC, '2024-12-12', '2024-12-12', price=(2, 10)),
	V('Grok-2 Vision (Dec 2024)', ['Grok 2 Vision', 'grok-2-vision-1212', 'grok-2-vision-latest'], GROK2_DEC, '2024-12-12', '2024-12-12', price=(2, 10)),
	V('Grok 2.5', ['Grok-2.5', 'Grok 2 (open weights)', 'xai-org/grok-2'], 'https://huggingface.co/xai-org/grok-2', '2025-08-23', '2025-08-23', weights='2025-08-23', params=(270 * B, 115 * B)),
)
S('Grok-2 mini', ['Grok-2 Mini', 'Grok 2 mini', 'grok-2-mini'], GROK2, '2024-08-13', '2024-08-13')

# Grok 3
GROK3 = NEWS + 'grok-3'
M('Grok 3',
	V('Grok 3', ['Grok-3', 'Grok 3 Beta', 'grok-3', 'grok-3-beta', 'grok-3-latest', 'grok-3-fast', 'grok-3-fast-beta', 'grok-3-fast-latest', 'chocolate', 'early-grok-3'], GROK3, '2024-04-08', '2025-02-17', params=(3_000 * B, None), price=(3, 15)),
	V('Grok 3 (Think)', ['Grok 3 Think', 'Grok 3 Reasoning', 'Grok 3 Reasoning Beta', 'Grok-3 Reasoning', 'grok-3-reasoning'], GROK3, '2025-02-17', '2025-02-17', id='grok-3-think', params=(3_000 * B, None)),
)
S('Grok 3 mini', ['Grok-3 Mini', 'Grok 3 Mini', 'Grok 3 mini Reasoning', 'grok-3-mini', 'grok-3-mini-beta', 'grok-3-mini-latest', 'grok-3-mini-fast', 'grok-3-mini-fast-beta', 'grok-3-mini-fast-latest'], GROK3, '2025-02-17', '2025-04-09', price=(0.3, 0.5), efforts=LH)

# Grok 4
GROK4 = NEWS + 'grok-4'
M('Grok 4',
	V('Grok 4', ['Grok-4', 'grok-4', 'grok-4-0709', 'grok-4-latest'], GROK4, '2025-06-27', '2025-07-09', params=(3_000 * B, None), price=(3, 15)),
	V('Grok 4 Heavy', ['Grok-4 Heavy', 'grok-4-heavy'], GROK4, '2025-07-09', '2025-07-09', params=(3_000 * B, None)),
)
S('Grok Code Fast 1', ['Grok Code Fast', 'grok-code-fast-1', 'grok-code-fast', 'grok-code-fast-1-0825', 'sonic'], NEWS + 'grok-code-fast-1', '2025-08-28', '2025-08-28', price=(0.2, 1.5))
S('Grok 4 Fast', ['Grok-4 Fast', 'grok-4-fast', 'grok-4-fast-reasoning', 'grok-4-fast-non-reasoning', 'grok-4-fast-search', 'tahoe', 'menlo'], NEWS + 'grok-4-fast', '2025-09-19', '2025-09-19', price=(0.2, 0.5), efforts=R)

# Grok 4.1
S('Grok 4.1', ['Grok-4.1', 'grok-4.1', 'Grok 4.1 Thinking', 'Grok-4.1 Thinking', 'grok-4.1-thinking', 'quasarflux', 'tensor'], NEWS + 'grok-4-1', '2025-11-17', '2025-11-17', efforts=T)
S('Grok 4.1 Fast', ['Grok-4.1 Fast', 'grok-4.1-fast', 'grok-4-1-fast', 'grok-4-1-fast-reasoning', 'grok-4-1-fast-non-reasoning'], NEWS + 'grok-4-1-fast', '2025-11-19', '2025-11-19', price=(0.2, 0.5), efforts=R)

# Grok 4.20
G420 = DOCS + 'grok-4.20-0309-reasoning'
P420 = (500 * B, None)
M('Grok 4.20',
	V('Grok 4.20 Beta', ['Grok-4.20 Beta', 'Grok 4.20 (Beta)', 'Grok-4.20 Beta Reasoning', 'Grok-4.20 Beta Non-Reasoning', 'grok-4.20-beta', 'grok-4.20-beta-latest', 'grok-4.20-beta-reasoning', 'grok-4.20-beta-latest-reasoning', 'grok-4.20-experimental-beta-0304', 'grok-4.20-experimental-beta-0304-reasoning', 'grok-4.20-experimental-beta-latest', 'grok-4.20-experimental-beta-reasoning-latest'], GROK_NOTES, '2025-12-07', '2026-02-17', params=P420, efforts=R),
	V('Grok 4.20 Multi-agent', ['Grok 4.20 Multi-Agent', 'Grok-4.20 Multi-Agent Beta', 'Grok 4.20 Heavy', 'grok-4.20-multi-agent', 'grok-4.20-multi-agent-0309', 'grok-4.20-multi-agent-beta', 'grok-4.20-multi-agent-beta-0309'], 'https://docs.x.ai/developers/model-capabilities/text/multi-agent', '2026-02-18', '2026-02-18', params=P420, price=(1.25, 2.5), efforts=LMHX),
	V('Grok 4.20 (Mar 2026)', ['Grok 4.20 0309', 'grok-4.20-0309', 'grok-4.20-0309-reasoning', 'grok-4.20-0309-non-reasoning', 'grok-4.20-beta-0309', 'grok-4.20-beta-0309-reasoning'], G420, '2026-03-09', '2026-03-10', params=P420, price=(1.25, 2.5), efforts=R),
	V('Grok 4.20 (Apr 2026)', ['Grok 4.20 0309 v2', 'grok-4.20-reasoning-gv2'], G420, '2026-04-07', '2026-04-07', params=P420, price=(1.25, 2.5), efforts=R),
)

# Grok 4.3
M('Grok 4.3',
	V('Grok 4.3 Beta', ['Grok 4.3 (Beta)'], GROK_NOTES, '2026-04-17', '2026-04-17', params=P420),
	V('Grok 4.3', ['Grok-4.3', 'grok-4.3', 'grok-4.3-latest'], DOCS + 'grok-4.3', '2026-04-30', '2026-04-30', params=P420, price=(1.25, 2.5), efforts=NLMH),
)

# Grok Build
S('Grok Build 0.1', ['Grok Build 0.1 0616', 'grok-build-0.1', 'grok-build-0.1-0616'], NEWS + 'grok-build-0-1', '2026-05-14', '2026-05-14', price=(1, 2))

# Grok 4.5 to 4.7 (the Fast variants are the same model served on faster infrastructure)
S('Grok 4.5', ['Grok-4.5', 'grok-4.5', 'grok-4.5-latest'], NEWS + 'grok-4-5', '2026-04-18', '2026-07-09', params=(1_500 * B, None), price=(2, 6), efforts=LMH)
S('Grok 4.6', ['Grok-4.6', 'grok-4.6', 'Grok 4.6 Fast'], NEWS + 'grok-4-6', '2026-07-24', '2026-08-12', params=(1_500 * B, None), price=(2, 6), efforts=LMHX)
S('Grok 4.7', ['Grok-4.7', 'grok-4.7', 'Grok 4.7 Fast'], NEWS + 'grok-4-7', '2026-07-24', '2026-09-21', params=(2_100 * B, None), price=(2, 6), efforts=LMHX)

write('xai')
