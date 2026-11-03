"""Generates data/models/anthropic/*.json (one file per model).

All the data lives in this script: edit it and run `python scripts/models/gen_anthropic.py`.
Existing JSON files are deleted and rewritten, so manual edits to them are lost.
"""

from common import B, M, N, S, V, write

T = ['', 'Thinking']											# Extended thinking on/off switch (budget_tokens)
NLMH = ['None', 'Low', 'Medium', 'High']						# Opus 4.5: effort + thinking off
NLMHM = ['None', 'Low', 'Medium', 'High', 'Max']				# 4.6 models: adaptive thinking, can be disabled
LMHM = ['Low', 'Medium', 'High', 'Max']							# Mythos Preview: thinking always on, no xhigh
NLMHXM = ['None', 'Low', 'Medium', 'High', 'Xhigh', 'Max']		# Thinking can be disabled
LMHXM = ['Low', 'Medium', 'High', 'Xhigh', 'Max']				# Thinking always on

NEWS = 'https://www.anthropic.com/news/'

# Research models (papers only)
S('Anthropic-LM 52B', ['Anthropic LM 52B', 'Anthropic-LM v4-s3 (52B)', 'anthropic/stanford-online-all-v4-s3'], 'https://arxiv.org/abs/2112.00861', '2021-12-01', None, params=(52 * B, 52 * B))
S('Anthropic-LM 175B', ['Anthropic LM 175B'], 'https://arxiv.org/abs/2302.07459', '2023-02-15', None, params=(175 * B, 175 * B))

# Claude 1 and Claude Instant
CLAUDE1 = NEWS + 'introducing-claude'
CTX100K = NEWS + '100k-context-windows'
M('Claude 1',
	V('Claude 1.0', ['Claude v1.0', 'claude-1.0', 'claude-v1.0'], CLAUDE1, '2023-03-14', '2023-03-14', price=(11.02, 32.68)),
	V('Claude 1.3', ['Claude v1.3', 'Anthropic Claude v1.3', 'claude-1.3', 'claude-v1.3', 'claude-1.3-100k', 'claude-v1.3-100k', 'claude-1'], 'https://x.com/AnthropicAI/status/1648353600350060545', '2023-04-18', '2023-04-18', price=(11.02, 32.68)),
)
M('Claude Instant',
	V('Claude Instant 1.0', ['Claude Instant v1.0', 'claude-instant-1.0', 'claude-instant-v1.0'], CLAUDE1, '2023-03-14', '2023-03-14', price=(1.63, 5.51)),
	V('Claude Instant 1.1', ['Claude Instant v1.1', 'claude-instant-1.1', 'claude-instant-v1.1', 'claude-instant-v1.1-100k'], CTX100K, '2023-05-11', '2023-05-11', price=(1.63, 5.51)),
	V('Claude Instant 1.2', ['Claude Instant v1.2', 'claude-instant-1.2', 'claude-instant-v1.2'], NEWS + 'releasing-claude-instant-1-2', '2023-08-09', '2023-08-09', price=(1.63, 5.51)),
)

# Claude 2
S('Claude 2', ['Claude 2.0', 'claude-2.0', 'claude-2'], NEWS + 'claude-2', '2023-07-11', '2023-07-11', price=(11.02, 32.68))
S('Claude 2.1', ['claude-2.1'], NEWS + 'claude-2-1', '2023-11-21', '2023-11-21', price=(8, 24))

# Claude 3
CLAUDE3 = NEWS + 'claude-3-family'
S('Claude 3 Opus', ['Claude Opus 3', 'claude-3-opus-20240229', 'claude-3-opus', 'claude-3-opus-latest'], CLAUDE3, '2024-03-04', '2024-03-04', price=(15, 75))
S('Claude 3 Sonnet', ['Claude Sonnet 3', 'claude-3-sonnet-20240229', 'claude-3-sonnet'], CLAUDE3, '2024-03-04', '2024-03-04', price=(3, 15))
S('Claude 3 Haiku', ['Claude Haiku 3', 'claude-3-haiku-20240307', 'claude-3-haiku'], CLAUDE3, '2024-03-04', '2024-03-13', price=(0.25, 1.25))

# Claude 3.5 and 3.7
NEW35 = NEWS + '3-5-models-and-computer-use'
M('Claude 3.5 Sonnet',
	V('Claude 3.5 Sonnet (Jun 2024)', ['Claude 3.5 Sonnet (June 2024)', "Claude 3.5 Sonnet (June '24)", "Claude 3.5 Sonnet (Jun '24)", 'Claude Sonnet 3.5 (Jun 2024)', 'claude-3-5-sonnet-20240620'], NEWS + 'claude-3-5-sonnet', '2024-06-20', '2024-06-20', price=(3, 15)),
	V('Claude 3.5 Sonnet (Oct 2024)', ['Claude 3.5 Sonnet (October 2024)', "Claude 3.5 Sonnet (Oct '24)", 'Claude 3.5 Sonnet (new)', 'Claude 3.5 Sonnet (New)', 'Claude 3.6 Sonnet', 'Claude Sonnet 3.5 (Oct 2024)', 'claude-3-5-sonnet-20241022', 'claude-3-5-sonnet-latest'], NEW35, '2024-10-22', '2024-10-22', price=(3, 15)),
)
S('Claude 3.5 Haiku', ['Claude Haiku 3.5', 'claude-3-5-haiku-20241022', 'claude-3-5-haiku-latest'], NEW35, '2024-10-22', '2024-11-04', price=(0.8, 4))
S('Claude 3.7 Sonnet', ['Claude Sonnet 3.7', 'claude-3-7-sonnet-20250219', 'claude-3-7-sonnet-latest'], NEWS + 'claude-3-7-sonnet', '2025-02-24', '2025-02-24', price=(3, 15), efforts=T)

# Claude 4
CLAUDE4 = NEWS + 'claude-4'
S('Claude Sonnet 4', ['Claude 4 Sonnet', 'Sonnet 4', 'claude-sonnet-4-20250514', 'claude-sonnet-4-0'], CLAUDE4, '2025-05-22', '2025-05-22', price=(3, 15), efforts=T)
S('Claude Opus 4', ['Claude 4 Opus', 'Opus 4', 'claude-opus-4-20250514', 'claude-opus-4-0'], CLAUDE4, '2025-05-22', '2025-05-22', price=(15, 75), efforts=T)
S('Claude Gov', ['Claude Gov models'], NEWS + 'claude-gov-models-for-u-s-national-security-customers', '2025-06-05', None)
S('Claude Opus 4.1', ['Claude 4.1 Opus', 'Opus 4.1', 'claude-opus-4-1-20250805', 'claude-opus-4-1'], NEWS + 'claude-opus-4-1', '2025-08-05', '2025-08-05', price=(15, 75), efforts=T)
S('Claude Sonnet 4.5', ['Claude 4.5 Sonnet', 'Sonnet 4.5', 'claude-sonnet-4-5-20250929', 'claude-sonnet-4-5'], NEWS + 'claude-sonnet-4-5', '2025-09-29', '2025-09-29', price=(3, 15), efforts=T)
S('Claude Haiku 4.5', ['Claude 4.5 Haiku', 'Haiku 4.5', 'claude-haiku-4-5-20251001', 'claude-haiku-4-5'], NEWS + 'claude-haiku-4-5', '2025-10-15', '2025-10-15', price=(1, 5), efforts=T)
S('Claude Opus 4.5', ['Claude 4.5 Opus', 'Opus 4.5', 'claude-opus-4-5-20251101', 'claude-opus-4-5'], NEWS + 'claude-opus-4-5', '2025-11-24', '2025-11-24', price=(5, 25), efforts=NLMH)
S('Claude Opus 4.6', ['Claude 4.6 Opus', 'Opus 4.6', 'claude-opus-4-6'], NEWS + 'claude-opus-4-6', '2026-02-05', '2026-02-05', price=(5, 25), efforts=NLMHM)
S('Claude Sonnet 4.6', ['Claude 4.6 Sonnet', 'Sonnet 4.6', 'claude-sonnet-4-6'], NEWS + 'claude-sonnet-4-6', '2026-02-17', '2026-02-17', price=(3, 15), efforts=NLMHM)
S('Claude Opus 4.7', ['Claude 4.7 Opus', 'Opus 4.7', 'claude-opus-4-7'], NEWS + 'claude-opus-4-7', '2026-04-16', '2026-04-16', price=(5, 25), efforts=NLMHXM)
S('Claude Opus 4.8', ['Claude 4.8 Opus', 'Opus 4.8', 'claude-opus-4-8'], NEWS + 'claude-opus-4-8', '2026-05-28', '2026-05-28', price=(5, 25), efforts=NLMHXM)

# Claude 5 (Mythos-class: Fable is the public release, Mythos the same model with fewer safeguards for trusted access)
FABLE5 = NEWS + 'claude-fable-5-mythos-5'
M('Claude Fable 5',
	V('Claude Mythos Preview', ['Mythos Preview', 'claude-mythos-preview', 'claude-mythos-preview-early'], 'https://www.anthropic.com/glasswing', '2026-03-26', None, price=(25, 125), efforts=LMHM),
	V('Claude Fable 5', ['Fable 5', 'claude-fable-5'], FABLE5, '2026-06-09', '2026-06-09', price=(10, 50), efforts=LMHXM),
	V('Claude Mythos 5', ['Mythos 5', 'claude-mythos-5'], FABLE5, '2026-06-09', None, price=(10, 50), efforts=LMHXM),
)
S('Claude Sonnet 5', ['Sonnet 5', 'claude-sonnet-5'], NEWS + 'claude-sonnet-5', '2026-06-30', '2026-06-30', price=(2, 10), efforts=NLMHXM)
S('Claude Opus 5', ['Opus 5', 'claude-opus-5'], NEWS + 'claude-opus-5', '2026-07-24', '2026-07-24', price=(5, 25), efforts=NLMHXM)
FABLE51 = 'https://www.anthropic.com/claude-fable-and-mythos-5-1'
M('Claude Fable 5.1',
	V('Claude Fable 5.1', ['Fable 5.1', 'claude-fable-5-1'], FABLE51, '2026-09-01', '2026-09-01', price=(10, 50), efforts=LMHXM),
	V('Claude Mythos 5.1', ['Mythos 5.1', 'claude-mythos-5-1'], FABLE51, '2026-09-01', None, price=(10, 50), efforts=LMHXM),
)

# Claude 5.5
S('Claude Opus 5.5', ['Opus 5.5', 'claude-opus-5-5'], 'https://www.anthropic.com/claude-opus-5-5', '2026-09-22', '2026-09-22', price=(4, 20), efforts=LMHXM)
S('Claude Sonnet 5.5', ['Sonnet 5.5', 'claude-sonnet-5-5'], 'https://www.anthropic.com/claude-sonnet-5-5', '2026-09-28', '2026-09-28', price=(2, 10), efforts=LMHXM)
S('Claude Haiku 5.5', ['Haiku 5.5', 'claude-haiku-5-5'], 'https://www.anthropic.com/claude-haiku-5-5', '2026-10-07', '2026-10-07', price=(0.1, 0.5), efforts=NLMHXM)

write('anthropic')
