"""Helpers shared by scripts/evaluations/models/common.py and scripts/evaluations/people/common.py.

The generators never import this file directly: they import the common.py of their directory, which re-exports what
they need from here.
"""

import json, os, sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

# Common source URL prefixes
A = 'https://arxiv.org/abs/'
HF = 'https://huggingface.co/'


# ---------------------------------------------------------------- uncertainty helpers (values in the evaluation's unit)
def CI(low, high, level=0.95):
	"""Confidence interval given by its bounds (absolute values, not offsets from the score)."""
	return {'type': 'CI', 'low': float(low), 'high': float(high), 'level': float(level)}


def PM(half_width, level=0.95):
	"""Symmetric confidence interval given as score ± half_width (resolved into bounds by result())."""
	return {'type': 'PM', 'half_width': float(half_width), 'level': float(level)}


def SE(value):
	"""Standard error of the score."""
	return {'type': 'SE', 'value': float(value)}


def STD(value, runs=None):
	"""Standard deviation of the score across runs (runs: number of runs averaged, None if not given)."""
	return {'type': 'STD', 'value': float(value), 'runs': runs}


def _uncertainty(u, score):
	if u is None:
		return None
	if u['type'] == 'PM':
		u = CI(score - u['half_width'], score + u['half_width'], u['level'])
	if u['type'] == 'CI':
		assert u['low'] <= score <= u['high'] and 0 < u['level'] < 1, (score, u)
		return {'type': 'CI', 'low': round(u['low'], 2), 'high': round(u['high'], 2), 'level': u['level']}
	assert u['value'] >= 0, u
	if u['type'] == 'SE':
		return {'type': 'SE', 'value': round(u['value'], 2)}
	assert u['type'] == 'STD' and (u['runs'] is None or (isinstance(u['runs'], int) and u['runs'] >= 1)), u
	return {'type': 'STD', 'value': round(u['value'], 2), 'runs': u['runs']}


# ---------------------------------------------------------------- results
def _effort(index, subject, version, effort):
	# 'auto' -> '' when the version has a default ('') effort (every version when not specified), None otherwise
	if effort == 'auto':
		if version is None:
			return '' if all(e == [''] for e in index[subject].values()) else None
		return '' if '' in index[subject][version] else None
	if effort is not None:
		assert version is not None and effort in index[subject][version], f'unknown effort {effort!r} for {subject}/{version}'
	return effort


def result(index, subject, version, score, shots, url, creator, maj_k, pass_k, effort, uncertainty, cost, part, eval_version):
	"""Builds a RawResult row (keys in the order of types.py). index: subject id -> {version id -> efforts}."""
	assert subject in index, f'unknown subject {subject}'
	assert version is None or version in index[subject], f'unknown version {subject}/{version}'
	assert isinstance(shots, int) and shots >= 0 and maj_k >= 1 and pass_k >= 1, (subject, version, shots, maj_k, pass_k)
	if cost is not None:
		assert cost[1] in ('TASK', 'EVALUATION'), cost
		cost = {'cost': float(cost[0]), 'type': cost[1]}
	return {
		'subject': {'id': subject, 'version': version, 'effort': _effort(index, subject, version, effort)},
		'score': round(float(score), 2),
		'uncertainty': _uncertainty(uncertainty, float(score)),
		'conditions': {'shots': shots, 'pass_k': pass_k, 'maj_k': maj_k},
		'cost': cost,
		'part': part,
		'version': eval_version,
		'source': {'url': url, 'creator': creator},
	}


# ---------------------------------------------------------------- output
def _without_url(row):
	row = dict(row)
	if isinstance(row['source'], dict):
		row['source'] = {k: v for k, v in row['source'].items() if k != 'url'}
	else:
		del row['source']	# leftovers: source is only the URL
	return row


def without_identical_lines(rows):
	"""No uniqueness is imposed: several scores for the same subject are all kept (the selection is done later from this
	raw data). A row is only dropped when an earlier one has exactly the same fields except source.url (the same number
	copied by another source): a single other difference (score, effort, conditions, source.creator...) keeps it."""
	out, seen = [], set()
	for row in rows:
		key = json.dumps(_without_url(row), sort_keys=True)
		if key not in seen:
			seen.add(key)
			out.append(row)
	return out


def check(evaluation, rows):
	"""Checks every part / evaluation version against data/evaluations/<eval>.json.
	Scores are not checked against min / max: those bounds are only used for normalisation."""
	path = os.path.join(ROOT, 'data', 'evaluations', evaluation + '.json')
	assert os.path.isfile(path), f'missing data/evaluations/{evaluation}.json'
	with open(path, encoding='utf-8') as fp:
		config = json.load(fp)
	for row in rows:
		assert row['part'] in config['parts'], f'unknown part {row["part"]!r}: {row}'
		assert row['version'] in config['versions'], f'unknown evaluation version {row["version"]!r}: {row}'


def write_jsonl(outputs):
	"""outputs: [(default path, rows)]. The i-th default path can be overridden by the i-th command line argument."""
	for i, (default, rows) in enumerate(outputs):
		out = sys.argv[i + 1] if len(sys.argv) > i + 1 else default
		os.makedirs(os.path.dirname(out), exist_ok=True)
		with open(out, 'w', encoding='utf-8', newline='\n') as fp:
			for row in rows:
				fp.write(json.dumps(row, ensure_ascii=False) + '\n')
