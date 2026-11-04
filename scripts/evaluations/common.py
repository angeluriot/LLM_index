"""Shared helpers for the scripts/evaluations/gen_<eval>.py generators.

A generator declares its results with R() and its leftovers with L(), then calls write('<eval>').
write() validates everything against data/evaluations/<eval>.json, sorts and rewrites data/results/<eval>.jsonl and
data/leftovers/<eval>.jsonl, so the generator is the single source of truth for them. No uniqueness is imposed on the
results: every score found is kept (the selection is done later from this raw data), except rows identical to an
earlier one in every field but source.url.
"""

import glob, json, os, sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

# Common source URL prefixes
A = 'https://arxiv.org/abs/'
HF = 'https://huggingface.co/'

# ---------------------------------------------------------------- models index
MODELS = {}		# model id -> {version id -> efforts}
DATES = {}		# (model id, version id) -> first_mention, (model id, None) -> first_mention of the model's first version
for _path in glob.glob(os.path.join(ROOT, 'data', 'models', '*', '*.json')):
	with open(_path, encoding='utf-8') as _fp:
		_model = json.load(_fp)
	_mid = os.path.basename(_path)[:-5]
	MODELS[_mid] = {vid: v['efforts'] for vid, v in _model['versions'].items()}
	for _vid, _v in _model['versions'].items():
		DATES[_mid, _vid] = _v['history']['first_mention']
	DATES[_mid, None] = min(DATES[_mid, vid] for vid in _model['versions'])

RESULTS = []
LEFTOVERS = []


# ---------------------------------------------------------------- uncertainty helpers (values in the evaluation's unit)
def CI(low, high, level=0.95):
	"""Confidence interval given by its bounds."""
	return {'type': 'CI', 'low': float(low), 'high': float(high), 'level': float(level)}


def PM(half_width, level=0.95):
	"""Symmetric confidence interval given as score ± half_width (resolved into bounds by R)."""
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


def R(model, version, score, shots, url, creator, maj_k=1, pass_k=1, effort='auto', uncertainty=None, cost=None, part='', eval_version=''):
	"""A result of a model of data/models.

	version: None when the source does not say which version was evaluated.
	effort: 'auto' -> '' when the version has a default ('') effort, None otherwise (and when version is None).
	shots: 0 when the source does not give it.
	uncertainty: None, or CI(low, high, level) / PM(half_width, level) / SE(value) / STD(value, runs) as given by the source.
	cost: None, or (dollars, 'TASK' | 'EVALUATION') when the source gives the real cost of the run (never from API prices).
	part / eval_version: the evaluation's part and version ('' = the main one).
	"""
	assert model in MODELS, f'unknown model {model}'
	assert version is None or version in MODELS[model], f'unknown version {model}/{version}'
	assert isinstance(shots, int) and shots >= 0 and maj_k >= 1 and pass_k >= 1, (model, version, shots, maj_k, pass_k)
	if effort == 'auto':
		if version is None:
			effort = '' if all(e == [''] for e in MODELS[model].values()) else None
		else:
			effort = '' if '' in MODELS[model][version] else None
	elif effort is not None:
		assert version is not None and effort in MODELS[model][version], f'unknown effort {effort!r} for {model}/{version}'
	if cost is not None:
		assert cost[1] in ('TASK', 'EVALUATION'), cost
		cost = {'cost': float(cost[0]), 'type': cost[1]}
	RESULTS.append({
		'model': {'id': model, 'version': version, 'effort': effort},
		'score': round(float(score), 2),
		'uncertainty': _uncertainty(uncertainty, float(score)),
		'conditions': {'shots': shots, 'pass_k': pass_k, 'maj_k': maj_k},
		'cost': cost,
		'part': part,
		'version': eval_version,
		'source': {'url': url, 'creator': creator},
	})


def L(name, score, url, part='', eval_version=''):
	"""A result of a model that is not in data/models (name = its real name, not an id)."""
	LEFTOVERS.append({'model': name, 'score': round(float(score), 2), 'part': part, 'version': eval_version, 'source': url})


def _without_url(row):
	row = dict(row)
	if isinstance(row['source'], dict):
		row['source'] = {k: v for k, v in row['source'].items() if k != 'url'}
	else:
		del row['source']	# leftovers: source is only the URL
	return row


def _without_identical_lines(rows):
	# No uniqueness is imposed: several scores for the same model are all kept (the selection is done later from this
	# raw data). A row is only dropped when an earlier one has exactly the same fields except source.url (the same number
	# copied by another source): a single other difference (score, effort, conditions, source.creator...) keeps it.
	out, seen = [], set()
	for row in rows:
		key = json.dumps(_without_url(row), sort_keys=True)
		if key not in seen:
			seen.add(key)
			out.append(row)
	return out


def _sort():
	results, leftovers = _without_identical_lines(RESULTS), _without_identical_lines(LEFTOVERS)
	# Results: chronological by first_mention of the version (of the model's first version when not specified)
	results.sort(key=lambda r: (DATES[r['model']['id'], r['model']['version']], r['model']['id'], r['model']['version'] or '', r['score']))
	# Leftovers: alphabetical by model name
	leftovers.sort(key=lambda l: (l['model'].lower(), l['model'], l['score']))
	return results, leftovers


def write(evaluation):
	"""Validates and sorts the results, then rewrites data/results/<eval>.jsonl and data/leftovers/<eval>.jsonl
	(or the files given as first and second arguments)."""
	path = os.path.join(ROOT, 'data', 'evaluations', evaluation + '.json')
	assert os.path.isfile(path), f'missing data/evaluations/{evaluation}.json'
	with open(path, encoding='utf-8') as fp:
		config = json.load(fp)
	# Scores are not checked against min / max: those bounds are only used for normalisation
	for row in RESULTS + LEFTOVERS:
		assert row['part'] in config['parts'], f'unknown part {row["part"]!r}: {row}'
		assert row['version'] in config['versions'], f'unknown evaluation version {row["version"]!r}: {row}'
	results, leftovers = _sort()
	outputs = [
		(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'data', 'results', evaluation + '.jsonl'), results),
		(sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, 'data', 'leftovers', evaluation + '.jsonl'), leftovers),
	]
	for out, rows in outputs:
		os.makedirs(os.path.dirname(out), exist_ok=True)
		with open(out, 'w', encoding='utf-8', newline='\n') as fp:
			for row in rows:
				fp.write(json.dumps(row, ensure_ascii=False) + '\n')
	print(f'{len(results)} results ({len({r["model"]["id"] for r in results})} models), {len(leftovers)} leftovers')
