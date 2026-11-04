"""Helpers for the scripts/evaluations/models/gen_<eval>.py generators (results of the models of data/models).

A generator declares its results with R() and its leftovers (models that are not in data/models) with L(), then calls
write('<eval>'). write() validates everything against data/evaluations/<eval>.json, sorts and rewrites
data/results/models/<eval>.jsonl and data/leftovers/<eval>.jsonl, so the generator is the single source of truth for
them. No uniqueness is imposed on the results: every score found is kept (the selection is done later from this raw
data), except rows identical to an earlier one in every field but source.url.
"""

import glob, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared import A, CI, HF, PM, ROOT, SE, STD, check, result, without_identical_lines, write_jsonl	# noqa: E402,F401

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


def R(model, version, score, shots, url, creator, maj_k=1, pass_k=1, effort='auto', uncertainty=None, cost=None, part='', eval_version=''):
	"""A result of a model of data/models.

	version: None when the source does not say which version was evaluated.
	effort: 'auto' -> '' when the version has a default ('') effort, None otherwise (and when version is None).
	shots: 0 when the source does not give it.
	uncertainty: None, or CI(low, high, level) / PM(half_width, level) / SE(value) / STD(value, runs) as given by the source.
	cost: None, or (dollars, 'TASK' | 'EVALUATION') when the source gives the real cost of the run (never from API prices).
	part / eval_version: the evaluation's part and version ('' = the main one).
	"""
	RESULTS.append(result(MODELS, model, version, score, shots, url, creator, maj_k, pass_k, effort, uncertainty, cost, part, eval_version))


def L(name, score, url, part='', eval_version=''):
	"""A result of a model that is not in data/models (name = its real name, not an id)."""
	LEFTOVERS.append({'model': name, 'score': round(float(score), 2), 'part': part, 'version': eval_version, 'source': url})


def write(evaluation):
	"""Validates and sorts the results, then rewrites data/results/models/<eval>.jsonl and data/leftovers/<eval>.jsonl
	(or the files given as first and second arguments)."""
	check(evaluation, RESULTS + LEFTOVERS)
	results, leftovers = without_identical_lines(RESULTS), without_identical_lines(LEFTOVERS)
	# Results: chronological by first_mention of the version (of the model's first version when not specified)
	results.sort(key=lambda r: (DATES[r['subject']['id'], r['subject']['version']], r['subject']['id'], r['subject']['version'] or '', r['score']))
	# Leftovers: alphabetical by model name
	leftovers.sort(key=lambda l: (l['model'].lower(), l['model'], l['score']))
	write_jsonl([
		(os.path.join(ROOT, 'data', 'results', 'models', evaluation + '.jsonl'), results),
		(os.path.join(ROOT, 'data', 'leftovers', evaluation + '.jsonl'), leftovers),
	])
	print(f'{len(results)} results ({len({r["subject"]["id"] for r in results})} models), {len(leftovers)} leftovers')
