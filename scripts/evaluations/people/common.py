"""Helpers for the scripts/evaluations/people/gen_<eval>.py generators (results of the groups of people of data/people,
i.e. human baselines).

A generator declares its results with P(), then calls write('<eval>'). write() validates everything against
data/evaluations/<eval>.json and data/people, sorts and rewrites data/results/people/<eval>.jsonl, so the generator is
the single source of truth for it. There are no leftovers: a group of people / version that does not exist yet is added
by hand to data/people before being used here. No uniqueness is imposed on the results, except rows identical to an
earlier one in every field but source.url.
"""

import glob, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared import A, CI, HF, PM, ROOT, SE, STD, check, result, without_identical_lines, write_jsonl	# noqa: E402,F401

# ---------------------------------------------------------------- people index
PEOPLE = {}		# people id -> {version id -> efforts}
for _path in glob.glob(os.path.join(ROOT, 'data', 'people', '*.json')):
	with open(_path, encoding='utf-8') as _fp:
		_people = json.load(_fp)
	PEOPLE[os.path.basename(_path)[:-5]] = {vid: v['efforts'] for vid, v in _people['versions'].items()}

RESULTS = []


def P(people, version, score, url, creator=False, shots=0, maj_k=1, pass_k=1, effort='auto', uncertainty=None, cost=None, part='', eval_version=''):
	"""A result of a group of people of data/people.

	version: the version (subgroup) of the group that was evaluated.
	creator: False (people have no creator).
	shots: number of solved examples shown to the people before the tasks (0 by default).
	maj_k: number of people whose answers are aggregated by majority vote into one answer (1 = individual answers).
	pass_k: number of attempts per task (1 by default).
	effort: 'auto' -> '' when the version has a default ('') effort.
	uncertainty: None, or CI(low, high, level) / PM(half_width, level) / SE(value) / STD(value, runs) as given by the source.
	cost: None, or (dollars, 'TASK' | 'EVALUATION') when the source gives what the people were really paid.
	part / eval_version: the evaluation's part and version ('' = the main one).
	"""
	RESULTS.append(result(PEOPLE, people, version, score, shots, url, creator, maj_k, pass_k, effort, uncertainty, cost, part, eval_version))


def write(evaluation):
	"""Validates and sorts the results, then rewrites data/results/people/<eval>.jsonl (or the file given as argument)."""
	check(evaluation, RESULTS)
	results = without_identical_lines(RESULTS)
	results.sort(key=lambda r: (r['subject']['id'], r['subject']['version'] or '', r['score']))
	write_jsonl([(os.path.join(ROOT, 'data', 'results', 'people', evaluation + '.jsonl'), results)])
	print(f'{len(results)} results ({len({r["subject"]["id"] for r in results})} groups of people)')
