"""Shared helpers for the scripts/models/gen_<company>.py generators.

A generator declares its models with M() / S() / V(), then calls write('<company>').
write() validates everything, deletes the company's existing JSON files and rewrites them,
so the generator is the single source of truth for data/models/<company>/.
"""

import glob, json, os, re, sys
from datetime import date

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
B = 1_000_000_000
N = ['']	# No configurable effort

MODELS = {}


def slug(s):
	return re.sub(r'[^a-z0-9]+', '-', s.lower()).strip('-')


def V(display, names, link, first, access, id=None, weights=None, params=(None, None), price=(None, None), efforts=N):
	"""A model version. Its id defaults to the slug of its display name."""
	f = lambda x: None if x is None else float(x)
	return id or slug(display), {
		'display_name': display,
		'names': [display] + [n for n in names if n != display],
		'link': link,
		'history': {'first_mention': first, 'access': access, 'weights': weights},
		'parameters': {'all': params[0], 'active': params[1]},
		'price': {'input': f(price[0]), 'output': f(price[1])},
		'efforts': efforts,
	}


def M(display, *versions, id=None):
	"""A model grouping one or more versions. Its id (= file name) defaults to the slug of its display name."""
	id = id or slug(display)
	assert id not in MODELS, f'duplicate model id {id}'
	ids = [v[0] for v in versions]
	assert len(set(ids)) == len(ids), f'duplicate version ids in {id}: {ids}'
	# Versions sorted by ascending first mention (stable: ties keep the declaration order)
	versions = sorted(versions, key=lambda v: v[1]['history']['first_mention'])
	MODELS[id] = {'display_name': display, 'versions': dict(versions)}


def S(display, *args, **kw):
	"""A single-version model whose model and version share the same display name and id."""
	v = V(display, *args, **kw)
	M(display, v, id=v[0])


def _other_companies_names(company):
	names = {}
	for path in glob.glob(os.path.join(ROOT, 'data', 'models', '*', '*.json')):
		if os.path.basename(os.path.dirname(path)) == company:
			continue
		with open(path, encoding='utf-8') as fp:
			for version in json.load(fp)['versions'].values():
				for name in version['names']:
					names[name] = path
	return names


def _validate(company):
	assert os.path.isfile(os.path.join(ROOT, 'data', 'companies', company + '.json')), f'missing data/companies/{company}.json'
	others = _other_companies_names(company)
	seen = {}
	for mid, m in MODELS.items():
		for vid, v in m['versions'].items():
			where = f'{mid}/{vid}'
			h = v['history']
			dates = {k: date.fromisoformat(d) for k, d in h.items() if d is not None}
			assert 'first_mention' in dates, f'{where}: first_mention is required'
			assert all(d >= dates['first_mention'] for d in dates.values()), f'{where}: a date precedes first_mention'
			assert dates['first_mention'] >= date(2019, 2, 14), f'{where}: predates GPT-2'
			assert v['efforts'], f'{where}: efforts must not be empty'
			assert v['link'].startswith('http'), f'{where}: bad link'
			p = v['parameters']
			assert p['all'] is None or p['active'] is None or p['active'] <= p['all'], f'{where}: active > all parameters'
			assert len(set(v['names'])) == len(v['names']), f'{where}: duplicate names'
			for n in v['names']:
				assert n not in seen, f'name {n!r} used by both {seen.get(n)} and {where}'
				assert n not in others, f'name {n!r} of {where} already used in {others.get(n)}'
				seen[n] = where


def write(company):
	"""Validates MODELS and rewrites data/models/<company>/ (or the directory given as first argument)."""
	_validate(company)
	out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'data', 'models', company)
	os.makedirs(out, exist_ok=True)
	for path in glob.glob(os.path.join(out, '*.json')):
		os.remove(path)
	for mid, m in MODELS.items():
		with open(os.path.join(out, mid + '.json'), 'w', encoding='utf-8', newline='\n') as fp:
			fp.write(json.dumps(m, indent='\t', ensure_ascii=False) + '\n')
	print(len(MODELS), 'models,', sum(len(m['versions']) for m in MODELS.values()), 'versions')
	for mid, m in MODELS.items():
		print(f'  {mid:28}', ', '.join(m['versions']))
