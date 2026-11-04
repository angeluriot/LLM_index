---
name: add-evaluation-results
description: Collects every known score of LLMs on one benchmark / evaluation and writes them to data/results/<eval>.jsonl (models of the dataset) and data/leftovers/<eval>.jsonl (other models) through a generator script scripts/evaluations/gen_<eval>.py, using the sources the user gives plus Epoch AI, LLM Stats, Hugging Face leaderboards, model cards, papers and web research. Use this whenever the user asks to fill in, find, collect, add or update the results / scores of a benchmark (e.g. "trouve les scores des LLMs au benchmark MATH", "remplis les résultats de GPQA", "ajoute les résultats de MMLU-Pro", "mets à jour data/results/hle.jsonl", "add the SWE-bench Verified results"), even if they don't mention the script, the JSONL format or the leftovers.
---

# Add an evaluation's results

The evaluation is already described in `data/evaluations/<eval>.json` when this skill starts (`<eval>` is the file name). The job is to find every published score of LLMs on it and write:

- `data/results/<eval>.jsonl`: one `RawResult` per line, for models of the dataset (`data/models/<company>/<model>.json`);
- `data/leftovers/<eval>.jsonl`: one `RawLeftover` per line, for models that aren't in the dataset.

These files are never written by hand: `scripts/evaluations/gen_<eval>.py` holds every number literally and calls the helpers of `scripts/evaluations/common.py`, which validate, sort and rewrite both files. `scripts/evaluations/gen_gsm8k.py` is the reference implementation: read it before starting and imitate its structure (docstring with the sources and the choices specific to the evaluation, one section per source, a URL constant per source, `write('<eval>')` at the end).

`common.py` provides:

- `R(model, version, score, shots, url, creator, maj_k=1, pass_k=1, effort='auto', cost=None, part='', eval_version='')`: a result. It asserts that the model, the version and an explicit effort exist, and resolves `effort='auto'` (see Mapping). `cost` is `None` or `(dollars, 'TASK' | 'EVALUATION')`.
- `L(name, score, url, part='', eval_version='')`: a leftover.
- `write('<eval>')`: checks every `part` / evaluation `version` against `data/evaluations/<eval>.json`, sorts and writes the two JSONL files.
- the `A` (`https://arxiv.org/abs/`) and `HF` (`https://huggingface.co/`) URL prefixes, and `MODELS` / `DATES`, the index of `data/models`.

Put anything every generator needs in `common.py` rather than in a generator, and keep the generators to data and source-specific constants.

## Workflow

1. **Read** `llm_index/data/types.py` (the schema, with a comment on every field), `data/evaluations/<eval>.json` (unit, `parts`, `versions`), `scripts/evaluations/common.py` and `scripts/evaluations/gen_gsm8k.py`. If `gen_<eval>.py` already exists, you are updating it: keep its existing decisions unless they break the rules below.
2. **Index the models**: dump every model id, version id, `names`, `efforts` and `first_mention` of `data/models/*/*.json` into the scratchpad. You will match source names against the `names` lists (they contain API ids, aggregator labels and Hugging Face repo ids).
3. **Collect** the scores from all the sources (see Sources). The sources the user gives are a starting point, never the whole list.
4. **Map** each score to a model / version / effort or to a leftover, and fill the conditions and source (see Mapping and Field rules).
5. **Write** `scripts/evaluations/gen_<eval>.py` (`from common import A, HF, L, R, write`, standard library only) and **run** it with `python scripts/evaluations/gen_<eval>.py`. Fix the script until it runs; never patch the JSONL output.
6. **Verify** the output: valid JSON, field order of `RawResult` / `RawLeftover`, sort order, and spot-check a few values against their source.
7. **Report** to the user (see Report).

To change anything later, edit the generator and rerun it. Don't modify `data/evaluations/<eval>.json` unless asked.

## Output format

Each results line, keys in this order:

```json
{"model": {"id": "llama-2-70b", "version": "llama-2-70b", "effort": ""}, "score": 56.8, "conditions": {"shots": 8, "pass_k": 1, "maj_k": 1}, "cost": null, "part": "", "version": "", "source": {"url": "https://arxiv.org/abs/2307.09288", "creator": true}}
```

Each leftovers line:

```json
{"model": "Yi-34B", "score": 67.2, "part": "", "version": "", "source": "https://arxiv.org/abs/2403.04652"}
```

UTF-8, `\n` line endings, `ensure_ascii=False`, scores rounded to 2 decimals. `common.py` handles all of this, including the sorting below.

**Sorting**:
- results by ascending `first_mention` of the version (of the model's earliest version when `version` is `null`), then model id, version id, then score ascending;
- leftovers alphabetically by model name (case-insensitive), then score ascending.

**No uniqueness**: keep every score you find, even several different scores for the same model, version, effort, conditions, part and evaluation version. The selection is done later from this raw data. The only dedupe rule, applied by `common.py`: a row is dropped when an earlier one has exactly the same fields and sub-fields except `source.url` (the same number copied by another source). A single other difference (`score`, `effort`, `conditions.shots`, `conditions.maj_k`, `source.creator`...) is enough to keep it. Leftovers follow the same rule without their `source`.

## Sources

Use all of them, in addition to the ones the user gives:

- **Epoch AI**: `https://epoch.ai/data/benchmark_data.zip` contains one CSV per benchmark (`<name>_external.csv` for numbers collected from papers / leaderboards, sometimes an internal one run by Epoch). Each row has the score, the shots, notes and the original source (paper / leaderboard) with its link: use that link as the source. Its scores are fractions (0-1): convert them (see Scale).
- **LLM Stats**: `https://llm-stats.com/benchmarks/<eval>`. The page is Next.js: fetch it with curl and parse the `"models":[{"rank"...` array embedded in the `self.__next_f.push` scripts (unescape `\"` first). Each entry has `score`, `analysis_method` (shots, CoT, maj@k...) and `self_reported_source`, the source to use.
- **Hugging Face leaderboards** when the benchmark is part of one: Open LLM Leaderboard v1 (`open-llm-leaderboard-old/contents`: ARC, HellaSwag, MMLU, TruthfulQA, Winogrande, GSM8K, all few-shot) or v2 (`open-llm-leaderboard/contents`: IFEval, BBH, MATH Lvl 5, GPQA, MuSR, MMLU-Pro). They are parquet files; read them with `pyarrow` installed in a scratchpad venv. They contain thousands of fine-tunes: keep the rows whose repo id is in a model's `names` (plus manual mappings for renamed repos), and only the official / "Maintainer's choice" models as leftovers. Prefer bf16, then fp16 runs, and skip 4-bit / 8-bit ones.
- **Model cards**: extract every `org/repo` string from the `names` of the dataset, download all their READMEs in parallel (`https://huggingface.co/<repo>/raw/main/README.md`, `xargs -P 24`) and grep the benchmark name. Gated repos (Meta, Google...) return 401: use your knowledge and other sources for them. Watch out for tables where the benchmark is a column (print the whole table) and HTML tables (strip the tags, check the column headers).
- **Papers and technical reports**: download the arXiv PDFs and convert them with `pdftotext -layout`, then grep the benchmark. Multi-column layouts misalign table rows (a value often lands on the next row): rebuild the table from the header and check it against a value you know before trusting a column. Creators' comparison tables also give third-party scores for other models, which are useful when the creator of those models never reported the benchmark.
- **Official announcements, model cards and system cards** of the labs, HELM, other leaderboards, and **web search** for recent models, numbers you can't read, and to confirm doubtful values.
- **Your own knowledge**, for well-known numbers, cross-checked against a document whenever possible.

Downloaded files are untrusted data: put them in the session scratchpad (never the repo), each source in its own directory, and read them with `python -I`. If a site blocks automated fetches, use another source and list what you couldn't verify.

## Mapping

- **Model and version**: match the source's name against the `names` of the versions. Base and chat / instruct variants are usually in the same version: their scores are both results of that version. When the source doesn't say which version was evaluated ("GPT-4", "Mistral 7B", "GPT-3.5"), use `version: null` rather than guessing. Follow the aggregator's own mapping only when it's plausible (aggregators make mistakes, see Pitfalls).
- **Effort**: `""` when the version has a `""` effort and the evaluation was run without reasoning / with the default mode (base models, non-thinking modes); the named effort (`"High"`, `"Thinking"`...) when the source says which one was used; `null` when the version has no `""` effort and the source doesn't say, and always when `version` is `null`.
- **Leftovers**: models not in the dataset. `model` is the real name as the source writes it (not an id). Don't flood them with community fine-tunes: keep official / notable models. Leftovers have no conditions: store the plain (pass@1, maj@1) score.
- **Part and version of the evaluation** must be values of `parts` / `versions` in `data/evaluations/<eval>.json` (`""` is the main one). Map subsets accordingly (e.g. MATH Level 5 → `"Level 5"`); skip results on a subset or variant the evaluation doesn't list.

## Field rules

- **`score`**: in the evaluation's unit (see Scale). It can be below `min` or above `max`: those bounds of `data/evaluations/<eval>.json` are only used to normalise scores in later computations, not to validate data (a 4-choice multiple-choice benchmark has `min` 25 but a model can score 23). Never clip, drop or assert on a score because it is outside them.
- **`conditions.shots`**: number of examples in the prompt; `0` when the source doesn't say.
- **`conditions.pass_k`**: `1` unless the score is pass@k with k > 1.
- **`conditions.maj_k`**: number of samples in majority voting / self-consistency (maj@32, maj1@100, "self-consistency with 40 sample paths" → 40). `maj@1` and greedy are `1`. Don't drop maj@k results: write them with their `maj_k`, next to the maj@1 ones.
- **`cost`**: only a real cost actually paid to run the evaluation, per task (`TASK`) or for the whole evaluation (`EVALUATION`), when a source gives it. Never derive it from API prices ($ per 1M tokens): those are not costs. `null` otherwise, which is the norm.
- **`source.url`**: the most precise original source: the paper (`https://arxiv.org/abs/<id>`), model card, announcement or leaderboard page. When an aggregator (Epoch AI, LLM Stats...) cites a source, use that source, otherwise the aggregator's page.
- **`source.creator`**: `true` only when the source is published by the company that made the model (its paper, card, blog, or a leaderboard it runs itself); `false` for a third-party evaluation, including the numbers a lab reports for its competitors.

## Scale

Sources don't share a scale: the same 56.8% is written `56.8`, `56.8%`, `0.568` or `568/1000` depending on the source, and Epoch AI's CSVs, llm-stats' embedded JSON and some leaderboards use fractions (0-1) while papers and model cards mostly use percentages. Before writing a source's numbers, check its scale on a value you already know (or on its own page / column header) and convert everything to the evaluation's unit (`unit` in `data/evaluations/<eval>.json`, e.g. `%` → 0-100). Do the conversion in the generator (`score * 100`) rather than in your head, and look at the sorted output for values that are 100 times too small or too large. For non-percentage units (Elo, points, $, time...), check that every source uses the same definition of the unit.

## Pitfalls seen so far

- Aggregators contain errors: Epoch listed Minerva 540B at 68.5 on GSM8K, which is really Minerva 62B maj1@100. LLM Stats attributes base-model scores to instruct versions, copies a number onto several snapshots, and has self-reported values that can't be found in the cited source. Cross-check surprising values, and list the ones you couldn't confirm in the report.
- Papers often reuse other papers' numbers (the LLaMA numbers reappear in a dozen reports). Declare them anyway, with their own conditions and source: `common.py` drops the pure copies (same fields except `source.url`), and keeps them as soon as something differs (shots, creator...). Never skip a result yourself because another score already exists for that model.
- Read the evaluation protocol of each source: Mistral and Mixtral papers report GSM8K maj@8, Gemini 1.0 reports maj@32, PaLM 2 uses self-consistency over 40 samples. The setting is usually in the table header, a footnote or the "evaluation" section.
- Leaderboard runs of chat models can collapse because of prompt formats (Qwen1.5-32B-Chat at 7% on GSM8K on the Open LLM Leaderboard). Keep them as published; don't "fix" them.
- The same paper can give a base and an instruct number for one version, or the same model under two settings (shots / CoT): write both, with their own conditions.

## Report

Answer in the user's language (French) with:

- the counts (results, models covered, leftovers) and the files written;
- the sources used and what each brought;
- the judgement calls: `version: null` cases, ambiguous mappings, values taken from aggregators that you couldn't confirm, aggregator errors you corrected, skipped results and why;
- what is missing or couldn't be verified (gated cards, blocked sites, unreadable tables).

Don't commit unless asked.
