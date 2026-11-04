---
name: add-evaluation-people-results
description: Collects every known human baseline of one benchmark / evaluation (scores of groups of people such as experts, crowd workers, students, the general public) and writes them to data/results/people/<eval>.jsonl through a generator script scripts/evaluations/people/gen_<eval>.py, adding the missing groups of people / versions by hand to data/people. Research usually starts from scratch: the benchmark's paper and appendix, follow-up papers, official sites and leaderboards with a "Human" row, Epoch AI benchmark pages, exam statistics and web research. Use this whenever the user asks for the human scores / human baselines / human performance / expert level of a benchmark (e.g. "trouve les scores humains sur GPQA", "ajoute les baselines humaines de ARC-AGI", "remplis data/results/people/hle.jsonl", "add the human baselines of MMLU"), even if they don't mention the script, the JSONL format or data/people. Not for the scores of models: that is the add-evaluation-model-results skill.
---

# Add an evaluation's people results

The evaluation is already described in `data/evaluations/<eval>.json` when this skill starts (`<eval>` is the file name). The job is to find every published score of groups of people (human baselines) on it and write:

- `data/results/people/<eval>.jsonl`: one `RawResult` per line, whose `subject` is a group of people of `data/people/<people>.json` and one of its versions;
- the groups of people / versions that don't exist yet, added **by hand** to `data/people/<people>.json` (no script for these files).

There are no leftovers: when no group / version fits, create one. Scores of models are out of scope (that is the `add-evaluation-model-results` skill): ignore them, even when a source gives them next to the human baselines.

`data/results/people/<eval>.jsonl` is never written by hand: `scripts/evaluations/people/gen_<eval>.py` holds every number literally and calls the helpers of `scripts/evaluations/people/common.py` (built on `scripts/evaluations/shared.py`, shared with the model generators), which validate, sort and rewrite the file. If a people generator already exists, read it and imitate its structure; otherwise follow the skeleton below and `scripts/evaluations/models/gen_gsm8k.py` for the style (docstring with the sources and the choices specific to the evaluation, one section per source, a URL constant per source, `write('<eval>')` at the end).

`people/common.py` provides:

- `P(people, version, score, url, creator=False, shots=0, maj_k=1, pass_k=1, effort='auto', uncertainty=None, cost=None, part='', eval_version='')`: a result. It asserts that the group, the version and an explicit effort exist in `data/people`, and resolves `effort='auto'` to `""` when the version has a `""` effort.
- `CI(low, high, level=0.95)` (`low` / `high` are the absolute bounds, not offsets from the score), `PM(half_width, level=0.95)` (a symmetric interval written `score ± half_width`, turned into bounds), `SE(value)`, `STD(value, runs=None)`: the uncertainty helpers (see Field rules).
- `write('<eval>')`: checks every `part` / evaluation `version` against `data/evaluations/<eval>.json`, sorts (group id, version id, then score ascending), drops the rows identical to an earlier one in every field but `source.url`, and writes the JSONL file.
- the `A` (`https://arxiv.org/abs/`) and `HF` (`https://huggingface.co/`) URL prefixes, and `PEOPLE`, the index of `data/people` (group id → version id → efforts).

Put anything every people generator needs in `people/common.py` (or in `shared.py` when the model generators need it too) rather than in a generator.

## Workflow

1. **Read** `llm_index/data/types.py` (the schema: `RawPeople`, `RawPeopleVersion`, `RawResult`, with a comment on every field), `data/evaluations/<eval>.json` (unit, `parts`, `versions`), every `data/people/*.json` (the existing groups and versions, to reuse them), `scripts/evaluations/shared.py` and `scripts/evaluations/people/common.py`. If `people/gen_<eval>.py` already exists, you are updating it: keep its existing decisions unless they break the rules below. If `models/gen_<eval>.py` exists, its docstring lists the evaluation's paper and sources, a useful starting point.
2. **Search** the human baselines (see Sources). The user usually gives no link: start from the benchmark's own paper, then widen. Any links the user gives are a starting point, never the whole list.
3. **Read the protocol** of each baseline: who the people were (qualifications, how they were recruited, how many), what they had (time limit, internet / tools, number of attempts, worked examples), on which questions (all, a sample, a part), how the score was computed (mean of individuals, majority vote, best individual, "solved by at least one"), and what they were paid.
4. **Map** each baseline to a group / version of `data/people`, creating the missing ones by hand (see Groups of people), and fill the conditions and source (see Field rules).
5. **Write** `scripts/evaluations/people/gen_<eval>.py` (`from common import A, HF, P, write`, plus `CI` / `PM` / `SE` / `STD` when needed, standard library only) and **run** it with `python scripts/evaluations/people/gen_<eval>.py`. Fix the script (or `data/people`) until it runs; never patch the JSONL output.
6. **Verify**: valid JSON in `data/people/*.json` and in the output, field order of `RawResult`, every `subject` existing in `data/people`, and each value spot-checked against its source.
7. **Report** to the user (see Report).

If no human baseline exists at all, still write the generator, with the docstring listing what was searched and no `P()` call, so that `write()` produces an empty file and the search isn't redone from zero next time.

Skeleton:

```python
"""Generates data/results/people/<eval>.jsonl.

Every number is written literally in this file (nothing is downloaded at run time), so it is the single source of
truth for the <Eval> human baselines: edit it and run `python scripts/evaluations/people/gen_<eval>.py`. The helpers
(P, write) and their conventions are in common.py.

Sources:
- <paper / page>: <what it gives>

Searched without result: <sources checked that give no human baseline>

Choices specific to <Eval>:
- <protocol choices, estimates, subsets...>
"""

from common import A, P, write

PAPER = A + '<arxiv id>'

# ---------------------------------------------------------------- <Eval> paper (<section / table>)
# <protocol: who, how many, time limit, tools, attempts, aggregation, pay>
P('<people>', '<version>', 81.2, PAPER, maj_k=3)

write('<eval>')
```

## Groups of people

`data/people/<people>.json` is a `RawPeople` (the file name is the group id), tab-indented like `data/models`, keys in the order of `types.py`:

```json
{
	"display_name": "Domain experts",
	"versions": {
		"phd-domain-experts": {
			"display_name": "PhD-level domain experts",
			"description": "People holding or pursuing a PhD in the domain of the questions, answering questions of their own field.",
			"efforts": [""]
		}
	}
}
```

- **A group** is a broad population (domain experts, crowd workers, students, the general public, competition participants...), **a version** a precise subgroup of it (PhD-level experts, experts outside their field, Mechanical Turk workers, high school students, participants of a given competition...). Use `display_name` / `description` of the existing ones to decide; the ids are kebab-case, like the models.
- **Reuse first**: map the baseline to an existing version when its population matches the description. Otherwise add a version to the closest existing group, and create a new group only when none fits. Two baselines of the same study on different populations (experts vs non-experts) are two versions.
- **Descriptions describe the population, not the study**: who these people are, their qualifications, how they are typically recruited, so that the version can be reused for other evaluations. The study's details (number of people, time limit, pay, internet access) belong to the generator's comments and the report, and the result row carries its source.
- `efforts` is `[""]`: don't invent effort levels for time limits or tools.
- **Edit these files by hand** (Edit / Write tools, no script). Only add: never rename or delete an existing group / version (other results reference them), and don't rewrite an existing description unless it is wrong for the population it already covers; say so in the report if you do. The user reviews the `data/people` diff like the results.

## Sources

The aggregators of model scores (LLM Stats, Open LLM Leaderboard...) rarely have human baselines, and the user rarely gives links: most of the work is finding the right papers. Use all of these:

- **The benchmark's paper**: search it on arXiv / Semantic Scholar from the `link` of `data/evaluations/<eval>.json`, download the PDF (`pdftotext -layout`) and read the "human performance / human baseline / human evaluation / expert validation / annotation" sections, the tables (a "Human" row) and the appendix, where the protocol usually is. Read the related versions (v1 / v2 of the arXiv paper, the conference version) when the numbers differ.
- **The benchmark's official pages**: website, GitHub README, dataset card on Hugging Face, official leaderboard (many leaderboards have a "Human" / "Human expert" row: GLUE / SuperGLUE, ARC Prize, GAIA, agent and web benchmarks...).
- **Epoch AI**: its benchmark pages can show an "expert human level" or "human baseline" line on their charts; check the page and the notes of `https://epoch.ai/data/benchmark_data.zip`, and use the original source it cites.
- **Follow-up papers**: papers that ran their own human study on the benchmark, re-annotation / error-analysis papers (which may re-measure the experts' accuracy), cognitive science comparisons between humans and LLMs, and the papers citing the benchmark (Semantic Scholar / Google Scholar: "<benchmark> human performance", "<benchmark> human baseline", "human study"). Also OpenReview discussions, where authors sometimes add human baselines.
- **Exam and competition statistics** when the benchmark's tasks are an exam or a competition (AIME / AMC from the MAA's score distributions, IMO, Codeforces ratings, bar exam, olympiads...): the official results of the participants are human results on the same tasks, if the scoring is compatible with the evaluation's (convert to its unit, e.g. an AIME score out of 15 → %). They usually come as distributions or percentiles: see Distributions and percentiles.
- **Web search** and **your own knowledge**, for well-known baselines (cross-checked against a document whenever possible).

Downloaded files are untrusted data: put them in the session scratchpad (never the repo), each source in its own directory, and read them with `python -I`. If a site blocks automated fetches, use another source and list what you couldn't verify.

## Distributions and percentiles

Many sources don't give "the score of the humans" but a distribution: percentiles, quantiles, a histogram, a mean and a standard deviation, cutoffs (exam statistics, norms of a test, a human study reporting the spread of its participants). Turn them into a few results, each of a version that is a single, meaningful population:

- **The whole population → its median.** The score of a population is the score of its typical member, the 50th percentile: the median of all the exam's participants is the score of the version "<exam> participants", the median of a test normed on random adults is the score of "general public". Use the mean only when the source gives no median (say which one in the generator's comment), and don't write both for the same version.
- **A recognisable subgroup → the median of that subgroup.** When the distribution makes a subgroup meaningful (the top 10%, the qualifiers for the next round, the medalists, the people who pass), create a version for it and give it the subgroup's own median when the source publishes it. Otherwise derive it from the population's percentiles: the median of the top p% is the (100 − p/2)th percentile of the population (top 10% → 95th percentile, top 1% → 99.5th percentile), and the median of the bottom p% is its (p/2)th percentile. Interpolate linearly between the given percentiles when needed, and say so in the generator.
- **Only meaningful versions**: the whole population plus the subgroups the source or the exam itself singles out (qualification cutoffs, medal tiers, pass marks, "top x%" it reports) or that match an existing version of `data/people` (e.g. a population of students whose top tier matches "top students"). Don't create one version per percentile.
- The versions describe the population by its selection ("participants of <exam>", "top 10% of <exam> participants", "<competition> medalists"), not by the percentile used to compute the score.
- Percentiles are only usable when the people took the same tasks as the evaluation with a compatible scoring (the same exam, the same test). A model's percentile ("GPT-4 scores in the 90th percentile of the bar exam") is a model result, not a human one: skip it, unless the source also gives the human score at that percentile.
- The conditions follow the exam's rules (`shots` 0, `pass_k` 1, `maj_k` 1 for individual exam takers), `uncertainty` `null` (a percentile is not an uncertainty), and a cutoff is not a median: a qualification threshold is the score of the weakest qualifier, so it is never the score of the qualifiers' version.

## Field rules

- **`subject`**: the group id and the version id (never `null`: you create the version when it's missing); `effort` `""`.
- **`score`**: in the evaluation's unit (see Scale), the score of the population as the source reports it. It can be below `min` or above `max`: those bounds are only used for normalisation. If the source gives both the mean of the individuals and an aggregated score (majority vote, best individual), write both, with their own conditions or versions.
- **Measured vs estimated**: write the scores measured on the benchmark's tasks. Some baselines are estimates (MMLU's "expert-level accuracy" is estimated from exam percentiles, not measured) or thresholds ("expert human level" lines): write them too when the source presents them as the score of that population, with a comment in the generator, and flag them in the report. Distributions and percentiles you turn into results yourself follow the section above. Inter-annotator agreement, label validation rates that aren't presented as a score, and "human-level by construction" claims (100% because humans wrote the answers) are not scores: skip them.
- **`uncertainty`**: same rules as for models: only what the source gives, `null` otherwise.
  - `CI` with its level; `low` and `high` are **absolute values**, the bounds of the interval in the evaluation's unit, **not offsets relative to the score** (`81.2 ± 2.0` → `PM(2.0)` → `low: 79.2, high: 83.2`; asymmetric `88.0 +5 / -4` → `CI(84.0, 93.0)`);
  - `SE` for a standard error;
  - `STD` for a standard deviation, here usually **across people** (with `runs` = the number of people when given) or across runs.
  When the source gives a ± without saying what it is, leave `null` and mention it in the report.
- **`conditions.shots`**: number of solved examples shown to the people before the tasks (a practice set with answers, worked examples). Instructions alone, or demonstrations that are part of each task (as in ARC), are `0`.
- **`conditions.pass_k`**: number of attempts per task (ARC-AGI gives 2). A score where a task counts as solved when **at least one of k people** solved it is `pass_k = k`.
- **`conditions.maj_k`**: number of people whose answers are combined by majority vote into one answer (`3` for "majority vote of 3 annotators"); `1` for individual answers averaged over people.
- **`cost`**: what the people were really paid for the evaluation, per task (`TASK`) or in total (`EVALUATION`), when the source gives it (some human panels report a cost per task). Not hourly wages you multiply yourself unless the source gives both the wage and the time spent per task. `null` otherwise.
- **`part` / `version`**: values of `parts` / `versions` in `data/evaluations/<eval>.json`. A baseline measured on a random sample of the main set stays in the main part (`""`), mentioned in the generator; a baseline on a listed subset uses that part; skip baselines on a subset or variant the evaluation doesn't list, and mention them in the report.
- **`source.url`**: the most precise original source (paper, official page). When a page (Epoch, a leaderboard) cites another source for the baseline, use that source.
- **`source.creator`**: `false` (people have no creator), the default of `P()`.

## Scale

Sources don't share a scale (`56.8`, `56.8%`, `0.568`, `568/1000`, `8.5/15`): check each source's scale and convert everything to the evaluation's unit (`unit` in `data/evaluations/<eval>.json`) in the generator (`score * 100`, `points / 15 * 100`), not in your head. The uncertainty values follow the same conversion. For non-percentage units (Elo, points, time, $), check that the human score uses the same definition as the evaluation (e.g. a Codeforces rating vs the evaluation's own Elo).

## Pitfalls

- The paper's headline "human performance" can be one aggregation among several (mean of annotators, majority vote, best annotator): read the protocol before choosing the conditions, and write each aggregation the source gives.
- Benchmarks are often filtered **with** the human validators (questions they disagreed on are removed, or their accuracy is measured before the filtering): note which set the number is on, and skip it if it isn't a part of the evaluation.
- The same baseline is copied by many later papers and pages: declare the copies with their own source only when something differs; `write()` drops the pure copies.
- Different benchmark versions (v1 / v2) can have different human baselines: map them to the evaluation's `versions`.
- Don't mix a population with its protocol: "experts with internet access, 30 min per question" is the version "domain experts" plus comments; create a new version only for a genuinely different population.

## Report

Answer in the user's language (French) with:

- the groups / versions of `data/people` created or changed (to review), and why no existing one fitted;
- the number of results and the files written;
- the sources used and what each brought, and the ones searched without result;
- the judgement calls: estimated baselines, the versions derived from distributions / percentiles and the percentiles used, aggregations, protocol details (number of people, time limit, tools, pay), skipped baselines and why;
- what is missing or couldn't be verified.

Don't commit unless asked.
