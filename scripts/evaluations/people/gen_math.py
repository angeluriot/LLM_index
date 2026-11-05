"""Generates data/results/people/math.jsonl.

Every number is written literally in this file (nothing is downloaded at run time), so it is the single source of
truth for the MATH human baselines: edit it and run `python scripts/evaluations/people/gen_math.py`. The helpers
(P, write) and their conventions are in common.py.

Parts of the evaluation (data/evaluations/math.json): '' (the full MATH test set), '-500' (MATH-500, the PRM800K
subset), 'Level 5' (the 1,324 hardest test problems).

Sources:
- MATH paper (Hendrycks et al., NeurIPS 2021 Datasets and Benchmarks, arXiv 2103.03874), section 3.1 "Human-Level
  Performance" and Appendix A.6: scores of 6 university students on 20 random problems of the MATH test set (the same
  numbers in arXiv v1, v2 and the NeurIPS version)

Searched without result:
- "Let's Verify Step by Step" (Lightman et al., 2023, arXiv 2305.20050, MATH-500 / PRM800K): human labelers graded the
  steps of model solutions, no human score on the problems
- Epoch AI MATH Level 5 page and benchmark_data.zip (math_level_5.csv, benchmark_metadata.csv, README): no human
  baseline
- Open LLM Leaderboard v2 (MATH Lvl 5), Artificial Analysis / Vals AI / Kaggle MATH-500: model scores only
- Web search for other human studies on MATH, MATH-500 or MATH Level 5 (students, crowd workers, olympiad
  participants): only copies of the MATH paper's numbers (40% / 90%); HARP (arXiv 2412.08819) is another benchmark
- AMC / AIME statistics: MATH mixes single problems of these exams with other AoPS problems, so the exams' score
  distributions are not scores on MATH's tasks

Choices specific to MATH:
- the baseline is on 20 problems sampled at random from the full test set: kept in the main part (''), and duplicated
  on '-500' because MATH-500 is also a representative random sample of the test set (so that the baseline appears on
  both); not on 'Level 5', which only has the hardest problems
- the paper reports 6 individuals, not a group score: the 4 participants without olympiad background form the
  computer-science-university-students version, scored by their median (the skill's rule for a population; their mean
  is 62.5%), and the 2 olympiad participants each give the score of their own olympiad population
"""

from statistics import median

from common import A, P, write

PAPER = A + '2103.03874'
PARTS = ('', '-500')	# full test set and MATH-500, both representative of the test set

# ---------------------------------------------------------------- MATH paper (section 3.1, Appendix A.6)
# 6 university students studying computer science (4 authors of the paper, 2 friends / acquaintances), 20 problems
# sampled at random from the MATH test set, 1 hour for the 20 problems, calculations by hand (no calculator), no worked
# examples. Remuneration offered but every participant volunteered (no pay). Individual scores out of 20.
# - a participant who does not like mathematics (the "CS PhD student" of the abstract): 8/20, one ambivalent toward
#   mathematics: 13/20, two who like mathematics: 14/20 and 15/20 -> median 13.5/20
# - a participant with a perfect score on the AMC 10 who attended USAMO several times: 18/20
# - a three-time IMO gold medalist: 18/20, the missed problems being only small arithmetic errors
for part in PARTS:
	P('university-students', 'computer-science-university-students', median([8, 13, 14, 15]) / 20 * 100, PAPER, part=part)
	P('competition-participants', 'usamo-qualifiers', 18 / 20 * 100, PAPER, part=part)
	P('competition-participants', 'imo-gold-medalists', 18 / 20 * 100, PAPER, part=part)

write('math')
