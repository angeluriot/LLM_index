"""Generates data/results/people/gsm8k.jsonl.

Every number is written literally in this file (nothing is downloaded at run time), so it is the single source of
truth for the GSM8K human baselines: edit it and run `python scripts/evaluations/people/gen_gsm8k.py`. The helpers
(P, write) and their conventions are in common.py.

Sources:
- GSM-Plus paper (Li et al., ACL 2024, arXiv 2402.19255), Table 4 / Table 8 "Human" row and Appendix C.1: accuracy of
  qualified crowd workers on a random sample of the GSM8K test set

Searched without result:
- GSM8K paper (Cobbe et al., 2021, arXiv 2110.14168): no human score; Appendix A only says that 1.7% of the problems
  still produce disagreements among the contractors who re-solved them (a validation rate, not a score)
- GSM1k paper (Zhang et al., 2024, arXiv 2405.00332), section 3.2.2: 14 annotators solved 4.07 +- 0.93 GSM8K problems in
  15 minutes (a number of problems solved under time pressure, not an accuracy)
- Epoch AI benchmark data (gsm8k_external.csv, README): no human baseline
- GSM8K-V (arXiv 2509.25160): human accuracy on a visual version of the problems, not on GSM8K
- Web search for other human studies on GSM8K (students, teachers, crowd workers): nothing found

Choices specific to GSM8K:
- the GSM-Plus baseline is on a random sample of 50 GSM8K test questions: kept in the main part
- the paper doesn't say how the 3 annotators' answers are aggregated: written as individual answers (maj_k=1)
"""

from common import A, P, write

GSM_PLUS = A + '2402.19255'

# ---------------------------------------------------------------- GSM-Plus paper (Table 4 / Table 8, Appendix C.1)
# 3 annotators of the Tencent crowdsourcing platform, each with more than 1,000 completed HITs, a 9-question
# qualification exam passed and at least a bachelor's degree. 50 seed questions sampled at random from the GSM8K test
# set (with their 400 GSM-Plus variations, 450 questions in total). No time limit, tools or pay given.
P('educated-adults', 'qualified-online-workers', 96.77, GSM_PLUS)

write('gsm8k')
