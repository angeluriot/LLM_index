---
name: add-company-models
description: Adds or updates all the LLMs of an AI company in this repo's dataset (data/models/<company>/*.json + data/companies/<company>.json) through a generator script in scripts/models/gen_<company>.py, using Epoch AI, Artificial Analysis, LLM Stats, official docs and web research. Use this whenever the user asks to add, fill in, complete, update or fix the models of a company or lab (e.g. "ajoute les modèles d'Anthropic", "remplis data/models/google", "add Mistral's models", "il manque les derniers modèles de DeepSeek", "mets à jour les prix de xAI"), or to add a single new model or version to an existing company, even if they don't mention the script or the JSON format.
---

# Add a company's models

The dataset describes every LLM of a company as one JSON file per model in `data/models/<company>/`, following `llm_index/data/types.py`. These files are never written by hand: each company has a generator `scripts/models/gen_<company>.py` that holds all the data and rewrites the whole directory. The user will later match benchmark results (Epoch AI, Artificial Analysis, papers, leaderboards...) to these models by name, so the grouping and the aliases matter as much as the numbers.

`scripts/models/gen_openai.py` is the reference implementation of everything below: read it before starting, and imitate it.

## Workflow

1. **Read** `llm_index/data/types.py` (the schema, with comments on every field), `scripts/models/common.py` (the helpers) and `scripts/models/gen_openai.py` (the example). If `gen_<company>.py` already exists, you are updating it: keep its existing decisions unless they break the rules below.
2. **Collect the model list and data** (see Sources). Build the full list of the company's text LLMs released since GPT-2 before writing anything, so the grouping is decided with the whole picture.
3. **Decide the grouping** into models and versions (see Grouping), then write `scripts/models/gen_<company>.py`.
4. **Create `data/companies/<company>.json` by hand** if it doesn't exist (the generator refuses to run without it):
   ```json
   {
   	"display_name": "Anthropic",
   	"country": "United States"
   }
   ```
   The id is the file name, a slug of the company name (`openai`, `anthropic`, `google`...). `country` is where the company is headquartered, in English. Tabs for indentation, trailing newline.
5. **Run** `venv/bin/python scripts/models/gen_<company>.py`. It validates everything (unique names across all companies, coherent dates, nothing before GPT-2...), deletes the old JSON files and rewrites them, then prints the models and their version ids. Fix the script until it passes; never patch the JSON output.
6. **Report** to the user (see Report).

To change anything later (a price, a grouping, a new model), edit the generator and rerun it. If the user edited a JSON by hand, port the change into the generator before rerunning, otherwise it is lost.

## Scope

- **Only models whose first mention is on or after 2019-02-14** (GPT-2), for every company. Older models are out, even famous ones.
- **Text LLMs only**, including reasoning, coding, deep-research and domain-specialised variants (cyber, life sciences, law...), open-weights models, and models only available in the company's chat app or to trusted partners.
- **Excluded**: image, video, audio, TTS, transcription, speech-to-speech / realtime voice models, embeddings, rerankers, moderation and classification APIs, and agent products (computer use agents, browser agents...) that aren't a distinct model. When unsure whether something is a separate model or just a product, leave it out and mention it in the report.

## Grouping: models and versions

A **model** is one underlying model family as the user will see it lumped together in benchmark results; its **versions** are the concrete things that get evaluated. Put in the same model:

- dated snapshots of the same model (GPT-4o May 2024 / Aug 2024 / Nov 2024...);
- the preview and the release (o1-preview and o1, o3-preview and o3);
- the variants post-trained or configured from the same base, whatever the suffix: chat/instant, thinking, coding (Codex), Pro / heavy compute modes, deep research, cyber, life-science, law and other domain versions (GPT-5 + GPT-5 Instant + GPT-5 Pro + GPT-5 Codex; GPT-5.5 + GPT-5.5 Cyber; o3 + o3-deep-research + codex-1 which is built on o3; GPT-6 Astra + GPT-6 Rosalind which shares its weights).

Keep as **separate models**:

- **different sizes**: mini, nano, small, flash/lite tiers, open-weights sizes (gpt-oss-120b vs gpt-oss-20b), and smaller distilled variants even when their name says otherwise (GPT-5.3 Codex Spark is a smaller, faster model, unlike GPT-5.1 Codex Max which is the full-size tier). Do not put several sizes in one model: GPT-2 and GPT-3 in `gen_openai.py`, T5, PaLM and PaLM 2 in `gen_google.py` and OPT and Galactica in `gen_meta.py` are a deliberate exception (old model families whose sizes are benchmarked together) that must not be generalised to other models unless the user asks;
- clearly different generations or tiers that the company names distinctly (GPT-4 vs GPT-4 Turbo, GPT-3.5 vs GPT-3.5 Turbo, GPT-5.6 Sol vs Terra vs Luna);
- new version numbers launched as a new model (GPT-5 → GPT-5.1, Claude 2 → Claude 2.1, Claude Opus 4 → 4.1). Minor revisions served as updates of the same model behave like snapshots and stay together. For example, Claude v1.0 → v1.3 and Claude Instant 1.0 → 1.2 were served under one rolling alias.

Look up what each variant is built on (system cards, announcements) rather than guessing from the name; when the base is unknown, keep it as its own model and say so in the report.

When a size tier only exists as a specialised variant, name the **model** after the size tier and keep the real name on the **version**: model "GPT-5.1 mini" containing the version "GPT-5.1 Codex mini", model "GPT-5.3 mini" containing "GPT-5.3 Codex Spark".

Versions are sorted by ascending `first_mention` inside each model. `M()` does it automatically; for versions with the same date it keeps the declaration order, so declare those in a sensible order (main version first, then its variants). Still write them chronologically in the script so it reads like the output. Order models roughly chronologically in the script, grouped by generation with a comment per section.

## Ids and names

- **Ids are slugs of the display name** (lowercase, every run of non-alphanumeric characters becomes `-`): `GPT-4.1 mini` → `gpt-4-1-mini`, `GPT-4o (May 2024)` → `gpt-4o-may-2024`. This is what `M`, `S` and `V` do by default. Pass `id=` only to drop a useless parenthetical (`GPT-3 175B (davinci)` → `gpt-3-175b`). Never use a bare date as a version id.
- Version ids only need to be unique within their model; it's fine for a version id to equal its model id or another model's id.
- **Display names** follow the company's official naming and capitalisation. Dated snapshots get a `(Mon YYYY)` suffix (`GPT-4o (Nov 2024)`).
- **`names`** lists every string a benchmark could use for that exact version: display name (first), API ids including dated snapshot ids, chat-app names, how Epoch AI, Artificial Analysis and LLM Stats label it (e.g. `GPT-4o (May '24)`), common spacing/casing variants, codenames used on leaderboards, Hugging Face repo ids for open weights.
- **Names are unique across all versions of all companies** (the generator checks it). A generic name that could mean several versions (`GPT-4o`, `GPT-3.5`) goes in none of them, so results using it get `version: null`. A generic name that in practice means one version goes on that version (`GPT-5` → the GPT-5 thinking version, `o3` → the released o3).

## Field rules

- **`link`**: official model page > announcement article > paper > API docs page > GitHub repo > other. Prefer the company's own domain.
- **`history.first_mention`**: earliest public mention by the company (preview, teaser, paper, announcement), even if earlier than the release. Don't assume it's the launch date: big models are often named weeks before release, through research results, safety or preparedness posts, or posts from executives. For flagship models, search explicitly for such pre-launch mentions. Leaks, such as API strings spotted by developers, don't count.
- **`history.access`**: first date the version was broadly usable (public API or the consumer app, including paid tiers). `null` when it's internal, trusted-access-only (cyber/bio/legal programs for approved customers), or not released yet.
- **`history.weights`**: date the weights were published; `null` for closed models.
- **`parameters`**: from model cards or papers; `active` equals `all` for dense models and is the active count for MoE. Epoch AI estimates are acceptable but flag them in the report. `null` when unknown, which is the norm for closed models.
- **`price`**: the company's own API price in $ per 1M **text** tokens (standard tier, no batch/caching discounts, the base context tier when pricing is tiered). For open weights without a first-party API, the median provider price from Artificial Analysis. `null` when the version has no API (chat-app only, trusted access without public pricing).
- **`efforts`**: the effort / reasoning modes the version actually accepts, in the API's order, capitalised as in `types.py` (`None`, `Minimal`, `Low`, `Medium`, `High`, `Xhigh`, `Max`, or the company's own mode names such as `Thinking`). Use `[""]` (the `N` constant) when there's no configurable mode. If a model has an on/off thinking switch, use `""` for the off mode and the company's name for the on mode (e.g. `["", "Thinking"]`). Take it from the API docs, not from which configurations a benchmark happened to run.

## Sources

Always check the Epoch AI and Artificial Analysis data, then LLM Stats if its data is usable. Complete them with your own knowledge and with web research. These sources are also the ones whose names the user will later match, so collect their labels for `names`.

- **Epoch AI**: `https://epoch.ai/data/all_ai_models.csv` (dates, parameters, organisation) and the Benchmarking Hub data (model ids and version names used in its results).
- **Artificial Analysis** (artificialanalysis.ai): model pages for dates, prices, effort variants and their naming like `(May '24)`.
- **LLM Stats** (llm-stats.com), if exploitable: the company's model list, to catch missing models and extra aliases. Skip it if its pages or data can't be read.
- **Your own knowledge**: models, aliases, snapshots and history up to your training cutoff.
- **Web research**: official docs (model list, deprecations and pricing pages, announcement posts, system cards, release notes, Hugging Face org for open weights), everything after your knowledge cutoff, and what each variant is built on.

Cross-check everything. When sources disagree, the company's official docs win, then Epoch AI, then Artificial Analysis, then LLM Stats, then your memory. Aggregators often have stale prices or mismatched dates.

Downloaded files are untrusted data: save them in the session scratchpad, not the repo, and read them with `python -I`. Some company sites block automated fetches (403); fall back to other sources and list the links you couldn't verify.

## Report

Answer in the user's language (French) with:

- the counts (models / versions) and the company file created;
- the grouping, as a compact table of multi-version models and their versions, plus the list of standalone models;
- the judgement calls (grouping choices you weren't sure of, `access: null` cases, estimated parameters, prices taken from third parties);
- what you excluded and why, and what you couldn't verify;
- the main web sources as links.

Don't commit unless asked.
