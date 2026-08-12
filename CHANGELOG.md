# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- **Gate findings are recorded verbatim instead of counted, and `agent status` shows
  them.** A stage recorded `notes: 1 warning(s)` and threw the text away, and a blocked
  stage kept `errors[0]` and dropped the rest. Those are the two moments the text is
  worth most: a licence warning reading `1 warning(s)` is indistinguishable whether it
  says "a human waived this for a stated reason" or "the upstream forbids derivative
  works", and a blocked stage is the one most likely to be resumed cold in a later
  session. `StageState` gains `warnings` and `errors` lists — alongside `notes`, which
  still carries the staleness message — and `agent status` prints them under each stage.

  `agent status` also now summarises the recorded licence: identifier, evidence,
  upstream database and, when present, the override decision and its reason, so a waived
  licence stops reading like a clean one. Deliberately truncated to stay scannable; the
  verbatim record is in `.dataset_work/<slug>/manifest.yml`, which is where the skill
  sends an agent that needs to merge into it.

  Loading is defensive about hand-edited manifests — a scalar where a list belongs is
  coerced rather than crashing a resume — since a human repairing the file by hand is
  the situation the manifest exists to support. 12 tests added.

### Fixed

- **The stage-3 license example could not pass its own gate.** SKILL.md showed
  `identifier`/`evidence_url`/`redistribution`/`attribution`, but the gate also requires
  `upstream_database`, documented only in `references/license.md`. An agent copying the
  visible example was blocked on a field it had never seen. The example now carries
  `"upstream_database": null` for the case where the publication is the original source,
  with a second example for a named upstream — labelled as one that *escalates* rather
  than passes, since a non-permissive upstream is exactly what only a human may resolve.
  Verified against the gate: the old example fails, both new ones behave as documented.

- **The `-1` sentinel rule could silently corrupt a single-target dataset.** "Skip rows
  with missing targets, preferring the `-1` sentinel" asks for two incompatible things
  when there is only one target, and is unsafe whenever `-1` is a value the target can
  really take. This is not hypothetical: `elanos_bp` contains a compound whose boiling
  point is exactly `-1.0` °C, so following the old advice would have made a real
  measurement indistinguishable from a missing one. The sentinel is now documented as
  multi-target only, and forbidden on any target whose range reaches `-1`. Corrected in
  SKILL.md, `metadata.md` and `script-anatomy.md` together, since all three repeated it.

- **The override example in `license.md` destroyed the evidence it was recording.**
  `cmmanage agent set` replaces the whole `license` object, so submitting the documented
  override JSON deleted `upstream_database`, `upstream_terms_url`, `upstream_license`
  and `attribution` — leaving a record that a human approved something, without saying
  what. Both override examples now show every field being re-sent, and say to read the
  current value back and merge.

- **`metadata.md` contradicted the metadata gate.** It said `target_descriptions` was
  "required whenever there is more than one target"; the gate requires a description for
  every index from `0` to `target_count - 1`, single-target included — which is the case
  where units matter most, there being no sibling column to infer them from.

- **`script-anatomy.md` called `TARGET_COLUMNS` documentation.** It said the parameter is
  only read by the default `load_dataset`, so overriding that hook makes it inert. The
  base template's `save_csv` hook reads it to name the exported CSV's target columns, so
  a stale value ships a raw CSV with `target_0` headers. It is load-bearing.

- **`source_url` was being overwritten, disabling the preprint/published check.** SKILL.md
  listed it among the freely settable scalar keys, and a run that helpfully replaced it
  with the direct download URL discovered in stage 1 made the source and evidence hosts
  identical — silently neutering the host-divergence warning added alongside the
  provenance check. It is now documented as set once by `init` and not to be overwritten,
  with the download URL belonging in `download_recipe.md` and the script instead.

- **Escalation was hardcoded to `AskUserQuestion`**, which does not exist for a run
  operating as a subagent or in any non-interactive context — the skill's own stage 3 and
  stage 6 both mandate subagents. Phrased tool-agnostically, with the rule that no
  channel means recording the blocker and stopping, never deciding on the human's behalf.

  All six were found by running the skill end to end against `elanos_bp`, a dataset
  already in the collection, from a scrubbed sandbox with the ground truth and git
  history removed. It reproduced the dataset exactly — 5432 compounds against the
  published 5431, the single difference being one multi-component SMILES that the
  published script drops and the skill forbids dropping — and independently established
  that the target is in degrees Celsius, which is how the shipped script's "Kelvin"
  target description was found to be wrong.

- **`half_life` re-sourced onto a license we can actually redistribute under.** The
  script downloaded the ChemRxiv preprint's supplementary workbook, which is
  **CC-BY-NC-ND-4.0** — NoDerivatives makes redistributing a processed graph dataset
  derived from it unsafe. The published ACS version of the same paper
  (`10.1021/acs.estlett.3c00526`) is **CC-BY-4.0** and carries identical data, so the
  script now fetches that instead, via Europe PMC (`PMC10569042`), because the
  pubs.acs.org supplementary URL 403s every scripted client.

  The dataset itself is unchanged: replaying the original filter logic against the new
  source yields exactly the same 892 compounds and the same single `Geometric mean
  DT50` target, confirmed by a full re-run. `sources` now names the published DOI, the
  Europe PMC endpoint and enviPath, and a new `notes` field records the CC-BY
  attribution requirement, enviPath's academic/non-commercial restriction, and the fact
  that values at the extremes of the target range are censoring artefacts rather than
  measurements. The description now also states that the SMILES are desalted and
  stereochemistry-stripped, so stereoisomers are merged into one entry.

  The two heuristic filters (multi-component SMILES, single-atom molecules) are retained
  and explicitly commented as legacy, so the published compound count stays stable at
  892. They would not be applied to a dataset added today.

### Added

- **`cmmanage agent tools` checks and installs the external page readers.** The prepare
  stage depends on `crwl` and `defuddle`, neither of which is a Python dependency, so
  nothing about installing this project brings them along. The command reports both by
  default and installs the missing ones with `--install`. It reports a `crwl` whose
  Chromium was never downloaded as a *missing browser* rather than as a missing `crwl`,
  because that is by far the more common failure and its own error message
  (`Executable doesn't exist at .../chromium-*`) does not name the command that fixes
  it.

- **`crwl` is now the default page reader, with `defuddle` and `WebFetch` as
  fallbacks.** The skill previously led with `defuddle` and reached for `crwl` only
  after a failure. Measured against a ChemRxiv article page, `defuddle` returns 403 and
  so does `WebFetch` — they are the same class of plain-HTTP client and fail on the same
  hosts — while `crwl`, which drives a real Chromium, returns the full article. On an
  ordinary Zenodo record all three work.

  So the old ordering was wrong in the only place it mattered: it spent a 403, a
  diagnosis and a retry on exactly the publisher pages the prepare stage most needs to
  read. `crwl` costs about three seconds more per page (≈4s against ≈1s) and reads
  everything, which is the cheaper trade in a pipeline where a failed fetch costs a
  whole retry cycle. `references/data-sources.md` now sends ChemRxiv straight to `crwl`,
  and `references/troubleshooting.md` gains the "403 from defuddle but the page loads in
  a browser" case, which is now a sign of having started from the fallback.

  The install instructions are corrected with it: the browser step is `crawl4ai-setup`,
  which targets the same Playwright installation crawl4ai itself uses, rather than a
  bare `playwright install chromium` from an unrelated environment. The note that
  `npm install -g` binds `defuddle` to the currently active nvm Node version — so it
  disappears after an `nvm use` — is recorded where it will be read.

  The ACS-versus-Europe-PMC contrast is now stated as measured numbers rather than as
  an assertion: the publisher SI URL returns 403 with 5.6 kB of HTML, the Europe PMC
  `supplementaryFiles` endpoint returns 200 with a 2.1 MB ZIP, for the same open-access
  paper.

- 23 tests covering the tool checks, the browser detection across platforms, and the
  install paths — including that a present-but-browserless `crwl` triggers a browser
  download rather than a redundant crawl4ai reinstall, that the browser is never
  downloaded twice in one run, and that the reader order stays crwl-first so a change to
  the documented default has to be deliberate.

- **The license gate now checks provenance, not just an identifier.** A permissive
  license on a publication says nothing about a database the publication merely
  aggregated, and that is where the binding restriction usually lives. The license
  stage must now state `upstream_database` explicitly — `null` when the publication is
  the original source — and naming one additionally requires `upstream_terms_url` and
  `upstream_license`, with anything non-permissive escalating to a human.

  This closes a real miss found by running the skill end to end: a CC-BY-4.0 paper
  republishing data from enviPath, whose terms permit only academic and non-commercial
  use and forbid systematic redistribution of substantial portions of the database. The
  gate passed it outright.

- **The license gate warns on preprint/published divergence.** When the evidence URL and
  the run's source URL are on different hosts, it flags that the same work often carries
  different licenses in its two forms — exactly the CC-BY-NC-ND versus CC-BY split
  above.

- **Europe PMC and ChemRxiv routes documented** in `references/data-sources.md`. The
  Europe PMC `supplementaryFiles` endpoint serves publisher-deposited SI as a plain ZIP
  with no bot check, and is the standard way around a 403 on a publisher's own SI URL.
  It was the highest-leverage discovery of the end-to-end test and was absent from the
  skill. The ChemRxiv section records that the article id is not the DOI, that the old
  `public-api/v1` endpoint is dead, and that OpenAlex resolves the opaque id when
  Crossref and Europe PMC search do not.

- 13 tests covering the provenance chain and the host-divergence warning, including a
  regression test reproducing the exact case the gate originally missed.

- **The extraction workflow is now six gated stages.** `prepare` (locate and download),
  `metadata` (assemble the descriptive fields), `license` (establish that
  redistribution is permitted), `script` (write and smoke-test on 50 elements),
  `execute` (full run and drop-rate analysis), `verification` (final independent pass).
  Each stage ends at a gate that must pass before the next may begin, and a stage whose
  predecessors have not passed refuses to evaluate rather than producing a meaningless
  verdict.

- **`cmmanage agent init | status | gate | set`.** `init` starts a run and creates
  `.dataset_work/<slug>/manifest.yml`; `status` shows where every stage stands and what
  is blocking; `gate <stage> <slug>` checks a stage's post-conditions, records the
  outcome, and exits non-zero on failure; `set` records a stage's conclusions into the
  manifest for the next stage and the gates to consume. The existing `agent verify` is
  unchanged and still works on a bare results directory.

- **A resumable per-run manifest** (`chem_mat_data/agent/manifest.py`). Stage status,
  blockers, the metadata draft, the license finding and the run paths all survive
  between sessions, so a run blocked on something only a human can resolve can be
  picked up later. Re-running a stage marks every later stage `stale`: re-deciding the
  metadata after the script was written means the script no longer reflects what was
  approved, and that becomes visible rather than being silently accepted.

- **Per-stage gates** (`chem_mat_data/agent/gates.py`) that check concrete facts rather
  than asking whether a stage went well. Notably: whether a downloaded file is secretly
  an HTML landing page (the most common way `prepare` fails while appearing to
  succeed), whether `Experiment.extend` was handed an absolute path, whether the script
  reaches back into `.dataset_work` instead of downloading its own data, whether the
  smoke run's target count matches the approved metadata, whether `__TESTING__` was
  left enabled, and whether what landed in `metadata.yml` matches what the metadata
  stage approved.

- **A license stage with an enforced policy.** Public domain, CC-BY, MIT, BSD and
  Apache pass. NonCommercial, ShareAlike, copyleft, custom terms, and **no license at
  all** escalate to a human, whose decision is recorded as an explicit
  `human_override` carrying a mandatory reason. The matcher is written so that
  `CC-BY-SA` cannot be accidentally matched by the `CC-BY` prefix. Policy and research
  guidance are documented in `references/license.md`.

- **A drop-rate check on the execute gate.** The final compound count is compared
  against the recorded input row count: above 2% it warns, above 10% it escalates,
  overridable per dataset with `max_drop_rate` plus a required `drop_rate_reason`.
  This is the only thing that catches a run which completed cleanly while quietly
  discarding most of its input, since molecules that fail conversion are logged and
  skipped rather than raising.

- **Interactive escalation.** When a stage hits something it cannot fix - a paywall, an
  ambiguous license, an unparseable format - the skill stops and asks the user with a
  concrete question stating what is blocking, what was tried, and what is needed,
  rather than working around it or failing silently.

- **Independent subagents for the license and verification stages.** Both are judgement
  calls where the main agent has a conflict of interest: an agent invested in making a
  dataset work reads an ambiguous license charitably and calls a marginal run good. The
  other four stages stay in one thread of context, because prepare, metadata and script
  share an understanding of the data that a subagent boundary would force to be
  re-derived.

- 81 tests in `tests/test_agent_pipeline.py` covering the manifest state machine, all
  six gates, and the license and drop-rate policies.

- **`cmmanage agent venv`** — builds a clean `uv` virtual environment at
  `.dataset_work/.venv` with the project installed editable, so generated scripts run
  in isolation from the development environment. Prints the interpreter path on the
  last line for capture by a calling script.

- **`cmmanage agent verify <results-dir>`** — asserts the hard post-conditions of a
  processing run: a non-empty `.mpack` and a parseable `metadata.yml` exist, the
  compound and target counts are sane, and `target_descriptions` covers every target
  index. Exits non-zero on error so it can gate a workflow; `--strict` also fails on
  warnings. Warnings cover missing tags/sources/description and a compound count at or
  below 50, which is the signature of a run left in `__TESTING__` mode.

  This check is implemented in Python rather than left to the agent deliberately: a
  processing run can exit cleanly while producing an unusable dataset, because
  molecules that fail conversion are logged and skipped rather than raising.

- `chem_mat_data/agent/verification.py` with `verify_results()` and a
  `VerificationReport` that partitions issues into errors and warnings, plus 20 unit
  tests in `tests/test_agent_verification.py` — the agent module previously had no
  test coverage at all.
### Changed

- **The skill's page-reading tool table now reflects reality.** All three documented
  options failed simultaneously in testing — `defuddle` was not installed, `crwl` had no
  Chromium (`playwright install chromium` is a separate step whose absence the error
  message does not explain), and `WebFetch` is 403'd by ChemRxiv. `defuddle` is now the
  primary with `crwl` and `WebFetch` behind it, each carrying its install command, and
  the skill says to check a tool exists before relying on it. Troubleshooting gained an
  escalation order for 403s that puts Europe PMC first, plus a warning against Playwright
  click-throughs: they may fetch a file once, but a generated script cannot use a route
  that needs a scripted browser.

- **Two documentation contradictions removed.** The canonical template in
  `script-anatomy.md` used integer `target_descriptions` keys while `metadata.md`
  mandates strings. And "`METADATA` must reproduce the approved `metadata_draft`
  exactly" was misleading — the draft's `description` reaches `metadata.yml` through the
  `DESCRIPTION` parameter rather than through `METADATA`, and `dataset_name`,
  `dataset_type` and `target_count` do not belong in `METADATA` at all. Following it
  literally fails the verification gate at the final stage; the skill now carries the
  field-by-field mapping.

- **Dataset extraction moved from Python prompts into a Claude Code skill.** The
  instructions that drive the publication → processing-script workflow now live in
  `.claude/skills/dataset-extraction/` as editable prose rather than as format strings
  in `chem_mat_data/agent/prompts.py`. The skill is human-invoked from Claude Code
  (`/dataset-extraction <url>`) and takes a publication URL or a concrete paper title
  as its input; deciding whether a paper is worth extracting happens beforehand.

  The skill uses progressive disclosure: a workflow-level `SKILL.md` plus five
  reference files (`script-anatomy.md`, `metadata.md`, `license.md`,
  `data-sources.md`, `troubleshooting.md`) and four worked examples covering the
  archetypes — a plain SMILES CSV, an archive/custom text format, an `.xyz` bundle,
  and a transition-metal complex dataset. The TMC example documents the fact that TMC
  scripts do *not* extend the base template but build a standalone `Experiment` around
  `MetalOrganicProcessing`.

  It ends at a verified local script; uploading to the remote share remains a
  deliberate `cmmanage metadata upload` step.

- **Generated scripts are written to `chem_mat_data/scripts/`** alongside the existing
  `create_graph_datasets__*.py` files, instead of to an artifacts folder inside the
  package. Because they now sit next to the base template, `Experiment.extend()` takes
  a *relative* path, which removes the previous requirement to hardcode a machine-
  specific absolute path.

- **Working files moved out of the package.** Downloads and identify results go to a
  gitignored `.dataset_work/` at the repository root rather than
  `chem_mat_data/agent/artifacts/`, which was inside the installed package and caused
  agent runs to write into the source tree. The run venv moved likewise, to
  `.dataset_work/.venv`.

- **Filtering policy made consistent.** The prompts forbade heuristic row filtering
  while the bundled example scripts filtered multi-component SMILES and single-atom
  molecules. The skill states one policy — skip only rows with missing/unparseable
  SMILES or missing targets, preferring the `-1` sentinel — and flags the older
  filtering scripts as legacy.

- **`target_descriptions` keys standardised on strings.** Both `'0'` and `0` occur in
  the existing metadata (24 string vs 2 int); the skill documents the string form and
  `verify` accepts either.
### Removed

- **The OpenCode backend and the whole runtime agent layer.** Deleted
  `agent/backend.py`, `agent/opencode_client.py`, `agent/prompts.py` (731 lines of
  prompt templates in two near-duplicate backend variants) and
  `agent/extraction_agent.py`. Dropped `opencode-ai` from the dependencies and the
  `claude` optional-dependency group along with `claude-agent-sdk`. Skills are a Claude
  Code concept, so maintaining the dual prompt variants no longer bought anything.

- **`cmmanage agent identify|prepare|generate|install-plugins`.** These invoked the
  deleted backends; their behaviour is now the skill's three stages. `manage.py` lost
  ~660 lines.

- `agent/plugins.py`, replaced by the leaner `agent/environment.py` which retains only
  the venv management. The MCP-server and crawl4ai installers went with the
  `install-plugins` command — the skill installs crawl4ai with a one-line
  `uv tool install crawl4ai` when it is wanted, and falls back to `defuddle`/`WebFetch`
  when it is not.

- `Config.get_agent_backend()`, `get_agent_backend_config()` and
  `get_agent_prompt_override()`, which configured the removed backends.

- `chem_mat_data/agent/example_scripts/`, whose three files were byte-identical copies
  of scripts already present in `chem_mat_data/scripts/`.
### Fixed

- **`create_agent_venv` installed into the wrong environment.** The function built an
  environment dict setting `VIRTUAL_ENV` but never passed it to the subprocess, so
  `uv pip install -e .[dev]` was not guaranteed to target the freshly created venv.
  `_run()` now accepts and forwards `env`, and `CONDA_PREFIX` is cleared so an active
  conda environment cannot capture the install.

## [1.10.1] - Unreleased

### Changed

- **`release.sh` reduced to build & publish.** The release script no longer bumps the version,
  runs tests, or creates git commits/tags and GitHub releases. It now only cleans `dist/`, runs
  `uv build`, and publishes with `uv publish`. Version bumping is handled separately. The script
  no longer takes a `<major|minor|patch>` argument.

### Fixed

- **Python 3.12 support.** The `requires-python` upper bound was changed from `<=3.12` to
  `<3.13`. Under PEP 440, `<=3.12` excludes patch releases such as `3.12.7`
  (since `3.12.7 > 3.12`), which meant common Python 3.12 installations were unable to install
  the package. `<3.13` correctly includes all of the 3.12.x series.
- **Python 3.9 import compatibility.** `chem_mat_data/connectors.py` used PEP 604 union
  syntax (`int | None`) in function signatures, which is evaluated at runtime and raised a
  `TypeError` on import under Python 3.9. Added `from __future__ import annotations` so the
  annotations are not evaluated at runtime.

## [1.10.0] - 2026-04-12

### Changed

- **CLI logo coloring.** The "CM" portion of the CMDATA ASCII art banner is now rendered in
  mint (`#5fd7af`) while "DATA" remains bold white, giving the logo a more distinctive look.
- **Improved `cmdata --help` text.** Rewrote the top-level help for the `cmdata` CLI using
  the same `render_help()` / `CODE_STYLE` pill pattern as `cmmanage`. The help now has an
  introductory paragraph, a description of the `cmdata` interface, and one paragraph per
  command category (Discover, Download, Cache, Configuration) with inline code pills for
  command names. Includes a single example block at the end. Fixed "materials sience" typo.
- **Agent web tool: crawl4ai CLI replaces broken MCP server.** All Claude-backend agent
  prompts (`IDENTIFY_PROMPT`, `CLAUDE_LINK_PROMPT`) now instruct the agent to use
  `crwl '<url>' -o markdown-fit` via Bash as the primary web-fetching tool, with `WebFetch`
  as fallback. The previous `crawl4ai-mcp-server` PyPI package was broken (shipped no source
  files) and the `mcp__web-search` MCP server caused the agent to hang on large pages.
- **`cmmanage agent install-plugins`** now installs crawl4ai as a `uv tool`
  (`uv tool install crawl4ai`) instead of the broken `pip install crawl4ai-mcp-server`.
  The crawl4ai MCP server entry is no longer written to the config file since the agent
  uses the `crwl` CLI directly.
- **Tightened identify-agent verification step.** The `IDENTIFY_PROMPT` procedure now limits
  the agent to visiting at most one repository link and two URLs total, preventing the
  previous behavior of spiraling through chains of redirects and hanging.

### Removed

- Removed `web-search` MCP server from `.mcp.json`. This custom MCP server was providing
  `mcp__web-search__get-single-web-page-content` which the agent would use instead of the
  intended crawl4ai tool, often hanging on large or slow pages.
- Removed broken `crawl4ai` MCP server entry from the default agent config
  (`[agent.claude.mcp_servers.crawl4ai]`).

### Added

- **Dataset discovery pipeline.** New `cmmanage discover` command group that integrates the
  upstream [Dataset-Discovery-Pipeline](https://github.com/aimat-lab/Dataset-Discovery-Pipeline)
  as a cheap first-stage filter. It walks the OpenAlex index of chemistry and materials-science
  papers and applies a fine-tuned TinyLlama-1.1B classifier to flag those likely to release a
  reusable dataset. Five subcommands cover the full workflow:
  - `cmmanage discover model-info`: Show the classifier cache location, whether the adapter
    files are present, and whether the required ML dependencies are importable. Read-only
    diagnostic — never downloads or modifies anything.
  - `cmmanage discover model-download`: Download the ~13 MB TinyLlama LoRA adapter archive
    from the Nextcloud URL configured under `[discovery] classifier_url` and extract it into
    the user config directory. Supports `--force` to re-download and `--model-url` to override
    the configured URL.
  - `cmmanage discover fetch`: Query the OpenAlex `/works` endpoint with cursor pagination,
    reconstruct abstracts from inverted-index form, and stream results into a JSONL file.
    Sibling `.cursor` and `.ids` files make runs resumable and de-duplicated across invocations.
    Supports `--reset` to discard existing state, `--year-min` to filter by publication year,
    and `--limit` to cap the number of new records per run.
  - `cmmanage discover score`: Read a JSONL file (typically from `fetch`), load the classifier,
    and score each record's `title + abstract`. Writes two JSONL outputs: `--out-all` with every
    record gaining a `prediction` field, and `--out-flagged` with only records above the
    threshold. Default threshold of 0.05 matches best upstream test-set performance.
  - `cmmanage discover stream`: Combine `fetch` and `score` into a single streaming pass,
    writing scored results to a CSV as they come in. Columns: `openalex_id`, `url`, `score`,
    `flagged`, `publication_year` (plus optional `title`). Rich live terminal progress with
    toast notifications. Supports `--flagged-only`, `--include-title`, and `--resume/--no-resume`.
- **`chem_mat_data.discovery` subpackage** with three modules:
  - `io.py`: `FetchState` class for persistent cursor + seen-ID tracking with context manager
    support, `iter_jsonl`/`write_jsonl_line` helpers, `build_csv_row` projection with DOI URL
    normalization and OpenAlex landing-page fallback.
  - `openalex.py`: OpenAlex API client with cursor pagination, abstract reconstruction from
    inverted index, configurable concept-ID and publication-type filters, exponential-backoff
    retries, and consecutive empty-page detection to cross stale cursor positions.
  - `classifier.py`: TinyLlama-1.1B-Chat LoRA adapter loading with CPU fallback and optional
    4-bit quantization via bitsandbytes. `ensure_classifier_model()` downloads and extracts
    the tar.gz archive with atomic rename and cleanup on failure. `Classifier` class with
    `predict_score` and `predict_batch` methods. Suppresses HuggingFace Hub and transformers
    noise by default (env vars, logging levels, stdout/stderr redirection, offline mode when
    base model is cached).
- **Config file extensions.** New `[discovery]` section in `config.toml` with `classifier_url`
  pointing to the Nextcloud-hosted adapter archive. New `get_discovery_classifier_url()` and
  `get_discovery_model_dir()` config accessors.
- **ML dependencies in dev extras.** Added `transformers>=4.40.0`, `peft>=0.10.0`,
  `safetensors>=0.4.0`, `accelerate>=0.29.0`, and `bitsandbytes>=0.43.0` (Linux + Python 3.10+)
  to the `[dev]` extras group.
- **Rich CLI help text** for all five discovery subcommands, using the `RichHelpCommand` pattern
  with `render_help()` paragraphs covering purpose, workflow context, and key options.
- **Comprehensive test suite.** 69 tests across five test files:
  - `test_discovery_io.py` (13 tests): `iter_jsonl`, `write_jsonl_line`, `FetchState` lifecycle.
  - `test_discovery_openalex.py` (15 tests): abstract reconstruction, filter building,
    pagination, empty-page skipping, DOI passthrough.
  - `test_discovery_classifier.py` (13 tests + 1 slow integration): model download, extraction,
    classifier loading with mocked torch/transformers/peft.
  - `test_discovery_cli.py` (7 tests): group registration, panel title, command routing.
  - `test_discovery_stream.py` (21 tests): `build_csv_row` unit tests, stream CLI end-to-end
    with canned fetch and classifier, flagged-only filtering, resume/skip semantics.
- **`slow` pytest marker.** Added alongside the existing `network` marker; default `addopts`
  excludes both.

## [1.9.0] - Unreleased

### Added

- **LLM-powered dataset extraction agent.** New `cmmanage agent` command group that automates
  dataset onboarding from scientific publications using an LLM agent. The workflow is split into
  three stages:
  - `cmmanage agent identify <url-or-title>`: Determines whether a publication publishes
    molecular or materials property data suitable for dataset extraction. Accepts either a
    URL or a paper title (the agent searches for it). Outputs a yes/no decision with a
    reasoning paragraph and saves a JSON artifact to `agent/artifacts/identify/`.
  - `cmmanage agent prepare <url>`: Navigates the publication, finds associated dataset
    repositories (Zenodo, Figshare, GitHub, OSF, etc.), and downloads the raw data files
    into `agent/artifacts/downloads/`. The agent writes a `download_recipe.md` documenting
    the programmatic download steps for reproducibility.
  - `cmmanage agent generate`: Inspects previously downloaded dataset files, reads example
    processing scripts for reference, and generates a ChemMatData-compatible PyComex script
    that extends the base processing template. Sets up a fresh virtual environment with `uv`,
    runs the generated script, and iterates on errors up to 3 times.
- **Pluggable agent backend system.** Abstract `AgentBackend` base class with two concrete
  implementations: `OpenCodeBackend` (delegates to a locally running OpenCode server) and
  `ClaudeBackend` (uses the Claude Agent SDK with API key or OAuth token authentication).
  Backends are selected via the `--backend` flag or the `[agent]` config section.
- **Claude Agent SDK integration.** The `claude` backend wraps the async `claude_agent_sdk.query()`
  API with a synchronous interface, supports both API key (`ANTHROPIC_API_KEY`) and OAuth token
  (`CLAUDE_CODE_OAUTH_TOKEN`) authentication for Claude Pro/Max subscriptions, and streams
  intermediate agent steps (text narrations, tool calls) to the CLI via an `on_event` callback.
  Runs with `permission_mode="bypassPermissions"` for non-interactive execution.
- **Agent prompt template system.** Backend-specific prompt templates for the link-discovery
  and script-generation stages (tool names differ between backends), and a shared template
  for the identify stage. Prompts can be overridden per-user via the config file.
- **MCP plugin installation.** `cmmanage agent install-plugins` command that installs
  `crawl4ai-mcp-server` (pip) and `mcp-file-downloader` (Node.js) for enhanced web crawling
  and file download capabilities. Automatically configures MCP server entries in the config
  file for the Claude backend.
- **Virtual environment management for script execution.** `create_agent_venv()` uses `uv` to
  create a fresh venv and install the project in editable mode before each generate run, ensuring
  generated scripts execute in a clean, reproducible environment.
- **Example processing scripts.** Three self-contained reference scripts (COMPAS-1x, Open Melting
  Point, TM-Bio) in `agent/example_scripts/` that demonstrate the full download-and-process
  pattern the script-generation agent should follow. Each script includes built-in data download
  with caching logic.
- **Rich CLI output for agent commands.** Agent commands display styled header panels, live
  streaming of intermediate agent steps (text as bullets, tool calls as one-line summaries with
  friendly tool names), and result panels with download/generation status. Custom Rich help
  widgets for all agent subcommands and the agent group itself.
- **Config file extensions.** New `[agent]` section in `config.toml` with `backend` selection,
  prompt override keys (`link_prompt`, `script_generation_prompt`, `identify_prompt`), and
  per-backend configuration (`[agent.opencode]`, `[agent.claude]` with `model`, `api_key`,
  `oauth_token`, `allowed_tools`, `mcp_servers`).

### Changed

- **Agent command ordering.** Commands in the `cmmanage agent` group now appear in workflow
  order: `install-plugins`, `identify`, `prepare`, `generate`.
- **Agent CLI renamed.** The `process` subcommand was renamed to `prepare` and `generate` was
  split into its own command for clarity.

## [1.8.0] - 2026-04-02

### Added

- **Interactive dataset explorer landing page.** The GitHub Pages site now features a full
  interactive landing page at the root (`/`) with MkDocs documentation served under `/docs/`.
  The landing page includes a searchable, sortable dataset table with expandable detail rows,
  category/task/format filters, copy-to-clipboard code examples, direct download links to the
  raw data on bwSyncAndShare, and source reference lists.
- **`cmmanage docs compile` command.** Fetches dataset metadata from the remote file share server
  and compiles it into a `datasets.json` file in the `pages/` directory. This JSON powers the
  interactive landing page at runtime.
- **`cmmanage docs deploy` command.** Builds the full GitHub Pages site locally (MkDocs docs +
  landing page) and deploys it to the `gh-pages` branch via `ghp-import`. Supports `--preview`
  flag for local inspection with a built-in HTTP server.
- **Direct dataset download links.** Expanded dataset details on the landing page include download
  buttons that link directly to raw dataset files (CSV, XYZ) on the bwSyncAndShare file share.
- **Format column and filter.** The dataset table shows a Format column (csv, xyz_bundle) with a
  corresponding filter in the filter bar alongside Category and Task filters.

### Changed

- **GitHub Pages deployment restructured.** The site is now composed of a custom landing page at
  the root and MkDocs documentation at `/docs/`. The GitHub Actions workflow and MkDocs config
  (`site_url`) were updated accordingly.

## [1.7.1] - 2026-04-02

### Changed

- **Redesigned `cmdata info` view.** Modern card-style layout with a colored hero panel driven by
  dataset category (green for organic, magenta for TMC). The panel title shows the dataset name as
  a colored badge with the category separated by dashes. Includes a short description line, compound
  count with size indicator, and two selector widgets (Task and Data) that highlight active values
  against all options. Sections below (Description, Notes, Targets, References, Tags) use lightweight
  rules instead of nested panels. Tags render as emoji-prefixed badges with subtle backgrounds.
- **Redesigned `cmdata list` view.** Cleaner table with fewer columns (dropped Tags), formatted
  compound counts with comma separators, right-aligned numeric columns, abbreviated type badges
  (REG/CLS), and a dataset count in the centered title bar. Each row includes a `LevelIndicator`
  widget showing dataset size at a glance.
- **Category-driven color theming.** Both list and info views color dataset names and accents based
  on the dataset category (`organic` = green, `tmc` = magenta, fallback = cyan).

### Added

- **`LevelIndicator` widget.** Reusable Rich display component that renders a numeric value as
  filled/empty squares (e.g. `■■■□□`) against configurable thresholds. Used in both the info hero
  panel and the list table's Compounds column.
- **Task type selector widget** in `cmdata info` showing regression, classification, and bioactivity
  as highlighted/dimmed badges. Bioactivity is auto-inferred from dataset tags.
- **Data format selector widget** in `cmdata info` showing CSV, XYZ, TMC, and Graph availability
  as highlighted/dimmed badges.

## [1.7.0] - 2026-04-02

### Added

- **Transition metal complex (TMC) support.** New `MetalOrganicProcessing` class in `tmc_processing.py`
  for converting decomposed TMC representations (metal + ligand SMILES + connecting atom indices) into
  graph dicts with 91-dim node features, 18-dim edge features, and 5-dim graph features. Includes
  dative bond encoding, metal-specific features (d-electron count, electronegativity, covalent radius),
  and lenient RDKit sanitization for ligand fragments with unusual valences.
- New `load_tmc_dataset()` function in `tmc.py` for loading TMC datasets in the decomposed tabular
  format with automatic JSON column parsing.
- Dataset processing script for **tmQMg** (~63,000 mononuclear TMCs with 20 QM properties). Downloads
  from the ELECTRUM validation repository, infers donor atoms heuristically, and produces CSV + mpack
  output following the standard PyComex experiment pattern.
- Custom encoder classes for TMC features: `LookupEncoder`, `PeriodicTableEncoder`, `MetalFlagEncoder`,
  `DElectronCountEncoder`, with bundled Pauling electronegativity and periodic table group lookup tables.
- `check_version_compatible()` utility function in `utils.py` for semantic version comparison against the
  installed package version.
- **Dataset `category` metadata field.** Optional field (default: `"organic"`) to distinguish dataset
  types. TMC datasets use `category: "tmc"`. Displayed in CLI list and detail views. Soft validation
  warning in `load_tmc_dataset()` when category doesn't match.
- **Dataset `min_version` metadata field.** Optional minimum package version requirement per dataset.
  Incompatible datasets are shown dimmed with a version marker in `cmdata list`. Python loaders
  (`load_smiles_dataset`, `load_graph_dataset`, `load_tmc_dataset`) log warnings when the installed
  version is below the requirement. Non-blocking by design.
- Architecture Decision Records: ADR 006 (TMC support) and ADR 007 (category and version compatibility).
- Comprehensive unit tests for TMC processing (28 tests covering encoders, feature dimensions, dative
  bonds, multiple metal types, and error handling).

### Changed

- `cmdata list` now shows a "Category" column for all datasets.
- `cmdata info` now shows "Category" and "Min Version" in the dataset detail view.
- `cmmanage docs collect-datasets` includes a "Category" column in the generated documentation table.

## [1.6.0] - 2026-03-26

### Added

- `.zenodo.json` for Zenodo-GitHub integration to enable DOI generation for releases.
- `CHANGELOG.md` replacing the previous `HISTORY.rst` file, now following the
  [Keep a Changelog](https://keepachangelog.com/) convention.
- Dataset processing script for the DUD-E (Directory of Useful Decoys - Enhanced) multi-target
  virtual screening benchmark with 102 protein targets and tri-state labels.
- Additional MUV helper script (`create_graph_datasets__muv_.py`).

### Changed

- Dropped Python 3.8 support; minimum supported version is now Python 3.9.
- Substantially reworked the MUV dataset processing script to support multi-target classification
  with tri-state labels (active/decoy/no data) across 17 biological targets.
- Rewrote the BBBP dataset processing script with proper documentation, metadata, and updated
  data source handling.
- Updated DUD-E compound count in `metadata.yml` (1,111,394 to 400,040).
- Added missing source reference for the BBBP dataset in `metadata.yml`.

### Removed

- `HISTORY.rst` (replaced by `CHANGELOG.md`).

## [1.5.0] - 2025-11-27

### Added

- New `cmdata stats` command which downloads a given dataset to compute common
  statistics such as the distribution of elements, graph sizes, and the most common motifs.

## [1.4.1] - 2025-11-10

### Changed

- Changed the way in which the `--help` string is being printed to be more informative.

### Fixed

- Changed the way in which the Nextcloud remote accesses the public endpoint which has changed for
  the transition to NextcloudHub 10.0. This had been a breaking change which prevented accessing the
  remote file share location at all.

## [1.4.0] - 2025-10-06

### Added

- Implemented **StreamingDataset** architecture for memory-efficient access to large molecular datasets:
  - `SmilesDataset`: Lazy-loading of SMILES strings from CSV files with minimal memory footprint.
  - `XyzDataset`: Lazy-loading of 3D molecular structures from XYZ file bundles, supporting multiple
    format parsers (default, qm9, hopv15).
  - `GraphDataset`: On-the-fly conversion from raw molecular representations (SMILES or XYZ) to graph dicts.
    - Automatic detection of dataset format (SMILES vs XYZ) for transparent handling.
    - Sequential mode (`num_workers=0`) for optimal performance with typical molecules (~2000 mol/s).
    - Parallel mode (`num_workers>0`) with multi-process architecture for complex molecules or custom processing.
    - Deadlock-free producer-collector-worker design that maintains dataset order while enabling true CPU parallelism.
  - `ShuffleDataset`: Approximate shuffling using fixed-size buffer for training with shuffled data
    while maintaining low memory usage.
- Comprehensive streaming datasets documentation (`docs/api_streaming_datasets.md`) covering:
  - Motivation and use cases for streaming vs pre-processed datasets.
  - Detailed usage examples for all streaming dataset classes.
  - Performance considerations and when to use sequential vs parallel processing.
  - Integration with deep learning frameworks and training workflows.
  - Guidance on choosing between SMILES and XYZ formats.
- Architecture Decision Record (`docs/architecture_decisions/004_streaming_datasets.md`) documenting:
  - Design rationale for streaming architecture.
  - Detailed explanation of parallel processing implementation and deadlock prevention.
  - Trade-offs between streaming and pre-processed datasets.
  - Auto-detection mechanism for SMILES vs XYZ datasets.
- Comprehensive unit tests for streaming datasets:
  - `tests/test_dataset.py`: Core functionality tests for SmilesDataset, XyzDataset, GraphDataset, and ShuffleDataset.
  - `tests/test_xyz_dataset.py`: XYZ-specific functionality and format parser tests.
  - `tests/test_xyz_bundle.py`: XYZ bundle file handling tests.
  - `tests/test_dataset_benchmark.py`: Performance benchmarks for sequential vs parallel processing modes.

### Changed

- Updated existing tests (`test_docs.py`, `test_main.py`, `test_web.py`) to accommodate streaming dataset functionality.

### Removed

- Deprecated `tests/test_datasets.py` in favor of new dataset-specific test files.

## [1.3.0] - 2025-09-12

### Added

- `HOPV15_exp` dataset which contains experimental values for organic photovoltaic materials.
- Missing target descriptions for the QM9 dataset.
- `melting_point` dataset which contains melting points for small organic molecules.

### Changed

- Minimum required version of pycomex to `0.23.0` to support the most recent features
  such as the caching system which has also been implemented in the dataset processing scripts.
- CLI logo displayed at the beginning of the help message to "CMDATA" in another ASCII font
  and added a logo image in ANSI art.

## [1.2.1] - 2025-09-22

### Changed

- Modified the syntax of type annotations so that the package is now compatible with
  Python 3.9 through Python 3.12.
- Using `nox` now for the testing sessions instead of `tox` due to the much faster uv backend
  to create the virtual environments.

## [1.2.0] - 2025-09-04

### Added

- `CLAUDE.md` file which contains information that can be used by AI agents such as
  Claude to understand and work with the package.
- `remote show` command which displays useful information for the currently registered
  file share location such as the URL and additional parameters.
- `remote diff` command which allows comparing the local and remote file share versions
  of the metadata.yml file and prints the difference to the console.
- `metadata diff` command in the manage CLI which allows comparing the local and remote
  versions of the metadata.yml file.
- `tadf` dataset associated with OLED design.

### Changed

- Default SSL verification in the `web.py` module set to `True` to avoid security issues
  when downloading files from the internet.

## [1.1.2] - 2025-09-01

### Changed

- Default template for the `config.toml` file to include commented out example values for the
  Nextcloud remote file share configuration and to fix the default download location.

## [1.1.1] - 2025-07-07

### Added

- `prettytable` as a dependency to create markdown tables in the documentation.
- `docs` command group in the manage CLI to manage the documentation:
  - `collect-datasets` which collects all datasets listed in the metadata.yml file and creates
    a new markdown docs file with a table containing all those datasets.

### Changed

- `list` command now also prints the verbose name / short description of the datasets.

## [1.1.0] - 2025-07-07

### Added

- `AGENTS.md` file which contains information that can be used by AI agents such as
  ChatGPT Codex to understand and work with the package.
- `manage.py` script which exposes an additional command line interface for the management
  and maintenance of the database:
  - `metadata` command group to interact with the local and remote version of the metadata.yml file.
  - `dataset` command group used to trigger the creation and upload of the local datasets.
- `remote` command group in the `cmdata` CLI:
  - `upload` command to upload arbitrary files to the file share server.
- New datasets:
  - `skin_irritation`: binary classification dataset on skin irritation.
  - `skin_sensitizers`: binary classification dataset on skin sensitization.
  - `elanos_bp`: regression of boiling point.
  - `elanos_vp`: regression of vapor pressure.

## [1.0.0] - 2025-05-01

- First official release of the package.

## [0.2.0] - 2024-12-12

### Added

- `HISTORY.rst` to start a changelog of the changes for each version.
- `DEVELOP.rst` which contains information about the development environment of the project.
- `ruff.toml` file to configure the Ruff linter and code formatter.

### Changed

- Replaced the `tox.ini` with a `tox.toml` file.
- Ported the `pyproject.toml` file from using Poetry to using `uv` and `hatchling` as
  the build backend.
