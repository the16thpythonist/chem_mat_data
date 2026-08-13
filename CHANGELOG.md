# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [1.10.1] - Unreleased

### Fixed

- **`node_atoms` recorded the encoder's view of an element rather than the element.** The
  field exists so that a graph says what it is made of — the comment above it puts it as
  letting "a human reconstruct the molecule from the graph structure". It was built as
  `symbol_encoder.encode_string(atom.GetSymbol())`, passing the true symbol through the
  one-hot encoder and storing what came back, which is the literal string `unknown` for
  anything outside its 19-element vocabulary (C, O, N, S, P, F, Cl, Br, I, Si, B, Na, Mg,
  Ca, Fe, Al, Cu, Zn, K). The real symbol was available on the same line and discarded.

  Everything else became one token: hydrogen, which is not in the list at all, and so
  selenium, germanium, tin, tellurium, arsenic, antimony, lead, lithium, caesium and the
  transition metals. A phosphorus-31 NMR dataset built recently came out with 49% of its
  atoms labelled `unknown` — every explicit hydrogen, and every selenium in it, despite
  phosphine selenides being a standard compound class for that measurement. A selenium and
  a hydrogen were the same value.

  `node_atoms` now stores `atom.GetSymbol()`. The one-hot block in `node_attributes` is
  untouched, so models see exactly what they saw before and the encoder's bucketing is
  still recoverable from there; this field is simply the ground truth beside it. This also
  makes the two processing paths agree — `tmc_processing` has always written the real
  symbol, and its tests assert `node_atoms[0] == 'Ru'`, an element the encoder does not
  know.

  Existing published datasets keep the old values until they are regenerated. 3 tests added.

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
