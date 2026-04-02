# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [1.7.0] - Unreleased

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
