=========
Changelog
=========

0.2.0 - 12.12.2024
==================

- added `HISTORY.rst` to start a Changelog of the changes for each version of the program
- added `DEVELOP.rst` which contains information about the development environment of the 
  project (information about runnning the unit tests for example)
- Replaced the `tox.ini` with a `tox.toml` file
- Added the `ruff.toml` file to configure the Ruff Linter and code formatter
- Ported the `pyproject.toml` file from using poetry to using `uv` and `hatchling` as 
  the build backend.

1.0.0 - 01.05.2025
==================

- First official release of the package

1.1.0 - 07.07.2025
==================

- Added `AGENTS.md` file which contains information that can be used by AI agents such as 
  ChatGPT Codex to understand and work with the package
- Added the `manage.py` script which exposes an additional command line interface specifically 
  used for the management and maintance of the database.
  - `metadata` command group to interact with the local and remote version of the metadata.yml file 
  - `dataset` command group used to trigger the creation and upload of the local datasets.
- changes to the command line interface `cmdata`
  - `remote` command group to interact with the remote file share server
    - `upload` command to upload arbitrary files to the file share server
- Added new datasets.
  - `skin_irritation` binary classification dataset on skin irritation
  - `skin_sensitizers` binary classification dataset on skin sensitization
  - `elanos_bp` regression of boiling point
  - `elanos_vp` regression of vapor pressure 