# Dataset Category and Version Compatibility Metadata

## Status

implemented

## Context

With the introduction of transition metal complex (TMC) support (see ADR 006), the package now hosts 
datasets that are fundamentally different from the original organic molecule datasets: different feature 
dimensions (91 vs 44 node dims), different raw column schemas (decomposed metal + ligand SMILES vs single 
SMILES column), different processing classes (``MetalOrganicProcessing`` vs ``MoleculeProcessing``), and 
different loader functions (``load_tmc_dataset`` vs ``load_smiles_dataset``).

Two problems arise:

1. **Delineation.** There is no way for a user (or the system) to know whether a dataset in the registry 
   is an organic molecule dataset or a TMC dataset. A user calling ``load_smiles_dataset('tmqmg')`` would 
   get a DataFrame with unexpected columns (``metal``, ``ligand_smiles`` instead of ``smiles``). The CLI 
   ``cmdata list`` shows all datasets in a flat list with no indication of their type.

2. **Version compatibility.** TMC datasets require ``MetalOrganicProcessing`` which was introduced in 
   version 1.7.0. A user on version 1.6.0 who sees a TMC dataset in ``cmdata list`` and tries to download 
   and load it would get confusing errors. Similarly, XYZ-format datasets require the XYZ parsing support 
   introduced in version 1.1.0.

## Decision

### Category Field

A ``category`` string field was added to the per-dataset metadata in ``metadata.yml``. The field is 
**optional** — datasets that lack it default to ``"organic"`` throughout the system. Currently defined 
categories:

- ``"organic"`` — standard small molecule datasets loaded via ``load_smiles_dataset()`` / ``load_graph_dataset()``
- ``"tmc"`` — transition metal complex datasets loaded via ``load_tmc_dataset()`` / ``load_graph_dataset()``

The category is displayed in both the CLI dataset list (as a dedicated column) and the dataset detail 
view. The ``load_tmc_dataset()`` function performs a soft validation check: if the dataset metadata has 
a category other than ``"tmc"``, it logs a warning suggesting the user may want ``load_smiles_dataset()`` 
instead. This is advisory only — the load proceeds regardless.

The documentation generation command (``cmmanage docs collect-datasets``) also includes the category 
column in the generated dataset table.

### Minimum Version Field

A ``min_version`` string field was added to the per-dataset metadata. The field is **optional** — datasets 
that lack it are treated as compatible with all versions. The value is a semantic version string (e.g., 
``"1.7.0"``).

Version comparison uses ``packaging.version.Version`` (a transitive dependency already available via 
pip/setuptools) via the new ``check_version_compatible(min_version)`` utility function in ``utils.py``, 
which compares against the installed version from the ``VERSION`` file.

The compatibility check is integrated at three levels:

- **CLI list (``cmdata list``).** Incompatible datasets are shown with dimmed text and a 
  ``(requires >=X.Y.Z)`` marker appended to the name. They are still visible — not hidden — so users 
  know what is available after upgrading.
- **CLI detail (``cmdata info``).** The min_version is displayed in the metadata table, colored green 
  if compatible or red with the installed version if not.
- **Python loaders.** ``load_smiles_dataset()``, ``load_graph_dataset()``, and ``load_tmc_dataset()`` 
  each attempt a lightweight metadata fetch and log a ``logging.warning`` if the installed version is 
  below the requirement. The warning is non-blocking — loading proceeds regardless, since the dataset 
  might still work partially with an older version.

All checks are wrapped in ``try/except`` so they never prevent loading even if metadata is unavailable 
(e.g., for local-only datasets or when the file share is unreachable).

### Version Assignments

- **Existing organic CSV datasets:** No ``min_version`` field (always compatible).
- **XYZ-format datasets** (e.g., ``_test2``): ``min_version: "1.1.0"`` (when XYZ support was introduced).
- **TMC datasets** (e.g., ``tmqmg``): ``min_version: "1.7.0"`` (when ``MetalOrganicProcessing`` is introduced).

New dataset processing scripts should include ``min_version`` in their ``METADATA`` dict so it is 
automatically written to the generated ``metadata.yml``.

## Consequences

### Advantages

**Backwards compatible.** Both fields are optional with sensible defaults (``"organic"`` for category, 
always-compatible for min_version). Existing metadata files, experiment archives, and API calls work 
without modification. The system gracefully handles datasets from before these fields existed.

**User-friendly.** Instead of cryptic import errors or shape mismatches, users get clear signals: the 
CLI shows which datasets need a newer version, and the loaders warn when category or version don't match.

**Extensible.** New categories (e.g., ``"mof"``, ``"polymer"``) and new version requirements can be 
added to future datasets without any code changes — only metadata entries.

### Disadvantages

**Metadata fetch overhead.** The version check in Python loaders requires fetching metadata before 
loading the actual dataset. This adds a small overhead (typically a local cache hit). The check is 
wrapped in ``try/except`` so it never blocks, but it does add latency on the first call when metadata 
must be downloaded from the file share.

**Soft enforcement only.** Both the category and version checks are warnings, not errors. A determined 
user can still load a TMC dataset with ``load_smiles_dataset()`` or load a dataset requiring 1.7.0 on 
version 1.6.0. This is a deliberate trade-off: hard errors would be more disruptive than the problems 
they prevent, and there are legitimate cases where partial loading is useful (e.g., inspecting raw CSV 
columns).
