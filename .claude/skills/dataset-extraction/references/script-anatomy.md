# Script anatomy

Everything about the structure of a `create_graph_datasets__<name>.py` script.

> **The template is truth.** This file describes intent and conventions. Before you
> rely on a parameter name or hook signature, confirm it against
> `chem_mat_data/scripts/create_graph_datasets.py`. If the two disagree, the template
> wins and this file is out of date.

## The extension hierarchy

```
create_graph_datasets.py              base — SMILES → graph, multiprocessing, save, metadata
├── create_graph_datasets__<name>.py  most datasets extend this directly
└── create_graph_datasets__from_xyz.py    intermediate base for .xyz bundles
    └── create_graph_datasets__from_xyz__<name>.py

create_graph_datasets__tmqm.py        standalone — does NOT extend; own Experiment + worker loop
```

Scripts live in `chem_mat_data/scripts/`, so `Experiment.extend()` takes a **relative**
filename. An absolute path is machine-specific and will break for everyone else.

## What the base template gives you

The base runs this sequence, and you only override the parts you need:

1. Spawns one `ProcessingWorker` per CPU. Each holds a `MoleculeProcessing` instance
   and converts `smiles` (or `mol`) → `GraphDict`, attaching `graph_labels` from
   `targets`. Workers start *before* the dataset loads — deliberately, so the dataset
   is not copied into every child process.
2. `load_dataset` hook → `Dict[int, dict]`.
3. `save_csv` hook → writes `<name>.csv` and `.csv.gz` into the results folder.
4. Truncates to 50 elements when `__TESTING__` is True.
5. Feeds the queue, collects graphs, logs ETA every 1000 elements.
6. Saves `<name>.mpack`, then `<name>.mpack.gz` when `COMPRESS`.
7. Builds `metadata.yml` from auto-derived fields (`compounds`, `targets`,
   `target_type`, `description`, `raw`, `sources`) then merges your `METADATA` over it.
8. `after_save` hook.

A molecule that throws during processing is logged and dropped — the worker returns
`None` and the main loop skips it. So a script can "succeed" while silently losing
rows. Compare the `compounds` count in `metadata.yml` against the input row count and
mention the difference in your report if it is more than a percent or two.

## Parameters

Declare these at module level, before `Experiment.extend()`. They override the
template defaults.

| Parameter | Type | Notes |
|---|---|---|
| `DATASET_NAME` | `str` | Short, lowercase, underscored: `esol`, `hopv15`, `open_melting_point`. Becomes the filename and the identifier on the remote share. |
| `DESCRIPTION` | `str` | 2–5 sentences. See `metadata.md`. |
| `METADATA` | `dict` | Merged over the auto-generated metadata. See `metadata.md`. |
| `DATASET_TYPE` | `str` | `'regression'` or `'classification'`. |
| `TARGET_COLUMNS` | `List[str]` | Column names, ordered to match your `targets` list. **Load-bearing even when you override `load_dataset`** — the default `save_csv` hook reads it to name the target columns of the exported CSV, so a stale or placeholder value ships a raw CSV with headers like `target_0`. Keep it correct. |
| `SMILES_COLUMN` | `str` | Read by the default `load_dataset` only; once you override that hook it is documentation. |
| `SOURCE_PATH` | `str` | Local input path. Irrelevant once `load_dataset` downloads its own data. |
| `USE_COORDINATES` | `bool` | Have RDKit generate 3D coordinates into `node_coordinates`. Slow, and fails on some molecules. Leave `False` unless the task needs geometry. |
| `COMPRESS` | `bool` | Leave `True`. |
| `__TESTING__` | `bool` | `True` truncates to 50 elements. Use for smoke runs; commit as `False`. |
| `__DEBUG__` | `bool` | `True` (the template default) reuses `results/<script>/debug/` instead of creating a timestamped folder. Leave it alone. |

For the `.xyz` path, `create_graph_datasets__from_xyz.py` adds `PARSER_CLASS` (a
subclass of `AbstractXyzParser`, e.g. `QM9XyzParser` from `chem_mat_data.data`) and
expects `SOURCE_PATH` to point at an xyz bundle — a folder or archive of `.xyz` files.

## Hooks

Always decorate with `@experiment.hook('<name>', default=False, replace=True)` in a
sub-experiment. (`default=True, replace=False` is what the *base* uses to register a
fallback; using it in a sub-experiment means your version may not take effect.)

### `load_dataset(e: Experiment) -> Dict[int, dict]` — required

The one hook you always write. Returns a dict keyed by contiguous integer index; each
value is a dict containing at least:

- `smiles: str` — a SMILES string RDKit can parse. (Alternatively `mol: Chem.Mol`;
  the worker accepts either, which is how the xyz path works.)
- `targets: List[float]` — flat, ordered, one entry per target. Single-target datasets
  use `[value]`. Missing entries in a *multi-target* dataset use `-1.0` as sentinel —
  but only where `-1.0` cannot be a real measurement for that target; otherwise skip
  the row. See the filtering rules in SKILL.md stage 4.

Any other keys pass through untouched and are visible to `add_graph_metadata`.

### `add_graph_metadata(e, data: dict, graph: dict) -> None` — optional

Runs inside the worker, after the graph exists. Attach per-graph provenance:

```python
graph['graph_doi'] = data.get('doi', '')
```

Keep values msgpack-serializable — str, int, float, bool, list, numpy array. Return
value is ignored; mutate `graph` in place.

### `save_csv(e, dataset: Dict[int, dict]) -> None` — optional

Override when the default flattening loses something you want in the raw CSV. The
default writes `smiles` plus one column per target, named from `TARGET_COLUMNS`.

### `after_save(e, index_data_map, graphs, **kwargs) -> None` — optional

Runs after the `.mpack` is written. `create_graph_datasets__from_xyz.py` uses it to
emit a `meta.csv` of scalar per-molecule fields. Accept `**kwargs`.

## Canonical shape

```python
"""
One-paragraph summary: what the dataset is, where it comes from, what the targets are.

**Source**: <Author et al., Journal Year>. DOI: <doi>
**Targets**: <list them with units>
"""
import os
from typing import Dict

import requests
import pandas as pd
import rdkit.Chem as Chem
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

DATASET_NAME: str = 'example'
DATASET_TYPE: str = 'regression'
DESCRIPTION: str = (
    'Two to five sentences describing what was measured or computed, on what, '
    'by what method, and what the dataset is typically used for.'
)
METADATA: dict = {
    'verbose': 'Human Readable Dataset Name',
    'tags': ['Molecules', 'SMILES', '...'],
    'sources': ['https://doi.org/...'],
    'target_descriptions': {
        # String keys - see references/metadata.md. Both forms occur in the existing
        # metadata, but string is the convention and what the gates expect.
        '0': 'column_name - what it is, with units',
    },
}

DOWNLOAD_URL: str = 'https://.../data.csv'

__TESTING__ = False

experiment = Experiment.extend(
    'create_graph_datasets.py',        # relative — the script lives next to it
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)


@experiment.hook('load_dataset', default=False, replace=True)
def load_dataset(e: Experiment) -> Dict[int, dict]:
    """
    Download the dataset (cached) and return it in the index → record format.

    :param e: The running experiment.
    :returns: Mapping of integer index to record dicts with 'smiles' and 'targets'.
    """
    # -- download with caching --
    path = os.path.join(e.path, 'raw_data.csv')
    if not os.path.exists(path):
        e.log('downloading dataset...')
        response = requests.get(DOWNLOAD_URL, timeout=300)
        response.raise_for_status()
        with open(path, 'wb') as file:
            file.write(response.content)
        e.log(f'downloaded {os.path.getsize(path) / 1024**2:.1f} MB')
    else:
        e.log(f'using cached file @ {path}')

    df = pd.read_csv(path)
    e.log(f'loaded {len(df)} rows')

    dataset: Dict[int, dict] = {}
    skipped = 0
    for record in df.to_dict('records'):
        smiles = record.get('smiles')
        if not smiles or pd.isna(smiles) or Chem.MolFromSmiles(smiles) is None:
            skipped += 1
            continue

        record['smiles'] = smiles
        record['targets'] = [record['target_column']]
        dataset[len(dataset)] = record

    e.log(f'built dataset with {len(dataset)} molecules ({skipped} skipped)')
    return dataset


experiment.run_if_main()
```

Last line is always `experiment.run_if_main()`.

## Download idioms

Cache into `e.path` (the results folder) so a re-run does not re-fetch.

```python
# small file
response = requests.get(url, timeout=300)
response.raise_for_status()

# large file — stream it
with requests.get(url, stream=True, timeout=300) as response:
    response.raise_for_status()
    with open(path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

# Figshare, or anywhere the certificate chain misbehaves
from chem_mat_data.connectors import FileDownloadSource
with FileDownloadSource(url, verbose=True, ssl_verify=False) as source:
    df = pd.read_excel(source.fetch(), sheet_name=0)

# archive
import zipfile
with zipfile.ZipFile(archive_path) as zf:
    e.log(f'members: {zf.namelist()}')
    zf.extract('data/measurements.csv', path=e.path)
```

## Conventions worth matching

- Log with `e.log(...)`, not `print` — it lands in `experiment_out.log`.
- reST docstrings with `:param:` / `:returns:`, per the project's `CLAUDE.md`.
- Comment blocks in the `# == SECTION ==` and `# :param NAME:` style the existing
  scripts use — `cmmanage` parses `# :param:` comments for documentation.
- Silence RDKit's console spam when a dataset has many unparseable entries:
  ```python
  from rdkit import RDLogger
  RDLogger.DisableLog('rdApp.*')
  ```
