"""
ARCHETYPE 1 of 4 — a CSV/Excel file with a SMILES column and one or more target columns.

This is the common case: roughly two thirds of the datasets in ChemMatData look like
this. Modelled on ``create_graph_datasets__compas_1x.py`` (COMPAS-1x, ~34k
cata-condensed polybenzenoid hydrocarbons with 9 DFT-computed properties).

What this example demonstrates:

- Extending the base template with a *relative* path.
- A ``load_dataset`` hook that downloads with caching into the results folder.
- Building a multi-target ``targets`` list in a fixed, documented order.
- Attaching per-graph provenance via ``add_graph_metadata``.

Copy this file, rename it to ``create_graph_datasets__<name>.py``, place it in
``chem_mat_data/scripts/``, and adapt.
"""

import os
from typing import Dict, List

import requests
import pandas as pd
import rdkit.Chem as Chem
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

# == SOURCE PARAMETERS ==

# :param DOWNLOAD_URL:
#       Direct download URL for the raw CSV. Must be a *direct* file URL, not a
#       repository landing page - see references/data-sources.md.
DOWNLOAD_URL: str = (
    'https://gitlab.com/porannegroup/compas/-/raw/main/COMPAS-1/compas-1x.csv'
    '?ref_type=heads&inline=false'
)

# :param SMILES_COLUMN:
#       Name of the CSV column holding the SMILES strings.
SMILES_COLUMN: str = 'smiles'

# :param TARGET_COLUMNS:
#       The target columns, in the exact order they will appear in the target vector.
#       This order must match the keys of METADATA['target_descriptions'].
TARGET_COLUMNS: List[str] = [
    'HOMO_eV',
    'LUMO_eV',
    'GAP_eV',
    'Dipmom_Debye',
    'Etot_eV',
    'aEA_eV',
    'aIP_eV',
    'NFOD',
    'n_rings',
]

# :param DATASET_TYPE:
#       'regression' or 'classification'.
DATASET_TYPE: str = 'regression'

# :param DESCRIPTION:
#       Human-readable summary: what, how many, measured/computed how, used for what.
DESCRIPTION: str = (
    'The COMPAS-1x dataset is part of the COMPAS Project, a computational database of '
    'polycyclic aromatic systems. It contains optimized ground-state structures and '
    'molecular properties for approximately 34,000 cata-condensed polybenzenoid '
    'hydrocarbons, computed at the GFN2-xTB level of theory. Molecules range from 1 to '
    '11 rings, with properties including HOMO/LUMO energies, energy gaps, dipole '
    'moments and electron affinities. It supports data-driven investigation of '
    'structure-property relationships in aromatic systems and machine learning '
    'applications in organic electronics and materials design.'
)

# :param METADATA:
#       Merged over the auto-generated metadata - see references/metadata.md.
METADATA: dict = {
    'verbose': 'Cata-Condensed Polybenzenoid Hydrocarbons (COMPAS-1x)',
    'tags': [
        'Molecules',
        'SMILES',
        'Quantum Chemistry',
        'Molecular Properties',
        'Polycyclic Aromatic Hydrocarbons',
        'PAH',
        'GFN2-xTB',
    ],
    'sources': [
        'https://doi.org/10.1021/acs.jcim.2c00503',
        'https://gitlab.com/porannegroup/compas/-/tree/main/COMPAS-1',
    ],
    'target_descriptions': {
        '0': 'HOMO_eV - Energy of the highest occupied molecular orbital (eV)',
        '1': 'LUMO_eV - Energy of the lowest unoccupied molecular orbital (eV)',
        '2': 'GAP_eV - HOMO-LUMO energy gap (eV)',
        '3': 'Dipmom_Debye - Dipole moment (Debye)',
        '4': 'Etot_eV - Total energy of the molecule (eV)',
        '5': 'aEA_eV - Adiabatic electron affinity (eV)',
        '6': 'aIP_eV - Adiabatic ionization potential (eV)',
        '7': 'NFOD - Fractional occupation density, indicating polyradical character',
        '8': 'n_rings - Number of aromatic rings in the molecule',
    },
}

# == EXPERIMENT PARAMETERS ==

# :param __TESTING__:
#       True limits processing to 50 elements. Use for smoke runs; commit as False.
__TESTING__ = False

# The path is relative because this script sits next to the base template in
# chem_mat_data/scripts/. Never hardcode an absolute path here.
experiment = Experiment.extend(
    'create_graph_datasets.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)


@experiment.hook('add_graph_metadata', default=False, replace=True)
def add_graph_metadata(e: Experiment, data: dict, graph: dict) -> None:
    """
    Attach the source identifier to each graph so individual molecules stay traceable
    back to the original dataset.

    Runs inside the processing worker. Values must be msgpack-serializable. Mutate
    ``graph`` in place; the return value is ignored.

    :param e: The running experiment.
    :param data: The original record dict from ``load_dataset``.
    :param graph: The graph dict produced by ``MoleculeProcessing``.
    """
    graph['graph_id'] = str(data.get('molecule', ''))


@experiment.hook('load_dataset', default=False, replace=True)
def load_dataset(e: Experiment) -> Dict[int, dict]:
    """
    Download the COMPAS-1x CSV (cached) and convert it into the index → record mapping
    that the base experiment consumes.

    The download is cached inside the experiment results folder, so re-running the
    script does not re-fetch the file.

    :param e: The running experiment.

    :returns: Mapping of integer index to record dicts, each containing at least a
        ``smiles`` string and a ``targets`` list of floats.
    """
    # -- download with caching --
    csv_path = os.path.join(e.path, 'raw_data.csv')
    if not os.path.exists(csv_path):
        e.log(f'downloading dataset from {DOWNLOAD_URL} ...')
        response = requests.get(DOWNLOAD_URL, timeout=300)
        response.raise_for_status()
        with open(csv_path, 'wb') as file:
            file.write(response.content)
        e.log(f'downloaded {os.path.getsize(csv_path) / 1024**2:.1f} MB')
    else:
        e.log(f'using cached CSV @ {csv_path}')

    df = pd.read_csv(csv_path)
    e.log(f'loaded {len(df)} rows with columns: {list(df.columns)}')

    # -- build the dataset mapping --
    # Only genuinely unusable rows are skipped: no SMILES, or SMILES that RDKit
    # cannot parse. No heuristic filtering - see the filtering policy in SKILL.md.
    dataset: Dict[int, dict] = {}
    skipped_empty: int = 0
    skipped_invalid: int = 0

    for record in df.to_dict('records'):
        smiles = record.get(e.SMILES_COLUMN)

        if not smiles or pd.isna(smiles):
            skipped_empty += 1
            continue

        if Chem.MolFromSmiles(smiles) is None:
            skipped_invalid += 1
            continue

        record['smiles'] = smiles
        record['targets'] = [float(record[column]) for column in e.TARGET_COLUMNS]

        dataset[len(dataset)] = record

    e.log(
        f'built dataset with {len(dataset)} molecules '
        f'({skipped_empty} without SMILES, {skipped_invalid} unparseable)'
    )
    return dataset


experiment.run_if_main()
