"""
This experiment module creates a graph dataset from the polymer gas separation membrane
dataset for multi-target regression tasks.

The dataset contains 778 polymers with experimentally measured gas permeabilities for
up to 6 gases: He, H2, O2, N2, CO2, and CH4. Permeabilities are reported in Barrers
(log10 scale). Not all polymers have measurements for all gases — missing values are
filled with -1 as a sentinel.

Gas separation membranes are critical for industrial processes like nitrogen generation,
hydrogen recovery, and carbon capture. Predicting permeability from polymer structure
enables rational design of next-generation membrane materials.

Polymer SMILES use ``*`` wildcards for polymerisation attachment points, which are
replaced with ``[H]`` for RDKit parsing.

References:
- Yuan, Q. et al., Sci. Adv., 8 (2022) eabn9545
  DOI: 10.1126/sciadv.abn9545
"""

import os
import shutil
import subprocess
from typing import Dict

import pandas as pd
import rdkit.Chem as Chem
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

# == DATASET METADATA ==

DATASET_NAME: str = 'polymer_gas_permeability'

DESCRIPTION: str = (
    'Polymer gas separation membrane dataset with 778 polymers and experimentally '
    'measured gas permeabilities (log10 Barrer) for He, H2, O2, N2, CO2, and CH4. '
    'Not all gases are measured for all polymers — missing values are set to -1. '
    'Curated by Yuan et al. (2022) for ML-enabled interpretable discovery of '
    'innovative gas separation membrane polymers.'
)

METADATA: dict = {
    'tags': [
        'Molecules',
        'SMILES',
        'Polymers',
        'Gas Permeability',
        'Membranes',
        'Multi-target Regression',
        'Materials Design',
    ],
    'verbose': 'Polymer Gas Separation Membranes (778 polymers)',
    'sources': [
        'https://doi.org/10.1126/sciadv.abn9545',
    ],
    'notes': [
        'Multi-target regression: 6 gas permeabilities (log10 Barrer).',
        'Missing permeability values encoded as -1 sentinel.',
        'Polymer SMILES use * for attachment points; replaced with [H] for RDKit parsing.',
        'Experimental data from literature, curated by Yuan et al. (2022).',
    ],
    'target_descriptions': {
        '0': 'log10(He permeability) - Helium permeability in log10(Barrer); -1 if not measured',
        '1': 'log10(H2 permeability) - Hydrogen permeability in log10(Barrer); -1 if not measured',
        '2': 'log10(O2 permeability) - Oxygen permeability in log10(Barrer); -1 if not measured',
        '3': 'log10(N2 permeability) - Nitrogen permeability in log10(Barrer); -1 if not measured',
        '4': 'log10(CO2 permeability) - Carbon dioxide permeability in log10(Barrer); -1 if not measured',
        '5': 'log10(CH4 permeability) - Methane permeability in log10(Barrer); -1 if not measured',
    },
}

# == DOWNLOAD SOURCE ==

# The original data is in the Science Advances supplementary materials
# but is easier to access from the GitHub repository
DATASET_URL = 'https://raw.githubusercontent.com/jsunn-y/PolymerGasMembraneML/main/datasetA_imputed_all.csv'

GAS_COLS = ['log10_He', 'log10_H2', 'log10_O2', 'log10_N2', 'log10_CO2', 'log10_CH4']

# == EXPERIMENT PARAMETERS ==

DATASET_TYPE: str = 'regression'

__DEBUG__ = True
__TESTING__ = False

experiment = Experiment.extend(
    'create_graph_datasets.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)


def _sanitize_polymer_smiles(smiles: str) -> str:
    """Replace ``*`` attachment-point wildcards with ``[H]``."""
    return smiles.replace('*', '[H]')


@experiment.hook('load_dataset', default=False, replace=True)
def load_dataset(e: Experiment) -> Dict[int, dict]:
    """
    Download and process the polymer gas permeability dataset.

    Downloads the experimental polymer gas permeability CSV, sanitises polymer
    SMILES for RDKit, and returns a dictionary with 6 gas permeabilities
    (log10 Barrer) as regression targets. Missing values are set to -1.

    :returns: Dictionary mapping indices to polymer data dictionaries.
    """
    assets_path = os.path.join(
        os.path.dirname(__file__), 'assets', 'polymer_gas_permeability.csv'
    )
    csv_path = os.path.join(e.path, 'polymer_gas_permeability.csv')

    if os.path.exists(assets_path) and os.path.getsize(assets_path) > 1000:
        shutil.copy2(assets_path, csv_path)
        e.log('Using cached dataset from assets/')
    elif not os.path.exists(csv_path) or os.path.getsize(csv_path) < 1000:
        e.log('Downloading polymer gas permeability dataset...')
        result = subprocess.run(
            ['curl', '-sL', '-o', csv_path, DATASET_URL],
            timeout=60,
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f'curl failed: {result.stderr.decode()}')

    df = pd.read_csv(csv_path)
    e.log(f'Loaded {len(df)} polymers')

    dataset: Dict[int, dict] = {}
    index = 0
    invalid = 0

    for _, row in df.iterrows():
        raw_smiles = str(row['Smiles']).strip()
        if not raw_smiles or raw_smiles == 'nan':
            continue

        smiles = _sanitize_polymer_smiles(raw_smiles)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            invalid += 1
            continue

        # Use -1 sentinel for missing gas permeabilities
        targets = []
        for col in GAS_COLS:
            val = row.get(col, None)
            targets.append(float(val) if pd.notna(val) else -1.0)

        # Skip polymers with no measurements at all
        if all(t == -1.0 for t in targets):
            continue

        polymer_name = str(row.get('Name', ''))
        if polymer_name == 'nan':
            polymer_name = ''

        dataset[index] = {
            'smiles': smiles,
            'targets': targets,
            'raw_smiles': raw_smiles,
            'polymer_name': polymer_name,
        }
        index += 1

    e.log(f'Processed {len(dataset)} polymers ({invalid} invalid SMILES skipped)')

    # Log coverage per gas
    for i, gas in enumerate(['He', 'H2', 'O2', 'N2', 'CO2', 'CH4']):
        measured = sum(1 for d in dataset.values() if d['targets'][i] != -1.0)
        e.log(f'  {gas}: {measured} measured ({100 * measured / len(dataset):.0f}%)')

    return dataset


@experiment.hook('add_graph_metadata', default=False, replace=True)
def add_graph_metadata(_e: Experiment, data: dict, graph: dict) -> None:
    """Add polymer metadata to each graph."""
    graph['graph_raw_smiles'] = data['raw_smiles']
    graph['graph_polymer_name'] = data['polymer_name']


experiment.run_if_main()
