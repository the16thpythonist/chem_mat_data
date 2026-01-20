"""
This experiment module creates a graph dataset from the Blood-Brain Barrier Penetration (BBBP)
dataset for binary classification tasks.

The BBBP dataset is extracted from a study on the modeling and prediction of blood-brain barrier
permeability. As a membrane separating circulating blood and brain extracellular fluid, the
blood-brain barrier blocks most drugs, hormones and neurotransmitters. Thus penetration of the
barrier forms a long-standing issue in development of drugs targeting the central nervous system.

This dataset includes binary labels for over 2000 compounds on their permeability properties,
compiled by Martins et al. (2012) and adopted by MoleculeNet as a benchmark dataset for
machine learning.

References:
- Martins, I.F. et al., J. Chem. Inf. Model., 52 (2012) 1686-1697
  DOI: 10.1021/ci300124c
- MoleculeNet: Wu, Z. et al., Chem. Sci., 9 (2018) 513-530
  DOI: 10.1039/C7SC02664A
"""
import os
import csv
from typing import Dict

import rdkit.Chem as Chem
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

from chem_mat_data.connectors import FileDownloadSource

# == DATASET METADATA ==

DATASET_NAME: str = 'bbbp'

DESCRIPTION: str = (
    'The Blood-Brain Barrier Penetration (BBBP) dataset is extracted from a study on the '
    'modeling and prediction of barrier permeability. As a membrane separating circulating '
    'blood and brain extracellular fluid, the blood-brain barrier blocks most drugs, hormones '
    'and neurotransmitters. Thus penetration of the barrier forms a long-standing issue in '
    'development of drugs targeting the central nervous system. This dataset includes binary '
    'labels for over 2000 compounds on their permeability properties. The target is encoded '
    'as a one-hot vector where index 0 indicates non-penetrating and index 1 indicates '
    'penetrating compounds.'
)

METADATA: dict = {
    'tags': [
        'Molecules',
        'SMILES',
        'Drug Discovery',
        'Blood-Brain Barrier',
        'BBB',
        'Permeability',
        'Binary Classification',
        'ADMET',
        'Pharmacokinetics',
        'MoleculeNet',
        'Benchmark',
    ],
    'verbose': 'Blood-Brain Barrier Penetration (BBBP)',
    'sources': [
        'https://doi.org/10.1021/ci300124c',  # Martins et al. 2012
        'https://doi.org/10.1039/C7SC02664A',  # MoleculeNet
        'http://moleculenet.ai/',
    ],
    'notes': [
        'Binary classification task: penetrating (1) vs non-penetrating (0).',
        'Target is one-hot encoded: [non-penetrating, penetrating].',
        'Original dataset compiled by Martins et al. (2012).',
        'Adopted as benchmark dataset by MoleculeNet.',
        'Compounds with salts (containing ".") and single atoms are filtered out.',
    ],
    'target_descriptions': {
        '0': 'Non-penetrating - compound does not cross the blood-brain barrier',
        '1': 'Penetrating - compound crosses the blood-brain barrier',
    },
}

# == DOWNLOAD SOURCE ==

# HuggingFace mirror of the MoleculeNet BBBP dataset
DATASET_URL = 'https://huggingface.co/datasets/chao1224/MoleculeSTM/resolve/main/MoleculeNet_data/bbbp/raw/BBBP.csv'

# == EXPERIMENT PARAMETERS ==

DATASET_TYPE: str = 'classification'

__DEBUG__ = True
__TESTING__ = False

experiment = Experiment.extend(
    'create_graph_datasets.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)


@experiment.hook('load_dataset', default=False, replace=True)
def load_dataset(e: Experiment) -> Dict[int, dict]:
    """
    Download and process the BBBP dataset for binary classification.

    This function:
    1. Downloads the BBBP CSV file from HuggingFace (MoleculeNet mirror)
    2. Saves the file to the experiment archive folder
    3. Parses the CSV to extract compound names, SMILES, and labels
    4. Filters out salts and invalid molecules
    5. Creates one-hot encoded target vectors

    :returns: Dictionary mapping indices to compound data dictionaries
    """
    e.log('=' * 80)
    e.log('DOWNLOADING BBBP DATASET')
    e.log('=' * 80)

    csv_path = os.path.join(e.path, 'BBBP.csv')

    if os.path.exists(csv_path):
        e.log(f'Dataset already exists at {csv_path}')
    else:
        e.log(f'Downloading dataset from {DATASET_URL}...')
        with FileDownloadSource(DATASET_URL, verbose=True) as source:
            downloaded_path = source.fetch()
            e.log(f'  Downloaded to temporary location')

            # Copy to experiment archive folder
            import shutil
            shutil.copy(downloaded_path, csv_path)
            e.log(f'  Saved to {csv_path}')

    # == PARSE CSV FILE ==
    e.log('')
    e.log('=' * 80)
    e.log('PARSING BBBP DATA')
    e.log('=' * 80)

    dataset: Dict[int, dict] = {}
    index = 0
    penetrating = 0
    non_penetrating = 0

    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            smiles = row['smiles']
            name = row['name']
            p_np = int(row['p_np'])

            # == MOLECULE FILTERS ==
            # Remove salts (molecules with '.' in SMILES)
            if '.' in smiles:
                continue

            # Validate SMILES with RDKit
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                continue

            # Filter out single atoms
            if len(mol.GetAtoms()) < 2:
                continue

            # == CREATE ONE-HOT TARGET VECTOR ==
            # Index 0: non-penetrating (p_np == 0)
            # Index 1: penetrating (p_np == 1)
            targets = [p_np == i for i in range(2)]

            if p_np == 1:
                penetrating += 1
            else:
                non_penetrating += 1

            data = {
                'smiles': smiles,
                'name': name,
                'targets': targets,
                'p_np': p_np,
            }

            dataset[index] = data
            index += 1

    # == PRINT STATISTICS ==
    e.log('')
    e.log('=' * 80)
    e.log('DATASET STATISTICS')
    e.log('=' * 80)
    e.log(f'Total compounds: {len(dataset)}')
    e.log(f'  - Penetrating (1): {penetrating} ({100*penetrating/len(dataset):.1f}%)')
    e.log(f'  - Non-penetrating (0): {non_penetrating} ({100*non_penetrating/len(dataset):.1f}%)')
    e.log('')

    return dataset


@experiment.hook('add_graph_metadata', default=False, replace=True)
def add_graph_metadata(_e: Experiment, data: dict, graph: dict) -> None:
    """
    Add additional metadata fields to each graph representation.

    This includes:
    - The compound name
    """
    graph['graph_name'] = data['name']


experiment.run_if_main()
