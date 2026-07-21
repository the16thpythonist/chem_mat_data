"""
This experiment module creates a graph dataset from the Open-Source QSAR pKa dataset
for regression tasks.

The dataset contains 7,912 compounds with experimental aqueous pKa values compiled
from the iBonD database, covering both acidic and basic ionisation. Each compound
has a measured pKa value, temperature, measurement method, and acid/base classification.

The pKa (negative log of the acid dissociation constant) is a fundamental molecular
property governing solubility, membrane permeability, and bioavailability. This dataset
was curated for benchmarking open-source QSAR models against commercial tools
(ACD/Labs, Chemaxon).

References:
- Mansouri, K. et al., J. Cheminform., 11 (2019) 60
  DOI: 10.1186/s13321-019-0384-1
"""

import os
import io
import zipfile
import subprocess
from typing import Dict

import pandas as pd
import rdkit.Chem as Chem
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

# == DATASET METADATA ==

DATASET_NAME: str = 'pka_water'

DESCRIPTION: str = (
    'Aqueous pKa dataset with 7,912 experimental acid dissociation constants from the '
    'iBonD database. Covers both acidic and basic ionisation across diverse chemical '
    'space. Curated by Mansouri et al. (2019) for benchmarking open-source QSAR pKa '
    'prediction models against commercial tools (ACD/Labs, Chemaxon). The target is '
    'the experimental pKa value (dimensionless, typically ranging from -10 to 16).'
)

METADATA: dict = {
    'tags': [
        'Molecules',
        'SMILES',
        'pKa',
        'Acid Dissociation',
        'QSAR',
        'Physical Chemistry',
        'Regression',
        'Benchmark',
    ],
    'verbose': 'Aqueous pKa (iBonD, 7912 compounds)',
    'sources': [
        'https://doi.org/10.1186/s13321-019-0384-1',
    ],
    'notes': [
        'Regression task: predict experimental aqueous pKa.',
        'Dataset includes both acidic and basic pKa values.',
        'pKa range: approximately -10 to 16.',
        'Source: iBonD database, curated by Mansouri et al. (2019).',
    ],
    'target_descriptions': {
        '0': 'pKa - Negative log of acid dissociation constant in water (dimensionless); lower values indicate stronger acids, higher values indicate weaker acids / stronger bases',
    },
}

# == DOWNLOAD SOURCE ==

DATASET_URL = (
    'https://static-content.springer.com/esm/'
    'art%3A10.1186%2Fs13321-019-0384-1/MediaObjects/'
    '13321_2019_384_MOESM1_ESM.zip'
)

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


def _download_and_extract(e: Experiment) -> pd.DataFrame:
    """
    Download the supplementary ZIP and load the pKa text file as a DataFrame.

    :param e: The running experiment instance.
    :returns: DataFrame with columns including Smiles and pKa.
    """
    zip_path = os.path.join(e.path, 'pka_supplementary.zip')
    if not os.path.exists(zip_path):
        e.log(f'Downloading pKa dataset from Springer...')
        result = subprocess.run(
            ['curl', '-sL', '-o', zip_path, DATASET_URL],
            timeout=120,
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f'curl failed: {result.stderr.decode()}'
            )
        e.log(f'Downloaded {os.path.getsize(zip_path) / 1e6:.1f} MB')

    with zipfile.ZipFile(zip_path, 'r') as zf:
        with zf.open('pKaInWater_7912.txt') as f:
            df = pd.read_csv(f, sep='\t')

    return df


@experiment.hook('load_dataset', default=False, replace=True)
def load_dataset(e: Experiment) -> Dict[int, dict]:
    """
    Download and process the aqueous pKa dataset.

    Downloads the supplementary ZIP, extracts the tab-separated pKa file with
    7,912 compounds, validates SMILES with RDKit, and returns a dictionary with
    the experimental pKa as the regression target.

    :returns: Dictionary mapping indices to compound data dictionaries.
    """
    df = _download_and_extract(e)
    e.log(f'Loaded {len(df)} entries')

    dataset: Dict[int, dict] = {}
    index = 0
    invalid = 0

    for _, row in df.iterrows():
        smiles = str(row['Smiles']).strip()
        if not smiles or smiles == 'nan':
            continue

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            invalid += 1
            continue

        pka = row['pKa']
        if pd.isna(pka):
            continue

        dataset[index] = {
            'smiles': smiles,
            'targets': [float(pka)],
            'acid_or_basic': str(row.get('basicOrAcidic', '')),
            'temperature': str(row.get('temp', '')),
            'method': str(row.get('method', '')),
            'pka_type': str(row.get('type', '')),
            'functional_group': str(row.get('group', '')),
        }
        index += 1

    e.log(f'Processed {len(dataset)} compounds ({invalid} invalid SMILES skipped)')

    # Log acid/base distribution
    acidic = sum(1 for d in dataset.values() if d['acid_or_basic'] == 'acidic')
    basic = sum(1 for d in dataset.values() if d['acid_or_basic'] == 'basic')
    e.log(f'  Acidic: {acidic}, Basic: {basic}')

    return dataset


@experiment.hook('add_graph_metadata', default=False, replace=True)
def add_graph_metadata(_e: Experiment, data: dict, graph: dict) -> None:
    """Add pKa measurement metadata to each graph."""
    graph['graph_acid_or_basic'] = data['acid_or_basic']
    graph['graph_pka_type'] = data['pka_type']
    graph['graph_functional_group'] = data['functional_group']


experiment.run_if_main()
