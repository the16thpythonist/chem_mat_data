"""
This experiment module creates a graph dataset from the PPARgamma binding affinity
QSAR dataset for regression tasks.

The dataset contains 1,735 compounds with experimentally determined binding affinities
for the peroxisome proliferator-activated receptor gamma (PPARgamma). PPARgamma is a
nuclear receptor involved in lipid metabolism, glucose homeostasis, and inflammation,
making it a key drug target for type 2 diabetes (e.g. thiazolidinediones).

Two training sets are combined: TS-1 (419 compounds with pKd/pKi values) and TS-2
(1,316 compounds with pIC50 values). Both are on a negative-log scale, so higher
values indicate stronger binding.

References:
- Wang, Z. et al., Environ. Sci. Technol., 55 (2021) 6857-6866
  DOI: 10.1021/acs.est.0c07040
"""

import os
import subprocess
from typing import Dict

import pandas as pd
import rdkit.Chem as Chem
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

# == DATASET METADATA ==

DATASET_NAME: str = 'pparg_binding'

DESCRIPTION: str = (
    'PPARgamma binding affinity dataset with 1,735 compounds and experimentally '
    'determined binding affinities (pKd/pKi/pIC50) for the peroxisome proliferator-activated '
    'receptor gamma. PPARgamma is a nuclear receptor target for type 2 diabetes drugs. '
    'Combines two curated training sets from ChEMBL: TS-1 (419, pKd/pKi) and TS-2 '
    '(1,316, pIC50). Curated by Wang et al. (2021) with defined applicability domains.'
)

METADATA: dict = {
    'tags': [
        'Molecules',
        'SMILES',
        'Binding Affinity',
        'PPARgamma',
        'Nuclear Receptor',
        'QSAR',
        'Drug Discovery',
        'Regression',
        'ChEMBL',
    ],
    'verbose': 'PPARgamma Binding Affinity (1,735 compounds)',
    'sources': [
        'https://doi.org/10.1021/acs.est.0c07040',
    ],
    'notes': [
        'Regression task: predict binding affinity to PPARgamma.',
        'Target is negative-log binding affinity (pKd/pKi or pIC50); higher = stronger binding.',
        'TS-1 (419 compounds): pKd or pKi values.',
        'TS-2 (1,316 compounds): pIC50 values.',
        'All affinities from ChEMBL, curated by Wang et al. (2021).',
    ],
    'target_descriptions': {
        '0': 'Binding affinity - Negative log of binding constant (pKd/pKi) or inhibitory concentration (pIC50) for PPARgamma; higher values indicate stronger binding (typical range 4-10)',
    },
}

# == DOWNLOAD SOURCE ==

DATASET_URL = (
    'https://pubs.acs.org/doi/suppl/10.1021/acs.est.0c07040/suppl_file/es0c07040_si_002.xlsx'
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


@experiment.hook('load_dataset', default=False, replace=True)
def load_dataset(e: Experiment) -> Dict[int, dict]:
    """
    Download and process the PPARgamma binding affinity dataset.

    Downloads the supplementary Excel file, reads TS-1 and TS-2 sheets,
    combines them, validates SMILES with RDKit, and returns a dictionary
    with binding affinity as the regression target.

    :returns: Dictionary mapping indices to compound data dictionaries.
    """
    # Cache outside experiment dir (which gets wiped on each run)
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts', 'assets')
    os.makedirs(cache_dir, exist_ok=True)
    xlsx_path = os.path.join(cache_dir, 'pparg_dataset.xlsx')

    if not os.path.exists(xlsx_path) or os.path.getsize(xlsx_path) < 100_000:
        e.log('Downloading PPARgamma dataset from ACS...')
        result = subprocess.run(
            [
                'curl', '-sL',
                '-H', 'User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
                '-H', 'Referer: https://pubs.acs.org/',
                '-o', xlsx_path, DATASET_URL,
            ],
            timeout=120,
            capture_output=True,
        )
        if result.returncode != 0 or os.path.getsize(xlsx_path) < 100_000:
            raise RuntimeError(
                f'ACS download blocked. Download manually from {DATASET_URL} '
                f'and place at {xlsx_path}'
            )

    # Load TS-1 (pKd/pKi) and TS-2 (pIC50)
    df1 = pd.read_excel(xlsx_path, sheet_name='Table S1', header=1)
    df2 = pd.read_excel(xlsx_path, sheet_name='Table S2', header=1)
    e.log(f'Loaded TS-1: {len(df1)} compounds (pKd/pKi), TS-2: {len(df2)} compounds (pIC50)')

    dataset: Dict[int, dict] = {}
    index = 0
    invalid = 0
    seen_chembl = set()

    for source_name, df, affinity_col in [
        ('TS-1', df1, 'pKd or pKi'),
        ('TS-2', df2, 'pIC50'),
    ]:
        for _, row in df.iterrows():
            chembl_id = str(row['Compound ChEMBL ID']).strip()

            # Deduplicate by ChEMBL ID
            if chembl_id in seen_chembl:
                continue
            seen_chembl.add(chembl_id)

            smiles = str(row['Neutralized SMILES code']).strip()
            if not smiles or smiles == 'nan':
                continue

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                invalid += 1
                continue

            affinity = row[affinity_col]
            if pd.isna(affinity):
                continue

            dataset[index] = {
                'smiles': smiles,
                'targets': [float(affinity)],
                'chembl_id': chembl_id,
                'source': source_name,
                'affinity_type': affinity_col,
            }
            index += 1

    e.log(f'Processed {len(dataset)} compounds ({invalid} invalid SMILES skipped)')

    ts1_count = sum(1 for d in dataset.values() if d['source'] == 'TS-1')
    ts2_count = sum(1 for d in dataset.values() if d['source'] == 'TS-2')
    e.log(f'  TS-1: {ts1_count}, TS-2: {ts2_count}')

    return dataset


@experiment.hook('add_graph_metadata', default=False, replace=True)
def add_graph_metadata(_e: Experiment, data: dict, graph: dict) -> None:
    """Add compound metadata to each graph."""
    graph['graph_chembl_id'] = data['chembl_id']
    graph['graph_affinity_type'] = data['affinity_type']


experiment.run_if_main()
