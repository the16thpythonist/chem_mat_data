"""
This experiment module creates a graph dataset from the Contaminant Reactivity QSAR
dataset for regression tasks.

The dataset contains ~1,700 contaminant-oxidant reaction rate constants (log k) for
four common water treatment oxidants: hypochlorous acid (HClO), chlorine dioxide
(ClO2), ozone (O3), and sulfate radical (SO4-). Each entry has a contaminant SMILES,
the measured log k value, and the reaction pH.

Predicting oxidant reactivity is important for water treatment design — knowing how
fast a contaminant reacts with a given oxidant determines the required dose and
contact time.

References:
- Zhong, S. et al., Environ. Sci. Technol., 55 (2021) 16655-16667
  DOI: 10.1021/acs.est.1c04883
"""

import os
import subprocess
from typing import Dict

import pandas as pd
import rdkit.Chem as Chem
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

# == DATASET METADATA ==

DATASET_NAME: str = 'contaminant_reactivity'

DESCRIPTION: str = (
    'Contaminant oxidant reactivity QSAR dataset with ~1,700 reaction rate constants '
    '(log k) for four water treatment oxidants: HClO (195 entries), ClO2 (191), O3 (759), '
    'and SO4 radical (557). Each entry has a contaminant SMILES, measured second-order '
    'rate constant (log k in log M-1 s-1), and reaction pH. Curated from multiple '
    'QSAR studies by Zhong et al. (2021) with ML-assisted knowledge transfer.'
)

METADATA: dict = {
    'tags': [
        'Molecules',
        'SMILES',
        'Reactivity',
        'Oxidation',
        'Water Treatment',
        'Rate Constants',
        'QSAR',
        'Regression',
        'Environmental Chemistry',
    ],
    'verbose': 'Contaminant Oxidant Reactivity (4 oxidants, ~1,700 entries)',
    'sources': [
        'https://doi.org/10.1021/acs.est.1c04883',
    ],
    'notes': [
        'Regression task: predict log k (log M-1 s-1) for contaminant-oxidant reactions.',
        'Four oxidants combined: HClO, ClO2, O3, SO4 radical.',
        'pH is stored as graph metadata, not as a target.',
        'Oxidant identity is stored as graph metadata.',
        'Curated from multiple literature QSAR studies by Zhong et al. (2021).',
    ],
    'target_descriptions': {
        '0': 'log k - Second-order reaction rate constant in log10(M-1 s-1); higher values indicate faster degradation of the contaminant by the oxidant',
    },
}

# == DOWNLOAD SOURCE ==

DATASET_URL = (
    'https://pubs.acs.org/doi/suppl/10.1021/acs.est.1c04883/suppl_file/es1c04883_si_001.xlsx'
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

# Sheets to load and their oxidant labels
_OXIDANT_SHEETS = [
    ('HClO_data', 'HClO'),
    ('ClO2_data', 'ClO2'),
    ('O3_data', 'O3'),
    ('SO4_data', 'SO4'),
]


@experiment.hook('load_dataset', default=False, replace=True)
def load_dataset(e: Experiment) -> Dict[int, dict]:
    """
    Download and process the contaminant oxidant reactivity dataset.

    Downloads the supplementary Excel file from ACS, reads all four oxidant
    sheets, validates SMILES with RDKit, and returns a combined dictionary
    with log k as the regression target and oxidant/pH as metadata.

    :returns: Dictionary mapping indices to contaminant data dictionaries.
    """
    # Try cached copy in assets, otherwise download
    assets_path = os.path.join(
        os.path.dirname(__file__), 'assets', 'contaminant_reactivity.xlsx'
    )
    xlsx_path = os.path.join(e.path, 'contaminant_reactivity.xlsx')

    if os.path.exists(assets_path) and os.path.getsize(assets_path) > 50_000:
        import shutil
        shutil.copy2(assets_path, xlsx_path)
        e.log(f'Using cached dataset from assets/')
    elif not os.path.exists(xlsx_path) or os.path.getsize(xlsx_path) < 50_000:
        e.log('Downloading contaminant reactivity dataset from ACS...')
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
        if result.returncode != 0 or os.path.getsize(xlsx_path) < 50_000:
            raise RuntimeError(
                f'ACS download blocked. Download manually from {DATASET_URL} '
                f'and place at {assets_path}'
            )

    dataset: Dict[int, dict] = {}
    index = 0
    invalid = 0

    for sheet_name, oxidant in _OXIDANT_SHEETS:
        df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
        e.log(f'{oxidant}: {len(df)} entries')

        for _, row in df.iterrows():
            smiles = str(row['SMILES']).strip()
            if not smiles or smiles == 'nan':
                continue

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                invalid += 1
                continue

            logk = row['logk']
            if pd.isna(logk):
                continue

            ph = row.get('pH', None)
            ph_val = float(ph) if pd.notna(ph) else -1.0

            dataset[index] = {
                'smiles': smiles,
                'targets': [float(logk)],
                'oxidant': oxidant,
                'ph': ph_val,
                'compound_name': str(row.get('chemical compound', '')),
            }
            index += 1

    e.log(f'Processed {len(dataset)} total entries ({invalid} invalid SMILES skipped)')

    from collections import Counter
    ox_counts = Counter(d['oxidant'] for d in dataset.values())
    for ox in ['HClO', 'ClO2', 'O3', 'SO4']:
        e.log(f'  {ox}: {ox_counts.get(ox, 0)}')

    return dataset


@experiment.hook('add_graph_metadata', default=False, replace=True)
def add_graph_metadata(_e: Experiment, data: dict, graph: dict) -> None:
    """Add oxidant and pH metadata to each graph."""
    graph['graph_oxidant'] = data['oxidant']
    graph['graph_ph'] = data['ph']
    graph['graph_compound_name'] = data['compound_name']


experiment.run_if_main()
