"""
This experiment module creates a graph dataset from the Polymer Glass Transition
Temperature benchmark for regression tasks.

The dataset contains 566 polymers with MD-simulated glass transition temperatures (Tg)
from the benchmark study by Tao et al. (2021). Polymers are represented as repeat-unit
SMILES with ``*`` wildcards indicating polymerisation attachment points, which are
replaced with ``[H]`` for RDKit parsing.

The glass transition temperature is a key thermal property of polymers — below Tg the
polymer is glassy and brittle, above it the polymer becomes rubbery and flexible.
Predicting Tg from structure is important for polymer design in applications ranging
from packaging to aerospace composites.

References:
- Tao, L. et al., J. Chem. Inf. Model., 61 (2021) 5395-5413
  DOI: 10.1021/acs.jcim.1c01031
"""

# == BLOCKED: DO NOT RUN ==
# This dataset is on hold on licensing grounds and must not be processed or
# published. The script is kept for its provenance and processing logic, and
# because the decision may be revisited - see docs/rejected_datasets.md for the
# evidence, who decided, and exactly what would have to change.
LICENSE_BLOCKED: str = (
    'The source is the Supporting Information of an ACS article '
    '(doi:10.1021/acs.jcim.1c01031) that carries an explicit copyright notice, '
    'sits behind a paywall, and whose SI terms permit download "by article for '
    'research use" only - the clause allowing other uses is conditional on a '
    'public use licence being linked to the article, and none exists. The '
    'upstream, PoLyInfo (NIMS MatNavi), separately prohibits bulk acquisition.'
)

if __name__ == '__main__':
    raise SystemExit(
        f'BLOCKED: this dataset is on hold on licensing grounds and must not '
        f'be processed. {LICENSE_BLOCKED} '
        f'Decided 2026-08-19 by the maintainer; see docs/rejected_datasets.md.'
    )


import os
import shutil
import subprocess
from typing import Dict

import pandas as pd
import rdkit.Chem as Chem
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

# == DATASET METADATA ==

DATASET_NAME: str = 'polymer_tg'

DESCRIPTION: str = (
    'Polymer glass transition temperature (Tg) benchmark with 566 polymers and '
    'MD-simulated Tg values in degrees Celsius. Polymers are represented as repeat-unit '
    'SMILES. Tg ranges from -148 to 472 C across diverse polymer chemistries. '
    'Curated by Tao et al. (2021) as a benchmark for comparing ML representations '
    '(fingerprints, descriptors, SMILES-based, graph-based) for polymer informatics.'
)

METADATA: dict = {
    'tags': [
        'Molecules',
        'SMILES',
        'Polymers',
        'Glass Transition',
        'Thermal Properties',
        'Regression',
        'Benchmark',
        'MD Simulation',
    ],
    'verbose': 'Polymer Glass Transition Temperature (566 polymers)',
    'sources': [
        'https://doi.org/10.1021/acs.jcim.1c01031',
    ],
    'notes': [
        'Regression task: predict glass transition temperature (Tg) in degrees Celsius.',
        'Tg values from molecular dynamics simulations.',
        'Polymer SMILES use * for attachment points; replaced with [H] for RDKit parsing.',
        'Benchmark for comparing ML representations for polymer property prediction.',
        'Curated by Tao et al. (2021).',
    ],
    'target_descriptions': {
        '0': 'Tg - Glass transition temperature in degrees Celsius from MD simulation; temperature at which polymer transitions from glassy to rubbery state (range: -148 to 472 C)',
    },
}

# == DOWNLOAD SOURCE ==

DATASET_URL = (
    'https://pubs.acs.org/doi/suppl/10.1021/acs.jcim.1c01031/suppl_file/ci1c01031_si_003.xlsx'
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


def _sanitize_polymer_smiles(smiles: str) -> str:
    """
    Replace ``*`` attachment-point wildcards with ``[H]`` so RDKit can parse
    the polymer repeat-unit SMILES as a valid molecule fragment.
    """
    return smiles.replace('*', '[H]')


@experiment.hook('load_dataset', default=False, replace=True)
def load_dataset(e: Experiment) -> Dict[int, dict]:
    """
    Download and process the polymer Tg benchmark dataset.

    Downloads the supplementary Excel file containing 566 polymers with
    repeat-unit SMILES and MD-simulated Tg values. Sanitises polymer SMILES
    for RDKit and returns a dictionary with Tg as the regression target.

    :returns: Dictionary mapping indices to polymer data dictionaries.
    """
    assets_path = os.path.join(
        os.path.dirname(__file__), 'assets', 'polymer_tg_predictions.xlsx'
    )
    xlsx_path = os.path.join(e.path, 'polymer_tg.xlsx')

    if os.path.exists(assets_path) and os.path.getsize(assets_path) > 10_000:
        shutil.copy2(assets_path, xlsx_path)
        e.log('Using cached dataset from assets/')
    elif not os.path.exists(xlsx_path) or os.path.getsize(xlsx_path) < 10_000:
        e.log('Downloading polymer Tg dataset from ACS...')
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
        if result.returncode != 0 or os.path.getsize(xlsx_path) < 10_000:
            raise RuntimeError(
                f'ACS download blocked. Download manually from {DATASET_URL} '
                f'and place at {assets_path}'
            )

    df = pd.read_excel(xlsx_path)
    e.log(f'Loaded {len(df)} polymers')

    dataset: Dict[int, dict] = {}
    index = 0
    invalid = 0

    for _, row in df.iterrows():
        raw_smiles = str(row['SMILES_RepeatUnit_1']).strip()
        if not raw_smiles or raw_smiles == 'nan':
            continue

        smiles = _sanitize_polymer_smiles(raw_smiles)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            invalid += 1
            continue

        tg = row['MD']
        if pd.isna(tg):
            continue

        monomer = str(row.get('SMILES_Monomer_0', ''))
        if monomer == 'nan':
            monomer = ''

        dataset[index] = {
            'smiles': smiles,
            'targets': [float(tg)],
            'raw_smiles': raw_smiles,
            'monomer_smiles': monomer,
        }
        index += 1

    e.log(f'Processed {len(dataset)} polymers ({invalid} invalid SMILES skipped)')
    return dataset


@experiment.hook('add_graph_metadata', default=False, replace=True)
def add_graph_metadata(_e: Experiment, data: dict, graph: dict) -> None:
    """Add polymer metadata to each graph."""
    graph['graph_raw_smiles'] = data['raw_smiles']
    graph['graph_monomer_smiles'] = data['monomer_smiles']


experiment.run_if_main()
