"""
This experiment module creates a graph dataset from the MoleculeACE activity cliff
benchmark for regression tasks.

The dataset contains 48,714 compounds across 30 ChEMBL protein targets, each with
experimental binding affinity (pEC50 or pKi). Compounds are annotated as activity
cliff molecules (structurally similar but with large activity differences) or non-cliff
molecules, making this a challenging benchmark for molecular property prediction.

Activity cliffs are a fundamental problem in drug discovery — small structural changes
can cause dramatic changes in bioactivity, and models that fail to capture these
structure-activity relationships are unreliable for lead optimization.

The benchmark was developed by the MoleculeACE project and used by ACtriplet
(Xin et al., 2025) for improved activity cliff prediction.

References:
- van Tilborg, D. et al., J. Cheminform., 14 (2022) 30 (MoleculeACE)
- Xin, L. et al., J. Pharm. Anal., 15 (2025) 101317 (ACtriplet)
"""

import os
import subprocess
from typing import Dict, List

import pandas as pd
import rdkit.Chem as Chem
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

# == DATASET METADATA ==

DATASET_NAME: str = 'moleculeace'

CHEMBL_TARGETS: List[str] = [
    'CHEMBL1862_Ki', 'CHEMBL1871_Ki', 'CHEMBL2034_Ki', 'CHEMBL2047_EC50',
    'CHEMBL204_Ki', 'CHEMBL2147_Ki', 'CHEMBL214_Ki', 'CHEMBL218_EC50',
    'CHEMBL219_Ki', 'CHEMBL228_Ki', 'CHEMBL231_Ki', 'CHEMBL233_Ki',
    'CHEMBL234_Ki', 'CHEMBL235_EC50', 'CHEMBL236_Ki', 'CHEMBL237_EC50',
    'CHEMBL237_Ki', 'CHEMBL238_Ki', 'CHEMBL239_EC50', 'CHEMBL244_Ki',
    'CHEMBL262_Ki', 'CHEMBL264_Ki', 'CHEMBL2835_Ki', 'CHEMBL287_Ki',
    'CHEMBL2971_Ki', 'CHEMBL3979_EC50', 'CHEMBL4005_Ki', 'CHEMBL4203_Ki',
    'CHEMBL4616_EC50', 'CHEMBL4792_Ki',
]

DESCRIPTION: str = (
    'MoleculeACE activity cliff benchmark with 48,714 compounds across 30 ChEMBL '
    'protein targets. Each compound has an experimental binding affinity (pEC50 or pKi) '
    'and is annotated as an activity cliff molecule or non-cliff molecule. Activity '
    'cliffs — structurally similar molecules with dramatically different bioactivities — '
    'are a key challenge for molecular property prediction in drug discovery.'
)

METADATA: dict = {
    'license': {
        'identifier': 'CC-BY-SA-3.0',
        'url': 'https://creativecommons.org/licenses/by-sa/3.0/',
        'note': (
            'The molecular structures derive from ChEMBL, which is licensed CC BY-SA '
            '3.0. The ShareAlike condition propagates, so this dataset is distributed '
            'under the same licence. The terms of the intermediate publication were '
            'not separately established.'
        ),
    },
    'tags': [
        'Molecules',
        'SMILES',
        'Bioactivity',
        'Activity Cliffs',
        'Drug Discovery',
        'ChEMBL',
        'Regression',
        'Benchmark',
        'Multi-target',
    ],
    'verbose': 'MoleculeACE Activity Cliff Benchmark (30 targets, 48K compounds)',
    'sources': [
        'https://doi.org/10.1186/s13321-022-00613-0',
        'https://doi.org/10.1016/j.jpha.2025.101317',
        'https://github.com/molML/MoleculeACE',
        'https://www.ebi.ac.uk/chembl/',
    ],
    'notes': [
        (
            'Molecular structures in this dataset derive from ChEMBL '
            '(https://www.ebi.ac.uk/chembl/), which is licensed CC BY-SA 3.0. Cite '
            'Mendez et al., Nucleic Acids Res. 47, D930-D940 (2019), '
            'doi:10.1093/nar/gky1075.'
        ),
        'Regression task: predict binding affinity (pEC50 or pKi).',
        '30 ChEMBL protein targets combined into one dataset.',
        '~39% of compounds are activity cliff molecules.',
        'Target protein ID stored as graph metadata.',
        'Cliff annotation (0/1) stored as graph metadata.',
        'Developed by MoleculeACE (van Tilborg et al., 2022).',
    ],
    'target_descriptions': {
        '0': 'Binding affinity - pEC50 or pKi (negative log of EC50 or Ki in M); higher values indicate stronger binding (typical range 4-10)',
    },
}

# == DOWNLOAD SOURCE ==

GITHUB_BASE = 'https://raw.githubusercontent.com/molML/MoleculeACE/main/MoleculeACE/Data/benchmark_data'

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
    Download and process all 30 MoleculeACE benchmark datasets.

    Downloads CSVs from the MoleculeACE GitHub repository (one per ChEMBL target),
    combines them, validates SMILES with RDKit, and returns a dictionary with
    binding affinity (pEC50/pKi) as the regression target.

    :returns: Dictionary mapping indices to compound data dictionaries.
    """
    # Check for cached copies in assets/
    assets_dir = os.path.join(os.path.dirname(__file__), 'assets', 'moleculeace')
    cache_dir = os.path.join(e.path, 'moleculeace')
    os.makedirs(cache_dir, exist_ok=True)

    dataset: Dict[int, dict] = {}
    index = 0
    invalid = 0

    for target_name in CHEMBL_TARGETS:
        # Try assets cache first, then download
        assets_path = os.path.join(assets_dir, f'{target_name}.csv')
        csv_path = os.path.join(cache_dir, f'{target_name}.csv')

        if os.path.exists(assets_path) and os.path.getsize(assets_path) > 100:
            import shutil
            shutil.copy2(assets_path, csv_path)
        elif not os.path.exists(csv_path) or os.path.getsize(csv_path) < 100:
            url = f'{GITHUB_BASE}/{target_name}.csv'
            result = subprocess.run(
                ['curl', '-sL', '-o', csv_path, url],
                timeout=30,
                capture_output=True,
            )
            if result.returncode != 0:
                e.log(f'WARNING: Failed to download {target_name}')
                continue

        df = pd.read_csv(csv_path)

        # The pEC50/pKi column name varies
        affinity_col = 'y [pEC50/pKi]' if 'y [pEC50/pKi]' in df.columns else 'y'
        assay_type = 'pEC50' if 'EC50' in target_name else 'pKi'
        chembl_id = target_name.split('_')[0]

        for _, row in df.iterrows():
            smiles = str(row['smiles']).strip()
            if not smiles or smiles == 'nan':
                continue

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                invalid += 1
                continue

            affinity = row[affinity_col]
            if pd.isna(affinity):
                continue

            cliff = int(row.get('cliff_mol', 0))

            dataset[index] = {
                'smiles': smiles,
                'targets': [float(affinity)],
                'target_name': target_name,
                'chembl_target': chembl_id,
                'assay_type': assay_type,
                'is_cliff': cliff,
            }
            index += 1

        e.log(f'  {target_name}: {len(df)} entries')

    e.log(f'Processed {len(dataset)} compounds ({invalid} invalid SMILES skipped)')

    cliff_count = sum(1 for d in dataset.values() if d['is_cliff'] == 1)
    e.log(f'  Activity cliff molecules: {cliff_count} ({100 * cliff_count / len(dataset):.1f}%)')
    e.log(f'  Non-cliff molecules: {len(dataset) - cliff_count}')

    return dataset


@experiment.hook('add_graph_metadata', default=False, replace=True)
def add_graph_metadata(_e: Experiment, data: dict, graph: dict) -> None:
    """Add activity cliff and target metadata to each graph."""
    graph['graph_target_name'] = data['target_name']
    graph['graph_chembl_target'] = data['chembl_target']
    graph['graph_assay_type'] = data['assay_type']
    graph['graph_is_cliff'] = data['is_cliff']


experiment.run_if_main()
