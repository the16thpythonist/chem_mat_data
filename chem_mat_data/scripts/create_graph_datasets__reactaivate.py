"""
This experiment module creates a graph dataset from the ReactAIvate dataset for
multi-class classification of reaction elementary steps.

The dataset contains 100,000 elementary reaction steps from transition-metal
catalysed cross-coupling reactions, each classified into one of 8 mechanistic
step types (oxidative addition, transmetalation, reductive elimination, etc.).
SMILES represent the full reaction complex including catalyst, substrates, and
intermediates as a multi-component string.

Since the SMILES contain multiple molecular fragments separated by ``.``, the
largest fragment is used for graph construction (typically the metal complex or
the largest organic intermediate).

References:
- Hoque, A. et al., arXiv:2407.10090 (2024)
  DOI: 10.48550/arXiv.2407.10090
"""

import os
import subprocess
from typing import Dict, List

import pandas as pd
import rdkit.Chem as Chem
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

# == DATASET METADATA ==

DATASET_NAME: str = 'reactaivate'

CLASS_LABELS: List[str] = [
    'Oxidative addition',
    'Ligand exchange',
    'Isomerization',
    'Transmetalation',
    'Beta-hydride elimination',
    'Reductive elimination',
    'Migratory insertion',
    'Other',
]

DESCRIPTION: str = (
    'ReactAIvate dataset of 100,000 elementary reaction steps from transition-metal '
    'catalysed cross-coupling reactions (Suzuki, Heck, Buchwald-Hartwig, etc.). '
    'Each step is classified into one of 8 mechanistic types: oxidative addition, '
    'ligand exchange, isomerization, transmetalation, beta-hydride elimination, '
    'reductive elimination, migratory insertion, or other. The largest molecular '
    'fragment from each multi-component reaction SMILES is used for graph construction.'
)

METADATA: dict = {
    'tags': [
        'Molecules',
        'SMILES',
        'Reactions',
        'Catalysis',
        'Transition Metal',
        'Cross-Coupling',
        'Classification',
        'Mechanistic Steps',
    ],
    'verbose': 'ReactAIvate Elementary Reaction Steps',
    'sources': [
        'https://doi.org/10.48550/arXiv.2407.10090',
        'https://github.com/alhqlearn/ReactAIvate',
    ],
    'notes': [
        'Multi-class classification: 8 elementary reaction step types.',
        'Target is one-hot encoded over 8 classes.',
        'SMILES are multi-component (reactant mixtures); largest fragment used for graph.',
        'Contains palladium-catalysed cross-coupling reaction mechanisms.',
        'Class 6 (other) is rare (~257 examples vs ~10-22K for other classes).',
    ],
    'target_descriptions': {
        '0': 'Oxidative addition - Metal inserts into a C-X bond, increasing oxidation state by 2',
        '1': 'Ligand exchange - One ligand on the metal centre is replaced by another',
        '2': 'Isomerization - Spatial rearrangement of ligands without bond breaking/forming',
        '3': 'Transmetalation - Organic group transfers from one metal to another',
        '4': 'Beta-hydride elimination - Metal abstracts beta-H forming an alkene + metal-hydride',
        '5': 'Reductive elimination - Two ligands on metal couple together, reducing oxidation state by 2',
        '6': 'Other - Elementary steps not fitting the above categories',
        '7': 'Migratory insertion - Ligand migrates to an adjacent coordinated unsaturated substrate',
    },
    'license': {
        'identifier': 'MIT',
        'url': 'https://github.com/alhqlearn/ReactAIvate/blob/main/LICENSE',
        'note': (
            'MIT terms come from the LICENSE file at the root of the authors\' GitHub '
            'repository (github.com/alhqlearn/ReactAIvate), the sole data record named '
            'in the paper\'s Data Availability statement, which covers the whole repo '
            'including elementary_step_100000.csv; redistribution requires including the '
            'copyright and permission notice. Some input reagent structures used to build '
            'the reaction templates were drawn from PubChem, over which NCBI places no '
            'redistribution restriction, and from primary literature; no additional '
            'licence obligation travels with these facts.'
        ),
    },
}

# == DOWNLOAD SOURCE ==

DATASET_URL = 'https://github.com/alhqlearn/ReactAIvate/raw/main/elementary_step_100000.csv'

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


def _largest_fragment(smiles: str) -> str:
    """
    Return the SMILES of the largest fragment in a multi-component SMILES string.

    Multi-component SMILES like ``A.B.C`` are split on ``.`` and the fragment
    with the most heavy atoms (as parsed by RDKit) is returned.
    """
    parts = smiles.split('.')
    if len(parts) == 1:
        return smiles

    best_smiles = parts[0]
    best_size = 0
    for part in parts:
        mol = Chem.MolFromSmiles(part)
        if mol is not None:
            size = mol.GetNumHeavyAtoms()
            if size > best_size:
                best_size = size
                best_smiles = part

    return best_smiles


@experiment.hook('load_dataset', default=False, replace=True)
def load_dataset(e: Experiment) -> Dict[int, dict]:
    """
    Download and process the ReactAIvate dataset.

    Downloads the 100K elementary steps CSV from GitHub, extracts the largest
    molecular fragment from each multi-component SMILES, validates with RDKit,
    and returns a dictionary with one-hot encoded mechanistic class labels.

    :returns: Dictionary mapping indices to reaction step data dictionaries.
    """
    csv_path = os.path.join(e.path, 'elementary_step_100000.csv')
    if not os.path.exists(csv_path):
        e.log('Downloading ReactAIvate dataset from GitHub...')
        result = subprocess.run(
            ['curl', '-sL', '-o', csv_path, DATASET_URL],
            timeout=120,
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f'curl failed: {result.stderr.decode()}')
        e.log(f'Downloaded {os.path.getsize(csv_path) / 1e6:.1f} MB')

    df = pd.read_csv(csv_path)
    e.log(f'Loaded CSV with {len(df)} rows')

    num_classes = len(CLASS_LABELS)
    dataset: Dict[int, dict] = {}
    index = 0
    invalid = 0

    for _, row in df.iterrows():
        raw_smiles = str(row['smiles']).strip()
        if not raw_smiles or raw_smiles == 'nan':
            continue

        frag_smiles = _largest_fragment(raw_smiles)
        mol = Chem.MolFromSmiles(frag_smiles)
        if mol is None:
            invalid += 1
            continue

        class_idx = int(row['class_label'])
        targets = [int(i == class_idx) for i in range(num_classes)]

        dataset[index] = {
            'smiles': frag_smiles,
            'targets': targets,
            'raw_smiles': raw_smiles,
            'class_label': class_idx,
            'class_name': CLASS_LABELS[class_idx] if class_idx < len(CLASS_LABELS) else 'Unknown',
        }
        index += 1

    e.log(f'Processed {len(dataset)} reaction steps ({invalid} invalid SMILES skipped)')

    from collections import Counter
    counts = Counter(d['class_name'] for d in dataset.values())
    for label in CLASS_LABELS:
        e.log(f'  {label}: {counts.get(label, 0)}')

    return dataset


@experiment.hook('add_graph_metadata', default=False, replace=True)
def add_graph_metadata(_e: Experiment, data: dict, graph: dict) -> None:
    """Add reaction metadata to each graph."""
    graph['graph_raw_smiles'] = data['raw_smiles']
    graph['graph_class_name'] = data['class_name']


experiment.run_if_main()
