"""
This experiment module creates a graph dataset from the BORON1000 C-H borylation
selectivity dataset for binary classification tasks.

The dataset contains ~2,300 reaction entries for iridium-catalysed C-H borylation
reactions, each labelled with the regioselectivity outcome (which positional isomer
of the borylated product is favoured). The substrate SMILES (the molecule undergoing
C-H activation) is used for graph construction.

C-H borylation is a key reaction in drug discovery and materials chemistry, and
predicting which C-H bond reacts is a fundamental selectivity problem. This dataset
was curated from literature sources and an internal Roche dataset.

References:
- Kotlyarevsky, R. et al., J. Chem. Inf. Model., 64 (2024)
  DOI: 10.1021/acs.jcim.4c00137
"""

import os
import subprocess
from typing import Dict

import rdkit.Chem as Chem
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

# == DATASET METADATA ==

DATASET_NAME: str = 'boron1000'

DESCRIPTION: str = (
    'BORON1000 C-H borylation regioselectivity dataset with ~2,300 reaction entries '
    'for iridium-catalysed C-H borylation. Each entry is a reaction SMILES '
    '(substrate > reagents > product) with a binary label indicating the regioselectivity '
    'outcome (major positional isomer). The substrate molecule is used for graph '
    'construction. Curated from literature and internal Roche data by Kotlyarevsky et al. (2024).'
)

METADATA: dict = {
    'license': {
        'identifier': 'MIT',
        'url': 'https://github.com/ETHmodlab/lsfml/blob/master/LICENSE',
        'note': (
            'WARNING - THIS DATASET IS MISNAMED AND ITS DESCRIPTION AND SOURCES ARE '
            'WRONG. The script does not download BORON1000. Real BORON1000 is '
            'Reaxys-derived, proprietary (Elsevier) and was deliberately never '
            'published as structures, only as Reaxys IDs. What this script actually '
            'pulls (github.com/ruslankotl/rxn-data-proc, data/roche/classification) '
            'is a byte-verified copy of the Nippa et al. literature borylation set '
            'from github.com/ETHmodlab/lsfml (Nippa et al., Nat. Chem. 2024, '
            'doi:10.1038/s41557-023-01360-5; Zenodo 10.5281/zenodo.8118845). The MIT '
            'identifier is the root LICENSE of ETHmodlab/lsfml, the origin of the '
            'file; the immediate GitHub host carries no licence file of its own, so '
            'the permission is read as travelling with the copied data. The naming '
            'and provenance error is tracked separately and must be fixed before this '
            'dataset is published.'
        ),
    },
    'tags': [
        'Molecules',
        'SMILES',
        'Reactions',
        'Borylation',
        'C-H Activation',
        'Regioselectivity',
        'Binary Classification',
        'Drug Discovery',
    ],
    'verbose': 'BORON1000 C-H Borylation Selectivity',
    'sources': [
        'https://doi.org/10.1021/acs.jcim.4c00137',
        'https://github.com/ruslankotl/rxn-data-proc',
    ],
    'notes': [
        'Binary classification: regioselectivity of C-H borylation (positional isomer 0 vs 1).',
        'Target is one-hot encoded: [isomer_0, isomer_1].',
        'Substrate SMILES used for graph construction (molecule undergoing C-H activation).',
        'Iridium-catalysed borylation with B2pin2 reagent.',
        'Combined train + test splits from the original publication.',
    ],
    'target_descriptions': {
        '0': 'Isomer 0 - The first (typically less substituted) positional isomer of the borylation product is major',
        '1': 'Isomer 1 - The second (typically more substituted) positional isomer of the borylation product is major',
    },
}

# == DOWNLOAD SOURCES ==

TRAIN_SOURCE_URL = 'https://raw.githubusercontent.com/ruslankotl/rxn-data-proc/main/data/roche/classification/train.source'
TRAIN_TARGET_URL = 'https://raw.githubusercontent.com/ruslankotl/rxn-data-proc/main/data/roche/classification/train.target'
TEST_SOURCE_URL = 'https://raw.githubusercontent.com/ruslankotl/rxn-data-proc/main/data/roche/classification/test.source'
TEST_TARGET_URL = 'https://raw.githubusercontent.com/ruslankotl/rxn-data-proc/main/data/roche/classification/test.target'

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


def _download_file(url: str, path: str) -> None:
    """Download a file with curl if it doesn't already exist."""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    result = subprocess.run(
        ['curl', '-sL', '-o', path, url],
        timeout=60,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f'curl failed for {url}: {result.stderr.decode()}')


def _extract_substrate(rxn_smiles: str) -> str:
    """
    Extract the substrate SMILES from a reaction SMILES string.

    Reaction SMILES format: ``substrate>reagents>product``.
    Returns the substrate part (before the first ``>``).
    """
    parts = rxn_smiles.split('>')
    return parts[0].strip()


@experiment.hook('load_dataset', default=False, replace=True)
def load_dataset(e: Experiment) -> Dict[int, dict]:
    """
    Download and process the BORON1000 borylation selectivity dataset.

    Downloads train and test source/target files from GitHub, combines them,
    extracts substrate SMILES from reaction SMILES, validates with RDKit,
    and returns a dictionary with one-hot encoded binary selectivity labels.

    :returns: Dictionary mapping indices to reaction data dictionaries.
    """
    # Download all files
    train_src = os.path.join(e.path, 'train.source')
    train_tgt = os.path.join(e.path, 'train.target')
    test_src = os.path.join(e.path, 'test.source')
    test_tgt = os.path.join(e.path, 'test.target')

    e.log('Downloading BORON1000 dataset from GitHub...')
    _download_file(TRAIN_SOURCE_URL, train_src)
    _download_file(TRAIN_TARGET_URL, train_tgt)
    _download_file(TEST_SOURCE_URL, test_src)
    _download_file(TEST_TARGET_URL, test_tgt)

    # Load and combine train + test
    with open(train_src) as f:
        train_sources = f.read().strip().split('\n')
    with open(train_tgt) as f:
        train_targets = f.read().strip().split('\n')
    with open(test_src) as f:
        test_sources = f.read().strip().split('\n')
    with open(test_tgt) as f:
        test_targets = f.read().strip().split('\n')

    all_sources = train_sources + test_sources
    all_targets = train_targets + test_targets
    e.log(f'Loaded {len(train_sources)} train + {len(test_sources)} test = {len(all_sources)} total entries')

    dataset: Dict[int, dict] = {}
    index = 0
    invalid = 0

    for rxn_smiles, label_str in zip(all_sources, all_targets):
        substrate = _extract_substrate(rxn_smiles)
        if not substrate:
            continue

        mol = Chem.MolFromSmiles(substrate)
        if mol is None:
            invalid += 1
            continue

        label = int(label_str.strip())
        targets = [int(i == label) for i in range(2)]

        dataset[index] = {
            'smiles': substrate,
            'targets': targets,
            'rxn_smiles': rxn_smiles,
            'selectivity_label': label,
        }
        index += 1

    e.log(f'Processed {len(dataset)} reactions ({invalid} invalid substrate SMILES skipped)')

    label_0 = sum(1 for d in dataset.values() if d['selectivity_label'] == 0)
    label_1 = sum(1 for d in dataset.values() if d['selectivity_label'] == 1)
    e.log(f'  Isomer 0: {label_0}, Isomer 1: {label_1}')

    return dataset


@experiment.hook('add_graph_metadata', default=False, replace=True)
def add_graph_metadata(_e: Experiment, data: dict, graph: dict) -> None:
    """Add reaction SMILES metadata to each graph."""
    graph['graph_rxn_smiles'] = data['rxn_smiles']


experiment.run_if_main()
