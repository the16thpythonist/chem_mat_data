"""
This experiment module creates a graph dataset from the RDKit Benchmarking Platform
Subset II dataset for multi-target classification tasks with tri-state labels.

The RDKit Benchmarking Platform is a comprehensive virtual screening benchmark originally
published by Riniker & Landrum (2013). Subset II is an extended version focused on a
different virtual screening use case with 37 ChEMBL targets, described in:
S. Riniker, N. Fechner, G. Landrum, J. Chem. Inf. Model., 53, 2829 (2013)

This script processes Subset II, which contains 37 ChEMBL targets with diverse active sets.

The dataset is structured as a multi-label classification task where each molecule has a
37-dimensional tri-state target vector:
- Indices 0-36: Correspond to 37 ChEMBL biological targets
- Each position can have three values:
  * 0: Not mentioned for that target (no data)
  * 1: Active against that target
  * 2: Decoy for that target

This tri-state encoding properly captures the per-target nature of the decoy selection, where
a molecule can be active against one target while serving as a decoy for another. Decoys are
shared ZINC compounds selected to match physical properties while avoiding structural
similarity to actives.

This enables both multi-task learning and proper evaluation of similarity-based virtual
screening methods with target-specific negative examples.

References:
- S. Riniker, G. Landrum, J. Cheminf., 5, 26 (2013)
  DOI: 10.1186/1758-2946-5-26
- S. Riniker, N. Fechner, G. Landrum, J. Chem. Inf. Model., 53, 2829 (2013)
  DOI: 10.1021/ci400466r
- Repository: https://github.com/rdkit/benchmarking_platform
"""
import os
import gzip
import zipfile
import tempfile
import pickle
from typing import Dict
from rich.pretty import pprint
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace
import rdkit.Chem as Chem

from chem_mat_data.connectors import FileDownloadSource

# == DATASET METADATA ==

DATASET_NAME: str = 'riniker_2'

DESCRIPTION: str = (
    'The RDKit Benchmarking Platform Subset II is a multi-target classification dataset '
    'containing 37 biological targets from ChEMBL. This extended version was published by '
    'Riniker, Fechner & Landrum (2013) to evaluate a different virtual screening use case '
    'with diverse active sets (100 actives per target). Each molecule has a 37-dimensional '
    'target vector where each position can have three values: 0 (not mentioned for that target), '
    '1 (active against that target), or 2 (decoy for that target). This tri-state encoding '
    'properly captures the per-target nature of the decoy selection, where a molecule can be '
    'active against one target while serving as a decoy for another. Decoys are shared ZINC '
    'compounds (10,000 total) selected for all ChEMBL targets to match physical properties '
    'while avoiding structural similarity to actives. This enables both multi-task learning '
    'and proper evaluation of similarity-based virtual screening methods with target-specific '
    'negative examples.'
)

METADATA: dict = {
    'tags': [
        'Molecules',
        'SMILES',
        'Drug Discovery',
        'Virtual Screening',
        'Similarity',
        'Multi-Target',
        'Multi-Label Classification',
        'Multi-Task',
        'Bioactivity',
        'Classification',
        'ChEMBL',
        'QSAR',
        'Benchmark',
    ],
    'verbose': 'RDKit Benchmarking Platform Subset II (Multi-Target Classification)',
    'sources': [
        'https://github.com/rdkit/benchmarking_platform',
        'https://doi.org/10.1186/1758-2946-5-26',  # Original Riniker & Landrum paper
        'https://doi.org/10.1021/ci400466r',  # Extended paper with Subset II
        'http://www.jcheminf.com/content/5/1/26',
    ],
    'notes': [
        'This is a virtual screening benchmark dataset designed for evaluating molecular similarity methods.',
        'Target vectors use tri-state encoding: 0 = not mentioned for that target, 1 = active against that target, 2 = decoy for that target.',
        'Decoys are shared ZINC compounds selected to match physical properties while avoiding structural similarity to actives.',
        'All decoy molecules have value 2 for all 37 ChEMBL targets (assumed inactive against all targets).',
    ],
    'target_descriptions': {
        # ChEMBL targets (indices 0-36)
        '0': 'ChEMBL_10434 - ChEMBL 10434 (ChEMBL)',
        '1': 'ChEMBL_12209 - ChEMBL 12209 (ChEMBL)',
        '2': 'ChEMBL_25 - ChEMBL 25 (ChEMBL)',
        '3': 'ChEMBL_43 - ChEMBL 43 (ChEMBL)',
        '4': 'ChEMBL_130 - ChEMBL 130 (ChEMBL)',
        '5': 'ChEMBL_126 - ChEMBL 126 (ChEMBL)',
        '6': 'ChEMBL_12252 - ChEMBL 12252 (ChEMBL)',
        '7': 'ChEMBL_11575 - ChEMBL 11575 (ChEMBL)',
        '8': 'ChEMBL_11534 - ChEMBL 11534 (ChEMBL)',
        '9': 'ChEMBL_11631 - ChEMBL 11631 (ChEMBL)',
        '10': 'ChEMBL_165 - ChEMBL 165 (ChEMBL)',
        '11': 'ChEMBL_10193 - ChEMBL 10193 (ChEMBL)',
        '12': 'ChEMBL_15 - ChEMBL 15 (ChEMBL)',
        '13': 'ChEMBL_11489 - ChEMBL 11489 (ChEMBL)',
        '14': 'ChEMBL_121 - ChEMBL 121 (ChEMBL)',
        '15': 'ChEMBL_72 - ChEMBL 72 (ChEMBL)',
        '16': 'ChEMBL_259 - ChEMBL 259 (ChEMBL)',
        '17': 'ChEMBL_10188 - ChEMBL 10188 (ChEMBL)',
        '18': 'ChEMBL_108 - ChEMBL 108 (ChEMBL)',
        '19': 'ChEMBL_12952 - ChEMBL 12952 (ChEMBL)',
        '20': 'ChEMBL_93 - ChEMBL 93 (ChEMBL)',
        '21': 'ChEMBL_10980 - ChEMBL 10980 (ChEMBL)',
        '22': 'ChEMBL_19905 - ChEMBL 19905 (ChEMBL)',
        '23': 'ChEMBL_107 - ChEMBL 107 (ChEMBL)',
        '24': 'ChEMBL_87 - ChEMBL 87 (ChEMBL)',
        '25': 'ChEMBL_17045 - ChEMBL 17045 (ChEMBL)',
        '26': 'ChEMBL_11140 - ChEMBL 11140 (ChEMBL)',
        '27': 'ChEMBL_114 - ChEMBL 114 (ChEMBL)',
        '28': 'ChEMBL_90 - ChEMBL 90 (ChEMBL)',
        '29': 'ChEMBL_13001 - ChEMBL 13001 (ChEMBL)',
        '30': 'ChEMBL_65 - ChEMBL 65 (ChEMBL)',
        '31': 'ChEMBL_61 - ChEMBL 61 (ChEMBL)',
        '32': 'ChEMBL_10280 - ChEMBL 10280 (ChEMBL)',
        '33': 'ChEMBL_51 - ChEMBL 51 (ChEMBL)',
        '34': 'ChEMBL_100 - ChEMBL 100 (ChEMBL)',
        '35': 'ChEMBL_10260 - ChEMBL 10260 (ChEMBL)',
        '36': 'ChEMBL_11365 - ChEMBL 11365 (ChEMBL)',
    },
}

# == GITHUB REPOSITORY SOURCE ==

REPO_URL = 'https://github.com/rdkit/benchmarking_platform/archive/refs/heads/master.zip'

# Subset II target configuration (from configuration_file_II.py)
CHEMBL_IDS = [10434, 12209, 25, 43, 130, 126, 12252, 11575, 11534, 11631, 165, 10193, 15,
              11489, 121, 72, 259, 10188, 108, 12952, 93, 10980, 19905, 107, 87, 17045,
              11140, 114, 90, 13001, 65, 61, 10280, 51, 100, 10260, 11365]

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


def get_repository_path() -> str:
    """
    Get the path to the cached repository directory.
    Uses /tmp with a named path so it persists between runs during system lifetime
    but gets cleared on system restart.

    Returns:
        Path to the extracted repository root directory
    """
    tmp_dir = os.path.join(tempfile.gettempdir(), 'benchmarking_platform_cache')
    extracted_path = os.path.join(tmp_dir, 'benchmarking_platform-master')
    return extracted_path


def extract_repository(zip_path: str, target_dir: str) -> None:
    """
    Extract the downloaded ZIP file to the target directory.

    Args:
        zip_path: Path to the ZIP file
        target_dir: Directory where to extract the ZIP contents
    """
    os.makedirs(target_dir, exist_ok=True)

    # Extract ZIP file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)


@experiment.hook('load_dataset', default=False, replace=True)
def load_dataset(e: Experiment) -> Dict[int, dict]:
    """
    Download and process the RDKit Benchmarking Platform Subset II for multi-target classification.

    This function:
    1. Downloads the repository as a ZIP file from GitHub
    2. Extracts to /tmp/benchmarking_platform_cache (persists between runs)
    3. Parses all active compound pickle files for the 37 ChEMBL targets in Subset II
    4. Parses the shared ZINC decoys file
    5. Groups molecules by SMILES to get unique structures
    6. For each unique molecule, creates a 37-dimensional tri-state target vector
    7. Vector encoding: 0 (not mentioned), 1 (active), 2 (decoy)
    8. A molecule can be active for some targets and a decoy for others
    9. Returns dictionary with per-target tri-state target vectors
    """

    e.log('=' * 80)
    e.log('DOWNLOADING REPOSITORY FROM GITHUB')
    e.log('=' * 80)

    # Check if repository is already cached
    repo_path = get_repository_path()

    if os.path.exists(repo_path):
        e.log(f'Repository already cached at {repo_path}')
        e.log('  ✓ Skipping download')
    else:
        # Download the repository as ZIP
        e.log(f'Downloading repository from {REPO_URL}...')
        with FileDownloadSource(REPO_URL, verbose=True) as source:
            zip_path = source.fetch()
            e.log(f'  ✓ Downloaded repository ZIP')

            # Extract to temporary directory while still in context manager
            e.log('Extracting repository to temporary directory...')
            tmp_dir = os.path.dirname(repo_path)
            extract_repository(zip_path, tmp_dir)
            e.log(f'  ✓ Extracted to {repo_path}')

    compounds_path = os.path.join(repo_path, 'compounds')
    e.log(f'  ✓ Compound files location: {compounds_path}')

    # == BUILD TARGET INDEX MAPPING ==
    e.log('')
    e.log('=' * 80)
    e.log('BUILDING TARGET INDEX MAPPING')
    e.log('=' * 80)

    # Create ordered list of all targets with their indices
    target_list = []
    target_metadata = {}

    # Add ChEMBL targets (indices 0-36)
    for chembl_id in CHEMBL_IDS:
        target_key = f'ChEMBL_{chembl_id}'
        target_list.append(target_key)
        target_metadata[target_key] = {
            'source': 'ChEMBL',
            'id': chembl_id,
            'name': f'ChEMBL {chembl_id}',
        }

    # Create mapping from target key to vector index
    target_to_index = {target_key: idx for idx, target_key in enumerate(target_list)}
    num_targets = len(target_list)

    e.log(f'Total targets in Subset II: {num_targets}')
    e.log(f'  - All ChEMBL targets (indices 0-{num_targets-1})')
    e.log(f'Target vector dimension: {num_targets}')
    e.log(f'Target vector encoding: 0 = not mentioned, 1 = active, 2 = decoy')

    # == PARSE ACTIVE COMPOUNDS ==
    e.log('')
    e.log('=' * 80)
    e.log('PARSING ACTIVE COMPOUNDS FROM PICKLE FILES')
    e.log('=' * 80)

    molecule_target_data = {}  # SMILES -> {'active_targets': [target_keys], 'decoy_targets': [target_keys], 'external_ids': [...]}

    # Parse ChEMBL_II actives from pickle files
    e.log('Parsing ChEMBL Subset II actives...')
    chembl_ii_path = os.path.join(compounds_path, 'ChEMBL_II')

    for chembl_id in CHEMBL_IDS:
        target_key = f'ChEMBL_{chembl_id}'
        file_path = os.path.join(chembl_ii_path, f'Target_no_{chembl_id}.pkl')

        if not os.path.exists(file_path):
            e.log(f'  ! Warning: File not found: {file_path}')
            continue

        # Load pickle file (contains dict of clusters)
        with open(file_path, 'rb') as f:
            target_data = pickle.load(f)

        count = 0
        # Each target file contains a dict with cluster_id -> list of [chembl_id, smiles]
        for _cluster_id, molecules in target_data.items():
            for mol_info in molecules:
                external_id, smiles = mol_info[0], mol_info[1]

                if smiles not in molecule_target_data:
                    molecule_target_data[smiles] = {
                        'active_targets': [],
                        'decoy_targets': [],
                        'external_ids': [],
                    }
                molecule_target_data[smiles]['active_targets'].append(target_key)
                if external_id and external_id not in molecule_target_data[smiles]['external_ids']:
                    molecule_target_data[smiles]['external_ids'].append(external_id)
                count += 1

        e.log(f'  ✓ {target_key}: {count} actives')

    total_molecules_with_actives = len(molecule_target_data)
    total_active_pairs = sum(len(data['active_targets']) for data in molecule_target_data.values())
    e.log(f'\nTotal unique molecules with active annotations: {total_molecules_with_actives}')
    e.log(f'Total (molecule, active target) pairs: {total_active_pairs}')
    if total_molecules_with_actives > 0:
        e.log(f'Average active targets per molecule: {total_active_pairs / total_molecules_with_actives:.2f}')

    # == PARSE DECOY COMPOUNDS ==
    e.log('')
    e.log('=' * 80)
    e.log('PARSING DECOY COMPOUNDS (SHARED FOR ALL CHEMBL TARGETS)')
    e.log('=' * 80)

    total_decoy_pairs = 0
    new_molecules_from_decoys = 0

    # Parse ChEMBL shared ZINC decoys (same file as used in Subset I)
    chembl_path = os.path.join(compounds_path, 'ChEMBL')
    file_path = os.path.join(chembl_path, 'cmp_list_ChEMBL_zinc_decoys.dat.gz')

    if os.path.exists(file_path):
        with gzip.open(file_path, 'rt') as f:
            # Skip header
            f.readline()
            zinc_decoy_smiles = []

            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    external_id, smiles = parts[0], parts[2]
                    zinc_decoy_smiles.append((smiles, external_id))

                    if smiles not in molecule_target_data:
                        # New molecule only seen as decoy
                        molecule_target_data[smiles] = {
                            'active_targets': [],
                            'decoy_targets': [],
                            'external_ids': [],
                        }
                        new_molecules_from_decoys += 1

                    if external_id and external_id not in molecule_target_data[smiles]['external_ids']:
                        molecule_target_data[smiles]['external_ids'].append(external_id)

            # Add all ChEMBL targets as decoy targets for these molecules
            count = 0
            for smiles, external_id in zinc_decoy_smiles:
                for chembl_id in CHEMBL_IDS:
                    target_key = f'ChEMBL_{chembl_id}'
                    molecule_target_data[smiles]['decoy_targets'].append(target_key)
                    count += 1

            total_decoy_pairs += count
            e.log(f'  ✓ ChEMBL zinc_decoys: {len(zinc_decoy_smiles)} molecules × {len(CHEMBL_IDS)} targets = {count} decoy pairs')
    else:
        e.log(f'  ! Warning: Decoy file not found: {file_path}')

    e.log(f'\nTotal (molecule, decoy target) pairs: {total_decoy_pairs}')
    e.log(f'New molecules introduced by decoy files: {new_molecules_from_decoys}')
    e.log(f'Total unique molecules (actives + decoys): {len(molecule_target_data)}')

    # == BUILD MULTI-TARGET VECTORS ==
    e.log('')
    e.log('=' * 80)
    e.log('BUILDING MULTI-TARGET ACTIVITY VECTORS')
    e.log('=' * 80)

    dataset: Dict[int, dict] = {}
    index = 0

    stats = {
        'total': 0,
        'filtered_salt': 0,
        'filtered_invalid_smiles': 0,
        'filtered_single_atom': 0,
        'success': 0,
        'molecules_actives': 0,
        'molecules_decoys': 0,
        'molecules_both': 0,  # molecules that are both active and decoy (for different targets)
        'total_active_targets': 0,
        'total_decoy_targets': 0,
        'multi_target_actives': 0,  # molecules active against >1 target
    }

    for smiles, mol_data in molecule_target_data.items():
        stats['total'] += 1

        # Filter 1: Remove salts (molecules with '.' in SMILES)
        if '.' in smiles:
            stats['filtered_salt'] += 1
            continue

        # Filter 2: Validate SMILES with RDKit
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            stats['filtered_invalid_smiles'] += 1
            continue

        # Filter 3: Filter out single atoms
        if len(mol.GetAtoms()) < 2:
            stats['filtered_single_atom'] += 1
            continue

        # == BUILD num_targets-DIMENSIONAL TARGET VECTOR ==
        # Each position can be: 0 (not mentioned), 1 (active), 2 (decoy)
        target_vector = [0] * num_targets

        active_target_keys = []
        decoy_target_keys = []

        # Set 1 for each target the molecule is active against
        for target_key in mol_data['active_targets']:
            if target_key in target_to_index:
                target_idx = target_to_index[target_key]
                target_vector[target_idx] = 1
                active_target_keys.append(target_key)

        # Set 2 for each target the molecule is a decoy for
        for target_key in mol_data['decoy_targets']:
            if target_key in target_to_index:
                target_idx = target_to_index[target_key]
                target_vector[target_idx] = 2
                decoy_target_keys.append(target_key)

        num_active_targets = len(active_target_keys)
        num_decoy_targets = len(decoy_target_keys)

        # Update statistics
        if num_active_targets > 0:
            stats['molecules_actives'] += 1
            stats['total_active_targets'] += num_active_targets
            if num_active_targets > 1:
                stats['multi_target_actives'] += 1

        if num_decoy_targets > 0:
            stats['molecules_decoys'] += 1
            stats['total_decoy_targets'] += num_decoy_targets

        # Track molecules that are both active and decoy (for different targets)
        if num_active_targets > 0 and num_decoy_targets > 0:
            stats['molecules_both'] += 1

        # Create data dict
        data = {
            'smiles': smiles,
            'targets': target_vector,
            'num_active_targets': num_active_targets,
            'num_decoy_targets': num_decoy_targets,
            'active_target_keys': active_target_keys,
            'decoy_target_keys': decoy_target_keys,
            'external_ids': mol_data['external_ids'][:3],  # Keep first 3 external IDs
        }

        dataset[index] = data
        index += 1
        stats['success'] += 1

    # == PRINT STATISTICS ==
    e.log('')
    e.log('=' * 80)
    e.log('FILTERING AND PROCESSING STATISTICS')
    e.log('=' * 80)
    e.log(f'Total unique molecules: {stats["total"]}')
    e.log(f'  - Filtered (salts/dots): {stats["filtered_salt"]} ({100*stats["filtered_salt"]/stats["total"]:.1f}%)')
    e.log(f'  - Filtered (invalid SMILES): {stats["filtered_invalid_smiles"]} ({100*stats["filtered_invalid_smiles"]/stats["total"]:.1f}%)')
    e.log(f'  - Filtered (single atoms): {stats["filtered_single_atom"]} ({100*stats["filtered_single_atom"]/stats["total"]:.1f}%)')
    e.log(f'  ✓ Successfully processed: {stats["success"]} ({100*stats["success"]/stats["total"]:.1f}%)')
    e.log('')
    e.log('MULTI-TARGET STATISTICS:')
    e.log(f'  - Molecules with active targets: {stats["molecules_actives"]} ({100*stats["molecules_actives"]/stats["success"]:.1f}%)')
    e.log(f'  - Molecules with decoy targets: {stats["molecules_decoys"]} ({100*stats["molecules_decoys"]/stats["success"]:.1f}%)')
    e.log(f'  - Molecules with both active & decoy targets: {stats["molecules_both"]} ({100*stats["molecules_both"]/stats["success"]:.1f}%)')
    e.log(f'  - Total active target assignments: {stats["total_active_targets"]}')
    e.log(f'  - Total decoy target assignments: {stats["total_decoy_targets"]}')
    if stats["molecules_actives"] > 0:
        avg_active = stats["total_active_targets"] / stats["molecules_actives"]
        e.log(f'  - Average active targets per active molecule: {avg_active:.2f}')
        e.log(f'  - Multi-target actives (active against >1 target): {stats["multi_target_actives"]} ({100*stats["multi_target_actives"]/stats["molecules_actives"]:.1f}%)')
    if stats["molecules_decoys"] > 0:
        avg_decoy = stats["total_decoy_targets"] / stats["molecules_decoys"]
        e.log(f'  - Average decoy targets per decoy molecule: {avg_decoy:.2f}')
    e.log(f'  - Target vector dimension: {num_targets}')
    e.log(f'  - Target vector encoding: 0=not mentioned, 1=active, 2=decoy')
    e.log('')

    # Print example entries
    e.log('Sample processed entries:')
    for i in range(min(3, len(dataset))):
        e.log(f'\nEntry {i}:')
        entry = dataset[i].copy()
        # Show first 20 positions of target vector for readability
        entry['targets_preview'] = entry['targets'][:20] + ['...'] + [entry['targets'][-1]]
        pprint(entry, max_depth=2)

    return dataset


@experiment.hook('add_graph_metadata', default=False, replace=True)
def add_graph_metadata(_e: Experiment, data: dict, graph: dict) -> None:
    """
    Add additional metadata fields to each graph representation.

    This includes:
    - Number of active targets (how many targets in the vector are 1)
    - Number of decoy targets (how many targets in the vector are 2)
    - List of active target keys
    - List of decoy target keys
    - External IDs (ChEMBL IDs, ZINC IDs, etc.)
    """
    # Add multi-target information
    graph['graph_num_active_targets'] = int(data.get('num_active_targets', 0))
    graph['graph_num_decoy_targets'] = int(data.get('num_decoy_targets', 0))

    # Store active target keys as a comma-separated string (for reference)
    active_keys = data.get('active_target_keys', [])
    if active_keys:
        graph['graph_active_target_keys'] = ','.join(active_keys)
    else:
        graph['graph_active_target_keys'] = ''

    # Store decoy target keys as a comma-separated string (for reference)
    decoy_keys = data.get('decoy_target_keys', [])
    if decoy_keys:
        graph['graph_decoy_target_keys'] = ','.join(decoy_keys)
    else:
        graph['graph_decoy_target_keys'] = ''

    # Store external IDs (first 3) as comma-separated string
    external_ids = data.get('external_ids', [])
    if external_ids:
        graph['graph_external_ids'] = ','.join(str(id) for id in external_ids[:3])
    else:
        graph['graph_external_ids'] = ''


experiment.run_if_main()
