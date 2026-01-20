"""
This experiment module creates a graph dataset from the MUV (Maximum Unbiased Validation)
dataset for multi-target classification tasks with tri-state labels.

The MUV datasets are specifically designed for validation of virtual screening techniques,
addressing two major problems in virtual screening benchmarks: analogue bias (when actives
are too similar to each other) and artificial enrichment (when decoys are too dissimilar
from actives). Published by Rohrer & Baumann (2009), MUV uses refined nearest neighbor
analysis from spatial statistics to ensure unbiased dataset construction.

This script processes the 17 MUV targets from the RDKit Benchmarking Platform, structured
as a multi-label classification task where each molecule has a 17-dimensional tri-state
target vector:
- Indices 0-16: Correspond to 17 biological targets (MUV AIDs)
- Each position can have three values:
  * 0: Not mentioned for that target (no data)
  * 1: Active against that target
  * 2: Decoy for that target

This tri-state encoding properly captures the per-target nature of the decoy selection, where
a molecule can be active against one target while serving as a decoy for another. Decoys are
specifically selected for each target with a 500:1 ratio (15,000 decoys per 30 actives),
embedded around actives using refined nearest neighbor analysis to create truly challenging
benchmarks without bias.

References:
- S. G. Rohrer, K. Baumann, J. Chem. Inf. Model., 49 (2009) 169-184
  DOI: 10.1021/ci8002649
- Repository: https://github.com/rdkit/benchmarking_platform
"""
import os
import csv
import gzip
import zipfile
import tempfile
from typing import Dict
from rich.pretty import pprint
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace
import rdkit.Chem as Chem

from chem_mat_data.connectors import FileDownloadSource

# == DATASET METADATA ==

DATASET_NAME: str = 'muv'

DESCRIPTION: str = (
    'The MUV (Maximum Unbiased Validation) dataset is a carefully designed multi-target '
    'virtual screening benchmark containing 17 biological targets from PubChem BioAssay '
    'data. Published by Rohrer & Baumann (2009), this dataset addresses critical issues in '
    'virtual screening benchmarks by avoiding analogue bias and artificial enrichment through '
    'refined nearest neighbor analysis from spatial statistics. Each molecule has a '
    '17-dimensional target vector where each position can have three values: '
    '0 (not mentioned for that target), 1 (active against that target), or 2 (decoy for '
    'that target). This tri-state encoding properly captures the per-target nature of the '
    'decoy selection, where a molecule can be active against one target while serving as a '
    'decoy for another. With 30 carefully selected diverse actives and 15,000 embedded decoys '
    'per target (500:1 ratio), MUV provides a truly challenging and unbiased benchmark for '
    'evaluating similarity-based virtual screening methods and molecular fingerprints.'
)

# MUV target configuration (17 targets from PubChem BioAssay)
MUV_IDS = [466, 548, 600, 644, 652, 689, 692, 712, 713, 733, 737, 810, 832, 846, 852, 858, 859]

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
        'MUV',
        'PubChem',
        'QSAR',
        'Benchmark',
        'Unbiased',
    ],
    'verbose': 'MUV (Maximum Unbiased Validation) Multi-Target Classification',
    'sources': [
        'https://github.com/rdkit/benchmarking_platform',
        'https://doi.org/10.1021/ci8002649',  # Rohrer & Baumann 2009
        'https://pubs.acs.org/doi/10.1021/ci8002649',
        'https://www.tu-braunschweig.de/en/pharmchem/forschung/baumann/translate-to-english-muv',
    ],
    'notes': [
        'This is an unbiased virtual screening benchmark dataset designed to avoid analogue bias and artificial enrichment.',
        'Target vectors use tri-state encoding: 0 = not mentioned for that target, 1 = active against that target, 2 = decoy for that target.',
        'Actives are maximally spread (diverse) to avoid analogue bias, with 30 actives per target.',
        'Decoys are embedded around actives using refined nearest neighbor analysis (500:1 ratio, 15,000 per target).',
        'A molecule can be active against some targets while serving as a decoy for others, capturing the per-target nature of the data.',
        'All targets are based on PubChem BioAssay confirmatory screens.',
    ],
    'target_descriptions': {
        '0': 'MUV_466 - MUV AID 466 (PubChem BioAssay)',
        '1': 'MUV_548 - MUV AID 548 (PubChem BioAssay)',
        '2': 'MUV_600 - MUV AID 600 (PubChem BioAssay)',
        '3': 'MUV_644 - MUV AID 644 (PubChem BioAssay)',
        '4': 'MUV_652 - MUV AID 652 (PubChem BioAssay)',
        '5': 'MUV_689 - MUV AID 689 (PubChem BioAssay)',
        '6': 'MUV_692 - MUV AID 692 (PubChem BioAssay)',
        '7': 'MUV_712 - MUV AID 712 (PubChem BioAssay)',
        '8': 'MUV_713 - MUV AID 713 (PubChem BioAssay)',
        '9': 'MUV_733 - MUV AID 733 (PubChem BioAssay)',
        '10': 'MUV_737 - MUV AID 737 (PubChem BioAssay)',
        '11': 'MUV_810 - MUV AID 810 (PubChem BioAssay)',
        '12': 'MUV_832 - MUV AID 832 (PubChem BioAssay)',
        '13': 'MUV_846 - MUV AID 846 (PubChem BioAssay)',
        '14': 'MUV_852 - MUV AID 852 (PubChem BioAssay)',
        '15': 'MUV_858 - MUV AID 858 (PubChem BioAssay)',
        '16': 'MUV_859 - MUV AID 859 (PubChem BioAssay)',
    },
}

# == GITHUB REPOSITORY SOURCE ==

REPO_URL = 'https://github.com/rdkit/benchmarking_platform/archive/refs/heads/master.zip'

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
    Download and process the MUV dataset for multi-target classification.

    This function:
    1. Downloads the RDKit Benchmarking Platform repository as a ZIP file from GitHub
    2. Extracts to /tmp/benchmarking_platform_cache (persists between runs)
    3. Parses all active compound files for the 17 MUV targets
    4. Parses all decoy compound files (per-target specific, 500:1 ratio)
    5. Groups molecules by SMILES to get unique structures
    6. For each unique molecule, creates a 17-dimensional tri-state target vector
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

    # Add MUV targets (indices 0-16)
    for muv_id in MUV_IDS:
        target_key = f'MUV_{muv_id}'
        target_list.append(target_key)
        target_metadata[target_key] = {
            'source': 'MUV',
            'id': muv_id,
            'name': f'MUV AID {muv_id}',
        }

    # Create mapping from target key to vector index
    target_to_index = {target_key: idx for idx, target_key in enumerate(target_list)}
    num_targets = len(target_list)

    e.log(f'Total MUV targets: {num_targets}')
    e.log(f'Target vector dimension: {num_targets}')
    e.log(f'Target vector encoding: 0 = not mentioned, 1 = active, 2 = decoy')

    # == PARSE ACTIVE COMPOUNDS ==
    e.log('')
    e.log('=' * 80)
    e.log('PARSING ACTIVE COMPOUNDS')
    e.log('=' * 80)

    molecule_target_data = {}  # SMILES -> {'active_targets': [target_keys], 'decoy_targets': [target_keys], 'external_ids': [...]}

    # Parse MUV actives
    e.log('Parsing MUV actives...')
    muv_path = os.path.join(compounds_path, 'MUV')
    for muv_id in MUV_IDS:
        target_key = f'MUV_{muv_id}'
        file_path = os.path.join(muv_path, f'cmp_list_MUV_{muv_id}_actives.dat.gz')

        if not os.path.exists(file_path):
            e.log(f'  ! Warning: File not found: {file_path}')
            continue

        with gzip.open(file_path, 'rt') as f:
            reader = csv.DictReader(f, delimiter='\t')
            count = 0
            for row in reader:
                smiles = row['SMILES']
                if smiles not in molecule_target_data:
                    molecule_target_data[smiles] = {
                        'active_targets': [],
                        'decoy_targets': [],
                        'external_ids': [],
                    }
                molecule_target_data[smiles]['active_targets'].append(target_key)
                # MUV uses '# PUBCHEM_COMPOUND_CID' as column name (with # prefix)
                external_id = row.get('# PUBCHEM_COMPOUND_CID') or row.get('PUBCHEM_COMPOUND_CID')
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
    e.log('PARSING DECOY COMPOUNDS (PER-TARGET, 500:1 RATIO)')
    e.log('=' * 80)

    total_decoy_pairs = 0
    new_molecules_from_decoys = 0

    # Parse MUV decoys (per-target specific)
    e.log('Parsing MUV decoys (per-target)...')
    for muv_id in MUV_IDS:
        target_key = f'MUV_{muv_id}'
        file_path = os.path.join(muv_path, f'cmp_list_MUV_{muv_id}_decoys.dat.gz')

        if not os.path.exists(file_path):
            e.log(f'  ! Warning: File not found: {file_path}')
            continue

        with gzip.open(file_path, 'rt') as f:
            reader = csv.DictReader(f, delimiter='\t')
            count = 0
            for row in reader:
                smiles = row['SMILES']
                if smiles not in molecule_target_data:
                    # New molecule only seen as decoy
                    molecule_target_data[smiles] = {
                        'active_targets': [],
                        'decoy_targets': [],
                        'external_ids': [],
                    }
                    new_molecules_from_decoys += 1

                # Add this target as a decoy target
                molecule_target_data[smiles]['decoy_targets'].append(target_key)

                # MUV uses '# PUBCHEM_COMPOUND_CID' as column name (with # prefix)
                external_id = row.get('# PUBCHEM_COMPOUND_CID') or row.get('PUBCHEM_COMPOUND_CID')
                if external_id and external_id not in molecule_target_data[smiles]['external_ids']:
                    molecule_target_data[smiles]['external_ids'].append(external_id)
                count += 1

            total_decoy_pairs += count
            e.log(f'  ✓ {target_key}: {count} decoys')

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
    if stats["total"] > 0:
        e.log(f'  - Filtered (salts/dots): {stats["filtered_salt"]} ({100*stats["filtered_salt"]/stats["total"]:.1f}%)')
        e.log(f'  - Filtered (invalid SMILES): {stats["filtered_invalid_smiles"]} ({100*stats["filtered_invalid_smiles"]/stats["total"]:.1f}%)')
        e.log(f'  - Filtered (single atoms): {stats["filtered_single_atom"]} ({100*stats["filtered_single_atom"]/stats["total"]:.1f}%)')
        e.log(f'  ✓ Successfully processed: {stats["success"]} ({100*stats["success"]/stats["total"]:.1f}%)')
    else:
        e.log('  ! No molecules found to process')
    e.log('')
    e.log('MULTI-TARGET STATISTICS:')
    if stats["success"] > 0:
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

    # == PER-TARGET STATISTICS ==
    e.log('=' * 80)
    e.log('PER-TARGET STATISTICS')
    e.log('=' * 80)
    e.log(f'{"Target":<12} {"Actives":>10} {"Decoys":>10} {"Not Mentioned":>15} {"Ratio (D:A)":>12}')
    e.log('-' * 61)

    for target_key in target_list:
        target_idx = target_to_index[target_key]
        actives_count = sum(1 for d in dataset.values() if d['targets'][target_idx] == 1)
        decoys_count = sum(1 for d in dataset.values() if d['targets'][target_idx] == 2)
        not_mentioned_count = sum(1 for d in dataset.values() if d['targets'][target_idx] == 0)

        if actives_count > 0:
            ratio = f'{decoys_count / actives_count:.0f}:1'
        else:
            ratio = 'N/A'

        e.log(f'{target_key:<12} {actives_count:>10} {decoys_count:>10} {not_mentioned_count:>15} {ratio:>12}')

    e.log('-' * 61)
    e.log('')

    # Print example entries
    e.log('Sample processed entries:')
    for i in range(min(3, len(dataset))):
        e.log(f'\nEntry {i}:')
        entry = dataset[i].copy()
        # Show first 17 positions of target vector for readability
        entry['targets_preview'] = entry['targets'][:17]
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
    - External IDs (PubChem CID, etc.)
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
