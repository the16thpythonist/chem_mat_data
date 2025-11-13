"""
This experiment module creates a graph dataset from the RDKit Benchmarking Platform
Subset I dataset for multi-target classification tasks with tri-state labels.

The RDKit Benchmarking Platform is a comprehensive virtual screening benchmark originally
published by Riniker & Landrum (2013) containing 118 biological targets from three public
data sources: MUV (Maximum Unbiased Validation), DUD (Directory of Useful Decoys), and ChEMBL.

This script processes Subset I, which contains 88 targets described in:
S. Riniker, G. Landrum, J. Cheminf., 5, 26 (2013)

The dataset is structured as a multi-label classification task where each molecule has an
88-dimensional tri-state target vector:
- Indices 0-87: Correspond to 88 biological targets (17 MUV + 21 DUD + 50 ChEMBL)
- Each position can have three values:
  * 0: Not mentioned for that target (no data)
  * 1: Active against that target
  * 2: Decoy for that target

This tri-state encoding properly captures the per-target nature of the decoy selection, where
a molecule can be active against one target while serving as a decoy for another. Decoys are
specifically selected for each target using methods like DUD (Directory of Useful Decoys)
which matches physical properties while avoiding structural similarity to actives.

This enables both multi-task learning and proper evaluation of similarity-based virtual
screening methods with target-specific negative examples.

References:
- S. Riniker, G. Landrum, J. Cheminf., 5, 26 (2013)
  DOI: 10.1186/1758-2946-5-26
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

DATASET_NAME: str = 'riniker_1'

DESCRIPTION: str = (
    'The RDKit Benchmarking Platform Subset I is a comprehensive multi-target classification '
    'dataset containing 88 biological targets from three public data sources: MUV (Maximum '
    'Unbiased Validation, 17 targets), DUD (Directory of Useful Decoys, 21 targets), and '
    'ChEMBL (50 targets). Originally published by Riniker & Landrum (2013), this dataset is '
    'designed for evaluating molecular similarity methods and virtual screening approaches. '
    'Each molecule has an 88-dimensional target vector where each position can have three values: '
    '0 (not mentioned for that target), 1 (active against that target), or 2 (decoy for that target). '
    'This tri-state encoding properly captures the per-target nature of the decoy selection, where '
    'a molecule can be active against one target while serving as a decoy for another. Decoys are '
    'specifically selected for each target using methods like DUD (Directory of Useful Decoys) '
    'which matches physical properties while avoiding structural similarity to actives. This enables '
    'both multi-task learning and proper evaluation of similarity-based virtual screening methods '
    'with target-specific negative examples.'
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
        'MUV',
        'DUD',
        'ChEMBL',
        'QSAR',
        'Benchmark',
    ],
    'verbose': 'RDKit Benchmarking Platform Subset I (Multi-Target Classification)',
    'sources': [
        'https://github.com/rdkit/benchmarking_platform',
        'https://doi.org/10.1186/1758-2946-5-26',  # Original Riniker & Landrum paper
        'http://www.jcheminf.com/content/5/1/26',
    ],
    'notes': [
        'This is a virtual screening benchmark dataset designed for evaluating molecular similarity methods.',
        'Target vectors use tri-state encoding: 0 = not mentioned for that target, 1 = active against that target, 2 = decoy for that target.',
        'Decoys are specifically selected for each target to match physical properties while avoiding structural similarity to actives.',
        'A molecule can be active against some targets while serving as a decoy for others, capturing the per-target nature of the data.',
    ],
    'target_descriptions': {
        # MUV targets (indices 0-16)
        '0': 'MUV_466 - MUV AID 466 (MUV)',
        '1': 'MUV_548 - MUV AID 548 (MUV)',
        '2': 'MUV_600 - MUV AID 600 (MUV)',
        '3': 'MUV_644 - MUV AID 644 (MUV)',
        '4': 'MUV_652 - MUV AID 652 (MUV)',
        '5': 'MUV_689 - MUV AID 689 (MUV)',
        '6': 'MUV_692 - MUV AID 692 (MUV)',
        '7': 'MUV_712 - MUV AID 712 (MUV)',
        '8': 'MUV_713 - MUV AID 713 (MUV)',
        '9': 'MUV_733 - MUV AID 733 (MUV)',
        '10': 'MUV_737 - MUV AID 737 (MUV)',
        '11': 'MUV_810 - MUV AID 810 (MUV)',
        '12': 'MUV_832 - MUV AID 832 (MUV)',
        '13': 'MUV_846 - MUV AID 846 (MUV)',
        '14': 'MUV_852 - MUV AID 852 (MUV)',
        '15': 'MUV_858 - MUV AID 858 (MUV)',
        '16': 'MUV_859 - MUV AID 859 (MUV)',
        # DUD targets (indices 17-37)
        '17': 'DUD_ace - DUD ace (DUD)',
        '18': 'DUD_ache - DUD ache (DUD)',
        '19': 'DUD_ar - DUD ar (DUD)',
        '20': 'DUD_cdk2 - DUD cdk2 (DUD)',
        '21': 'DUD_cox2 - DUD cox2 (DUD)',
        '22': 'DUD_dhfr - DUD dhfr (DUD)',
        '23': 'DUD_egfr - DUD egfr (DUD)',
        '24': 'DUD_er_agonist - DUD er_agonist (DUD)',
        '25': 'DUD_fgfr1 - DUD fgfr1 (DUD)',
        '26': 'DUD_fxa - DUD fxa (DUD)',
        '27': 'DUD_gpb - DUD gpb (DUD)',
        '28': 'DUD_gr - DUD gr (DUD)',
        '29': 'DUD_hivrt - DUD hivrt (DUD)',
        '30': 'DUD_inha - DUD inha (DUD)',
        '31': 'DUD_na - DUD na (DUD)',
        '32': 'DUD_p38 - DUD p38 (DUD)',
        '33': 'DUD_parp - DUD parp (DUD)',
        '34': 'DUD_pdgfrb - DUD pdgfrb (DUD)',
        '35': 'DUD_sahh - DUD sahh (DUD)',
        '36': 'DUD_src - DUD src (DUD)',
        '37': 'DUD_vegfr2 - DUD vegfr2 (DUD)',
        # ChEMBL targets (indices 38-87)
        '38': 'ChEMBL_11359 - ChEMBL 11359 (ChEMBL)',
        '39': 'ChEMBL_28 - ChEMBL 28 (ChEMBL)',
        '40': 'ChEMBL_11536 - ChEMBL 11536 (ChEMBL)',
        '41': 'ChEMBL_8 - ChEMBL 8 (ChEMBL)',
        '42': 'ChEMBL_10434 - ChEMBL 10434 (ChEMBL)',
        '43': 'ChEMBL_12670 - ChEMBL 12670 (ChEMBL)',
        '44': 'ChEMBL_20014 - ChEMBL 20014 (ChEMBL)',
        '45': 'ChEMBL_234 - ChEMBL 234 (ChEMBL)',
        '46': 'ChEMBL_12261 - ChEMBL 12261 (ChEMBL)',
        '47': 'ChEMBL_12209 - ChEMBL 12209 (ChEMBL)',
        '48': 'ChEMBL_25 - ChEMBL 25 (ChEMBL)',
        '49': 'ChEMBL_36 - ChEMBL 36 (ChEMBL)',
        '50': 'ChEMBL_43 - ChEMBL 43 (ChEMBL)',
        '51': 'ChEMBL_219 - ChEMBL 219 (ChEMBL)',
        '52': 'ChEMBL_130 - ChEMBL 130 (ChEMBL)',
        '53': 'ChEMBL_105 - ChEMBL 105 (ChEMBL)',
        '54': 'ChEMBL_11336 - ChEMBL 11336 (ChEMBL)',
        '55': 'ChEMBL_20174 - ChEMBL 20174 (ChEMBL)',
        '56': 'ChEMBL_126 - ChEMBL 126 (ChEMBL)',
        '57': 'ChEMBL_11225 - ChEMBL 11225 (ChEMBL)',
        '58': 'ChEMBL_12252 - ChEMBL 12252 (ChEMBL)',
        '59': 'ChEMBL_11682 - ChEMBL 11682 (ChEMBL)',
        '60': 'ChEMBL_134 - ChEMBL 134 (ChEMBL)',
        '61': 'ChEMBL_116 - ChEMBL 116 (ChEMBL)',
        '62': 'ChEMBL_11265 - ChEMBL 11265 (ChEMBL)',
        '63': 'ChEMBL_10475 - ChEMBL 10475 (ChEMBL)',
        '64': 'ChEMBL_12679 - ChEMBL 12679 (ChEMBL)',
        '65': 'ChEMBL_10579 - ChEMBL 10579 (ChEMBL)',
        '66': 'ChEMBL_11575 - ChEMBL 11575 (ChEMBL)',
        '67': 'ChEMBL_18061 - ChEMBL 18061 (ChEMBL)',
        '68': 'ChEMBL_237 - ChEMBL 237 (ChEMBL)',
        '69': 'ChEMBL_276 - ChEMBL 276 (ChEMBL)',
        '70': 'ChEMBL_11534 - ChEMBL 11534 (ChEMBL)',
        '71': 'ChEMBL_10198 - ChEMBL 10198 (ChEMBL)',
        '72': 'ChEMBL_10498 - ChEMBL 10498 (ChEMBL)',
        '73': 'ChEMBL_12911 - ChEMBL 12911 (ChEMBL)',
        '74': 'ChEMBL_12968 - ChEMBL 12968 (ChEMBL)',
        '75': 'ChEMBL_100579 - ChEMBL 100579 (ChEMBL)',
        '76': 'ChEMBL_100126 - ChEMBL 100126 (ChEMBL)',
        '77': 'ChEMBL_10378 - ChEMBL 10378 (ChEMBL)',
        '78': 'ChEMBL_10417 - ChEMBL 10417 (ChEMBL)',
        '79': 'ChEMBL_10752 - ChEMBL 10752 (ChEMBL)',
        '80': 'ChEMBL_10773 - ChEMBL 10773 (ChEMBL)',
        '81': 'ChEMBL_11631 - ChEMBL 11631 (ChEMBL)',
        '82': 'ChEMBL_10927 - ChEMBL 10927 (ChEMBL)',
        '83': 'ChEMBL_11085 - ChEMBL 11085 (ChEMBL)',
        '84': 'ChEMBL_11442 - ChEMBL 11442 (ChEMBL)',
        '85': 'ChEMBL_11279 - ChEMBL 11279 (ChEMBL)',
        '86': 'ChEMBL_11488 - ChEMBL 11488 (ChEMBL)',
        '87': 'ChEMBL_12840 - ChEMBL 12840 (ChEMBL)',
    },
}

# == GITHUB REPOSITORY SOURCE ==

REPO_URL = 'https://github.com/rdkit/benchmarking_platform/archive/refs/heads/master.zip'

# Subset I target configuration (from configuration_file_I.py)
MUV_IDS = [466, 548, 600, 644, 652, 689, 692, 712, 713, 733, 737, 810, 832, 846, 852, 858, 859]
DUD_IDS = ['ace', 'ache', 'ar', 'cdk2', 'cox2', 'dhfr', 'egfr', 'er_agonist', 'fgfr1', 'fxa',
           'gpb', 'gr', 'hivrt', 'inha', 'na', 'p38', 'parp', 'pdgfrb', 'sahh', 'src', 'vegfr2']
CHEMBL_IDS = [11359, 28, 11536, 8, 10434, 12670, 20014, 234, 12261, 12209, 25, 36, 43, 219, 130,
              105, 11336, 20174, 126, 11225, 12252, 11682, 134, 116, 11265, 10475, 12679, 10579,
              11575, 18061, 237, 276, 11534, 10198, 10498, 12911, 12968, 100579, 100126, 10378,
              10417, 10752, 10773, 11631, 10927, 11085, 11442, 11279, 11488, 12840]

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
    Download and process the RDKit Benchmarking Platform Subset I for multi-target classification.

    This function:
    1. Downloads the repository as a ZIP file from GitHub
    2. Extracts to /tmp/benchmarking_platform_cache (persists between runs)
    3. Parses all active compound files for the 88 targets in Subset I
    4. Parses all decoy compound files (per-target specific)
    5. Groups molecules by SMILES to get unique structures
    6. For each unique molecule, creates an 88-dimensional tri-state target vector
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

    # Add DUD targets (indices 17-37)
    for dud_id in DUD_IDS:
        target_key = f'DUD_{dud_id}'
        target_list.append(target_key)
        target_metadata[target_key] = {
            'source': 'DUD',
            'id': dud_id,
            'name': f'DUD {dud_id}',
        }

    # Add ChEMBL targets (indices 38-87)
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

    e.log(f'Total targets in Subset I: {num_targets}')
    e.log(f'  - MUV targets: {len(MUV_IDS)} (indices 0-{len(MUV_IDS)-1})')
    e.log(f'  - DUD targets: {len(DUD_IDS)} (indices {len(MUV_IDS)}-{len(MUV_IDS)+len(DUD_IDS)-1})')
    e.log(f'  - ChEMBL targets: {len(CHEMBL_IDS)} (indices {len(MUV_IDS)+len(DUD_IDS)}-{num_targets-1})')
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

    # Parse DUD actives
    e.log('Parsing DUD actives...')
    dud_path = os.path.join(compounds_path, 'DUD')
    for dud_id in DUD_IDS:
        target_key = f'DUD_{dud_id}'
        file_path = os.path.join(dud_path, f'cmp_list_DUD_{dud_id}_actives.dat.gz')

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
                # DUD uses '# Mol_Title' as column name (with # prefix)
                external_id = row.get('# Mol_Title') or row.get('Mol_Title')
                if external_id and external_id not in molecule_target_data[smiles]['external_ids']:
                    molecule_target_data[smiles]['external_ids'].append(external_id)
                count += 1

            e.log(f'  ✓ {target_key}: {count} actives')

    # Parse ChEMBL actives
    e.log('Parsing ChEMBL actives...')
    chembl_path = os.path.join(compounds_path, 'ChEMBL')
    for chembl_id in CHEMBL_IDS:
        target_key = f'ChEMBL_{chembl_id}'
        file_path = os.path.join(chembl_path, f'cmp_list_ChEMBL_{chembl_id}_actives.dat.gz')

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
                # ChEMBL uses '# _Name' as column name (with # prefix)
                external_id = row.get('# _Name') or row.get('_Name')
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
    e.log('PARSING DECOY COMPOUNDS (PER-TARGET)')
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

    # Parse DUD decoys (per-target specific)
    e.log('Parsing DUD decoys (per-target)...')
    for dud_id in DUD_IDS:
        target_key = f'DUD_{dud_id}'
        file_path = os.path.join(dud_path, f'cmp_list_DUD_{dud_id}_decoys.dat.gz')

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

                # DUD uses '# Mol_Title' as column name (with # prefix)
                external_id = row.get('# Mol_Title') or row.get('Mol_Title')
                if external_id and external_id not in molecule_target_data[smiles]['external_ids']:
                    molecule_target_data[smiles]['external_ids'].append(external_id)
                count += 1

            total_decoy_pairs += count
            e.log(f'  ✓ {target_key}: {count} decoys')

    # Parse ChEMBL decoys (shared decoys for all ChEMBL targets)
    e.log('Parsing ChEMBL decoys (shared for all ChEMBL targets)...')
    file_path = os.path.join(chembl_path, 'cmp_list_ChEMBL_zinc_decoys.dat.gz')
    if os.path.exists(file_path):
        with gzip.open(file_path, 'rt') as f:
            reader = csv.DictReader(f, delimiter='\t')
            zinc_decoy_smiles = []
            for row in reader:
                smiles = row['SMILES']
                zinc_decoy_smiles.append(smiles)

                if smiles not in molecule_target_data:
                    # New molecule only seen as decoy
                    molecule_target_data[smiles] = {
                        'active_targets': [],
                        'decoy_targets': [],
                        'external_ids': [],
                    }
                    new_molecules_from_decoys += 1

                # ChEMBL uses '# _Name' as column name (with # prefix)
                external_id = row.get('# _Name') or row.get('_Name')
                if external_id and external_id not in molecule_target_data[smiles]['external_ids']:
                    molecule_target_data[smiles]['external_ids'].append(external_id)

            # Add all ChEMBL targets as decoy targets for these molecules
            count = 0
            for smiles in zinc_decoy_smiles:
                for chembl_id in CHEMBL_IDS:
                    target_key = f'ChEMBL_{chembl_id}'
                    molecule_target_data[smiles]['decoy_targets'].append(target_key)
                    count += 1

            total_decoy_pairs += count
            e.log(f'  ✓ ChEMBL zinc_decoys: {len(zinc_decoy_smiles)} molecules × {len(CHEMBL_IDS)} targets = {count} decoy pairs')

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
    - External IDs (PubChem CID, ZINC ID, ChEMBL ID, etc.)
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
