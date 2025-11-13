"""
This experiment module creates a graph dataset from the Briem & Lessel-like dataset
created by Greg Landrum using ChEMBL25 data for binary classification tasks.

This is a modern recreation of the classic Briem and Lessel dataset (2000) designed
for validating similarity-based virtual screening methods. The classification version
(_cls) focuses on binary prediction of active (Ki ≤ 1nM) vs inactive (Ki ≥ 100nM)
compounds, excluding intermediate activity compounds.

The dataset includes both measured inactive compounds from ChEMBL and assumed inactive
decoy compounds for similarity screening validation.

References:
- Original blog post: https://rdkit.blogspot.com/2019/10/a-new-lessel-and-briem-like-dataset.html
- Data source: https://github.com/greglandrum/rdkit_blog/tree/master/data
"""
import csv
import io
import rdkit.Chem as Chem
from typing import Dict
from rich.pretty import pprint
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

from chem_mat_data.connectors import FileDownloadSource

# == DATASET METADATA ==

DATASET_NAME: str = 'bl_chembl_cls'

DESCRIPTION: str = (
    'The Briem & Lessel ChEMBL Multi-Target Classification Dataset is a modern recreation of '
    'the classic Briem and Lessel dataset (originally published in 2000) using ChEMBL25 data. '
    'This multi-label classification version contains two classes: (1) selected high-potency '
    'active compounds from the curated selected_actives file, and (2) decoy compounds assumed '
    'to be inactive. Each molecule has a 35-dimensional tri-state target activity vector where '
    'each position can have three values: 0 (not mentioned), 1 (active), or 2 (decoy). For '
    'selected actives, a 1 in position i indicates the molecule is a high-potency active '
    '(Ki ≤ 1nM) against target i. For decoys, all positions are set to 2, indicating the '
    'molecule is assumed to be inactive (a decoy) against all 35 targets. This tri-state '
    'encoding enables multi-label multi-target activity prediction where a single molecule can '
    'be active against multiple targets simultaneously, with explicit decoy identification.'
)

METADATA: dict = {
    'tags': [
        'Molecules',
        'SMILES',
        'Drug Discovery',
        'Virtual Screening',
        'Similarity',
        'Multi-Target',
        'ChEMBL',
        'Ki',
        'Bioactivity',
        'Classification',
        'Multi-Label Classification',
        'Multi-Task',
    ],
    'verbose': 'Briem & Lessel ChEMBL Dataset (Multi-Target Classification)',
    'sources': [
        'https://rdkit.blogspot.com/2019/10/a-new-lessel-and-briem-like-dataset.html',
        'https://github.com/greglandrum/rdkit_blog/tree/master/data',
        'https://doi.org/10.1021/ci000197k'  # Original Briem & Lessel paper
    ],
    'notes': [
        'This is a virtual screening benchmark dataset derived from ChEMBL25 for evaluating molecular similarity methods.',
        'Target vectors use tri-state encoding: 0 = not mentioned for that target, 1 = active (high-potency, Ki ≤ 1nM) against that target, 2 = decoy for that target.',
        'Decoys are shared compounds assumed to be inactive against all 35 targets.',
        'All decoy molecules have value 2 for all target positions (assumed inactive against all targets).',
        'Active compounds are high-potency selected actives (Ki ≤ 1nM) from ChEMBL.',
    ],
    'target_descriptions': {
        '0': 'CHEMBL1862 - Tyrosine-protein kinase ABL (Homo sapiens)',
        '1': 'CHEMBL204 - Thrombin (Homo sapiens)',
        '2': 'CHEMBL205 - Carbonic anhydrase II (Homo sapiens)',
        '3': 'CHEMBL4794 - Vanilloid receptor (Homo sapiens)',
        '4': 'CHEMBL264 - Histamine H3 receptor (Homo sapiens)',
        '5': 'CHEMBL214 - Serotonin 1a (5-HT1a) receptor (Homo sapiens)',
        '6': 'CHEMBL217 - Dopamine D2 receptor (Homo sapiens)',
        '7': 'CHEMBL4552 - Peripheral-type benzodiazepine receptor (Rattus norvegicus)',
        '8': 'CHEMBL2147 - Serine/threonine-protein kinase PIM1 (Homo sapiens)',
        '9': 'CHEMBL224 - Serotonin 2a (5-HT2a) receptor (Homo sapiens)',
        '10': 'CHEMBL229 - Alpha-1a adrenergic receptor (Homo sapiens)',
        '11': 'CHEMBL233 - Mu opioid receptor (Homo sapiens)',
        '12': 'CHEMBL234 - Dopamine D3 receptor (Homo sapiens)',
        '13': 'CHEMBL236 - Delta opioid receptor (Homo sapiens)',
        '14': 'CHEMBL237 - Kappa opioid receptor (Homo sapiens)',
        '15': 'CHEMBL2366517 - Protease (Human immunodeficiency virus 1)',
        '16': 'CHEMBL4409 - Phosphodiesterase 10A (Homo sapiens)',
        '17': 'CHEMBL2835 - Tyrosine-protein kinase JAK1 (Homo sapiens)',
        '18': 'CHEMBL2971 - Tyrosine-protein kinase JAK2 (Homo sapiens)',
        '19': 'CHEMBL3952 - Kappa opioid receptor (Cavia porcellus)',
        '20': 'CHEMBL243 - Human immunodeficiency virus type 1 protease (Human immunodeficiency virus 1)',
        '21': 'CHEMBL244 - Coagulation factor X (Homo sapiens)',
        '22': 'CHEMBL339 - Dopamine D2 receptor (Rattus norvegicus)',
        '23': 'CHEMBL245 - Muscarinic acetylcholine receptor M3 (Homo sapiens)',
        '24': 'CHEMBL251 - Adenosine A2a receptor (Homo sapiens)',
        '25': 'CHEMBL253 - Cannabinoid CB2 receptor (Homo sapiens)',
        '26': 'CHEMBL256 - Adenosine A3 receptor (Homo sapiens)',
        '27': 'CHEMBL269 - Delta opioid receptor (Rattus norvegicus)',
        '28': 'CHEMBL270 - Mu opioid receptor (Rattus norvegicus)',
        '29': 'CHEMBL1946 - Melatonin receptor 1B (Homo sapiens)',
        '30': 'CHEMBL273 - Serotonin 1a (5-HT1a) receptor (Rattus norvegicus)',
        '31': 'CHEMBL1907596 - Neuronal acetylcholine receptor; alpha4/beta2 (Rattus norvegicus)',
        '32': 'CHEMBL3371 - Serotonin 6 (5-HT6) receptor (Homo sapiens)',
        '33': 'CHEMBL313 - Serotonin transporter (Rattus norvegicus)',
        '34': 'CHEMBL4860 - Apoptosis regulator Bcl-2 (Homo sapiens)',
    }
}

# == GITHUB DATA SOURCES ==
# URLs to the raw data files from Greg Landrum's RDKit blog repository

GITHUB_BASE_URL = 'https://raw.githubusercontent.com/greglandrum/rdkit_blog/master/data/'

URLS = {
    'actives': GITHUB_BASE_URL + 'BLSets_actives.txt',
    'selected_actives': GITHUB_BASE_URL + 'BLSets_selected_actives.txt',
    'decoys': GITHUB_BASE_URL + 'BLSets_singleic50_decoys.txt',
    'target_key': GITHUB_BASE_URL + 'BLSets_actives_target_key.txt',
}

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


@experiment.hook('load_dataset', default=False, replace=True)
def load_dataset(e: Experiment) -> Dict[int, dict]:
    """
    Download and process the Briem & Lessel ChEMBL dataset files for multi-target classification.

    This function:
    1. Downloads all required files from GitHub
    2. Parses the selected_actives file to get high-potency compounds
    3. Groups molecules by SMILES to get unique structures
    4. For each unique molecule, creates a 35-dimensional tri-state target activity vector
    5. For selected actives: sets 1 in positions where molecule is active, 0 elsewhere
    6. For decoys: sets all positions to 2 (decoy for all targets)
    7. Excludes all inactive and intermediate compounds
    8. Returns dictionary with tri-state multi-label target vectors
    """

    e.log('=' * 80)
    e.log('DOWNLOADING DATASET FILES FROM GITHUB')
    e.log('=' * 80)

    downloaded_files = {}

    # Download all required files
    for file_key, url in URLS.items():
        e.log(f'Downloading {file_key} from {url}...')
        with FileDownloadSource(url, verbose=True) as source:
            file_path = source.fetch()

            # Read the file content immediately and store it
            with open(file_path, 'r') as f:
                content = f.read()
            downloaded_files[file_key] = content

        e.log(f'  ✓ Downloaded {file_key} ({len(content)} bytes)')

    e.log('All files downloaded successfully!')

    # == PARSE TARGET KEY FILE ==
    e.log('')
    e.log('=' * 80)
    e.log('PARSING TARGET METADATA')
    e.log('=' * 80)

    target_metadata = {}
    target_list = []  # Ordered list of targets for indexing
    target_file = io.StringIO(downloaded_files['target_key'])
    reader = csv.DictReader(target_file, delimiter='\t')
    for row in reader:
        chembl_id = row['chembl_id']
        target_metadata[chembl_id] = {
            'tid': row['tid'],
            'name': row['pref_name'],
            'organism': row['organism']
        }
        target_list.append(chembl_id)

    # Create mapping from target ChEMBL ID to vector index (0-34)
    target_id_to_index = {chembl_id: idx for idx, chembl_id in enumerate(target_list)}
    num_targets = len(target_list)

    e.log(f'Loaded metadata for {num_targets} targets')
    e.log(f'Target order: {target_list[:5]}... (total {num_targets})')
    e.log(f'Target index mapping created for multi-target vector')

    # == PARSE SELECTED ACTIVES FILE FIRST ==
    e.log('')
    e.log('=' * 80)
    e.log('PARSING SELECTED ACTIVES FILE (HIGH-POTENCY COMPOUNDS)')
    e.log('=' * 80)

    # Build set of (SMILES, target) pairs that are selected actives
    selected_actives_map = {}  # smiles -> list of target_chembl_ids
    selected_file = io.StringIO(downloaded_files['selected_actives'])
    # Note: This file uses SPACE delimiter, not tab!
    reader = csv.DictReader(selected_file, delimiter=' ')

    for row in reader:
        smiles = row['smiles']
        target_id = row['target_chembl_id']

        if smiles not in selected_actives_map:
            selected_actives_map[smiles] = []
        selected_actives_map[smiles].append(target_id)

    e.log(f'Loaded {len(selected_actives_map)} unique molecules from selected_actives')
    total_selected_pairs = sum(len(targets) for targets in selected_actives_map.values())
    e.log(f'  Total (SMILES, target) pairs: {total_selected_pairs}')
    e.log(f'  Average targets per molecule: {total_selected_pairs / len(selected_actives_map):.2f}')

    # Build molecule_target_data from selected actives only
    molecule_target_data = {}
    for smiles, target_list in selected_actives_map.items():
        molecule_target_data[smiles] = {
            'targets': target_list,  # List of target ChEMBL IDs
            'is_decoy': False,
            'source': 'selected_actives'
        }

    # == PARSE DECOYS FILE ==
    e.log('')
    e.log('=' * 80)
    e.log('PARSING DECOYS FILE (BLSets_singleic50_decoys.txt)')
    e.log('=' * 80)

    decoys_file = io.StringIO(downloaded_files['decoys'])
    reader = csv.DictReader(decoys_file, delimiter='\t')
    num_decoys = 0

    for row in reader:
        smiles = row['smiles']
        # Decoys have no target information, will get all-zero vector
        if smiles not in molecule_target_data:
            molecule_target_data[smiles] = {
                'targets': [],  # Empty - no target information
                'is_decoy': True,
                'source': 'decoys',
                'chembl_id': row['chembl_id']
            }
            num_decoys += 1

    e.log(f'Added {num_decoys} unique decoy compounds')
    e.log(f'Total unique molecules: {len(molecule_target_data)}')

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
        'molecules_selected_actives': 0,
        'molecules_decoys': 0,
        'total_active_targets': 0,
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

        # == BUILD 35-DIMENSIONAL TRI-STATE TARGET VECTOR ==
        # Each position can be: 0 (not mentioned), 1 (active), 2 (decoy)
        # For actives: 1 where active, 0 elsewhere
        # For decoys: 2 in all positions (assumed inactive/decoy for all targets)

        active_target_ids = []

        if mol_data['is_decoy']:
            # Decoy: set all positions to 2 (decoy for all targets)
            target_vector = [2] * num_targets
        else:
            # Selected actives: start with all zeros, set 1 for active targets
            target_vector = [0] * num_targets
            # mol_data['targets'] is a list of target ChEMBL IDs
            for target_id in mol_data['targets']:
                if target_id in target_id_to_index:
                    target_idx = target_id_to_index[target_id]
                    target_vector[target_idx] = 1
                    active_target_ids.append(target_id)

        # Update statistics
        if mol_data['is_decoy']:
            stats['molecules_decoys'] += 1
        else:
            stats['molecules_selected_actives'] += 1
            stats['total_active_targets'] += len(active_target_ids)

        # Create data dict
        data = {
            'smiles': smiles,
            'targets': target_vector,
            'is_decoy': mol_data['is_decoy'],
            'source': mol_data['source'],
            'num_active_targets': len(active_target_ids),
            'active_target_ids': active_target_ids,
        }

        # Add ChEMBL ID for decoys
        if mol_data['is_decoy'] and 'chembl_id' in mol_data:
            data['chembl_id'] = mol_data['chembl_id']

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
    e.log(f'  - Selected active molecules (from selected_actives file): {stats["molecules_selected_actives"]} ({100*stats["molecules_selected_actives"]/stats["success"]:.1f}%)')
    e.log(f'  - Decoy molecules (all positions=2, assumed inactive): {stats["molecules_decoys"]} ({100*stats["molecules_decoys"]/stats["success"]:.1f}%)')
    e.log(f'  - Total active target assignments: {stats["total_active_targets"]}')
    if stats["molecules_selected_actives"] > 0:
        avg_active = stats["total_active_targets"] / stats["molecules_selected_actives"]
        e.log(f'  - Average active targets per selected molecule: {avg_active:.2f}')
    e.log(f'  - Target vector dimension: {num_targets}')
    e.log(f'  - Target vector encoding: 0=not mentioned, 1=active, 2=decoy (all positions)')
    e.log('')

    # Print example entries
    e.log('Sample processed entries:')
    for i in range(min(3, len(dataset))):
        e.log(f'\nEntry {i}:')
        entry = dataset[i].copy()
        # Show first 10 positions of target vector for readability
        entry['targets_preview'] = entry['targets'][:10] + ['...']
        pprint(entry, max_depth=2)

    return dataset


@experiment.hook('add_graph_metadata', default=False, replace=True)
def add_graph_metadata(_e: Experiment, data: dict, graph: dict) -> None:
    """
    Add additional metadata fields to each graph representation.

    This includes:
    - Number of active targets (how many targets in the vector are 1)
    - List of active target ChEMBL IDs
    - Source file marker (selected_actives vs decoys)
    - Decoy flag
    - ChEMBL ID for decoys
    """
    # Add multi-target information
    graph['graph_num_active_targets'] = int(data.get('num_active_targets', 0))

    # Add source tracking
    graph['graph_source'] = data.get('source', '')

    # Add decoy flag
    graph['graph_is_decoy'] = int(data.get('is_decoy', False))

    # Add ChEMBL ID for decoys
    if data.get('is_decoy') and 'chembl_id' in data:
        graph['graph_chembl_id'] = data.get('chembl_id', '')

    # Store active target IDs as a comma-separated string (for reference)
    active_ids = data.get('active_target_ids', [])
    if active_ids:
        graph['graph_active_target_ids'] = ','.join(active_ids)
    else:
        graph['graph_active_target_ids'] = ''


experiment.run_if_main()
