"""
This experiment module creates a graph dataset from the Briem & Lessel-like dataset
created by Greg Landrum using ChEMBL25 data. This is a modern recreation of the classic
Briem and Lessel dataset (2000) designed for validating similarity-based virtual screening
methods.

The dataset includes compounds with measured Ki values across 35 biological targets from
ChEMBL, along with activity classifications (active: Ki ≤ 1nM, inactive: Ki ≥ 100nM).

This regression version (_reg) uses pKi (negative log of Ki) as the target variable for
quantitative affinity prediction.

References:
- Original blog post: https://rdkit.blogspot.com/2019/10/a-new-lessel-and-briem-like-dataset.html
- Data source: https://github.com/greglandrum/rdkit_blog/tree/master/data
"""
import csv
import io
import numpy as np
import pandas as pd
import rdkit.Chem as Chem
from typing import Dict
from rich.pretty import pprint
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

from chem_mat_data.connectors import FileDownloadSource

# == DATASET METADATA ==

DATASET_NAME: str = 'bl_chembl_reg'

DESCRIPTION: str = (
    'The Briem & Lessel ChEMBL Multi-Target Regression Dataset is a modern recreation of the '
    'classic Briem and Lessel dataset (originally published in 2000) using ChEMBL25 data. '
    'Each molecule has a 35-dimensional target vector, where each dimension corresponds to one '
    'of 35 biological targets (GPCRs, kinases, proteases, etc.) from ChEMBL. The values are '
    'pKi (negative log of Ki binding affinity in molar units), with NaN indicating missing '
    'measurements. This enables multi-task regression where a single molecule can have measured '
    'affinities for multiple targets simultaneously. Higher pKi values indicate stronger binding. '
    'Typical range: 4.0-12.0 for measured values. This dataset is specifically designed for '
    'multi-task QSAR modeling and validating similarity-based virtual screening methods.'
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
        'QSAR',
        'Multi-Task',
        'Regression',
    ],
    'verbose': 'Briem & Lessel ChEMBL Dataset (Multi-Target Regression)',
    'sources': [
        'https://rdkit.blogspot.com/2019/10/a-new-lessel-and-briem-like-dataset.html',
        'https://github.com/greglandrum/rdkit_blog/tree/master/data',
        'https://doi.org/10.1021/ci000197k'  # Original Briem & Lessel paper
    ],
    'target_descriptions': {
        '0': 'CHEMBL1862 - Tyrosine-protein kinase ABL (Homo sapiens) - pKi value',
        '1': 'CHEMBL204 - Thrombin (Homo sapiens) - pKi value',
        '2': 'CHEMBL205 - Carbonic anhydrase II (Homo sapiens) - pKi value',
        '3': 'CHEMBL4794 - Vanilloid receptor (Homo sapiens) - pKi value',
        '4': 'CHEMBL264 - Histamine H3 receptor (Homo sapiens) - pKi value',
        '5': 'CHEMBL214 - Serotonin 1a (5-HT1a) receptor (Homo sapiens) - pKi value',
        '6': 'CHEMBL217 - Dopamine D2 receptor (Homo sapiens) - pKi value',
        '7': 'CHEMBL4552 - Peripheral-type benzodiazepine receptor (Rattus norvegicus) - pKi value',
        '8': 'CHEMBL2147 - Serine/threonine-protein kinase PIM1 (Homo sapiens) - pKi value',
        '9': 'CHEMBL224 - Serotonin 2a (5-HT2a) receptor (Homo sapiens) - pKi value',
        '10': 'CHEMBL229 - Alpha-1a adrenergic receptor (Homo sapiens) - pKi value',
        '11': 'CHEMBL233 - Mu opioid receptor (Homo sapiens) - pKi value',
        '12': 'CHEMBL234 - Dopamine D3 receptor (Homo sapiens) - pKi value',
        '13': 'CHEMBL236 - Delta opioid receptor (Homo sapiens) - pKi value',
        '14': 'CHEMBL237 - Kappa opioid receptor (Homo sapiens) - pKi value',
        '15': 'CHEMBL2366517 - Protease (Human immunodeficiency virus 1) - pKi value',
        '16': 'CHEMBL4409 - Phosphodiesterase 10A (Homo sapiens) - pKi value',
        '17': 'CHEMBL2835 - Tyrosine-protein kinase JAK1 (Homo sapiens) - pKi value',
        '18': 'CHEMBL2971 - Tyrosine-protein kinase JAK2 (Homo sapiens) - pKi value',
        '19': 'CHEMBL3952 - Kappa opioid receptor (Cavia porcellus) - pKi value',
        '20': 'CHEMBL243 - Human immunodeficiency virus type 1 protease (Human immunodeficiency virus 1) - pKi value',
        '21': 'CHEMBL244 - Coagulation factor X (Homo sapiens) - pKi value',
        '22': 'CHEMBL339 - Dopamine D2 receptor (Rattus norvegicus) - pKi value',
        '23': 'CHEMBL245 - Muscarinic acetylcholine receptor M3 (Homo sapiens) - pKi value',
        '24': 'CHEMBL251 - Adenosine A2a receptor (Homo sapiens) - pKi value',
        '25': 'CHEMBL253 - Cannabinoid CB2 receptor (Homo sapiens) - pKi value',
        '26': 'CHEMBL256 - Adenosine A3 receptor (Homo sapiens) - pKi value',
        '27': 'CHEMBL269 - Delta opioid receptor (Rattus norvegicus) - pKi value',
        '28': 'CHEMBL270 - Mu opioid receptor (Rattus norvegicus) - pKi value',
        '29': 'CHEMBL1946 - Melatonin receptor 1B (Homo sapiens) - pKi value',
        '30': 'CHEMBL273 - Serotonin 1a (5-HT1a) receptor (Rattus norvegicus) - pKi value',
        '31': 'CHEMBL1907596 - Neuronal acetylcholine receptor; alpha4/beta2 (Rattus norvegicus) - pKi value',
        '32': 'CHEMBL3371 - Serotonin 6 (5-HT6) receptor (Homo sapiens) - pKi value',
        '33': 'CHEMBL313 - Serotonin transporter (Rattus norvegicus) - pKi value',
        '34': 'CHEMBL4860 - Apoptosis regulator Bcl-2 (Homo sapiens) - pKi value'
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
    Download and process the Briem & Lessel ChEMBL dataset files for multi-target regression.

    This function:
    1. Downloads all required files from GitHub
    2. Parses TSV files and creates target index mapping
    3. Groups molecules by SMILES to detect duplicates
    4. For each unique molecule, creates a 35-dimensional pKi target vector
    5. Sets pKi values for positions where molecule has measurements for that target
    6. Uses NaN for missing target measurements
    7. Returns dictionary with multi-task target vectors
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
            # (since the temp file will be cleaned up when exiting the context)
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

    # == PARSE MAIN ACTIVES FILE ==
    e.log('')
    e.log('=' * 80)
    e.log('PARSING MAIN ACTIVES FILE (BLSets_actives.txt)')
    e.log('=' * 80)

    # Build mapping: smiles -> {target_chembl_id: pKi_value}
    molecule_target_data = {}
    actives_file = io.StringIO(downloaded_files['actives'])
    reader = csv.DictReader(actives_file, delimiter='\t')

    for row in reader:
        smiles = row['smiles']
        target_id = row['target_chembl_id']
        pki_str = row['pKi']

        # Only process if pKi value exists
        if pki_str and pki_str.strip():
            try:
                pki_value = float(pki_str)

                if smiles not in molecule_target_data:
                    molecule_target_data[smiles] = {
                        'target_pki_map': {},
                        'activity_labels': {},
                        'source': 'actives'
                    }

                # Store pKi value for this target
                molecule_target_data[smiles]['target_pki_map'][target_id] = pki_value

                # Store activity label if present
                label = row['label'] if row['label'] else 'unlabeled'
                molecule_target_data[smiles]['activity_labels'][target_id] = label

            except ValueError:
                # Skip invalid pKi values
                pass

    e.log(f'Loaded {len(molecule_target_data)} unique molecules from BLSets_actives.txt')

    # Count total measurements
    total_measurements = sum(len(data['target_pki_map']) for data in molecule_target_data.values())
    e.log(f'  Total (SMILES, target) pairs with pKi: {total_measurements}')
    e.log(f'  Average targets per molecule: {total_measurements / len(molecule_target_data):.2f}')

    # == PARSE SELECTED ACTIVES FILE ==
    e.log('')
    e.log('=' * 80)
    e.log('PARSING SELECTED ACTIVES FILE')
    e.log('=' * 80)

    selected_actives_set = set()
    selected_file = io.StringIO(downloaded_files['selected_actives'])
    # Note: This file uses SPACE delimiter, not tab!
    reader = csv.DictReader(selected_file, delimiter=' ')

    for row in reader:
        # Create a unique key combining SMILES and target
        key = (row['smiles'], row['target_chembl_id'])
        selected_actives_set.add(key)

    e.log(f'Identified {len(selected_actives_set)} selected high-potency actives')

    # == BUILD MULTI-TARGET VECTORS ==
    e.log('')
    e.log('=' * 80)
    e.log('BUILDING MULTI-TARGET pKi VECTORS')
    e.log('=' * 80)

    dataset: Dict[int, dict] = {}
    index = 0

    stats = {
        'total': 0,
        'filtered_salt': 0,
        'filtered_invalid_smiles': 0,
        'filtered_single_atom': 0,
        'success': 0,
        'total_measurements': 0,
        'measurements_per_molecule': [],
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

        # == BUILD 35-DIMENSIONAL TARGET VECTOR ==
        # Initialize all NaN
        target_vector = [float('nan')] * num_targets

        # Fill in measured pKi values
        target_pki_map = mol_data['target_pki_map']
        measured_targets = []

        for target_id, pki_value in target_pki_map.items():
            if target_id in target_id_to_index:
                target_idx = target_id_to_index[target_id]
                target_vector[target_idx] = pki_value
                measured_targets.append(target_id)

        num_measurements = len(measured_targets)
        stats['total_measurements'] += num_measurements
        stats['measurements_per_molecule'].append(num_measurements)

        # Check if any targets are selected actives
        selected_targets = []
        for target_id in measured_targets:
            key = (smiles, target_id)
            if key in selected_actives_set:
                selected_targets.append(target_id)

        # Create data dict
        data = {
            'smiles': smiles,
            'targets': target_vector,
            'source': mol_data['source'],
            'num_measurements': num_measurements,
            'measured_target_ids': measured_targets,
            'selected_target_ids': selected_targets,
            'num_selected': len(selected_targets),
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
    e.log(f'  - Total pKi measurements across all molecules: {stats["total_measurements"]}')
    avg_measurements = stats["total_measurements"] / stats["success"] if stats["success"] > 0 else 0
    e.log(f'  - Average measurements per molecule: {avg_measurements:.2f}')

    if stats['measurements_per_molecule']:
        measurements_array = np.array(stats['measurements_per_molecule'])
        e.log(f'  - Min measurements per molecule: {measurements_array.min()}')
        e.log(f'  - Max measurements per molecule: {measurements_array.max()}')
        e.log(f'  - Median measurements per molecule: {np.median(measurements_array):.1f}')

    e.log(f'  - Target vector dimension: {num_targets}')
    e.log('')

    # Print example entries
    e.log('Sample processed entries:')
    for i in range(min(3, len(dataset))):
        e.log(f'\nEntry {i}:')
        entry = dataset[i].copy()
        # Show non-NaN values from target vector for readability
        target_vec = entry['targets']
        non_nan_indices = [idx for idx, val in enumerate(target_vec) if not pd.isna(val)]
        entry['targets_preview'] = {
            f'target_{idx}': target_vec[idx]
            for idx in non_nan_indices[:5]  # Show first 5 measured targets
        }
        if len(non_nan_indices) > 5:
            entry['targets_preview']['...'] = f'({len(non_nan_indices) - 5} more)'
        pprint(entry, max_depth=2)

    return dataset


@experiment.hook('add_graph_metadata', default=False, replace=True)
def add_graph_metadata(_e: Experiment, data: dict, graph: dict) -> None:
    """
    Add additional metadata fields to each graph representation.

    This includes:
    - Number of measured targets (how many non-NaN values in the target vector)
    - List of measured target ChEMBL IDs
    - List of selected high-potency target ChEMBL IDs
    - Number of selected targets
    - Source file marker
    """
    # Add multi-target measurement information
    graph['graph_num_measurements'] = int(data.get('num_measurements', 0))
    graph['graph_num_selected'] = int(data.get('num_selected', 0))

    # Add source tracking
    graph['graph_source'] = data.get('source', '')

    # Store measured target IDs as a comma-separated string (for reference)
    measured_ids = data.get('measured_target_ids', [])
    if measured_ids:
        graph['graph_measured_target_ids'] = ','.join(measured_ids)
    else:
        graph['graph_measured_target_ids'] = ''

    # Store selected target IDs as a comma-separated string (for reference)
    selected_ids = data.get('selected_target_ids', [])
    if selected_ids:
        graph['graph_selected_target_ids'] = ','.join(selected_ids)
    else:
        graph['graph_selected_target_ids'] = ''


experiment.run_if_main()
