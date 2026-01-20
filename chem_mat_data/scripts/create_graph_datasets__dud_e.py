"""
This experiment module creates a graph dataset from the DUD-E (Directory of Useful
Decoys - Enhanced) dataset for multi-target classification tasks with tri-state labels.

DUD-E is a comprehensive virtual screening benchmark published by Mysinger et al. (2012)
containing 102 protein targets with 22,886 active compounds and over 1 million decoys.
The dataset is designed to help benchmark molecular docking programs by providing
challenging decoys that have similar physicochemical properties but dissimilar 2-D
topology to the actives.

The dataset is structured as a multi-label classification task where each molecule has a
102-dimensional tri-state target vector:
- Indices 0-101: Correspond to 102 biological targets
- Each position can have three values:
  * 0: Not mentioned for that target (no data)
  * 1: Active against that target
  * 2: Decoy for that target

This tri-state encoding properly captures the per-target nature of the decoy selection, where
a molecule can be active against one target while serving as a decoy for another. Decoys are
specifically selected for each target with a ratio of 50 decoys per active, matched for physical
properties while maintaining 2-D structural dissimilarity to actives.

This enables both multi-task learning and proper evaluation of similarity-based virtual
screening methods with target-specific negative examples.

References:
- Mysinger MM, Carchia M, Irwin JJ, Shoichet BK, J. Med. Chem., 2012, Jul 5.
  DOI: 10.1021/jm300687e
- Website: https://dude.docking.org/
"""
import os
import tarfile
import tempfile
from typing import Dict
from rich.pretty import pprint
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace
import rdkit.Chem as Chem

from chem_mat_data.connectors import FileDownloadSource

# == DATASET METADATA ==

DATASET_NAME: str = 'dud_e'

DESCRIPTION: str = (
    'The DUD-E (Directory of Useful Decoys - Enhanced) dataset is a comprehensive '
    'multi-target virtual screening benchmark containing 102 protein targets with '
    '22,886 active compounds and over 1 million carefully selected decoys. Published '
    'by Mysinger et al. (2012), this dataset is designed to benchmark molecular docking '
    'programs and similarity-based virtual screening methods. Each molecule has a '
    '102-dimensional target vector where each position can have three values: '
    '0 (not mentioned for that target), 1 (active against that target), or 2 (decoy '
    'for that target). This tri-state encoding properly captures the per-target nature '
    'of the decoy selection, where a molecule can be active against one target while '
    'serving as a decoy for another. Decoys are specifically selected for each target '
    'with a 50:1 ratio, matched for physicochemical properties while maintaining 2-D '
    'structural dissimilarity to actives. This enables both multi-task learning and '
    'proper evaluation of similarity-based virtual screening methods with target-specific '
    'negative examples.'
)

# DUD-E target definitions (102 targets)
# Format: (gene_name, pdb_id, full_name)
TARGETS = [
    ('aa2ar', '3eml', 'Adenosine A2a receptor'),
    ('abl1', '2hzi', 'Tyrosine-protein kinase ABL'),
    ('ace', '3bkl', 'Angiotensin-converting enzyme'),
    ('aces', '1e66', 'Acetylcholinesterase'),
    ('ada', '2e1w', 'Adenosine deaminase'),
    ('ada17', '2oi0', 'ADAM17'),
    ('adrb1', '2vt4', 'Beta-1 adrenergic receptor'),
    ('adrb2', '3ny8', 'Beta-2 adrenergic receptor'),
    ('akt1', '3cqw', 'Serine/threonine-protein kinase AKT'),
    ('akt2', '3d0e', 'Serine/threonine-protein kinase AKT2'),
    ('aldr', '2hv5', 'Aldose reductase'),
    ('ampc', '1l2s', 'Beta-lactamase'),
    ('andr', '2am9', 'Androgen Receptor'),
    ('aofb', '1s3b', 'Monoamine oxidase B'),
    ('bace1', '3l5d', 'Beta-secretase 1'),
    ('braf', '3d4q', 'Serine/threonine-protein kinase B-raf'),
    ('cah2', '1bcd', 'Carbonic anhydrase II'),
    ('casp3', '2cnk', 'Caspase-3'),
    ('cdk2', '1h00', 'Cyclin-dependent kinase 2'),
    ('comt', '3bwm', 'Catechol O-methyltransferase'),
    ('cp2c9', '1r9o', 'Cytochrome P450 2C9'),
    ('cp3a4', '3nxu', 'Cytochrome P450 3A4'),
    ('csf1r', '3krj', 'Macrophage colony stimulating factor receptor'),
    ('cxcr4', '3odu', 'C-X-C chemokine receptor type 4'),
    ('def', '1lru', 'Peptide deformylase'),
    ('dhi1', '3frj', '11-beta-hydroxysteroid dehydrogenase 1'),
    ('dpp4', '2i78', 'Dipeptidyl peptidase IV'),
    ('drd3', '3pbl', 'Dopamine D3 receptor'),
    ('dyr', '3nxo', 'Dihydrofolate reductase'),
    ('egfr', '2rgp', 'Epidermal growth factor receptor erbB1'),
    ('esr1', '1sj0', 'Estrogen receptor alpha'),
    ('esr2', '2fsz', 'Estrogen receptor beta'),
    ('fa10', '3kl6', 'Coagulation factor X'),
    ('fa7', '1w7x', 'Coagulation factor VII'),
    ('fabp4', '2nnq', 'Fatty acid binding protein adipocyte'),
    ('fak1', '3bz3', 'Focal adhesion kinase 1'),
    ('fgfr1', '3c4f', 'Fibroblast growth factor receptor 1'),
    ('fkb1a', '1j4h', 'FK506-binding protein 1A'),
    ('fnta', '3e37', 'Protein farnesyltransferase/geranylgeranyltransferase type I alpha subunit'),
    ('fpps', '1zw5', 'Farnesyl diphosphate synthase'),
    ('gcr', '3bqd', 'Glucocorticoid receptor'),
    ('glcm', '2v3f', 'Beta-glucocerebrosidase'),
    ('gria2', '3kgc', 'Glutamate receptor ionotropic, AMPA 2'),
    ('grik1', '1vso', 'Glutamate receptor ionotropic kainate 1'),
    ('hdac2', '3max', 'Histone deacetylase 2'),
    ('hdac8', '3f07', 'Histone deacetylase 8'),
    ('hivint', '3nf7', 'Human immunodeficiency virus type 1 integrase'),
    ('hivpr', '1xl2', 'Human immunodeficiency virus type 1 protease'),
    ('hivrt', '3lan', 'Human immunodeficiency virus type 1 reverse transcriptase'),
    ('hmdh', '3ccw', 'HMG-CoA reductase'),
    ('hs90a', '1uyg', 'Heat shock protein HSP 90-alpha'),
    ('hxk4', '3f9m', 'Hexokinase type IV'),
    ('igf1r', '2oj9', 'Insulin-like growth factor I receptor'),
    ('inha', '2h7l', 'Enoyl-[acyl-carrier-protein] reductase'),
    ('ital', '2ica', 'Leukocyte adhesion glycoprotein LFA-1 alpha'),
    ('jak2', '3lpb', 'Tyrosine-protein kinase JAK2'),
    ('kif11', '3cjo', 'Kinesin-like protein 1'),
    ('kit', '3g0e', 'Stem cell growth factor receptor'),
    ('kith', '2b8t', 'Thymidine kinase'),
    ('kpcb', '2i0e', 'Protein kinase C beta'),
    ('lck', '2of2', 'Tyrosine-protein kinase LCK'),
    ('lkha4', '3chp', 'Leukotriene A4 hydrolase'),
    ('mapk2', '3m2w', 'MAP kinase-activated protein kinase 2'),
    ('mcr', '2aa2', 'Mineralocorticoid receptor'),
    ('met', '3lq8', 'Hepatocyte growth factor receptor'),
    ('mk01', '2ojg', 'MAP kinase ERK2'),
    ('mk10', '2zdt', 'c-Jun N-terminal kinase 3'),
    ('mk14', '2qd9', 'MAP kinase p38 alpha'),
    ('mmp13', '830c', 'Matrix metalloproteinase 13'),
    ('mp2k1', '3eqh', 'Dual specificity mitogen-activated protein kinase kinase 1'),
    ('nos1', '1qw6', 'Nitric-oxide synthase, brain'),
    ('nram', '1b9v', 'Neuraminidase'),
    ('pa2ga', '1kvo', 'Phospholipase A2 group IIA'),
    ('parp1', '3l3m', 'Poly [ADP-ribose] polymerase-1'),
    ('pde5a', '1udt', 'Phosphodiesterase 5A'),
    ('pgh1', '2oyu', 'Cyclooxygenase-1'),
    ('pgh2', '3ln1', 'Cyclooxygenase-2'),
    ('plk1', '2owb', 'Serine/threonine-protein kinase PLK1'),
    ('pnph', '3bgs', 'Purine nucleoside phosphorylase'),
    ('ppara', '2p54', 'Peroxisome proliferator-activated receptor alpha'),
    ('ppard', '2znp', 'Peroxisome proliferator-activated receptor delta'),
    ('pparg', '2gtk', 'Peroxisome proliferator-activated receptor gamma'),
    ('prgr', '3kba', 'Progesterone receptor'),
    ('ptn1', '2azr', 'Protein-tyrosine phosphatase 1B'),
    ('pur2', '1njs', 'GAR transformylase'),
    ('pygm', '1c8k', 'Muscle glycogen phosphorylase'),
    ('pyrd', '1d3g', 'Dihydroorotate dehydrogenase'),
    ('reni', '3g6z', 'Renin'),
    ('rock1', '2etr', 'Rho-associated protein kinase 1'),
    ('rxra', '1mv9', 'Retinoid X receptor alpha'),
    ('sahh', '1li4', 'Adenosylhomocysteinase'),
    ('src', '3el8', 'Tyrosine-protein kinase SRC'),
    ('tgfr1', '3hmm', 'TGF-beta receptor type I'),
    ('thb', '1q4x', 'Thyroid hormone receptor beta-1'),
    ('thrb', '1ype', 'Thrombin'),
    ('try1', '2ayw', 'Trypsin I'),
    ('tryb1', '2zec', 'Tryptase beta-1'),
    ('tysy', '1syn', 'Thymidylate synthase'),
    ('urok', '1sqt', 'Urokinase-type plasminogen activator'),
    ('vgfr2', '2p2i', 'Vascular endothelial growth factor receptor 2'),
    ('wee1', '3biz', 'Serine/threonine-protein kinase WEE1'),
    ('xiap', '3hl5', 'Inhibitor of apoptosis protein 3'),
]

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
        'DUD-E',
        'QSAR',
        'Benchmark',
        'Molecular Docking',
    ],
    'verbose': 'DUD-E (Directory of Useful Decoys - Enhanced) Multi-Target Classification',
    'sources': [
        'https://dude.docking.org/',
        'https://doi.org/10.1021/jm300687e',  # Mysinger et al. 2012
        'http://jmc.acs.org/content/55/14/6582',
    ],
    'notes': [
        'This is a virtual screening benchmark dataset designed for evaluating molecular docking and similarity methods.',
        'Target vectors use tri-state encoding: 0 = not mentioned for that target, 1 = active against that target, 2 = decoy for that target.',
        'Decoys are specifically selected for each target (50:1 ratio) to match physicochemical properties while avoiding 2-D structural similarity to actives.',
        'A molecule can be active against some targets while serving as a decoy for others, capturing the per-target nature of the data.',
        'Active compounds have measured affinities ≤1 μM against their targets, extracted from ChEMBL09 database.',
    ],
    'target_descriptions': {
        str(idx): f'{gene.upper()} - {name} (PDB: {pdb})'
        for idx, (gene, pdb, name) in enumerate(TARGETS)
    },
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


def get_cache_dir() -> str:
    """
    Get the path to the cache directory for DUD-E downloads.
    Uses /tmp with a named path so it persists between runs during system lifetime
    but gets cleared on system restart.

    Returns:
        Path to the cache directory
    """
    cache_dir = os.path.join(tempfile.gettempdir(), 'dud_e_cache')
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def download_target(target_name: str, cache_dir: str, e: Experiment) -> str:
    """
    Download and extract a single DUD-E target.

    Args:
        target_name: The gene name of the target (e.g., 'aa2ar')
        cache_dir: Directory where to cache the downloaded files
        e: Experiment instance for logging

    Returns:
        Path to the extracted target directory
    """
    target_dir = os.path.join(cache_dir, target_name)

    # Check if already downloaded and extracted
    if os.path.exists(target_dir) and os.path.exists(os.path.join(target_dir, 'actives_final.ism')):
        return target_dir

    # Download the target tar.gz file
    url = f'https://dude.docking.org/targets/{target_name}/{target_name}.tar.gz'
    e.log(f'  Downloading {target_name}...')

    try:
        with FileDownloadSource(url, verbose=False, ssl_verify=False) as source:
            tar_path = source.fetch()

            # Extract tar.gz file
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(cache_dir)

            e.log(f'    ✓ Downloaded and extracted {target_name}')

    except Exception as ex:
        e.log(f'    ! Error downloading {target_name}: {ex}')
        return None

    return target_dir


def parse_ism_file(file_path: str) -> list:
    """
    Parse a DUD-E .ism file.

    .ism files are space-separated with format: SMILES ID ChEMBL_ID
    No header line.

    Args:
        file_path: Path to the .ism file

    Returns:
        List of tuples: (smiles, molecule_id, chembl_id)
    """
    molecules = []

    if not os.path.exists(file_path):
        return molecules

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Split by whitespace (spaces)
            parts = line.split()
            if len(parts) >= 2:
                smiles = parts[0]
                mol_id = parts[1]
                chembl_id = parts[2] if len(parts) >= 3 else ''
                molecules.append((smiles, mol_id, chembl_id))

    return molecules


@experiment.hook('load_dataset', default=False, replace=True)
def load_dataset(e: Experiment) -> Dict[int, dict]:
    """
    Download and process the DUD-E dataset for multi-target classification.

    This function:
    1. Downloads all 102 DUD-E targets from https://dude.docking.org/
    2. Caches downloads to /tmp/dud_e_cache (persists between runs)
    3. Parses actives_final.ism and decoys_final.ism for each target
    4. Groups molecules by SMILES to get unique structures
    5. For each unique molecule, creates a 102-dimensional tri-state target vector
    6. Vector encoding: 0 (not mentioned), 1 (active), 2 (decoy)
    7. A molecule can be active for some targets and a decoy for others
    8. Returns dictionary with per-target tri-state target vectors
    """

    e.log('=' * 80)
    e.log('DOWNLOADING DUD-E TARGETS')
    e.log('=' * 80)

    cache_dir = get_cache_dir()
    e.log(f'Cache directory: {cache_dir}')
    e.log(f'Total targets to download: {len(TARGETS)}')

    # Download all targets
    target_paths = {}
    for gene_name, pdb_id, full_name in TARGETS:
        target_dir = download_target(gene_name, cache_dir, e)
        if target_dir:
            target_paths[gene_name] = target_dir

    e.log(f'\n  ✓ Successfully downloaded {len(target_paths)}/{len(TARGETS)} targets')

    # == BUILD TARGET INDEX MAPPING ==
    e.log('')
    e.log('=' * 80)
    e.log('BUILDING TARGET INDEX MAPPING')
    e.log('=' * 80)

    target_list = [gene_name for gene_name, _, _ in TARGETS]
    target_to_index = {target_key: idx for idx, target_key in enumerate(target_list)}
    num_targets = len(target_list)

    e.log(f'Total targets: {num_targets}')
    e.log(f'Target vector dimension: {num_targets}')
    e.log(f'Target vector encoding: 0 = not mentioned, 1 = active, 2 = decoy')

    # == PARSE ACTIVE COMPOUNDS ==
    e.log('')
    e.log('=' * 80)
    e.log('PARSING ACTIVE COMPOUNDS')
    e.log('=' * 80)

    molecule_target_data = {}  # SMILES -> {'active_targets': [target_keys], 'decoy_targets': [target_keys], 'external_ids': [...]}

    total_active_pairs = 0
    for gene_name in target_list:
        if gene_name not in target_paths:
            e.log(f'  ! Skipping {gene_name} (not downloaded)')
            continue

        target_dir = target_paths[gene_name]
        actives_file = os.path.join(target_dir, 'actives_final.ism')

        molecules = parse_ism_file(actives_file)

        for smiles, mol_id, chembl_id in molecules:
            if smiles not in molecule_target_data:
                molecule_target_data[smiles] = {
                    'active_targets': [],
                    'decoy_targets': [],
                    'external_ids': [],
                }

            molecule_target_data[smiles]['active_targets'].append(gene_name)

            # Store external IDs
            if mol_id and mol_id not in molecule_target_data[smiles]['external_ids']:
                molecule_target_data[smiles]['external_ids'].append(mol_id)
            if chembl_id and chembl_id not in molecule_target_data[smiles]['external_ids']:
                molecule_target_data[smiles]['external_ids'].append(chembl_id)

            total_active_pairs += 1

        e.log(f'  ✓ {gene_name}: {len(molecules)} actives')

    total_molecules_with_actives = len(molecule_target_data)
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

    for gene_name in target_list:
        if gene_name not in target_paths:
            continue

        target_dir = target_paths[gene_name]
        decoys_file = os.path.join(target_dir, 'decoys_final.ism')

        molecules = parse_ism_file(decoys_file)

        for smiles, mol_id, chembl_id in molecules:
            if smiles not in molecule_target_data:
                # New molecule only seen as decoy
                molecule_target_data[smiles] = {
                    'active_targets': [],
                    'decoy_targets': [],
                    'external_ids': [],
                }
                new_molecules_from_decoys += 1

            # Add this target as a decoy target
            molecule_target_data[smiles]['decoy_targets'].append(gene_name)

            # Store external IDs
            if mol_id and mol_id not in molecule_target_data[smiles]['external_ids']:
                molecule_target_data[smiles]['external_ids'].append(mol_id)
            if chembl_id and chembl_id not in molecule_target_data[smiles]['external_ids']:
                molecule_target_data[smiles]['external_ids'].append(chembl_id)

            total_decoy_pairs += 1

        e.log(f'  ✓ {gene_name}: {len(molecules)} decoys')

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
    - External IDs (DUD-E IDs, ChEMBL IDs, etc.)
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
