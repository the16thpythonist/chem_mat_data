"""
This experiment creates the processed graph dataset for the **Kulik group spin-splitting**
dataset — 2,125 computationally generated octahedral transition metal complexes (Co, Cr,
Fe, Mn in oxidation states +2/+3) with DFT-computed spin-state energetics.

The raw data is downloaded from Zenodo (Meyer, Chu & Kulik 2024). Complex structures are
encoded in a molSimplify naming convention: ``metal_oxstate_lig1_lig2_lig3_lig4_lig5_lig6``
where each ligand identifier is either a SMILES-like string (e.g., ``[OH-]``, ``[NH2]-[NH2]``)
or a plain name (e.g., ``ammonia``, ``pyr``). A built-in dictionary maps plain names to SMILES
using molSimplify's ligand library.

**Source**: Meyer, Chu & Kulik, *J. Chem. Phys.* 2024; original dataset: Nandy et al.,
*Ind. Eng. Chem. Res.* 2018. DOI: 10.1021/acs.iecr.8b04015.

**Targets** (7 regression properties):
    Spin splitting energy (kcal/mol), HOMO energies (low/high spin, eV),
    LUMO energies (low/high spin, eV), HOMO-LUMO gaps (low/high spin, eV).

**Usage**::

    python create_graph_datasets__kulik_spin.py

Results are written to ``results/create_graph_datasets__kulik_spin/debug/``.
"""
import os
import json
import gzip
import time
import shutil
import zipfile
import datetime
import multiprocessing
from typing import Dict, List, Optional, Tuple

import yaml
import msgpack
import requests
import numpy as np
import rdkit.Chem as Chem
import pandas as pd
from rich.pretty import pprint
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import chem_mat_data._typing as typc
from chem_mat_data.tmc_processing import MetalOrganicProcessing
from chem_mat_data.data import default, ext_hook, save_graphs


# == SOURCE PARAMETERS ==

# :param DOWNLOAD_URL:
#       Zenodo archive containing training, validation, and test splits.
#       Approximately 229 MB. The CSV files are in ``many_body_ml copy/data/``.
DOWNLOAD_URL: str = 'https://zenodo.org/api/records/13331586/files/many_body_ml.zip/content'

# :param DATA_CSV_PATHS:
#       Paths within the zip for the 4 data splits. All are combined into a single
#       dataset — the original train/val/test split is not preserved.
DATA_CSV_PATHS: List[str] = [
    'many_body_ml copy/data/training_data.csv',
    'many_body_ml copy/data/validation_data.csv',
    'many_body_ml copy/data/composition_test_data.csv',
    'many_body_ml copy/data/ligand_test_data.csv',
]

TARGET_COLUMNS: List[str] = [
    'spin_splitting_kcal/mol',
    'energetic_homo_ls_eV',
    'energetic_homo_hs_eV',
    'energetic_lumo_ls_eV',
    'energetic_lumo_hs_eV',
    'energetic_gap_ls_eV',
    'energetic_gap_hs_eV',
]

DATASET_TYPE: str = 'regression'

DESCRIPTION: str = (
    'Graph dataset of 2,125 computationally generated octahedral transition metal '
    'complexes (Co, Cr, Fe, Mn) with 7 DFT-computed spin-state properties: spin splitting '
    'energy, HOMO/LUMO energies and HOMO-LUMO gaps for both low-spin and high-spin states. '
    'From Meyer, Chu & Kulik (J. Chem. Phys. 2024), expanding the original Nandy et al. '
    '(IECR 2018) dataset. Processed using MetalOrganicProcessing with ligand SMILES '
    'extracted via a molSimplify ligand dictionary.'
)

METADATA: dict = {
    'category': 'tmc',
    'min_version': '1.7.0',
    'tags': ['Molecules', 'TransitionMetals', 'TMC', 'SpinSplitting', 'DFT',
             'Octahedral', 'OpenShell'],
    'sources': [
        'https://doi.org/10.1021/acs.iecr.8b04015',
        'https://doi.org/10.5281/zenodo.13331586',
    ],
    'verbose': 'Kulik Octahedral TMC Spin-Splitting Properties',
    'target_descriptions': {
        str(i): col for i, col in enumerate(TARGET_COLUMNS)
    },
}

# == PROCESSING PARAMETERS ==

DATASET_NAME: str = 'kulik_spin'
COMPRESS: bool = True
MAX_ELEMENTS: Optional[int] = None

# == EXPERIMENT PARAMETERS ==

__DEBUG__ = True
__TESTING__ = False

experiment = Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)


# ======================================================================================
# MOLSIMPLIFY LIGAND NAME → SMILES MAPPING
# ======================================================================================
# Maps the 30 plain-text ligand names used in the dataset's name column to
# (SMILES, connecting_atom_indices) tuples. The connecting atom indices correspond
# to the canonical SMILES as produced by RDKit.
#
# SMILES and connecting atoms are sourced from molSimplify's ligands.dict and
# verified against RDKit parsing. Carbon-donor ligands (carbonyl, isocyanides)
# have explicit index 0 since the heuristic would incorrectly pick a heteroatom.

LIGAND_NAME_MAP: Dict[str, Tuple[str, List[int]]] = {
    'acac':               ('CC(=O)/C=C(/C)[O-]', None),   # bidentate, infer
    'acetonitrile':       ('CC#N', [2]),                    # N-donor
    'aminomethylene':     ('[CH]=[NH]', [1]),               # N-donor
    'ammonia':            ('N', [0]),                        # N-donor
    'benzisc':            ('[C-]#[N+]Cc1ccccc1', [0]),      # isocyanide C-donor
    'bipy':               ('c1ccc(-c2ccccn2)nc1', None),    # bidentate, infer
    'carbonyl':           ('C#O', [0]),                      # CO, C-donor
    'chloride':           ('[Cl-]', [0]),
    'cyanide':            ('[C-]#N', [0]),                   # C-donor
    'diazole':            ('c1c[nH]cn1', None),              # infer
    'en':                 ('NCCN', None),                    # bidentate, infer
    'fluoride':           ('[F-]', [0]),
    'furan':              ('c1ccoc1', None),                  # O-donor, infer
    'hydrogencyanide':    ('C#N', [1]),                      # HCN, N-donor
    'hydrogenisocyanide': ('[C-]#[N+]', [0]),                # HNC, C-donor
    'hydrogensulfide':    ('S', [0]),                         # H2S
    'imidazolidinone':    ('O=C1NCCN1', None),               # infer
    'iodide':             ('[I-]', [0]),
    'methanol':           ('CO', [1]),                        # O-donor
    'methylamine':        ('CN', [1]),                        # N-donor
    'misc':               ('[C-]#[N+]C', [0]),               # methyl isocyanide
    'ox':                 ('O=C([O-])C(=O)[O-]', None),      # oxalate bidentate, infer
    'phen':               ('c1cnc2c(c1)ccc1cccnc12', None),  # bidentate, infer
    'phosphine':          ('P', [0]),
    'phosphorine':        ('C1=CCPC=C1', None),              # P-donor, infer
    'pisc':               ('[C-]#[N+]c1ccc(C(C)(C)C)cc1', [0]),  # isocyanide C-donor
    'porphyrin':          None,                               # tetradentate — skip
    'pyr':                ('c1ccncc1', None),                 # pyridine, infer
    'sulfide':            ('[SH-]', [0]),
    'water':              ('O', [0]),
}


# ======================================================================================
# DONOR ATOM HEURISTIC
# ======================================================================================

DONOR_ELEMENTS: set = {7, 8, 15, 16, 34, 9, 17, 35, 53}


def infer_connecting_atoms(ligand_smiles: str) -> List[int]:
    """
    Heuristically determine donor atoms in a ligand SMILES.
    """
    mol = Chem.MolFromSmiles(ligand_smiles)
    if mol is None:
        mol = Chem.MolFromSmiles(ligand_smiles, sanitize=False)
        if mol is not None:
            try:
                Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
            except Exception:
                pass
    if mol is None:
        return [0]

    if mol.GetNumAtoms() == 1:
        return [0]

    candidates = []
    charged_candidates = []
    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        charge = atom.GetFormalCharge()
        if charge < 0:
            charged_candidates.append(atom.GetIdx())
        elif z in DONOR_ELEMENTS:
            candidates.append(atom.GetIdx())

    if charged_candidates:
        return charged_candidates
    if candidates:
        return candidates
    return [0]


def resolve_ligand(name: str) -> Optional[Tuple[str, List[int]]]:
    """
    Resolve a ligand identifier (plain name or SMILES-like string) to a
    ``(SMILES, connecting_atom_indices)`` tuple.

    For plain names, uses :data:`LIGAND_NAME_MAP`. For SMILES-like strings
    (containing brackets), parses them directly. Returns ``None`` for
    unresolvable ligands.
    """
    # Check plain name mapping first
    if name in LIGAND_NAME_MAP:
        entry = LIGAND_NAME_MAP[name]
        if entry is None:
            return None  # e.g., porphyrin — skip
        smiles, conn = entry
        if conn is None:
            conn = infer_connecting_atoms(smiles)
        return smiles, conn

    # Try parsing as SMILES directly (SMILES-like ligand names)
    mol = Chem.MolFromSmiles(name)
    if mol is None:
        mol = Chem.MolFromSmiles(name, sanitize=False)
        if mol is not None:
            try:
                Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
            except Exception:
                pass
    if mol is None:
        return None

    smiles = Chem.MolToSmiles(mol)
    conn = infer_connecting_atoms(smiles)
    return smiles, conn


def parse_complex_name(
    name: str,
    eq_dent: int,
    ax_dent: int,
) -> Optional[Tuple[str, int, List[str], List[List[int]]]]:
    """
    Parse a molSimplify complex name into decomposed TMC format.

    The name format is ``metal_oxstate_pos0_pos1_pos2_pos3_pos4_pos5`` where
    positions 0-3 are equatorial and 4-5 are axial. For bidentate ligands,
    adjacent positions share the same ligand (grouped by denticity).

    :returns: ``(metal, oxidation_state, ligand_smiles, connecting_atom_indices)``
        or ``None`` on failure.
    """
    parts = name.split('_')
    if len(parts) < 8:
        return None

    metal = parts[0].capitalize()  # 'fe' → 'Fe'
    oxidation_state = int(parts[1])
    positions = parts[2:]  # 6 position strings

    if len(positions) != 6:
        return None

    eq_positions = positions[:4]
    ax_positions = positions[4:]

    ligand_smiles: List[str] = []
    connecting_atom_indices: List[List[int]] = []

    # Process equatorial positions, grouped by denticity
    for i in range(0, 4, eq_dent):
        lig_name = eq_positions[i]
        result = resolve_ligand(lig_name)
        if result is None:
            return None
        smi, conn = result

        if eq_dent == 1:
            # Monodentate: use first connecting atom only
            ligand_smiles.append(smi)
            connecting_atom_indices.append([conn[0]])
        else:
            # Polydentate: use up to eq_dent connecting atoms
            ligand_smiles.append(smi)
            connecting_atom_indices.append(conn[:eq_dent])

    # Process axial positions, grouped by denticity
    for i in range(0, 2, ax_dent):
        lig_name = ax_positions[i]
        result = resolve_ligand(lig_name)
        if result is None:
            return None
        smi, conn = result

        if ax_dent == 1:
            ligand_smiles.append(smi)
            connecting_atom_indices.append([conn[0]])
        else:
            ligand_smiles.append(smi)
            connecting_atom_indices.append(conn[:ax_dent])

    return metal, oxidation_state, ligand_smiles, connecting_atom_indices


# ======================================================================================
# PROCESSING WORKER
# ======================================================================================

class TMCProcessingWorker(multiprocessing.Process):

    def __init__(self,
                 input_queue: multiprocessing.Queue,
                 output_queue: multiprocessing.Queue,
                 ):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.processing = MetalOrganicProcessing()

    def run(self):
        for data in iter(self.input_queue.get, None):
            try:
                graph: typc.GraphDict = self.processing.process(
                    metal=data['metal'],
                    ligand_smiles=data['ligand_smiles'],
                    connecting_atom_indices=data['connecting_atom_indices'],
                    oxidation_state=data.get('oxidation_state', 0),
                    total_charge=data.get('total_charge', 0),
                    spin_multiplicity=data.get('spin_multiplicity', 1),
                )
                graph['graph_labels'] = np.array(data['targets'])

                experiment.apply_hook(
                    'add_graph_metadata',
                    data=data,
                    graph=graph,
                )

            except Exception as exc:
                metal = data.get('metal', '?')
                cid = data.get('complex_id', '?')
                print(f' ! error processing {cid} ({metal}) - {exc.__class__.__name__}: {exc}')
                graph = None

            graph_encoded = msgpack.packb(graph, default=default)
            self.output_queue.put(graph_encoded)


# ======================================================================================
# HOOKS
# ======================================================================================

@experiment.hook('add_graph_metadata', default=True, replace=False)
def add_graph_metadata(e: Experiment, data: dict, graph: typc.GraphDict) -> None:
    if graph is not None:
        graph['graph_id'] = data.get('complex_id', '')
        graph['graph_metal'] = data.get('metal', '')


@experiment.hook('load_dataset', default=True, replace=False)
def load_dataset(e: Experiment) -> Dict[int, dict]:
    """
    Download the Zenodo archive, load all 4 splits, parse molSimplify complex names
    into decomposed TMC format, and collect regression targets.
    """
    # -- Download and cache --
    zip_path = os.path.join(e.path, 'many_body_ml.zip')
    if not os.path.exists(zip_path):
        e.log(f'downloading Zenodo archive (~229 MB) ...')
        response = requests.get(e.DOWNLOAD_URL, timeout=600, stream=True)
        response.raise_for_status()
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        e.log(f'downloaded {os.path.getsize(zip_path) / 1024 / 1024:.1f} MB')
    else:
        e.log(f'using cached archive')

    # -- Load and combine all splits --
    e.log('loading data splits...')
    dfs = []
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for csv_path in e.DATA_CSV_PATHS:
            with zf.open(csv_path) as csv_file:
                df = pd.read_csv(csv_file)
                dfs.append(df)
                e.log(f'  {os.path.basename(csv_path)}: {len(df)} rows')

    df = pd.concat(dfs, ignore_index=True)
    e.log(f'combined: {len(df)} rows')

    if e.MAX_ELEMENTS is not None:
        df = df.head(e.MAX_ELEMENTS)
        e.log(f'limited to {e.MAX_ELEMENTS} rows')

    # -- Parse each complex --
    dataset: Dict[int, dict] = {}
    skipped_parse = 0
    skipped_porphyrin = 0

    for idx, row in df.iterrows():
        name = row['name']

        # Get denticity info
        eq_dent = int(row.get('misc-dent-eq', 1))
        ax_dent = int(row.get('misc-dent-ax', 1))

        # Skip tetradentate (porphyrin) — too complex
        if eq_dent > 2 or ax_dent > 2:
            skipped_porphyrin += 1
            continue

        result = parse_complex_name(name, eq_dent, ax_dent)
        if result is None:
            skipped_parse += 1
            continue

        metal, oxidation_state, ligand_smiles, connecting_atom_indices = result

        # Collect targets
        targets = []
        has_nan = False
        for col in e.TARGET_COLUMNS:
            val = row.get(col, None)
            if pd.isna(val):
                has_nan = True
                break
            targets.append(float(val))

        if has_nan:
            skipped_parse += 1
            continue

        dataset[len(dataset)] = {
            'complex_id': name,
            'metal': metal,
            'ligand_smiles': ligand_smiles,
            'connecting_atom_indices': connecting_atom_indices,
            'total_charge': 0,
            'oxidation_state': oxidation_state,
            'spin_multiplicity': int(row.get('high_spin', 1)),
            'targets': targets,
        }

    e.log(f'parsed {len(dataset)} complexes '
          f'({skipped_parse} parse failures, '
          f'{skipped_porphyrin} tetradentate skipped)')
    return dataset


@experiment.hook('save_csv', default=True, replace=False)
def save_csv(e: Experiment, dataset: Dict[int, dict]) -> None:
    e.log('saving CSV...')

    rows = []
    for index, data in dataset.items():
        row = {
            'complex_id': data['complex_id'],
            'metal': data['metal'],
            'ligand_smiles': json.dumps(data['ligand_smiles']),
            'connecting_atom_indices': json.dumps(data['connecting_atom_indices']),
            'total_charge': data['total_charge'],
            'oxidation_state': data['oxidation_state'],
            'spin_multiplicity': data['spin_multiplicity'],
        }
        for i, col in enumerate(e.TARGET_COLUMNS):
            row[col] = data['targets'][i]
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(e.path, f'{e.DATASET_NAME}.csv')
    df.to_csv(csv_path, index=False)

    gz_path = csv_path + '.gz'
    with open(csv_path, 'rb') as f_in, gzip.open(gz_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

    e.log(f'saved CSV ({len(rows)} rows)')


# ======================================================================================
# MAIN EXPERIMENT
# ======================================================================================

@experiment
def experiment(e: Experiment):

    e.log('starting Kulik spin-splitting processing experiment...')
    e.log_parameters()

    e.log('creating TMC processing workers...')
    input_queue = multiprocessing.Queue(maxsize=100)
    output_queue = multiprocessing.Queue(maxsize=100)
    workers = []
    num_workers = os.cpu_count()
    for _ in range(num_workers):
        worker = TMCProcessingWorker(
            input_queue=input_queue,
            output_queue=output_queue,
        )
        worker.start()
        workers.append(worker)

    e.log(f'started {num_workers} workers')

    try:
        e.log('loading dataset...')
        dataset: Dict[int, dict] = e.apply_hook('load_dataset')
        num_elements = len(dataset)
        e.log(f'loaded {num_elements} complexes')

        e.apply_hook('save_csv', dataset=dataset)

        if e.__TESTING__:
            e.log('testing mode: limiting to 50 elements')
            num_elements = min(num_elements, 50)
            dataset = dict(list(dataset.items())[:num_elements])

        e.log('processing dataset...')
        indices = list(dataset.keys())
        num_indices = len(indices)
        graphs = []

        start_time = time.time()
        count = 0
        prev_count = 0

        while count < num_indices:

            while not input_queue.full() and len(indices) != 0:
                index = indices.pop()
                data = dataset[index]
                input_queue.put(data)

            while not output_queue.empty():
                graph_encoded = output_queue.get()
                graph = msgpack.unpackb(graph_encoded, ext_hook=ext_hook)
                if graph:
                    graphs.append(graph)
                count += 1

            if count % 500 == 0 and count != prev_count:
                prev_count = count
                time_passed = time.time() - start_time
                num_remaining = num_elements - (count + 1)
                time_per_element = time_passed / (count + 1)
                time_remaining = time_per_element * num_remaining
                eta = datetime.datetime.now() + datetime.timedelta(seconds=time_remaining)
                e.log(f' * {count:05d}/{num_elements} done'
                      f' - time passed: {time_passed / 60:.2f}m'
                      f' - eta: {eta:%a %d.%m %H:%M}')

        end_time = time.time()
        duration = end_time - start_time
        e.log(f'finished processing {len(graphs)} graphs in {duration:.1f}s')

    finally:
        e.log('stopping workers...')
        for worker in workers:
            input_queue.put(None)
            worker.terminate()
            worker.join()

        del input_queue
        del output_queue

    e.log(f'saving {len(graphs)} graphs to mpack...')
    dataset_path = os.path.join(e.path, e.DATASET_NAME + '.mpack')
    save_graphs(graphs, dataset_path)

    file_size = os.path.getsize(dataset_path)
    e.log(f'wrote mpack: {file_size / 1024 / 1024:.1f} MB')

    if e.COMPRESS:
        e.log('compressing...')
        compressed_path = os.path.join(e.path, e.DATASET_NAME + '.mpack.gz')
        with open(dataset_path, 'rb') as f_in, gzip.open(compressed_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

        compressed_size = os.path.getsize(compressed_path)
        e.log(f'compressed: {compressed_size / 1024 / 1024:.1f} MB')

    e.log('saving metadata...')
    example_graph = graphs[0]
    pprint(example_graph)

    metadata: dict = {
        'compounds': len(graphs),
        'targets': len(example_graph['graph_labels']),
        'target_type': [e.DATASET_TYPE],
        'description': e.DESCRIPTION,
        'raw': ['csv'],
        'sources': [],
    }
    metadata.update(e.METADATA)

    metadata_path = os.path.join(e.path, 'metadata.yml')
    with open(metadata_path, 'w') as f:
        yaml.dump(metadata, f)

    e.log(f'saved metadata @ {metadata_path}')
    e.log('done!')


experiment.run_if_main()
