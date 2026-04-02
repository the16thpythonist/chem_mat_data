"""
This experiment creates the processed graph dataset for **tmQMg*** — a dataset of ~74,000
mononuclear transition metal complexes with TD-DFT excited-state properties (UV-Vis-NIR
absorption wavelengths, oscillator strengths, and solvatochromic shifts).

The raw excited-state data is downloaded from the tmQMg* GitHub repository. Since tmQMg*
does not contain SMILES, the script also downloads the ELECTRUM tmQMg CSV (which shares
the same CSD identifiers) to obtain the decomposed metal + ligand SMILES representation.

**Source**: Kneiding et al., *Digital Discovery* 2024 (tmQMg* paper);
Kneiding et al., *Digital Discovery* 2023 (tmQMg); ELECTRUM repo (Orsi & Frei 2025).

**Targets** (26 regression properties):
    Ground-state: HOMO-LUMO gap, dipole moment, metal charge (gas phase).
    Excited-state: first 10 excitation wavelengths and oscillator strengths (gas phase).
    UV band max: wavelength and oscillator strength at UV absorption maximum.
    Solvatochromism: wavelength shift between gas phase and acetone solvent.

**Usage**::

    python create_graph_datasets__tmqmg_star.py

Results are written to ``results/create_graph_datasets__tmqmg_star/debug/``.
"""
import os
import json
import gzip
import time
import shutil
import datetime
import multiprocessing
from typing import Dict, List, Optional

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
from chem_mat_data.tmc_processing import MetalOrganicProcessing, TRANSITION_METAL_ATOMIC_NUMBERS
from chem_mat_data.data import default, ext_hook, save_graphs


# == SOURCE PARAMETERS ==

# :param DOWNLOAD_URL_STAR:
#       The URL for the tmQMg* excited-state CSV. Contains 166 columns including
#       30 excitation wavelengths, 30 oscillator strengths, band maxima, NTO data,
#       and solvatochromism properties. Approximately 82 MB.
DOWNLOAD_URL_STAR: str = 'https://raw.githubusercontent.com/uiocompcat/tmQMg_star/main/tmQMg*.csv'

# :param DOWNLOAD_URL_TMQMG:
#       The ELECTRUM tmQMg CSV providing the decomposed metal + ligand SMILES for
#       each complex. Joined with tmQMg* by the CSD identifier (``id`` column).
DOWNLOAD_URL_TMQMG: str = 'https://raw.githubusercontent.com/TheFreiLab/electrum_val/main/datasets/tmQMg.csv'

# :param TARGET_COLUMNS:
#       26 regression targets covering ground-state, excited-state, and solvatochromic
#       properties. All are ~100% populated across the 74,000 complexes.
TARGET_COLUMNS: List[str] = [
    # Ground-state properties (3)
    'homo_lumo_gap_gasphase',
    'dipole_moment_gasphase',
    'metal_charge_gasphase',
    # First 10 excitation wavelengths in nm (10)
    'lambda_1_gasphase',
    'lambda_2_gasphase',
    'lambda_3_gasphase',
    'lambda_4_gasphase',
    'lambda_5_gasphase',
    'lambda_6_gasphase',
    'lambda_7_gasphase',
    'lambda_8_gasphase',
    'lambda_9_gasphase',
    'lambda_10_gasphase',
    # First 10 oscillator strengths (10)
    'f_1_gasphase',
    'f_2_gasphase',
    'f_3_gasphase',
    'f_4_gasphase',
    'f_5_gasphase',
    'f_6_gasphase',
    'f_7_gasphase',
    'f_8_gasphase',
    'f_9_gasphase',
    'f_10_gasphase',
    # UV band maximum (2)
    'lambda_max_uv_gasphase',
    'f_max_uv_gasphase',
    # Solvatochromic shift (1)
    'lambda_delta',
]

DATASET_TYPE: str = 'regression'

DESCRIPTION: str = (
    'Dataset of ~74,000 mononuclear transition metal complexes from tmQMg* '
    '(Kneiding et al., Digital Discovery 2024) with 26 TD-DFT excited-state properties. '
    'Targets include ground-state HOMO-LUMO gap, dipole moment, and metal charge; '
    'the first 10 excitation wavelengths and oscillator strengths; UV absorption band '
    'maximum; and gas-to-acetone solvatochromic shifts. Useful for predicting '
    'photophysical and spectroscopic properties of metal complexes.'
)

METADATA: dict = {
    'category': 'tmc',
    'min_version': '1.7.0',
    'tags': ['Molecules', 'TransitionMetals', 'TMC', 'ExcitedStates', 'TDDFT', 'UVVis'],
    'sources': [
        'https://doi.org/10.1039/D4DD00216D',
        'https://github.com/uiocompcat/tmQMg_star',
        'https://github.com/TheFreiLab/electrum_val',
    ],
    'verbose': 'tmQMg* Transition Metal Complex Excited-State Properties',
    'target_descriptions': {
        str(i): col for i, col in enumerate(TARGET_COLUMNS)
    },
}

# == PROCESSING PARAMETERS ==

DATASET_NAME: str = 'tmqmg_star'
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
# DONOR ATOM HEURISTIC
# ======================================================================================

DONOR_ELEMENTS: set = {
    7,   # N
    8,   # O
    15,  # P
    16,  # S
    34,  # Se
    9,   # F
    17,  # Cl
    35,  # Br
    53,  # I
}


def infer_connecting_atoms(ligand_smiles: str) -> List[int]:
    """
    Heuristically determine which atoms in a ligand SMILES are donor atoms.

    :param ligand_smiles: SMILES string of a single ligand fragment.
    :returns: List of 0-based atom indices of inferred donor atoms.
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

    num_atoms = mol.GetNumAtoms()
    if num_atoms == 1:
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
    Download both the tmQMg* excited-state CSV and the ELECTRUM tmQMg CSV, join them
    by CSD identifier, and parse into the decomposed TMC format.

    The tmQMg* CSV contains excited-state properties but no SMILES. The ELECTRUM tmQMg
    CSV provides the decomposed metal + ligand SMILES. The join is by the ``id`` column
    (CSD refcode) which is shared between both datasets.
    """
    # -- Download tmQMg* excited-state data --
    star_csv_path = os.path.join(e.path, 'tmqmg_star_raw.csv')
    if not os.path.exists(star_csv_path):
        e.log(f'downloading tmQMg* dataset from {e.DOWNLOAD_URL_STAR} ...')
        response = requests.get(e.DOWNLOAD_URL_STAR, timeout=600)
        response.raise_for_status()
        with open(star_csv_path, 'w') as f:
            f.write(response.text)
        e.log(f'downloaded {len(response.content) / 1024 / 1024:.1f} MB')
    else:
        e.log(f'using cached tmQMg* CSV at {star_csv_path}')

    # -- Download ELECTRUM tmQMg for SMILES --
    tmqmg_csv_path = os.path.join(e.path, 'tmqmg_smiles.csv')
    if not os.path.exists(tmqmg_csv_path):
        e.log(f'downloading tmQMg SMILES from {e.DOWNLOAD_URL_TMQMG} ...')
        response = requests.get(e.DOWNLOAD_URL_TMQMG, timeout=120)
        response.raise_for_status()
        with open(tmqmg_csv_path, 'w') as f:
            f.write(response.text)
        e.log(f'downloaded {len(response.content) / 1024 / 1024:.1f} MB')
    else:
        e.log(f'using cached tmQMg CSV at {tmqmg_csv_path}')

    # -- Load and merge --
    e.log('parsing datasets...')
    df_star = pd.read_csv(star_csv_path)
    df_tmqmg = pd.read_csv(tmqmg_csv_path)

    # Join by CSD identifier
    df_merged = df_star.merge(df_tmqmg[['id', 'Metal', 'charge', 'LigandSmiles']],
                              on='id', how='inner')
    e.log(f'merged {len(df_merged)} complexes '
          f'(tmQMg*: {len(df_star)}, tmQMg: {len(df_tmqmg)})')

    if e.MAX_ELEMENTS is not None:
        df_merged = df_merged.head(e.MAX_ELEMENTS)
        e.log(f'limited to {e.MAX_ELEMENTS} complexes')

    dataset: Dict[int, dict] = {}
    skipped = 0

    for idx, row in df_merged.iterrows():
        metal = row['Metal']
        ligand_smiles_raw = str(row['LigandSmiles'])
        charge = int(row['charge'])

        ligand_smiles = ligand_smiles_raw.split('.')

        connecting_atom_indices = []
        valid = True
        for lig_smi in ligand_smiles:
            conn = infer_connecting_atoms(lig_smi)
            if not conn:
                valid = False
                break
            connecting_atom_indices.append(conn)

        if not valid:
            skipped += 1
            continue

        # Collect target values, skip rows with NaN in any target column
        targets = []
        has_nan = False
        for col in e.TARGET_COLUMNS:
            val = row.get(col, None)
            if pd.isna(val):
                has_nan = True
                break
            targets.append(float(val))

        if has_nan:
            skipped += 1
            continue

        dataset[len(dataset)] = {
            'complex_id': str(row['id']),
            'metal': metal,
            'ligand_smiles': ligand_smiles,
            'connecting_atom_indices': connecting_atom_indices,
            'total_charge': charge,
            'oxidation_state': 0,
            'spin_multiplicity': 1,
            'targets': targets,
            'ligand_smiles_raw': ligand_smiles_raw,
        }

    e.log(f'parsed {len(dataset)} complexes ({skipped} skipped)')
    return dataset


@experiment.hook('save_csv', default=True, replace=False)
def save_csv(e: Experiment, dataset: Dict[int, dict]) -> None:
    """
    Save the dataset as a CSV in decomposed TMC format with JSON-encoded list columns.
    """
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

    e.log('starting tmQMg* processing experiment...')
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

            if count % 1000 == 0 and count != prev_count:
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
