"""
This experiment creates the processed graph dataset for the **ELECTRUM Oxidation State**
dataset — a classification benchmark of 39,166 mononuclear transition metal complexes
labeled with the metal's formal oxidation state (7 classes: 0, +1, +2, +3, +4, +5, +6).

The raw data is downloaded from the ELECTRUM validation repository on GitHub. Each complex
is represented in a decomposed format: a metal element symbol and dot-separated ligand
SMILES strings. Connecting atom indices are inferred heuristically. Unlike the coordination
number dataset, this one also provides the actual oxidation state, which is passed to
``MetalOrganicProcessing`` for more accurate d-electron count encoding.

**Source**: Orsi & Frei, *Digital Discovery* 2025, 4, 3567 (ELECTRUM paper).

**Target**: Multi-class classification with 7 oxidation state classes (0, +1, +2, +3, +4, +5, +6).
    Stored as a one-hot vector of length 7 in ``graph_labels``.

**Usage**::

    python create_graph_datasets__electrum_oxstate.py

Results are written to ``results/create_graph_datasets__electrum_oxstate/debug/``.
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

# :param DOWNLOAD_URL:
#       The URL from which the raw oxidation state CSV file will be downloaded.
#       Points to the ELECTRUM validation repository. The file is approximately 6 MB
#       and contains ~39,000 rows.
DOWNLOAD_URL: str = 'https://raw.githubusercontent.com/TheFreiLab/electrum_val/main/datasets/oxidationstate_46k.csv'

# :param OS_CLASSES:
#       The oxidation state classes in logical order. The dataset is heavily imbalanced:
#       OS +2 accounts for ~52% of samples. Each class has at least 1,000 examples.
OS_CLASSES: List[int] = [0, 1, 2, 3, 4, 5, 6]

# :param DATASET_TYPE:
#       Classification task — predicting metal oxidation state from molecular structure.
DATASET_TYPE: str = 'classification'

DESCRIPTION: str = (
    'Classification dataset of ~39,000 mononuclear transition metal complexes from '
    'the Cambridge Structural Database, curated by Orsi & Frei (Digital Discovery 2025) '
    'for the ELECTRUM benchmark. The task is to predict the metal oxidation state as one '
    'of 7 classes (0 to +6). The dataset is imbalanced, with OS +2 accounting for ~52% '
    'of samples. Useful for evaluating models on chemical property prediction from '
    'molecular structure.'
)

METADATA: dict = {
    'category': 'tmc',
    'min_version': '1.7.0',
    'tags': ['Molecules', 'TransitionMetals', 'TMC', 'OxidationState', 'Classification', 'CSD'],
    'sources': [
        'https://doi.org/10.1039/D5DD00145E',
        'https://github.com/TheFreiLab/electrum_val',
    ],
    'verbose': 'ELECTRUM CSD Oxidation State Classification',
    'target_descriptions': {
        str(i): f'OS=+{os} class indicator' if os > 0 else 'OS=0 class indicator'
        for i, os in enumerate(OS_CLASSES)
    },
}

# == PROCESSING PARAMETERS ==

DATASET_NAME: str = 'electrum_oxstate'
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
                # NOTE: oxidation_state is intentionally NOT passed here because it is
                # the prediction target. Passing it would leak the label into the
                # input features via the d-electron count encoder.
                graph: typc.GraphDict = self.processing.process(
                    metal=data['metal'],
                    ligand_smiles=data['ligand_smiles'],
                    connecting_atom_indices=data['connecting_atom_indices'],
                    oxidation_state=0,
                    total_charge=data.get('total_charge', 0),
                    spin_multiplicity=data.get('spin_multiplicity', 1),
                    whole_complex_smiles=data.get('whole_complex_smiles', None),
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
    Download the oxidation state CSV from the ELECTRUM repo and parse it into the
    decomposed TMC format.

    The raw CSV has columns ``smiles`` (whole-complex), ``Name``, ``LigandSmiles``,
    ``Metal``, ``bondorder``, ``oxidation_states``, ``classification``. The
    ``oxidation_states`` column contains the actual oxidation state as a string
    (e.g., ``'+2'``, ``'0'``), which is parsed into an integer and passed to
    ``MetalOrganicProcessing`` for accurate d-electron count encoding.
    """
    csv_path = os.path.join(e.path, 'oxidationstate_raw.csv')
    if not os.path.exists(csv_path):
        e.log(f'downloading oxidation state dataset from {e.DOWNLOAD_URL} ...')
        response = requests.get(e.DOWNLOAD_URL, timeout=300)
        response.raise_for_status()
        with open(csv_path, 'w') as f:
            f.write(response.text)
        e.log(f'downloaded {len(response.content) / 1024 / 1024:.1f} MB')
    else:
        e.log(f'using cached raw CSV at {csv_path}')

    e.log('parsing dataset...')
    df = pd.read_csv(csv_path)

    if e.MAX_ELEMENTS is not None:
        df = df.head(e.MAX_ELEMENTS)
        e.log(f'limited to {e.MAX_ELEMENTS} complexes')

    # Map oxidation state string → integer
    os_to_index = {os_val: i for i, os_val in enumerate(e.OS_CLASSES)}

    dataset: Dict[int, dict] = {}
    skipped = 0

    for idx, row in df.iterrows():
        metal = str(row['Metal']).strip()
        ligand_smiles_raw = str(row['LigandSmiles'])

        # Parse oxidation state from the oxidation_states column (e.g., '+2', '0')
        os_str = str(row['oxidation_states']).strip()
        try:
            oxidation_state = int(os_str.replace('+', ''))
        except (ValueError, TypeError):
            skipped += 1
            continue

        if oxidation_state not in os_to_index:
            skipped += 1
            continue

        # Split dot-separated ligand SMILES into individual fragments
        ligand_smiles = ligand_smiles_raw.split('.')

        # Infer connecting atom indices heuristically
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

        # One-hot encode the oxidation state
        targets = [float(oxidation_state == os_val) for os_val in e.OS_CLASSES]

        # Get whole-complex SMILES for graph_repr
        whole_smiles = row.get('smiles', '')
        if pd.isna(whole_smiles):
            whole_smiles = ''

        dataset[len(dataset)] = {
            'complex_id': str(row['Name']),
            'metal': metal,
            'ligand_smiles': ligand_smiles,
            'connecting_atom_indices': connecting_atom_indices,
            'total_charge': 0,
            'oxidation_state': oxidation_state,
            'spin_multiplicity': 1,     # Not available
            'coordination_number': int(float(row['bondorder'])) if not pd.isna(row.get('bondorder', None)) else 0,
            'whole_complex_smiles': str(whole_smiles),
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
            'coordination_number': data['coordination_number'],
            'whole_complex_smiles': data['whole_complex_smiles'],
        }
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

    e.log('starting oxidation state processing experiment...')
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
