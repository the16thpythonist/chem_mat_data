"""
This experiment creates the processed graph dataset for **tmQMg** — a dataset of ~63,000
mononuclear transition metal complexes with 20 DFT-computed quantum-mechanical properties.

The raw data is downloaded from the ELECTRUM validation repository on GitHub. Each complex
is represented in a decomposed format: a metal element symbol and dot-separated ligand
SMILES strings. The script infers connecting atom indices heuristically (identifying
donor atoms with available lone pairs) and uses ``MetalOrganicProcessing`` to build the
molecular graphs.

**Source**: Kneiding et al., *Digital Discovery* 2023; ELECTRUM repo (Orsi & Frei 2025).

**Targets** (20 regression properties):
    HOMO/LUMO energies, HOMO-LUMO gap, electronic energy, dispersion energy, enthalpy,
    Gibbs energy, ZPE correction, heat capacity, entropy, dipole moment, polarizability,
    vibrational frequencies, and delta corrections.

**Usage**::

    python create_graph_datasets__tmqmg.py

Results are written to ``results/create_graph_datasets__tmqmg/debug/``.
"""
import os
import gzip
import time
import shutil
import datetime
import tempfile
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
# These parameters configure where the raw dataset is fetched from and how it is
# interpreted. The tmQMg dataset is hosted in the ELECTRUM validation repository
# on GitHub as a single CSV file with columns for CSD identifier, metal, charge,
# ligand SMILES (dot-separated), and 20 DFT-computed QM properties.

# :param DOWNLOAD_URL:
#       The URL from which the raw tmQMg CSV file will be downloaded. This points
#       to the ELECTRUM validation repository's ``datasets/tmQMg.csv`` on GitHub.
#       The file is approximately 20 MB and contains ~63,000 rows.
DOWNLOAD_URL: str = 'https://raw.githubusercontent.com/TheFreiLab/electrum_val/main/datasets/tmQMg.csv'

# :param TARGET_COLUMNS:
#       A list of the 20 column names in the raw CSV that contain the DFT-computed
#       quantum-mechanical properties. These become the regression targets (graph_labels)
#       in the processed graph dataset. The order here determines the order in the
#       target vector.
TARGET_COLUMNS: List[str] = [
    'tzvp_lumo_energy',
    'tzvp_homo_energy',
    'tzvp_homo_lumo_gap',
    'homo_lumo_gap_delta',
    'tzvp_electronic_energy',
    'electronic_energy_delta',
    'tzvp_dispersion_energy',
    'dispersion_energy_delta',
    'enthalpy_energy',
    'enthalpy_energy_correction',
    'gibbs_energy',
    'gibbs_energy_correction',
    'zpe_correction',
    'heat_capacity',
    'entropy',
    'tzvp_dipole_moment',
    'dipole_moment_delta',
    'polarisability',
    'lowest_vibrational_frequency',
    'highest_vibrational_frequency',
]

# :param DATASET_TYPE:
#       Either 'regression' or 'classification'. All 20 tmQMg targets are continuous
#       QM properties, so this is always 'regression'.
DATASET_TYPE: str = 'regression'

# :param DESCRIPTION:
#       A human-readable description of the dataset that will be stored in the
#       metadata.yml file alongside the processed dataset.
DESCRIPTION: str = (
    'Dataset of ~63,000 mononuclear transition metal complexes from the tmQMg dataset '
    '(Kneiding et al., Digital Discovery 2023) with 20 DFT-computed quantum-mechanical '
    'properties including HOMO/LUMO energies, electronic and dispersion energies, '
    'thermodynamic quantities (enthalpy, Gibbs energy, entropy, heat capacity), '
    'dipole moment, polarizability, and vibrational frequencies.'
)

# :param METADATA:
#       A dictionary of additional metadata fields that will be merged into the
#       auto-generated metadata.yml. Includes tags for discoverability, source
#       URLs for provenance, and per-target descriptions.
METADATA: dict = {
    'category': 'tmc',
    'min_version': '1.7.0',
    'tags': ['Molecules', 'TransitionMetals', 'TMC', 'QM', 'DFT'],
    'sources': [
        'https://doi.org/10.1039/D2DD00129B',
        'https://github.com/TheFreiLab/electrum_val',
    ],
    'verbose': 'tmQMg Transition Metal Complex QM Properties',
    'target_descriptions': {
        str(i): col for i, col in enumerate(TARGET_COLUMNS)
    },
}

# == PROCESSING PARAMETERS ==
# These parameters control how the dataset is processed and stored. Unlike the
# organic molecule pipeline which uses MoleculeProcessing, this script uses
# MetalOrganicProcessing which takes a decomposed input (metal + ligand SMILES)
# rather than a single whole-complex SMILES string.

# :param DATASET_NAME:
#       The base filename for all output files (CSV, mpack, metadata). The script
#       produces: ``{DATASET_NAME}.csv``, ``{DATASET_NAME}.mpack``,
#       ``{DATASET_NAME}.mpack.gz``, and ``metadata.yml``.
DATASET_NAME: str = 'tmqmg'

# :param COMPRESS:
#       If True, the mpack file is additionally compressed to gzip format. This
#       typically achieves 10-15x compression and is recommended for distribution.
COMPRESS: bool = True

# :param MAX_ELEMENTS:
#       Maximum number of complexes to process. Set to None to process the full
#       dataset (~63,000 complexes). Set to a smaller number (e.g. 100) for quick
#       verification runs during development.
MAX_ELEMENTS: Optional[int] = None

# == EXPERIMENT PARAMETERS ==

# :param __DEBUG__:
#       In debug mode, results overwrite the previous run in the ``debug/`` folder
#       instead of creating a new timestamped folder.
__DEBUG__ = True

# :param __TESTING__:
#       If True, the dataset is limited to 50 elements for fast testing regardless
#       of MAX_ELEMENTS.
__TESTING__ = False

experiment = Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)


# ======================================================================================
# DONOR ATOM HEURISTIC
# ======================================================================================
# The tmQMg dataset provides ligand SMILES but does NOT provide information about which
# atoms in each ligand coordinate to the metal. We need to infer this heuristically.
#
# The approach: in coordination chemistry, the atoms that donate electron pairs to the
# metal are almost always heteroatoms (N, O, P, S, Se) or halides (F, Cl, Br, I), and
# anionic donors (negatively charged atoms) take priority. This covers the vast majority
# of ligands in the dataset — amines, phosphines, carbonyls, halides, carboxylates, etc.
#
# The main limitation is carbon-donor ligands (NHC carbenes, cyclopentadienyl, CO) where
# carbon is the actual donor atom. For these, we fall back to index 0 which happens to be
# correct for CO (C#O, carbon is index 0) but may be wrong for more complex cases.

# Atomic numbers of common donor atoms in coordination chemistry.
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
    Heuristically determine which atoms in a ligand SMILES are donor atoms
    (i.e., atoms that coordinate to the metal center).

    The heuristic identifies atoms that:

    1. Are common donor elements (N, O, P, S, Se, halides), OR
    2. Carry a negative formal charge (anionic donors like carbanions).

    Among candidate donors, atoms with negative charge are prioritized. For neutral
    ligands where multiple candidates exist, all candidates are returned (the ligand
    may be polydentate).

    For single-atom ligands (e.g., ``Cl``, ``Br``), atom index 0 is always returned.

    :param ligand_smiles: SMILES string of a single ligand fragment.
    :returns: List of 0-based atom indices of inferred donor atoms.
    """
    mol = Chem.MolFromSmiles(ligand_smiles)
    if mol is None:
        # Some ligand fragments have unusual valences (e.g., B with 4 bonds in
        # tris(pyrazolyl)borate) because they were extracted from a metal complex.
        # Fall back to partial sanitization that skips the valence check.
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

    # Collect candidates in two tiers: negatively charged atoms (strongest donors)
    # and neutral heteroatoms (common donors).
    candidates = []
    charged_candidates = []
    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        charge = atom.GetFormalCharge()

        if charge < 0:
            charged_candidates.append(atom.GetIdx())
        elif z in DONOR_ELEMENTS:
            candidates.append(atom.GetIdx())

    # Prefer negatively charged atoms (anionic donors like carboxylate O-)
    if charged_candidates:
        return charged_candidates

    # Fall back to neutral donor atoms (amines, phosphines, etc.)
    if candidates:
        return candidates

    # Last resort for carbon-donor ligands (CO, isocyanides, NHC carbenes):
    # return index 0. For CO (C#O) this is correct since C is the donor.
    return [0]


# ======================================================================================
# PROCESSING WORKER
# ======================================================================================
# This worker mirrors the ProcessingWorker in the base ``create_graph_datasets.py`` but
# uses ``MetalOrganicProcessing`` instead of ``MoleculeProcessing``. The key difference
# is the input format: instead of a single SMILES string, each work item contains the
# decomposed representation (metal, ligand_smiles list, connecting_atom_indices).

class TMCProcessingWorker(multiprocessing.Process):
    """
    Multiprocessing worker that converts decomposed TMC data into graph representations
    using ``MetalOrganicProcessing``.

    Communication with the main process happens via two queues: the ``input_queue``
    delivers data dicts to be processed and the ``output_queue`` returns msgpack-encoded
    graph dicts (or None on failure) back to the main process.
    """

    def __init__(self,
                 input_queue: multiprocessing.Queue,
                 output_queue: multiprocessing.Queue,
                 ):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.processing = MetalOrganicProcessing()

    def run(self):
        """
        Infinite loop consuming data dicts from the input queue. For each item,
        calls ``MetalOrganicProcessing.process()`` with the decomposed TMC fields
        and sends the resulting graph (or None on error) to the output queue.
        A ``None`` sentinel on the input queue signals the worker to stop.
        """
        for data in iter(self.input_queue.get, None):
            try:
                graph: typc.GraphDict = self.processing.process(
                    metal=data['metal'],
                    ligand_smiles=data['ligand_smiles'],
                    connecting_atom_indices=data['connecting_atom_indices'],
                    oxidation_state=data.get('oxidation_state', 0),
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
    """
    This hook is invoked in the processing worker after the graph has been created.
    It adds the CSD identifier and metal symbol to each graph dict for traceability,
    so that processed graphs can be linked back to their source entries.
    """
    if graph is not None:
        graph['graph_id'] = data.get('complex_id', '')
        graph['graph_metal'] = data.get('metal', '')


@experiment.hook('load_dataset', default=True, replace=False)
def load_dataset(e: Experiment) -> Dict[int, dict]:
    """
    Download the tmQMg CSV from the ELECTRUM repo and parse it into the decomposed
    TMC format expected by ``MetalOrganicProcessing``.

    The raw CSV has a ``LigandSmiles`` column that contains all ligands for a complex
    concatenated with dots (the standard SMILES disconnection notation). For example,
    ``"C#O.C#O.Cl.Cl"`` represents two carbonyl and two chloride ligands. This hook
    splits them into individual ligand SMILES and infers which atoms in each ligand
    coordinate to the metal using the ``infer_connecting_atoms`` heuristic.

    Note that tmQMg does not provide oxidation states or spin multiplicities, so these
    default to 0 and 1 respectively.
    """
    # -- Download the raw CSV (cached after the first run) --
    csv_path = os.path.join(e.path, 'tmqmg_raw.csv')
    if not os.path.exists(csv_path):
        e.log(f'downloading tmQMg dataset from {e.DOWNLOAD_URL} ...')
        response = requests.get(e.DOWNLOAD_URL, timeout=120)
        response.raise_for_status()
        with open(csv_path, 'w') as f:
            f.write(response.text)
        e.log(f'downloaded {len(response.content) / 1024 / 1024:.1f} MB')
    else:
        e.log(f'using cached raw CSV at {csv_path}')

    # -- Parse each row into the decomposed TMC format --
    e.log('parsing dataset...')
    df = pd.read_csv(csv_path)

    if e.MAX_ELEMENTS is not None:
        df = df.head(e.MAX_ELEMENTS)
        e.log(f'limited to {e.MAX_ELEMENTS} complexes')

    dataset: Dict[int, dict] = {}
    skipped = 0

    for idx, row in df.iterrows():
        metal = row['Metal']
        ligand_smiles_raw = str(row['LigandSmiles'])
        charge = int(row['charge'])

        # The LigandSmiles column uses the SMILES dot notation to separate individual
        # ligand fragments. Each fragment between dots is one complete ligand structure.
        ligand_smiles = ligand_smiles_raw.split('.')

        # For each ligand fragment, heuristically determine which atoms are donor atoms
        # that coordinate to the metal center. This is necessary because tmQMg does not
        # provide explicit connecting atom information.
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

        # Collect the 20 QM property values as the regression target vector
        targets = []
        for col in e.TARGET_COLUMNS:
            targets.append(float(row[col]))

        dataset[len(dataset)] = {
            'complex_id': str(row['id']),
            'metal': metal,
            'ligand_smiles': ligand_smiles,
            'connecting_atom_indices': connecting_atom_indices,
            'total_charge': charge,
            'oxidation_state': 0,      # Not available in tmQMg
            'spin_multiplicity': 1,    # Not available in tmQMg
            'targets': targets,
            # Keep the original dot-separated string for CSV output
            'ligand_smiles_raw': ligand_smiles_raw,
        }

    e.log(f'parsed {len(dataset)} complexes ({skipped} skipped)')
    return dataset


@experiment.hook('save_csv', default=True, replace=False)
def save_csv(e: Experiment, dataset: Dict[int, dict]) -> None:
    """
    Save the dataset as a CSV file in the decomposed TMC format. This CSV serves as
    the "raw" format counterpart to the processed mpack graph file and can be loaded
    via ``load_tmc_dataset('tmqmg')``.
    """
    e.log('saving CSV...')

    rows = []
    for index, data in dataset.items():
        row = {
            'complex_id': data['complex_id'],
            'metal': data['metal'],
            'ligand_smiles': data['ligand_smiles_raw'],
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

    # Also save a compressed version for distribution
    gz_path = csv_path + '.gz'
    with open(csv_path, 'rb') as f_in, gzip.open(gz_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

    e.log(f'saved CSV ({len(rows)} rows)')


# ======================================================================================
# MAIN EXPERIMENT
# ======================================================================================

@experiment
def experiment(e: Experiment):

    e.log('starting tmQMg processing experiment...')
    e.log_parameters()

    # -- Create worker processes --
    # Workers are spawned before loading the dataset to avoid copying the full dataset
    # into each subprocess's memory (Python forks the current process, including its
    # memory, when spawning subprocesses).
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
        # -- Load and parse the dataset --
        e.log('loading dataset...')
        dataset: Dict[int, dict] = e.apply_hook('load_dataset')
        num_elements = len(dataset)
        e.log(f'loaded {num_elements} complexes')

        # -- Save the raw CSV alongside the processed data --
        e.apply_hook('save_csv', dataset=dataset)

        # In testing mode, limit to a small subset so the full pipeline can be
        # validated quickly without processing the entire dataset.
        if e.__TESTING__:
            e.log('testing mode: limiting to 50 elements')
            num_elements = min(num_elements, 50)
            dataset = dict(list(dataset.items())[:num_elements])

        # -- Feed data to workers and collect results --
        # The main process continuously fills the input queue and drains the output
        # queue in a loop. Workers convert each decomposed TMC entry into a GraphDict
        # using MetalOrganicProcessing. Failed conversions produce None and are silently
        # dropped from the final dataset (with an error message printed to stdout).
        e.log('processing dataset...')
        indices = list(dataset.keys())
        num_indices = len(indices)
        graphs = []

        start_time = time.time()
        count = 0
        prev_count = 0

        while count < num_indices:

            # Keep the input queue full so workers always have work available
            while not input_queue.full() and len(indices) != 0:
                index = indices.pop()
                data = dataset[index]
                input_queue.put(data)

            # Drain completed results from the output queue
            while not output_queue.empty():
                graph_encoded = output_queue.get()
                graph = msgpack.unpackb(graph_encoded, ext_hook=ext_hook)
                if graph:
                    graphs.append(graph)
                count += 1

            # Progress logging every 1000 elements with ETA estimate
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
        # Ensure workers are stopped even if an exception occurs. Each worker
        # terminates when it receives a None sentinel on its input queue.
        e.log('stopping workers...')
        for worker in workers:
            input_queue.put(None)
            worker.terminate()
            worker.join()

        del input_queue
        del output_queue

    # -- Save processed graphs as msgpack --
    e.log(f'saving {len(graphs)} graphs to mpack...')
    dataset_path = os.path.join(e.path, e.DATASET_NAME + '.mpack')
    save_graphs(graphs, dataset_path)

    file_size = os.path.getsize(dataset_path)
    e.log(f'wrote mpack: {file_size / 1024 / 1024:.1f} MB')

    # -- Compress the mpack file for distribution --
    if e.COMPRESS:
        e.log('compressing...')
        compressed_path = os.path.join(e.path, e.DATASET_NAME + '.mpack.gz')
        with open(dataset_path, 'rb') as f_in, gzip.open(compressed_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

        compressed_size = os.path.getsize(compressed_path)
        e.log(f'compressed: {compressed_size / 1024 / 1024:.1f} MB')

    # -- Generate and save metadata --
    # The metadata file combines auto-detected information (compound count, target count)
    # with the user-defined METADATA dict (tags, sources, descriptions).
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
