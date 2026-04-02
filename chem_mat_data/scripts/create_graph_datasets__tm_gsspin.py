"""
This experiment creates the processed graph dataset for **TM-GSspin** — a classification
dataset of 2,063 mononuclear first-row transition metal complexes labeled with their
DFT-computed ground-state spin multiplicity (2S+1).

The raw data (XYZ coordinates and property file) is downloaded from the Materials Cloud
Archive. Since the dataset provides only 3D structures (no SMILES), the script converts
each XYZ file to a decomposed TMC representation (metal + ligand SMILES + connecting atom
indices) using ASE for bond detection and RDKit's ``rdDetermineBonds`` for bond-order
assignment in the organic ligand fragments. ``MetalOrganicProcessing`` then builds the
molecular graphs.

**Source**: Cho et al., *Digital Discovery* 2024, DOI: 10.1039/D4DD00093E.

**Target**: Multi-class classification with 6 spin multiplicity classes (1–6, i.e.
    singlet through sextet). Stored as a one-hot vector of length 6 in ``graph_labels``.

**Usage**::

    python create_graph_datasets__tm_gsspin.py

Results are written to ``results/create_graph_datasets__tm_gsspin/debug/``.
"""
import os
import json
import gzip
import time
import shutil
import tarfile
import datetime
import multiprocessing
from typing import Dict, List, Optional, Tuple

import yaml
import msgpack
import requests
import numpy as np
import rdkit.Chem as Chem
from rdkit.Chem import rdDetermineBonds
import pandas as pd
import networkx as nx
from ase.io import read as ase_read
from ase.neighborlist import natural_cutoffs, NeighborList
from rich.pretty import pprint
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import chem_mat_data._typing as typc
from chem_mat_data.tmc_processing import MetalOrganicProcessing, TRANSITION_METAL_ATOMIC_NUMBERS
from chem_mat_data.data import default, ext_hook, save_graphs


# == SOURCE PARAMETERS ==

# :param ARCHIVE_URL:
#       Direct download URL for the tar.gz archive containing the 2,063 XYZ files
#       from the Materials Cloud record.
ARCHIVE_URL: str = 'https://archive.materialscloud.org/records/jw6j6-nn007/files/Ground_state_spin_dataset.tar.gz'

# :param PROPERTY_URL:
#       Direct download URL for the tab-separated property file containing
#       metal, charge, spin multiplicity, oxidation state, d-electron count,
#       coordination number, and geometry for each complex.
PROPERTY_URL: str = 'https://archive.materialscloud.org/records/jw6j6-nn007/files/property_2063.txt'

# :param SPIN_CLASSES:
#       The spin multiplicity values (2S+1) present in the dataset.
#       1=singlet, 2=doublet, 3=triplet, 4=quartet, 5=quintet, 6=sextet.
SPIN_CLASSES: List[int] = [1, 2, 3, 4, 5, 6]

# :param DATASET_TYPE:
#       Classification task — predicting ground-state spin multiplicity.
DATASET_TYPE: str = 'classification'

# :param DESCRIPTION:
#       Human-readable description stored in metadata.yml.
DESCRIPTION: str = (
    'Classification dataset of 2,063 mononuclear first-row transition metal complexes '
    'from the Cambridge Structural Database, curated by Cho et al. (Digital Discovery 2024). '
    'The task is to predict the ground-state spin multiplicity (2S+1) as one of 6 classes '
    '(singlet through sextet). Complexes cover 5 metals (Cr, Mn, Fe, Co, Ni) with diverse '
    'coordination geometries (octahedral, tetrahedral, square planar, etc.) and ligand '
    'environments. Spin states were determined at B3LYP*-D3BJ/def2-TZVP level. '
    'Structures were converted from XYZ coordinates to decomposed SMILES using '
    'ASE neighbor detection and RDKit DetermineBonds.'
)

# :param METADATA:
#       Additional metadata fields merged into metadata.yml.
METADATA: dict = {
    'category': 'tmc',
    'min_version': '1.8.0',
    'tags': ['Molecules', 'TransitionMetals', 'TMC', 'SpinState', 'Classification', 'CSD', 'DFT'],
    'sources': [
        'https://doi.org/10.1039/D4DD00093E',
        'https://doi.org/10.24435/materialscloud:jx-a5',
    ],
    'verbose': 'TM-GSspin Ground State Spin Classification',
    'target_descriptions': {
        str(i): f'spin_multiplicity={s} class indicator' for i, s in enumerate(SPIN_CLASSES)
    },
}


# == PROCESSING PARAMETERS ==

DATASET_NAME: str = 'tm_gsspin'
COMPRESS: bool = True
MAX_ELEMENTS: Optional[int] = None

# :param BOND_SCALE:
#       Multiplicative factor applied to ASE covalent radii for determining bonded
#       atom pairs. 1.3 is a standard cutoff for coordination chemistry that catches
#       most metal-ligand bonds without creating spurious connections.
BOND_SCALE: float = 1.3


# == EXPERIMENT PARAMETERS ==

__DEBUG__ = True
__TESTING__ = False

experiment = Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)


# ======================================================================================
# XYZ TO DECOMPOSED TMC FORMAT
# ======================================================================================

def xyz_to_decomposed(
    xyz_path: str,
    metal_symbol: str,
    total_charge: int = 0,
    bond_scale: float = 1.3,
) -> Optional[Tuple[str, List[str], List[List[int]], int]]:
    """
    Convert an XYZ file of a mononuclear TMC into the decomposed format
    (metal, ligand SMILES, connecting atom indices) needed by ``MetalOrganicProcessing``.

    The conversion proceeds in three steps:

    1. **Bond detection**: ASE's covalent-radii-based ``NeighborList`` identifies all
       bonded pairs, including metal-ligand bonds.
    2. **Fragmentation**: The metal atom is removed and the remaining atoms are split
       into connected components (one per ligand) using networkx.
    3. **SMILES generation**: Each ligand fragment is converted to SMILES using RDKit's
       ``rdDetermineBonds.DetermineBonds`` (the integrated xyz2mol algorithm) which
       assigns bond orders from 3D coordinates.

    Monatomic ligands (e.g. Cl, Br, O) are handled directly without xyz2mol.

    :param xyz_path: Path to an XYZ file with standard format (natoms / comment / coords).
    :param metal_symbol: Element symbol of the metal center (e.g. ``'Fe'``).
    :param total_charge: Total charge of the complex (used as fallback for bond order
        assignment in ligand fragments).
    :param bond_scale: Multiplicative factor for covalent radii cutoff (default 1.3).

    :returns: Tuple of ``(metal_symbol, ligand_smiles, connecting_atom_indices, total_charge)``
        or ``None`` if the conversion fails.
    """
    atoms = ase_read(xyz_path)

    # Find the single metal atom by matching the expected symbol
    metal_idx = None
    for i, atom in enumerate(atoms):
        if atom.symbol == metal_symbol:
            if metal_idx is not None:
                return None  # Multinuclear — not supported
            metal_idx = i

    if metal_idx is None:
        return None

    # Build neighbor list from covalent radii
    cutoffs = natural_cutoffs(atoms, mult=bond_scale)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)

    # Identify atoms bonded to the metal
    metal_neighbor_arr, _ = nl.get_neighbors(metal_idx)
    metal_neighbors = set(metal_neighbor_arr.tolist())

    if not metal_neighbors:
        return None

    # Build a graph of all non-metal atoms with intra-ligand bonds
    G = nx.Graph()
    for i in range(len(atoms)):
        if i == metal_idx:
            continue
        G.add_node(i)

    for i in range(len(atoms)):
        if i == metal_idx:
            continue
        neighbors_i, _ = nl.get_neighbors(i)
        for j in neighbors_i.tolist():
            if j != metal_idx and j > i and G.has_node(j):
                G.add_edge(i, j)

    # Each connected component is one ligand
    ligand_smiles: List[str] = []
    connecting_atom_indices: List[List[int]] = []

    for component in nx.connected_components(G):
        component_sorted = sorted(component)

        # Which atoms in this component were bonded to the metal?
        conn_in_component = [idx for idx in component_sorted if idx in metal_neighbors]
        if not conn_in_component:
            continue  # Counter-ion or solvent fragment — skip

        # Map original atom indices to fragment-local (0-based) indices
        orig_to_local = {orig: local for local, orig in enumerate(component_sorted)}
        local_conn = [orig_to_local[idx] for idx in conn_in_component]

        # Monatomic ligands: SMILES is just the bracketed element symbol
        if len(component_sorted) == 1:
            sym = atoms[component_sorted[0]].symbol
            ligand_smiles.append(f'[{sym}]')
            connecting_atom_indices.append([0])
            continue

        # Build an XYZ block string for this ligand fragment
        n_frag = len(component_sorted)
        xyz_lines = [str(n_frag), '']
        for orig_idx in component_sorted:
            sym = atoms[orig_idx].symbol
            x, y, z = atoms.positions[orig_idx]
            xyz_lines.append(f'{sym} {x:.6f} {y:.6f} {z:.6f}')
        xyz_block = '\n'.join(xyz_lines)

        # Convert the organic fragment to an RDKit mol with proper bond orders
        mol = _xyz_block_to_mol(xyz_block)
        if mol is None:
            continue

        smi = Chem.MolToSmiles(mol)
        if not smi:
            continue

        # Map local indices through SMILES canonicalization reordering
        try:
            output_order = list(mol.GetPropsAsDict()['_smilesAtomOutputOrder'])
        except (KeyError, TypeError):
            try:
                order_str = mol.GetProp('_smilesAtomOutputOrder')
                order_str = order_str.strip('[]').rstrip(',')
                output_order = [int(x) for x in order_str.split(',') if x.strip()]
            except Exception:
                output_order = list(range(mol.GetNumAtoms()))

        local_to_canonical = {local: canon for canon, local in enumerate(output_order)}
        canonical_conn = [local_to_canonical[c] for c in local_conn if c in local_to_canonical]

        if not canonical_conn:
            continue

        ligand_smiles.append(smi)
        connecting_atom_indices.append(canonical_conn)

    if not ligand_smiles:
        return None

    return metal_symbol, ligand_smiles, connecting_atom_indices, total_charge


def _xyz_block_to_mol(xyz_block: str, charge: int = 0) -> Optional[Chem.Mol]:
    """
    Convert an XYZ block string to an RDKit Mol with bond orders assigned.

    Tries ``rdDetermineBonds.DetermineBonds`` with charge 0 first, then -1 and +1
    as fallbacks (ligand charge is often unknown).

    :param xyz_block: Multi-line XYZ format string.
    :param charge: Initial charge guess for bond order assignment.
    :returns: An RDKit Mol object with bonds, or None on failure.
    """
    mol = Chem.MolFromXYZBlock(xyz_block)
    if mol is None:
        return None

    # Try charge 0 first, then -1 and +1 as fallbacks
    for try_charge in [charge, -1, 1, -2, 2]:
        try:
            mol_copy = Chem.RWMol(mol)
            rdDetermineBonds.DetermineBonds(mol_copy, charge=try_charge)
            # Validate: check that the result is chemically reasonable
            Chem.SanitizeMol(
                mol_copy,
                Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
            )
            return mol_copy.GetMol()
        except Exception:
            continue

    return None


# ======================================================================================
# PROCESSING WORKER
# ======================================================================================

class TMCProcessingWorker(multiprocessing.Process):
    """
    Multiprocessing worker that converts decomposed TMC data into graph representations.
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
    Download the XYZ archive and property file from Materials Cloud, convert each
    XYZ structure to the decomposed TMC format, and return the parsed dataset.

    The archive contains 2,063 individual XYZ files named by CSD refcode. The property
    file is tab-separated with columns: refcode, metal, total_charge, spin_multiplicity,
    elem_nr, m_ox, d_elec, CN, geometry, rel_m.
    """
    # -- Download property file --
    prop_path = os.path.join(e.path, 'property_2063.txt')
    if not os.path.exists(prop_path):
        e.log(f'downloading property file from {e.PROPERTY_URL} ...')
        response = requests.get(e.PROPERTY_URL, timeout=120)
        response.raise_for_status()
        with open(prop_path, 'wb') as f:
            f.write(response.content)
        e.log(f'downloaded property file ({len(response.content) / 1024:.1f} KB)')
    else:
        e.log(f'using cached property file at {prop_path}')

    # -- Download and extract XYZ archive --
    archive_path = os.path.join(e.path, 'Ground_state_spin_dataset.tar.gz')
    extract_dir = os.path.join(e.path, 'Ground_state_spin_dataset')

    if not os.path.exists(extract_dir):
        if not os.path.exists(archive_path):
            e.log(f'downloading XYZ archive from {e.ARCHIVE_URL} ...')
            response = requests.get(e.ARCHIVE_URL, timeout=300)
            response.raise_for_status()
            with open(archive_path, 'wb') as f:
                f.write(response.content)
            e.log(f'downloaded archive ({len(response.content) / 1024 / 1024:.1f} MB)')

        e.log('extracting XYZ archive...')
        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extractall(e.path)
        e.log(f'extracted to {extract_dir}')
    else:
        e.log(f'using cached XYZ directory at {extract_dir}')

    # -- Parse property file --
    e.log('parsing property file...')
    df = pd.read_csv(prop_path, sep='\t')

    if e.MAX_ELEMENTS is not None:
        df = df.head(e.MAX_ELEMENTS)
        e.log(f'limited to {e.MAX_ELEMENTS} complexes')

    # Build spin multiplicity → one-hot index mapping
    spin_to_index = {s: i for i, s in enumerate(e.SPIN_CLASSES)}

    # -- Convert each XYZ to decomposed format --
    e.log('converting XYZ files to decomposed TMC format...')
    dataset: Dict[int, dict] = {}
    skipped_xyz_missing = 0
    skipped_spin_class = 0
    skipped_conversion = 0
    total = len(df)

    for row_idx, row in df.iterrows():
        refcode = str(row['refcode']).strip()
        metal = str(row['metal']).strip()
        total_charge = int(row['total_charge'])
        spin_mult = int(row['spin_multiplicity'])
        oxidation_state = int(row['m_ox'])

        if spin_mult not in spin_to_index:
            skipped_spin_class += 1
            continue

        xyz_path = os.path.join(extract_dir, f'{refcode}.xyz')
        if not os.path.exists(xyz_path):
            skipped_xyz_missing += 1
            continue

        result = xyz_to_decomposed(
            xyz_path=xyz_path,
            metal_symbol=metal,
            total_charge=total_charge,
            bond_scale=e.BOND_SCALE,
        )

        if result is None:
            skipped_conversion += 1
            continue

        metal_sym, lig_smiles, conn_indices, charge = result

        # One-hot encode the spin multiplicity
        targets = [float(spin_mult == s) for s in e.SPIN_CLASSES]

        dataset[len(dataset)] = {
            'complex_id': refcode,
            'metal': metal_sym,
            'ligand_smiles': lig_smiles,
            'connecting_atom_indices': conn_indices,
            'total_charge': charge,
            'oxidation_state': oxidation_state,
            'spin_multiplicity': spin_mult,
            'targets': targets,
        }

        if (len(dataset)) % 200 == 0:
            e.log(f'  converted {len(dataset)}/{total} ...')

    e.log(f'parsed {len(dataset)} complexes from {total} entries')
    e.log(f'  skipped (XYZ missing): {skipped_xyz_missing}')
    e.log(f'  skipped (spin class):  {skipped_spin_class}')
    e.log(f'  skipped (conversion):  {skipped_conversion}')
    e.log(f'  conversion rate: {len(dataset) / total * 100:.1f}%')

    return dataset


@experiment.hook('save_csv', default=True, replace=False)
def save_csv(e: Experiment, dataset: Dict[int, dict]) -> None:
    """
    Save the dataset as a CSV in the decomposed TMC format with JSON-encoded list
    columns, compatible with :func:`load_tmc_dataset`.
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

    e.log('starting TM-GSspin processing experiment...')
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

        e.log('processing dataset into graphs...')
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

            if count % 200 == 0 and count != prev_count:
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
