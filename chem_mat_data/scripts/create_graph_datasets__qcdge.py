"""
Processing of the QCDGE database (Zhu et al., Sci. Data 2024) into the graph format.

QCDGE contains ground- and excited-state properties of about 443,000 closed-shell organic molecules with up to ten
C, N, O and F atoms, collected from QM9, GDB-11 and PubChemQC. Ground-state geometries were optimised at the
B3LYP-D3(BJ)/6-31G* level; the ten lowest singlet and ten lowest triplet excitations were computed at the
TD-wB97X-D/6-31G* level at these geometries. This script extracts, for every molecule, the excitation energies of
S1-S10 and T1-T10 (eV) and the oscillator strengths of S1-S10.

The data is distributed as four HDF5 files (A_9: QM9-derived, A_10: GDB-11-derived, B_9 and B_10: PubChemQC-derived,
about 50 GB in total) from the download server of the authors' group. The files are verified against the SHA-512 sums
published alongside them. Reading all molecules takes on the order of an hour (each molecule's excited-state
information is a JSON string inside the HDF5 group). Other properties contained in the files (ground-state
properties, transition moments, Mulliken charges, ...) are not extracted here.
"""
import hashlib
import json
import os
from typing import Dict, List, Optional

import h5py
import requests
import rdkit.Chem as Chem
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path

# :param DATASET_NAME:
#       This string determines the name of the message pack dataset file that is then
#       stored into the "results" folder of the experiment as the result of the
#       processing process. The corresponding file extensions will be added
#       automatically.
DATASET_NAME: str = 'qcdge'
# :param BASE_URL:
#       Directory on the authors' download server containing the HDF5 files and SHA512SUM.
BASE_URL: str = 'http://download_scnu.langroup.site/Researches/QCDGE_Database/'
# :param SUBSETS:
#       The HDF5 subsets to process (file names without extension).
SUBSETS: List[str] = ['A_9', 'A_10', 'B_9', 'B_10']
# :param LOCAL_DIR:
#       Optional directory that already contains the HDF5 files; files found there are not downloaded again.
LOCAL_DIR: Optional[str] = None
# :param VERIFY_SHA512:
#       Whether to verify downloaded files against the published SHA-512 sums.
VERIFY_SHA512: bool = True
# :param N_STATES:
#       Number of singlet and triplet excitations per molecule (QCDGE provides ten of each).
N_STATES: int = 10
# :param TARGET_COLUMNS:
#       Column names of the targets in the saved CSV file, in target order.
TARGET_COLUMNS: List[str] = (
    [f'S{i}_eV' for i in range(1, 11)] + [f'f_S{i}' for i in range(1, 11)] + [f'T{i}_eV' for i in range(1, 11)]
)
# :param METADATA:
#       A dictionary which will be used as the basis for the metadata that will be added
#       as additional information to the file share server.
METADATA: dict = {
    'verbose': 'QCDGE - Ground- and Excited-State Properties of 443k Molecules (excitation energies)',
    'description': (
        'QCDGE contains about 443,000 closed-shell organic molecules with up to ten C, N, O and F atoms from QM9, '
        'GDB-11 and PubChemQC. Geometries were optimised at the B3LYP-D3(BJ)/6-31G* level and the ten lowest '
        'singlet and triplet excited states were computed with TD-wB97X-D/6-31G*. This dataset provides the '
        'excitation energies of S1-S10 and T1-T10 in eV and the oscillator strengths of S1-S10, sorted by energy '
        'within each multiplicity. The graph id encodes the subset and the molecule index of the source files '
        '(A_9: QM9-derived, A_10: GDB-11-derived, B_9/B_10: PubChemQC-derived).'
    ),
    'target_type': ['regression'],
    'tags': [
        'SMILES',
        'Molecules',
        'Quantum Chemistry',
        'Excited States',
        'TDDFT',
        'Excitation Energies',
        'Oscillator Strengths',
        'Singlets',
        'Triplets',
    ],
    'sources': [
        'https://doi.org/10.1038/s41597-024-03788-x',
        'https://doi.org/10.6084/m9.figshare.25930255.v1',
        'http://langroup.site/QCDGE/',
    ],
    'target_descriptions': {
        **{i: f'S{i + 1}_eV - excitation energy of the {i + 1}. singlet excited state in eV (TD-wB97X-D/6-31G*)'
           for i in range(10)},
        **{10 + i: f'f_S{i + 1} - oscillator strength of the {i + 1}. singlet excited state' for i in range(10)},
        **{20 + i: f'T{i + 1}_eV - excitation energy of the {i + 1}. triplet excited state in eV (TD-wB97X-D/6-31G*)'
           for i in range(10)},
    },
}

__TESTING__ = False

experiment = Experiment.extend(
    'create_graph_datasets.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)


def sha512_of(path: str, chunk: int = 1 << 24) -> str:
    """
    Computes the SHA-512 hex digest of a file in chunks.

    :param path: Path of the file.
    :param chunk: Read size in bytes.

    :returns: The hex digest.
    """
    h = hashlib.sha512()
    with open(path, 'rb') as file:
        for block in iter(lambda: file.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def download(url: str, path: str) -> None:
    """
    Streams ``url`` to ``path`` (via a temporary file, so an interrupted download never looks complete).

    :param url: Source URL.
    :param path: Destination path.
    """
    tmp = path + '.part'
    with requests.get(url, stream=True, timeout=(30, 600)) as response:
        response.raise_for_status()
        with open(tmp, 'wb') as file:
            for block in response.iter_content(chunk_size=1 << 22):
                file.write(block)
    os.replace(tmp, path)


def parse_states(info_json: str, n_states: int) -> Optional[List[float]]:
    """
    Extracts S1..Sn energies, S1..Sn oscillator strengths and T1..Tn energies from QCDGE's excited-state JSON.

    :param info_json: The ``Info_of_AllExcitedStates`` string of one molecule.
    :param n_states: Number of states per multiplicity that must be present.

    :returns: The concatenated target list, or None if fewer than ``n_states`` states of either multiplicity exist.
    """
    info = json.loads(info_json)
    singlets, triplets = [], []
    for state in info.values():
        energy = float(str(state['excitation_e_eV']).split()[0])
        # the source spells the key 'oscillator_trength'; accept the correct spelling as well
        strength = float(state.get('oscillator_trength', state.get('oscillator_strength', float('nan'))))
        if state['state_type'].startswith('Singlet'):
            singlets.append((energy, strength))
        elif state['state_type'].startswith('Triplet'):
            triplets.append(energy)
    if len(singlets) < n_states or len(triplets) < n_states:
        return None
    singlets.sort(key=lambda x: x[0])
    triplets.sort()
    singlets, triplets = singlets[:n_states], triplets[:n_states]
    return [s[0] for s in singlets] + [s[1] for s in singlets] + triplets


@experiment.hook('add_graph_metadata', default=False, replace=True)
def add_graph_metadata(e: Experiment, data: dict, graph: dict) -> dict:
    """
    Stores '<subset>_<index>' (the group name in the source HDF5 file) as the graph id.
    """
    graph['graph_id'] = data['id']


@experiment.hook('load_dataset', default=False, replace=True)
def load_dataset(e: Experiment) -> Dict[int, dict]:
    """
    Obtains the QCDGE HDF5 subsets (local copy or download, SHA-512 verified) and extracts SMILES and targets.

    In testing mode only the first 200 molecules of the first subset are read, so the files still have to be
    present (pass ``LOCAL_DIR`` to avoid the download).

    :returns: A dict mapping integer indices to dicts with the keys 'smiles', 'targets' and 'id'.
    """
    sums = {}
    if e.VERIFY_SHA512:
        for line in requests.get(e.BASE_URL + 'SHA512SUM', timeout=60).text.splitlines():
            parts = line.split()
            if len(parts) == 2:
                sums[os.path.basename(parts[1].lstrip('*'))] = parts[0]

    dataset: Dict[int, dict] = {}
    index = 0
    for subset in e.SUBSETS:
        name = f'{subset}.hdf5'
        path = os.path.join(e.LOCAL_DIR, name) if e.LOCAL_DIR else None
        if path is None or not os.path.exists(path):
            path = os.path.join(e.path, name)
            if not os.path.exists(path):
                e.log(f'downloading {name} ...')
                download(e.BASE_URL + name, path)
        if e.VERIFY_SHA512 and not e.__TESTING__:
            if name in sums and sha512_of(path) != sums[name]:
                raise ValueError(f'SHA-512 mismatch for {path}')
            e.log(f' * {name}: SHA-512 ' + ('verified' if name in sums else 'not published, skipped'))

        n_skipped = 0
        with h5py.File(path, 'r') as file:
            keys = list(file.keys())
            if e.__TESTING__:
                keys = keys[:200]
            for key in keys:
                group = file[key]
                smiles = str(group.attrs.get('SMILES_RDKIT_CAN', group.attrs.get('SMILES_RDKIT', '')))
                if not smiles or '.' in smiles:
                    n_skipped += 1
                    continue
                mol = Chem.MolFromSmiles(smiles)
                if mol is None or mol.GetNumAtoms() < 2:
                    n_skipped += 1
                    continue
                raw = group['excited_state']['Info_of_AllExcitedStates'][0]
                targets = parse_states(raw.decode() if isinstance(raw, bytes) else str(raw), e.N_STATES)
                if targets is None:
                    n_skipped += 1
                    continue
                dataset[index] = {'smiles': smiles, 'targets': targets, 'id': f'{subset}_{key}'}
                index += 1
        e.log(f' * {name}: {len(keys) - n_skipped} molecules kept, {n_skipped} skipped')
        if e.__TESTING__:
            break

    return dataset


experiment.run_if_main()
