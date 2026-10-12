"""
Processing of the QCDGE database (Zhu et al., Sci. Data 2024) into the graph format.

QCDGE contains ground- and excited-state properties of about 443,000 closed-shell organic molecules with up to ten
C, N, O and F atoms, collected from QM9, GDB-11 and PubChemQC. Ground-state geometries were optimised at the
B3LYP-D3(BJ)/6-31G* level; the ten lowest singlet and ten lowest triplet excitations were computed at the
TD-wB97X-D/6-31G* level at these geometries. This script extracts, for every molecule, the excitation energies of
S1-S10 and T1-T10 (eV) and the oscillator strengths of S1-S10.

The data is distributed as four HDF5 files (A_9: QM9-derived, A_10: GDB-11-derived, B_9 and B_10: PubChemQC-derived,
about 50 GB in total), each in its own CC0 record of the figshare collection the article cites
(doi:10.6084/m9.figshare.c.7259125). They are downloaded with FigshareSource and verified against figshare's MD5 sums.
The authors' group also serves the files from its own web server, but that server states no terms, so it is not used.
Reading all molecules takes on the order of an hour (each molecule's excited-state information is a JSON string
inside the HDF5 group). Other properties contained in the files (ground-state
properties, transition moments, Mulliken charges, ...) are not extracted here.
"""
import hashlib
import json
import os
from typing import Dict, List, Optional, Tuple

import h5py
import rdkit.Chem as Chem
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path

from chem_mat_data.connectors import FigshareSource

# :param DATASET_NAME:
#       This string determines the name of the message pack dataset file that is then
#       stored into the "results" folder of the experiment as the result of the
#       processing process. The corresponding file extensions will be added
#       automatically.
DATASET_NAME: str = 'qcdge'
# :param FIGSHARE_FILES:
#       Figshare file id and MD5 sum of each HDF5 subset. Each subset is its own CC0 record in collection
#       doi:10.6084/m9.figshare.c.7259125 (A_9: 25929004, A_10: 25929001, B_9: 25929010, B_10: 25929013).
FIGSHARE_FILES: Dict[str, Tuple[int, str]] = {
    'A_9': (46666141, '6a89cdee022e909da50cacf364f269f2'),  # 13.2 GB
    'A_10': (46666012, '9d19c2ff91c0397edf982acaf1801228'),  # 13.5 GB
    'B_9': (46667431, '154e4eb0a60befd0ce80550d217fadcb'),  # 12.2 GB
    'B_10': (46666255, '58826eb582cd9ba577f0d04009c9a431'),  # 10.5 GB
}
# :param SUBSETS:
#       The HDF5 subsets to process (file names without extension).
SUBSETS: List[str] = ['A_9', 'A_10', 'B_9', 'B_10']
# :param LOCAL_DIR:
#       Optional directory that already contains the HDF5 files; files found there are not downloaded again.
LOCAL_DIR: Optional[str] = None
# :param DOWNLOAD_DIR:
#       Directory that missing subsets are downloaded to and kept in, so that a later run can pass it as LOCAL_DIR.
#       Defaults to the experiment's archive folder. The four files take about 50 GB, so it must be on a disk with
#       room for them; the download never passes through the system temp directory.
DOWNLOAD_DIR: Optional[str] = None
# :param VERIFY_MD5:
#       Whether to verify the HDF5 files against figshare's MD5 sums (skipped in testing mode, since hashing the
#       files takes about a minute each).
VERIFY_MD5: bool = True
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
    'license': {
        'identifier': 'CC0-1.0',
        'url': 'https://creativecommons.org/publicdomain/zero/1.0/',
        'note': (
            'Each of the four HDF5 subsets is its own figshare record dedicated under CC0 1.0 (A_9 '
            '10.6084/m9.figshare.25929004, A_10 25929001, B_9 25929010, B_10 25929013), in the collection '
            '10.6084/m9.figshare.c.7259125 that the article cites as its data reference; the files are '
            'downloaded from those records. The same files are also served by the authors\' group web server, '
            'which states no terms and is not used. The article itself is CC BY-NC-ND 4.0, which covers its '
            'text only. Upstream: A_9 derives from QM9 (CC0; see the qm9_smiles licence note for the GDB '
            'terms), A_10 from GDB-11 (deposited by the Reymond group on Zenodo under CC BY 4.0, '
            'doi:10.5281/zenodo.5172018) and B_9/B_10 from PubChemQC B3LYP (JCIM2017), which is licensed '
            'CC BY 4.0 and itself derives from PubChem. The two CC BY sources require attribution - see '
            'notes. Determined by an independent licence review on 2026-10-12.'
        ),
    },
    'verbose': 'QCDGE - Ground- and Excited-State Properties of 443k Molecules (excitation energies)',
    'description': (
        'QCDGE contains about 443,000 closed-shell organic molecules with up to ten C, N, O and F atoms from QM9, '
        'GDB-11 and PubChemQC. Geometries were optimised at the B3LYP-D3(BJ)/6-31G* level and the ten lowest '
        'singlet and triplet excited states were computed with TD-wB97X-D/6-31G*. This dataset provides the '
        'excitation energies of S1-S10 and T1-T10 in eV and the oscillator strengths of S1-S10, sorted by energy '
        'within each multiplicity. Molecules with fewer than ten states of either multiplicity are skipped. The graph '
        'id encodes the subset and the molecule index of the source files (A_9: QM9-derived, A_10: GDB-11-derived, '
        'B_9/B_10: PubChemQC-derived, where the index is the PubChem CID).'
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
        'https://doi.org/10.6084/m9.figshare.c.7259125.v1',
        'https://doi.org/10.6084/m9.figshare.25929004',
        'https://doi.org/10.6084/m9.figshare.25929001',
        'https://doi.org/10.6084/m9.figshare.25929010',
        'https://doi.org/10.6084/m9.figshare.25929013',
        'https://langroup.site/QCDGE/',
    ],
    'notes': [
        (
            'Cite Zhu, Y., Li, M., Xu, C. & Lan, Z. Quantum Chemistry Dataset with Ground- and Excited-state '
            'Properties of 450 Kilo Molecules. Sci. Data 11, 948 (2024), doi:10.1038/s41597-024-03788-x.'
        ),
        (
            'The A_10 subset derives from GDB-11 (CC BY 4.0): cite Fink, T. & Reymond, J.-L. J. Chem. Inf. '
            'Model. 47, 342-353 (2007).'
        ),
        (
            'The B_9 and B_10 subsets derive from PubChemQC B3LYP (JCIM2017, CC BY 4.0): cite Nakata, M. & '
            'Shimazaki, T. J. Chem. Inf. Model. 57, 1300-1308 (2017).'
        ),
        (
            'The A_9 subset derives from QM9: cite Ramakrishnan, R., Dral, P. O., Rupp, M. & von Lilienfeld, '
            'O. A. Sci. Data 1, 140022 (2014).'
        ),
    ],
    'target_descriptions': {
        **{str(i): f'S{i + 1}_eV - excitation energy of the {i + 1}. singlet excited state in eV (TD-wB97X-D/6-31G*)'
           for i in range(10)},
        **{str(10 + i): f'f_S{i + 1} - oscillator strength of the {i + 1}. singlet excited state' for i in range(10)},
        **{str(20 + i): f'T{i + 1}_eV - excitation energy of the {i + 1}. triplet excited state in eV (TD-wB97X-D/6-31G*)'
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


def md5_of(path: str, chunk: int = 1 << 24) -> str:
    """
    Computes the MD5 hex digest of a file in chunks, for comparison with figshare's ``computed_md5``.

    :param path: Path of the file.
    :param chunk: Read size in bytes.

    :returns: The hex digest.
    """
    h = hashlib.md5()
    with open(path, 'rb') as file:
        for block in iter(lambda: file.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


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
    Obtains the QCDGE HDF5 subsets (local copy or figshare download, MD5 verified) and extracts SMILES and targets.

    In testing mode only the first 200 molecules of the first subset are read, but that subset still has to be
    present in full (pass ``LOCAL_DIR`` to avoid the download).

    :returns: A dict mapping integer indices to dicts with the keys 'smiles', 'targets' and 'id'.
    """
    download_dir = e.DOWNLOAD_DIR or e.path

    dataset: Dict[int, dict] = {}
    index = 0
    for subset in e.SUBSETS:
        name = f'{subset}.hdf5'
        file_id, md5 = e.FIGSHARE_FILES[subset]
        path = os.path.join(e.LOCAL_DIR, name) if e.LOCAL_DIR else None
        if path is None or not os.path.exists(path):
            path = os.path.join(download_dir, name)
            if not os.path.exists(path):
                e.log(f'downloading {name} from figshare (file {file_id}) to {download_dir} ...')
                os.makedirs(download_dir, exist_ok=True)
                # The temporary directory is created inside download_dir, so the finished file is moved out by a
                # rename: the download never touches the system temp disk, and an interrupted one never leaves a
                # file under the final name.
                with FigshareSource(file_id=file_id, directory=download_dir, verbose=True) as source:
                    os.replace(source.fetch(), path)
        if e.VERIFY_MD5 and not e.__TESTING__:
            if md5_of(path) != md5:
                raise ValueError(f'MD5 mismatch for {path}: expected {md5} from figshare file {file_id}')
            e.log(f' * {name}: MD5 verified')

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
