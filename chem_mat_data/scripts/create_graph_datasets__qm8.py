"""
Processing of the QM8 electronic spectra dataset (Ramakrishnan et al., J. Chem. Phys. 2015) into the graph format.

QM8 contains the two lowest singlet excitation energies and oscillator strengths of 21,786 small organic molecules
(up to eight C, N, O, F atoms from GDB-17 / QM9) computed with RI-CC2, LR-TDPBE0 (def2-SVP and def2-TZVP) and
LR-TDCAM-B3LYP.

**Why not simply the MoleculeNet CSV?** The widely used MoleculeNet/DeepChem ``qm8.csv`` repeats the
TDPBE0/def2-SVP columns in place of the TDPBE0/def2-TZVP columns. This script therefore takes the target values
from the original data file published with the paper (``gdb8_22k_elec_spec.txt``) and only the SMILES strings from
the MoleculeNet CSV. Both files list the molecules in the same order; this is verified on the CC2 columns, which
are present in both, before the SMILES are attached.
"""
import io
from typing import Dict, List

import numpy as np
import pandas as pd
import requests
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path
from rdkit import Chem

# :param DATASET_NAME:
#       This string determines the name of the message pack dataset file that is then
#       stored into the "results" folder of the experiment as the result of the
#       processing process. The corresponding file extensions will be added
#       automatically.
DATASET_NAME: str = 'qm8'
# :param SPECTRA_URL:
#       URL of the original QM8 data file with the excitation energies and oscillator strengths.
SPECTRA_URL: str = (
    'https://raw.githubusercontent.com/raghurama123/ExcitedStatesQM8/main/'
    '22k_electronic_spectra_TDDFT_CC2/gdb8_22k_elec_spec.txt'
)
# :param SMILES_URL:
#       URL of the MoleculeNet version of QM8, used only for its SMILES column.
SMILES_URL: str = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm8.csv'
# :param HARTREE_TO_EV:
#       Conversion factor applied to all excitation energies (the source data is in atomic units).
HARTREE_TO_EV: float = 27.211386245988
# :param TARGET_NAMES:
#       Names of the 16 targets in the order of the source data file.
TARGET_NAMES: List[str] = [
    f'{q}-{m}'
    for m in ['CC2', 'PBE0_def2SVP', 'PBE0_def2TZVP', 'CAM']
    for q in ['E1', 'E2', 'f1', 'f2']
]
# :param TARGET_COLUMNS:
#       Column names of the targets in the saved CSV file.
TARGET_COLUMNS: List[str] = TARGET_NAMES
# :param METADATA:
#       A dictionary which will be used as the basis for the metadata that will be added
#       as additional information to the file share server.
METADATA: dict = {
    'verbose': 'QM8 Electronic Spectra (corrected PBE0/def2-TZVP columns)',
    'description': (
        'QM8 contains the two lowest singlet-singlet vertical excitation energies and oscillator strengths of '
        '21,786 small organic molecules with up to eight C, N, O and F atoms (a subset of QM9 / GDB-17), computed '
        'at the RI-CC2/def2-TZVP, LR-TDPBE0/def2-SVP, LR-TDPBE0/def2-TZVP and LR-TDCAM-B3LYP/def2-TZVP levels of '
        'theory at B3LYP/6-31G(2df,p) geometries. Excitation energies are given in eV, oscillator strengths in the '
        'length representation. The target values are taken from the original data file of Ramakrishnan et al.; '
        'unlike the MoleculeNet CSV, the TDPBE0/def2-TZVP columns therefore contain the actual def2-TZVP values '
        '(in the MoleculeNet file they duplicate the def2-SVP columns).'
    ),
    'target_type': ['regression'],
    'tags': [
        'SMILES',
        'Molecules',
        'Quantum Chemistry',
        'Excited States',
        'TDDFT',
        'CC2',
        'Excitation Energies',
        'Oscillator Strengths',
    ],
    'sources': [
        'https://doi.org/10.1063/1.4928757',
        'https://github.com/raghurama123/ExcitedStatesQM8',
        'https://moleculenet.org/datasets-1',
    ],
    'target_descriptions': {
        i: f'{name} - '
        + ('excitation energy in eV' if name.startswith('E') else 'oscillator strength (dimensionless, length form)')
        + {
            'CC2': ' at the RI-CC2/def2-TZVP level',
            'PBE0_def2SVP': ' at the LR-TDPBE0/def2-SVP level',
            'PBE0_def2TZVP': ' at the LR-TDPBE0/def2-TZVP level',
            'CAM': ' at the LR-TDCAM-B3LYP/def2-TZVP level',
        }[name.split('-', 1)[1]]
        for i, name in enumerate(TARGET_NAMES)
    },
}

__TESTING__ = False

experiment = Experiment.extend(
    'create_graph_datasets.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)


@experiment.hook('add_graph_metadata', default=False, replace=True)
def add_graph_metadata(e: Experiment, data: dict, graph: dict) -> dict:
    """
    Stores the GDB-9 index of the molecule (first column of the source file) as the graph id.
    """
    graph['graph_id'] = data['id']


@experiment.hook('load_dataset', default=False, replace=True)
def load_dataset(e: Experiment) -> Dict[int, dict]:
    """
    Downloads the original QM8 spectra file and the MoleculeNet SMILES and joins them row by row.

    The join is only performed after checking that the CC2 columns of both files agree for every row, so a change
    in the ordering of either source fails loudly instead of silently attaching wrong SMILES to the targets.

    :returns: A dict mapping integer indices to dicts with the keys 'smiles', 'targets' and 'id'.
    """
    e.log('downloading original QM8 spectra file...')
    text = requests.get(e.SPECTRA_URL, timeout=120).text
    rows = [line.split() for line in text.splitlines() if line.strip() and not line.lstrip().startswith('#')]
    spectra = np.array([[float(x) for x in row] for row in rows])
    e.log(f' * {spectra.shape[0]} molecules, {spectra.shape[1]} columns')

    e.log('downloading MoleculeNet QM8 csv (SMILES only)...')
    df = pd.read_csv(io.StringIO(requests.get(e.SMILES_URL, timeout=120).text))
    if len(df) != len(spectra):
        raise ValueError(f'row counts differ: MoleculeNet {len(df)}, original {len(spectra)}')
    cc2 = df[['E1-CC2', 'E2-CC2', 'f1-CC2', 'f2-CC2']].values
    if not np.allclose(cc2, spectra[:, 1:5], atol=1e-6):
        raise ValueError('the CC2 columns of the two files do not agree row by row; refusing to join them')
    e.log(' * CC2 columns agree row by row, joining SMILES to the original targets')

    energy_columns = [i for i, name in enumerate(e.TARGET_NAMES) if name.startswith('E')]
    dataset: Dict[int, dict] = {}
    index = 0
    for smiles, row in zip(df['smiles'], spectra):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or '.' in smiles or mol.GetNumAtoms() < 2:
            continue
        targets = row[1:].copy()
        targets[energy_columns] *= e.HARTREE_TO_EV
        dataset[index] = {'smiles': smiles, 'targets': targets.tolist(), 'id': int(row[0])}
        index += 1

    return dataset


experiment.run_if_main()
