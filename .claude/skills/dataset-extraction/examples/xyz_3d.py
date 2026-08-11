"""
ARCHETYPE 3 of 4 — a bundle of .xyz files carrying 3D geometry rather than SMILES.

Use this when the dataset ships atomic coordinates instead of (or alongside) SMILES
strings: QM9, tmQM geometries, and most DFT output collections. Modelled on
``create_graph_datasets__from_xyz__qm9.py``.

The distinguishing feature is the **two-level extension hierarchy**::

    create_graph_datasets.py                 base: worker loop, save, metadata
    └── create_graph_datasets__from_xyz.py   xyz bundle loading, meta.csv export
        └── this script                      dataset-specific parameters

So this script extends ``create_graph_datasets__from_xyz.py``, not the base template,
and in the simplest case it declares parameters only - the intermediate base already
implements ``load_dataset`` and ``after_save``.

An "xyz bundle" is a folder or archive containing one ``.xyz`` file per molecule, and
optionally a CSV of per-molecule metadata. The intermediate base loads it through
``chem_mat_data.main.load_xyz_dataset``, which parses each file with a ``PARSER_CLASS``
and produces RDKit ``Mol`` objects. The worker accepts a ``mol`` key in place of
``smiles``, which is what makes this path work at all.

Name the file ``create_graph_datasets__from_xyz__<name>.py`` to match the convention.
"""

import os
from typing import Dict, List

from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

from chem_mat_data.data import QM9XyzParser

# == SOURCE PARAMETERS ==

# :param SOURCE_PATH:
#       Absolute path of the xyz bundle - either a folder or an archive containing the
#       individual .xyz files. If the path does not exist locally, the intermediate
#       base experiment falls back to fetching a bundle of this name from the remote
#       file share.
SOURCE_PATH: str = os.path.join(
    os.path.expanduser('~'), 'Downloads', 'qm9.xyz_bundle'
)

# :param PARSER_CLASS:
#       The parser used for the individual .xyz files, a subclass of
#       AbstractXyzParser. Different producers write different .xyz dialects - the
#       comment line in particular is not standardized - so pick the parser matching
#       the source, or write a new one in chem_mat_data/data.py if none fits.
PARSER_CLASS: type = QM9XyzParser

# :param TARGET_COLUMNS:
#       Documents the target order. For xyz bundles the target values come from the
#       parser (usually out of the .xyz comment line) or from an accompanying CSV.
TARGET_COLUMNS: List[str] = [
    'mu',
    'alpha',
    'homo',
    'lumo',
    'gap',
    'r2',
    'zpve',
    'u0',
    'u298',
    'h298',
    'g298',
    'cv',
]

# :param DATASET_TYPE:
DATASET_TYPE: str = 'regression'

DESCRIPTION: str = (
    'QM9 contains approximately 134,000 small organic molecules with up to nine heavy '
    'atoms (C, O, N, F), each with a relaxed 3D geometry and 12 quantum-chemical '
    'properties computed at the B3LYP/6-31G(2df,p) level of theory. Properties include '
    'the dipole moment, isotropic polarizability, HOMO and LUMO energies, the HOMO-LUMO '
    'gap, zero-point vibrational energy, internal and free energies, and heat capacity. '
    'It is the most widely used benchmark for machine learning models that predict '
    'quantum-chemical properties from molecular structure.'
)

METADATA: dict = {
    'verbose': 'QM9 Quantum Chemistry Properties',
    'tags': [
        'Molecules',
        'SMILES',
        '3D coordinates',
        'Quantum Chemistry',
        'DFT',
        'Benchmark',
        'Multi-Target',
    ],
    'sources': [
        'https://doi.org/10.1038/sdata.2014.22',
        'https://figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904',
    ],
    'target_descriptions': {
        '0': 'mu - Dipole moment (Debye)',
        '1': 'alpha - Isotropic polarizability (Bohr^3)',
        '2': 'homo - Highest occupied molecular orbital energy (Hartree)',
        '3': 'lumo - Lowest unoccupied molecular orbital energy (Hartree)',
        '4': 'gap - HOMO-LUMO gap (Hartree)',
        '5': 'r2 - Electronic spatial extent (Bohr^2)',
        '6': 'zpve - Zero point vibrational energy (Hartree)',
        '7': 'u0 - Internal energy at 0 K (Hartree)',
        '8': 'u298 - Internal energy at 298.15 K (Hartree)',
        '9': 'h298 - Enthalpy at 298.15 K (Hartree)',
        '10': 'g298 - Free energy at 298.15 K (Hartree)',
        '11': 'cv - Heat capacity at 298.15 K (cal/(mol K))',
    },
}

# == PROCESSING PARAMETERS ==

# :param DATASET_NAME:
DATASET_NAME: str = 'qm9'

# :param USE_COORDINATES:
#       For an xyz dataset the coordinates come from the source files themselves, so
#       there is no need to have RDKit generate them.
USE_COORDINATES: bool = False

__TESTING__ = False

# Note the base: the *intermediate* xyz template, not create_graph_datasets.py.
experiment = Experiment.extend(
    'create_graph_datasets__from_xyz.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)


# --------------------------------------------------------------------------------------
# In the simple case nothing below is needed: the intermediate base already implements
# load_dataset (via load_xyz_dataset) and an after_save hook that writes meta.csv.
#
# Override load_dataset only when the bundle needs handling the base does not provide -
# for example downloading and unpacking the bundle first, or joining target values from
# a separate CSV that the parser does not read. The records it returns use a 'mol' key
# holding an RDKit Mol instead of a 'smiles' string; the worker accepts either.
# --------------------------------------------------------------------------------------

# @experiment.hook('load_dataset', default=False, replace=True)
# def load_dataset(e: Experiment) -> Dict[int, dict]:
#     """
#     Load the xyz bundle and attach target values from a companion CSV.
#
#     :param e: The running experiment.
#     :returns: Mapping of integer index to records with 'mol' and 'targets'.
#     """
#     import pandas as pd
#     from chem_mat_data.main import load_xyz_dataset
#
#     df = load_xyz_dataset(
#         dataset_name=os.path.basename(e.SOURCE_PATH),
#         folder_path=os.path.dirname(e.SOURCE_PATH),
#         parser_cls=e.PARSER_CLASS,
#         use_cache=True,
#     )
#
#     dataset: Dict[int, dict] = {}
#     for index, row in df.iterrows():
#         record = row.to_dict()
#         record['targets'] = [float(record[column]) for column in e.TARGET_COLUMNS]
#         dataset[index] = record
#
#     return dataset


experiment.run_if_main()
