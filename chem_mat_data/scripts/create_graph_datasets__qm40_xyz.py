"""
Creates the ``qm40_xyz`` graph dataset: QM40 with 3D geometry.

This is the geometric counterpart of ``qm40_smiles``. Same molecules, same 21 property
targets, but each graph additionally carries the B3LYP/6-31G(2df,p) optimised Cartesian
coordinates of every atom, the Mulliken partial charge of every atom, and the bond lengths
that follow from the geometry. Hydrogens are explicit here, so a molecule has roughly
twice as many nodes as the same molecule in ``qm40_smiles``.

**Source**: Madushanka, A., Moura, R.T. & Kraka, E. *QM40, Realistic Quantum Mechanical
Dataset for Machine Learning in Molecular Science.* Sci Data **11**, 1376 (2024).
DOI: 10.1038/s41597-024-04206-y — data record DOI: 10.6084/m9.figshare.25993060.v1

The archive is not distributed per file, so the whole 332 MB has to be fetched, and this
script needs **two** members of it: ``QM40_xyz.csv`` for geometry and charges, and
``QM40_main.csv`` for the scalar properties, which the xyz file does not contain. They
join on the ZINC identifier.

**Targets**: the same 21 as ``qm40_smiles`` and in the same order - 0-15 exactly as the
source publishes them, then ``A``, ``B``, ``C`` (rotational constants in GHz), ``gap``
(positive HOMO-LUMO gap in Hartree) and ``u0`` (zero-point-inclusive internal energy at
0 K in Hartree). See ``create_graph_datasets__qm40_smiles.py`` for why those five exist
and what is wrong with the columns they restate.

**1,152 molecules are excluded because their deposited geometry is wrong.** This is the
one substantive difference from ``qm40_smiles``, which keeps all 162,954. The published
``rot1``/``rot2``/``rot3`` are rotational temperatures derived from the optimised
structure, so they can be checked against the deposited coordinates: compute the inertia
tensor, convert to rotational temperatures, compare. A sweep of the full dataset found the
two populations almost disjoint - 99% of molecules agree to better than 0.3%, which is
just the 5-decimal rounding of the published values, while 1,152 disagree by more than 1%
and often by far more (worst case 265%). Only 20 molecules fall between 0.5% and 2%, so
the cutoff is not sensitive to where in that range it is placed.

For those molecules the deposited structure is not the structure their own properties were
computed from. That is tolerable in a dataset of scalar properties and not tolerable in a
dataset whose entire point is the geometry, which is why they are dropped here and kept
there. The check is recomputed at build time rather than shipped as a hardcoded list.

**Atom order.** The source lists atoms in the order RDKit produces from the SMILES - the
paper states structures were built as SMILES -> RDKit -> PDB, and this was verified
element-for-element on a sample with zero mismatches. Coordinates can therefore be
attached to a SMILES-built molecule positionally. The script re-verifies this per molecule
anyway and skips any molecule that disagrees, because a sample is not a proof.
"""

import os
import csv
import shutil
import hashlib
import zipfile
from typing import Dict, List

import requests
import numpy as np
import pandas as pd
import rdkit.Chem as Chem
from rdkit import RDLogger
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

RDLogger.DisableLog('rdApp.*')

# == SOURCE PARAMETERS ==

# :param DOWNLOAD_URL:
#       Direct download URL of the QM40 archive on Figshare.
DOWNLOAD_URL: str = 'https://ndownloader.figshare.com/files/47535647'
# :param ARCHIVE_MD5:
#       The MD5 Figshare publishes for the archive. Figshare's certificate chain does not
#       validate from every network, so the download falls back to unverified TLS; this
#       hash is what keeps that safe rather than merely convenient.
ARCHIVE_MD5: str = '337773160d22b762399e4745530df6c2'
# :param XYZ_MEMBER:
#       Member holding one row per atom: coordinates and Mulliken charge.
XYZ_MEMBER: str = 'QM40 dataset/QM40_xyz.csv'
# :param MAIN_MEMBER:
#       Member holding one row per molecule: the 16 published scalar properties.
MAIN_MEMBER: str = 'QM40 dataset/QM40_main.csv'

# :param SOURCE_TARGET_COLUMNS:
#       Columns of the main CSV copied straight through as targets 0-15. "spatial extent"
#       really is spelled with a space in the source file.
SOURCE_TARGET_COLUMNS: List[str] = [
    'Internal_E(0K)',
    'HOMO',
    'LUMO',
    'HL_gap',
    'Polarizability',
    'spatial extent',
    'dipol_mom',
    'ZPE',
    'rot1',
    'rot2',
    'rot3',
    'Inter_E(298)',
    'Enthalpy',
    'Free_E',
    'CV',
    'Entropy',
]
# :param DERIVED_TARGET_COLUMNS:
#       Targets 16-20, added by this collection so that QM40 can actually be compared
#       against QM9. They carry QM9's own names for that reason.
DERIVED_TARGET_COLUMNS: List[str] = ['A', 'B', 'C', 'gap', 'u0']
# :param TARGET_COLUMNS:
#       Full ordered target vector, also used to name the columns of the exported CSV.
TARGET_COLUMNS: List[str] = SOURCE_TARGET_COLUMNS + DERIVED_TARGET_COLUMNS

# :param KELVIN_PER_GHZ:
#       h / k_B as Kelvin per GHz, from the SI-exact CODATA values
#       h = 6.62607015e-34 J s and k_B = 1.380649e-23 J/K. Used both to recover the
#       rotational constants and to run the geometry consistency check.
KELVIN_PER_GHZ: float = 0.04799243073
# :param KCAL_MOL_PER_HARTREE:
#       Used to add the ZPE column, published in kcal/mol, onto an energy in Hartree.
KCAL_MOL_PER_HARTREE: float = 627.5094740631
# :param GHZ_PER_INV_INERTIA:
#       Rotational constant B[GHz] = GHZ_PER_INV_INERTIA / I[amu Angstrom^2].
GHZ_PER_INV_INERTIA: float = 505.379
# :param ATOMIC_MASSES:
#       Most-abundant-isotope masses, which is what Gaussian uses by default. Using
#       standard atomic weights instead shifts the recovered rotational constants by
#       ~750 ppm, which is small against the 1% tolerance but is free to get right.
ATOMIC_MASSES: Dict[str, float] = {
    'H': 1.00782503207,
    'C': 12.0,
    'N': 14.0030740048,
    'O': 15.9949146196,
    'F': 18.99840322,
    'S': 31.97207100,
    'Cl': 34.96885268,
}
# :param GEOMETRY_TOLERANCE:
#       Maximum relative disagreement between the rotational temperatures computed from
#       the deposited geometry and those published for the same molecule. Molecules above
#       this are excluded. The two populations are almost disjoint - only 20 of 162,954
#       molecules lie between 0.005 and 0.02 - so this value is not delicate.
GEOMETRY_TOLERANCE: float = 0.01

DATASET_NAME: str = 'qm40_xyz'
DATASET_TYPE: str = 'regression'

DESCRIPTION: str = (
    'QM40 3D geometries: the DFT-optimised Cartesian coordinates and Mulliken partial '
    'charges for the drug-like molecules of the QM40 dataset, carrying the same 21 '
    'property targets as qm40_smiles. Structures were optimised at the '
    'B3LYP/6-31G(2df,p) level of theory in Gaussian16 after a GFN2-xTB pre-optimisation, '
    'and are stored with explicit hydrogens, so a molecule here has substantially more '
    'nodes than the same molecule in qm40_smiles. Each atom carries its Mulliken partial '
    'charge as a node label, and bond lengths are derived from the geometry. Of the '
    '162,954 molecules the source publishes, 1,152 (0.71%) are excluded because their '
    'deposited geometry is not the geometry their own published properties were computed '
    'from: comparing the moments of inertia of each deposited structure against that '
    'molecule own published rotational values separates the two populations almost '
    'completely, with 99% of molecules agreeing to better than 0.3% and the excluded ones '
    'disagreeing by more than 1% and often far more. Targets 0 to 15 are the properties '
    'exactly as the source publishes them, and targets 16 to 20 are corrected, '
    'QM9-compatible restatements added by this collection, since three of the published '
    'columns cannot be compared against QM9 as they stand. A, B and C are the rotational '
    'constants in GHz, recovered from the published rotational temperatures in Kelvin '
    'that the publication mislabels as GHz; gap is the positive HOMO-LUMO gap in Hartree; '
    'and u0 is the zero-point-inclusive internal energy at 0 K in Hartree. Targets 16 to '
    '20 are exact functions of earlier targets and should not be used alongside their '
    'counterparts as independent targets.'
)

# :param METADATA:
#       Merged over the automatically derived metadata at save time.
METADATA: dict = {
    'license': {
        'identifier': 'CC-BY-4.0',
        'url': 'https://creativecommons.org/licenses/by/4.0/',
        'note': (
            'CC BY 4.0 is stated on the Figshare data record itself '
            '(10.6084/m9.figshare.25993060), which is the record the script downloads '
            'from - verified by matching the archive MD5. Note the accompanying '
            'Scientific Data article carries the more restrictive CC BY-NC-ND 4.0; '
            'the data record governs the data. The structures come from ZINC '
            '(wiki.docking.org), whose terms permit sharing "the results of a ZINC '
            'search or a screen of molecules from ZINC" and require written '
            'permission only for "major portions" of the database. The maintainer '
            'determined on 2026-08-19 that this subset is not a major portion, so '
            'ZINC is credited rather than treated as imposing terms. ZINC itself '
            'states no SPDX licence.'
        ),
    },
    'verbose': 'QM40 3D Geometry Dataset',
    'tags': [
        'Molecules',
        'SMILES',
        '3D coordinates',
        'Quantum Chemistry',
        'DFT',
        'Electronic Properties',
        'Drug Discovery',
        'Multi-Target',
    ],
    'sources': [
        'https://doi.org/10.1038/s41597-024-04206-y',
        'https://doi.org/10.6084/m9.figshare.25993060.v1',
        'https://github.com/Ayeshmadu/QM40_dataset_for_ML',
        'https://creativecommons.org/licenses/by/4.0/',
    ],
    'target_descriptions': {},  # filled in below from the shared definition
}

# The target descriptions are identical to those of qm40_smiles, and are written out in
# full rather than imported so that this script stays self-contained.
METADATA['license'] = {
    'identifier': 'CC-BY-4.0',
    'url': 'https://creativecommons.org/licenses/by/4.0/',
    'note': (
        'The data record is CC BY 4.0. Attribution: Mahamada Kalapuwage, A. M. (2024). '
        'QM40: A More Realistic QM Dataset for Machine Learning in Molecular Science. '
        'figshare. https://doi.org/10.6084/m9.figshare.25993060.v1 ; and the data '
        'descriptor Madushanka, A., Moura, R.T. & Kraka, E. Sci Data 11, 1376 (2024). '
        'https://doi.org/10.1038/s41597-024-04206-y . '
        'MODIFIED: this dataset is a reformatted subset of that record - 1,152 molecules '
        'with defective geometries are excluded, and five targets (A, B, C, gap, u0) are '
        'added that do not exist upstream. '
        'The molecules were screened from the ZINC database, which asks to be cited as '
        'Irwin, J. J. et al., J. Chem. Inf. Model. 52, 1757-1768 (2012), and whose terms '
        'of use (https://wiki.docking.org/index.php/Terms_And_Conditions) permit sharing '
        'the results of a screen of ZINC but not redistribution of major portions of it. '
        'The 3D coordinates are the authors own Gaussian16 output and are not ZINC '
        'content.'
    ),
}
METADATA['notes'] = [
    'Targets 16-20 (A, B, C, gap, u0) are exact deterministic functions of targets 0-15. '
    'They exist so that QM40 can be compared against QM9, and must not be used as '
    'independent targets alongside their counterparts.',
    'This dataset holds 161,800 of the 162,954 molecules the source publishes. The '
    'excluded ones failed one of two build-time checks: 1,152 whose deposited geometry '
    'is inconsistent with their own published rotational values by more than 1%, and a '
    'small number whose atom ordering did not match what RDKit produces from their '
    'SMILES. The sibling dataset qm40_smiles retains all 162,954, since without '
    'coordinates the first defect only makes three of its targets uncertain.',
    'Molecules carry explicit hydrogens, so node counts are roughly double those of the '
    'same molecule in qm40_smiles. Per-atom Mulliken partial charges are stored as '
    'node_labels, and bond lengths derived from the geometry as edge_lengths.',
    'The raw CSV export carries the scalar targets only. Coordinates and per-atom '
    'charges do not fit a flat table and are available in the message pack file.',
]

METADATA['target_descriptions'] = {
    '0': (
        'Internal_E(0K) - electronic (SCF) energy at the optimised geometry in Hartree, '
        'taken from the Gaussian SCF Done line; it follows the zero-point-exclusive '
        'convention, so use target 20 (u0) to compare against the zero-point-inclusive '
        'QM9 U0 target'
    ),
    '1': 'HOMO - energy of the highest occupied molecular orbital in Hartree',
    '2': 'LUMO - energy of the lowest unoccupied molecular orbital in Hartree',
    '3': (
        'HL_gap - HOMO minus LUMO in Hartree as published; in this sign convention it is '
        'always negative (range -0.375 to -0.009), so use target 19 (gap) for the '
        'conventional positive gap'
    ),
    '4': 'Polarizability - isotropic polarizability in cubic Bohr (a0^3)',
    '5': 'spatial extent - electronic spatial extent <R^2> in square Bohr (a0^2)',
    '6': 'dipol_mom - total dipole moment in Debye',
    '7': 'ZPE - zero-point vibrational energy in kcal/mol',
    '8': (
        'rot1 - largest rotational temperature in Kelvin as published; the source '
        'publication labels this a rotational constant in GHz, which it is not, so use '
        'target 16 (A) for the actual constant'
    ),
    '9': (
        'rot2 - second rotational temperature in Kelvin as published; see target 17 (B) '
        'for the actual rotational constant'
    ),
    '10': (
        'rot3 - smallest rotational temperature in Kelvin as published; see target 18 '
        '(C) for the actual rotational constant'
    ),
    '11': (
        'Inter_E(298) - internal energy at 298.15 K in Hartree, including zero-point and '
        'thermal corrections'
    ),
    '12': 'Enthalpy - enthalpy at 298.15 K in Hartree',
    '13': 'Free_E - Gibbs free energy at 298.15 K in Hartree',
    '14': 'CV - constant-volume heat capacity at 298.15 K in cal/(mol K)',
    '15': 'Entropy - total entropy at 298.15 K in cal/(mol K)',
    '16': (
        'A - CORRECTED, QM9-compatible: largest rotational constant in GHz, computed as '
        'rot1 divided by 0.04799243073 K per GHz because the published rot1 is a '
        'rotational temperature in Kelvin despite being labelled GHz. Directly '
        'comparable to the QM9 A target'
    ),
    '17': (
        'B - CORRECTED, QM9-compatible: second rotational constant in GHz, computed as '
        'rot2 divided by 0.04799243073 K per GHz. Directly comparable to the QM9 B target'
    ),
    '18': (
        'C - CORRECTED, QM9-compatible: smallest rotational constant in GHz, computed as '
        'rot3 divided by 0.04799243073 K per GHz. Directly comparable to the QM9 C target'
    ),
    '19': (
        'gap - CORRECTED, QM9-compatible: HOMO-LUMO gap in Hartree with the conventional '
        'positive sign, computed as LUMO minus HOMO, i.e. the negation of the published '
        'HL_gap. Directly comparable to the QM9 gap target'
    ),
    '20': (
        'u0 - CORRECTED, QM9-compatible: internal energy at 0 K in Hartree including the '
        'zero-point correction, computed as Internal_E(0K) plus ZPE divided by '
        '627.5094740631. Directly comparable to the QM9 u0 target'
    ),
}

__TESTING__ = False

experiment = Experiment.extend(
    'create_graph_datasets.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)


def _stream_download(e: Experiment, url: str, path: str, verify: bool = True) -> None:
    """
    Stream a large file to disk in chunks rather than buffering it in memory.

    :param e: The running experiment, used for logging.
    :param url: The URL to fetch.
    :param path: Local destination path.
    :param verify: Whether to verify the TLS certificate chain.

    :return: None
    """
    with requests.get(url, stream=True, timeout=600, verify=verify) as response:
        response.raise_for_status()
        with open(path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                file.write(chunk)


def rotational_temperatures(masses: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    """
    Principal rotational temperatures in Kelvin, in descending order.

    Shifts to the centre of mass, builds the inertia tensor, diagonalises it, converts the
    principal moments to rotational constants and then to rotational temperatures. This is
    the quantity the QM40 ``rot1``/``rot2``/``rot3`` columns actually hold, which is what
    makes it usable as a consistency check against the deposited geometry.

    :param masses: Atomic masses in amu, one per atom.
    :param coordinates: Cartesian coordinates in Angstrom, shape (n_atoms, 3).

    :return: Array of three rotational temperatures in Kelvin, descending.
    """
    centred = coordinates - (masses[:, None] * coordinates).sum(0) / masses.sum()
    inertia = np.zeros((3, 3))
    for mass, position in zip(masses, centred):
        inertia += mass * (
            np.dot(position, position) * np.eye(3) - np.outer(position, position)
        )
    moments = np.sort(np.linalg.eigvalsh(inertia))
    with np.errstate(divide='ignore'):
        return (GHZ_PER_INV_INERTIA / moments) * KELVIN_PER_GHZ


@experiment.hook('add_graph_metadata', default=False, replace=True)
def add_graph_metadata(e: Experiment, data: dict, graph: dict) -> None:
    """
    Attach the ZINC identifier and the per-atom Mulliken partial charges.

    The charges are stored as ``node_labels`` so that they can be used as atom-level
    regression targets. Their order matches the graph's node order because the molecule
    was built with the atoms in exactly the order the source lists them, which
    ``load_dataset`` verifies per molecule.

    :param e: The running experiment.
    :param data: The record dict this graph was built from.
    :param graph: The graph dict, mutated in place.

    :return: None
    """
    graph['graph_id'] = data['Zinc_id']
    graph['node_labels'] = np.array(data['charges'], dtype=float).reshape(-1, 1)


@experiment.hook('load_dataset', default=False, replace=True)
def load_dataset(e: Experiment) -> Dict[int, dict]:
    """
    Download the archive, join geometry with properties, and build the dataset.

    The xyz member is streamed and grouped by molecule rather than read whole - it is
    986 MB and 7.5 million rows. Molecules are rejected here for three reasons: an
    unparseable SMILES, an atom ordering that does not match what RDKit produces from that
    SMILES, or a deposited geometry inconsistent with the molecule's own published
    rotational values.

    :param e: The running experiment.

    :return: Mapping of contiguous integer index to record dicts, each carrying a 'mol'
        with a conformer, a 'targets' list of 21 floats, and per-atom 'charges'.
    """
    # -- download the archive (cached) --
    archive_path = os.path.join(e.path, 'QM40_dataset.zip')
    if not os.path.exists(archive_path):
        e.log('downloading QM40 archive (~332 MB), this takes a few minutes...')
        try:
            _stream_download(e, DOWNLOAD_URL, archive_path, verify=True)
        except requests.exceptions.SSLError:
            e.log('! TLS verification failed for Figshare; retrying unverified')
            e.log('! integrity is enforced by the MD5 check instead')
            _stream_download(e, DOWNLOAD_URL, archive_path, verify=False)
        e.log(f'downloaded {os.path.getsize(archive_path) / 1024**2:.1f} MB')
    else:
        e.log(f'using cached archive @ {archive_path}')

    digest = hashlib.md5()
    with open(archive_path, 'rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != ARCHIVE_MD5:
        raise RuntimeError(
            f'archive MD5 {digest.hexdigest()} does not match the expected '
            f'{ARCHIVE_MD5} - the download is corrupt or the record changed'
        )
    e.log(f'verified archive md5 {ARCHIVE_MD5}')

    # -- extract both members (cached) --
    paths: Dict[str, str] = {}
    for label, member in (('xyz', XYZ_MEMBER), ('main', MAIN_MEMBER)):
        path = os.path.join(e.path, os.path.basename(member))
        if not os.path.exists(path):
            e.log(f'extracting "{member}"...')
            with zipfile.ZipFile(archive_path) as archive:
                with archive.open(member) as source, open(path, 'wb') as file:
                    shutil.copyfileobj(source, file)
        paths[label] = path
        e.log(f'{label}: {os.path.getsize(path) / 1024**2:.1f} MB @ {path}')

    # -- the scalar properties, keyed by ZINC id --
    main_df = pd.read_csv(paths['main'])
    e.log(f'loaded {len(main_df)} property rows')

    targets_by_id: Dict[str, List[float]] = {}
    published_rotations: Dict[str, np.ndarray] = {}
    for record in main_df.to_dict('records'):
        values = [record[column] for column in SOURCE_TARGET_COLUMNS]
        if any(pd.isna(value) for value in values):
            continue
        values.extend(
            [
                record['rot1'] / KELVIN_PER_GHZ,
                record['rot2'] / KELVIN_PER_GHZ,
                record['rot3'] / KELVIN_PER_GHZ,
                record['LUMO'] - record['HOMO'],
                record['Internal_E(0K)'] + record['ZPE'] / KCAL_MOL_PER_HARTREE,
            ]
        )
        targets_by_id[record['Zinc_id']] = values
        published_rotations[record['Zinc_id']] = np.array(
            [record['rot1'], record['rot2'], record['rot3']], dtype=float
        )

    # -- stream the geometry, one molecule at a time --
    dataset: Dict[int, dict] = {}
    counts = {
        'no_properties': 0,
        'bad_smiles': 0,
        'atom_order': 0,
        'unknown_element': 0,
        'bad_geometry': 0,
    }

    def build(zinc_id, smiles, atoms, coordinates, charges) -> None:
        """Turn one buffered molecule into a dataset record, or count why it was not."""
        if zinc_id not in targets_by_id:
            counts['no_properties'] += 1
            return

        mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
        if mol is None:
            counts['bad_smiles'] += 1
            return
        mol = Chem.AddHs(mol)

        # The source lists atoms in RDKit's own order, so coordinates map positionally.
        # Verified per molecule rather than assumed - a mismatch would silently attach
        # every coordinate to the wrong atom, which no later check would catch.
        if [atom.GetSymbol() for atom in mol.GetAtoms()] != atoms:
            counts['atom_order'] += 1
            return

        try:
            masses = np.array([ATOMIC_MASSES[symbol] for symbol in atoms])
        except KeyError:
            counts['unknown_element'] += 1
            return

        positions = np.array(coordinates, dtype=float)

        # Is this geometry actually the geometry the properties were computed from?
        published = published_rotations[zinc_id]
        computed = rotational_temperatures(masses, positions)
        deviation = np.abs(computed - published) / np.maximum(np.abs(published), 5e-6)
        if deviation.max() > GEOMETRY_TOLERANCE:
            counts['bad_geometry'] += 1
            return

        conformer = Chem.Conformer(mol.GetNumAtoms())
        for index, position in enumerate(positions):
            conformer.SetAtomPosition(index, position.tolist())
        mol.AddConformer(conformer)

        dataset[len(dataset)] = {
            'Zinc_id': zinc_id,
            'smiles': smiles,
            'mol': mol,
            'charges': charges,
            'targets': targets_by_id[zinc_id],
        }

    e.log('streaming geometry file and grouping by molecule...')
    current_id = None
    current_smiles = None
    atoms: List[str] = []
    coordinates: List[tuple] = []
    charges: List[float] = []

    reader = pd.read_csv(
        paths['xyz'],
        usecols=['Zinc_id', 'smile', 'atom', 'final_x', 'final_y', 'final_z', 'charge'],
        chunksize=1_000_000,
    )
    for chunk in reader:
        for zinc_id, smiles, atom, x, y, z, charge in zip(
            chunk.Zinc_id.to_numpy(),
            chunk.smile.to_numpy(),
            chunk.atom.to_numpy(),
            chunk.final_x.to_numpy(),
            chunk.final_y.to_numpy(),
            chunk.final_z.to_numpy(),
            chunk.charge.to_numpy(),
        ):
            if zinc_id != current_id:
                if current_id is not None:
                    build(current_id, current_smiles, atoms, coordinates, charges)
                current_id = zinc_id
                current_smiles = smiles
                atoms, coordinates, charges = [], [], []
            atoms.append(atom)
            coordinates.append((x, y, z))
            charges.append(charge)
        e.log(f'  ... {len(dataset)} molecules built')

        # The base experiment truncates to 50 elements only *after* this hook returns,
        # which for a 986 MB per-atom file would mean streaming all 7.5 million rows to
        # smoke-test 50 molecules. Stop early instead.
        if e.__TESTING__ and len(dataset) >= 200:
            e.log('testing mode: stopping the stream early')
            break

    if current_id is not None:
        build(current_id, current_smiles, atoms, coordinates, charges)

    e.log(f'built dataset with {len(dataset)} molecules')
    for reason, count in counts.items():
        if count:
            e.log(f'  skipped {count} molecules: {reason}')

    return dataset


@experiment.hook('save_csv', default=False, replace=True)
def save_csv(e: Experiment, dataset: Dict[int, dict]) -> None:
    """
    Write the raw CSV export.

    The default hook expects a flat record with one key per target column, which does not
    hold here because the targets are assembled rather than read from columns. Geometry
    and per-atom charges are deliberately not exported - they do not fit a flat table and
    are available in the message pack file.

    The ``smiles`` column holds the *canonical* SMILES of the explicit-hydrogen molecule,
    which is what the processing step stores as each graph's ``graph_repr``. That looks
    redundant next to ``source_smiles`` but is load-bearing: the two artifacts of a run
    are checked for agreement by joining them on this column, and writing the source
    SMILES here instead makes every row look like a mismatch even though the ordering is
    correct. The source SMILES is kept alongside because the canonical explicit-H form is
    close to unreadable.

    :param e: The running experiment.
    :param dataset: The dataset mapping produced by ``load_dataset``.

    :return: None
    """
    path = os.path.join(e.path, f'{e.DATASET_NAME}.csv')
    with open(path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Zinc_id', 'smiles', 'source_smiles'] + TARGET_COLUMNS)
        for data in dataset.values():
            writer.writerow(
                [data['Zinc_id'], Chem.MolToSmiles(data['mol']), data['smiles']]
                + list(data['targets'])
            )
    e.log(f'saved csv @ {path}')


experiment.run_if_main()
