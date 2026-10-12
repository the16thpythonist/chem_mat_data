"""
Creates the ``qm40_smiles`` graph dataset from the QM40 main CSV file.

QM40 is a set of 162,954 neutral drug-like molecules screened out of the ZINC database
and annotated with 16 quantum-mechanical properties computed at B3LYP/6-31G(2df,p) in
Gaussian16. This script downloads the authors' Figshare archive, extracts the single
member that holds the per-molecule properties, and converts the SMILES strings into the
graph representation used throughout this package.

This dataset carries the molecular graphs and the scalar properties only. The 3D
geometries that the same archive publishes are a separate dataset, ``qm40_xyz``, built by
``create_graph_datasets__qm40_xyz.py`` - which is why this one is named for SMILES rather
than being called "main" as upstream does.

**Source**: Madushanka, A., Moura, R.T. & Kraka, E. *QM40, Realistic Quantum Mechanical
Dataset for Machine Learning in Molecular Science.* Sci Data **11**, 1376 (2024).
DOI: 10.1038/s41597-024-04206-y — data record DOI: 10.6084/m9.figshare.25993060.v1

**Targets**: 21 values per molecule.

Targets 0-15 are the source columns unchanged, in the order the CSV declares them —
electronic energy, HOMO, LUMO, HOMO-LUMO difference (all Hartree), isotropic
polarizability (a0^3), electronic spatial extent (a0^2), dipole moment (Debye),
zero-point energy (kcal/mol), three rotational temperatures (Kelvin), internal energy,
enthalpy and free energy at 298.15 K (Hartree), heat capacity and entropy (cal/mol/K).

Targets 16-20 — ``A``, ``B``, ``C``, ``gap``, ``u0`` — are added by this collection and
do not exist upstream. They restate three of the columns above in the form QM9 uses, so
that the two datasets can actually be concatenated, and they carry QM9's own names for
exactly that reason. Each is an exact deterministic function of an earlier target, so
**they must not be treated as independent targets alongside their originals** - a
multi-target model trained on all 21 would see five perfectly collinear labels.

**Two columns do not mean what the paper's Table 2 suggests.** Only the first is an
error; the second is a convention difference. Both were established against the data and
against the authors' own published parser, and both matter to anyone following the
paper's suggestion to concatenate QM40 with QM9.

- ``rot1``/``rot2``/``rot3`` are **mislabelled**. Table 2 documents them as rotational
  constants in GHz. They are rotational *temperatures* in Kelvin. Computing moments of
  inertia from the optimised geometries the authors also publish (``QM40_xyz.csv``) and
  comparing against these columns over 6,519 molecules recovers a ratio to the GHz
  constant of 0.047992566 against h/kB = 0.047992431 - agreement to 2.8 parts per
  million, with 99.9% of the residual accounted for by the 5-decimal rounding of the
  published values. The cm^-1 reading is off by -30.5% and is excluded outright.
  Gaussian16 prints "Rotational temperatures (Kelvin)" on the line directly above
  "Rotational constants (GHZ)", which is the likely cause. These columns are therefore
  *not* comparable to the QM9 ``A``/``B``/``C`` targets, which are true GHz constants.
- ``Internal_E(0K)`` is the **electronic (SCF) energy**, not a zero-point-inclusive
  U(0). This is a convention difference rather than a mistake - Table 2's stated unit
  (Hartree) is correct, and under the Born-Oppenheimer convention the electronic energy
  is the classical 0 K internal energy. It is confirmed directly by the authors'
  published extractor (``gaussian_info_extractor.py``), which parses this column from
  Gaussian's ``SCF Done`` line while taking ``Inter_E(298)`` from the separate,
  ZPE-inclusive "Sum of electronic and thermal Energies" keyword. QM9's ``U0`` *is*
  ZPE-inclusive, so the two differ by exactly the ``ZPE`` column: add
  ``ZPE / 627.5094740631`` Hartree to compare.

``HL_gap`` is worth flagging for a third reason, and here the paper is self-consistent:
it is defined as HOMO minus LUMO and so is always negative, the negative of the
conventional band gap. Verified on all 162,954 rows - zero are non-negative.

The source data needs no cleaning. All 162,954 rows parse, none carry missing values,
there are no duplicate structures, no multi-component SMILES and no charged species, so
the guards in ``load_dataset`` below are expected to reject nothing.
"""

import os
import csv
import hashlib
import shutil
import zipfile
from typing import Dict, List

import requests
import pandas as pd
import rdkit.Chem as Chem
from rdkit import RDLogger
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

RDLogger.DisableLog('rdApp.*')

# == SOURCE PARAMETERS ==

# :param DOWNLOAD_URL:
#       The direct download URL of the QM40 archive on Figshare. This is the only
#       distribution the authors offer - the three CSV files that make up QM40 are not
#       individually downloadable - so the full 332 MB archive has to be fetched even
#       though this script only parses the 35 MB member.
DOWNLOAD_URL: str = 'https://ndownloader.figshare.com/files/47535647'
# :param ARCHIVE_MEMBER:
#       The path of the properties CSV inside the archive. Note the space in the folder
#       name, which is part of the actual member path.
ARCHIVE_MEMBER: str = 'QM40 dataset/QM40_main.csv'
# :param ARCHIVE_MD5:
#       The MD5 Figshare publishes for the archive (its "supplied_md5", which also
#       matches its "computed_md5"). The download is checked against this. That matters
#       more than usual here: Figshare's certificate chain is unreliable from some
#       networks, so the download falls back to an unverified TLS connection, and this
#       hash is what makes that fallback safe rather than merely convenient.
ARCHIVE_MD5: str = '337773160d22b762399e4745530df6c2'
# :param SMILES_COLUMN:
#       The name of the CSV column that holds the SMILES strings.
SMILES_COLUMN: str = 'smile'
# :param SOURCE_TARGET_COLUMNS:
#       The CSV columns that are copied straight through as targets 0-15, in this exact
#       order. "spatial extent" really is spelled with a space in the source file, unlike
#       in the publication's Table 2 and the authors' own reader module.
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
#       Targets 16-20, which do not exist in the source file. They are corrected,
#       QM9-compatible restatements of columns that cannot be compared against QM9 as
#       published, and they deliberately carry QM9's own names so that the correspondence
#       is obvious to anyone concatenating the two datasets. See the module docstring for
#       what is wrong with each original and how it was established.
DERIVED_TARGET_COLUMNS: List[str] = ['A', 'B', 'C', 'gap', 'u0']
# :param TARGET_COLUMNS:
#       The full ordered target vector. The default "save_csv" hook uses these to name
#       the columns of the exported raw CSV, so the list has to stay correct even though
#       "load_dataset" is overridden below.
TARGET_COLUMNS: List[str] = SOURCE_TARGET_COLUMNS + DERIVED_TARGET_COLUMNS

# :param KELVIN_PER_GHZ:
#       h / k_B expressed as Kelvin per GHz, used to turn the published rotational
#       temperatures back into rotational constants. Derived from the SI-exact CODATA
#       values h = 6.62607015e-34 J s and k_B = 1.380649e-23 J/K.
KELVIN_PER_GHZ: float = 0.04799243073
# :param HARTREE_PER_KCAL_MOL:
#       Conversion used to add the ZPE column, which the source publishes in kcal/mol,
#       onto the electronic energy, which it publishes in Hartree.
KCAL_MOL_PER_HARTREE: float = 627.5094740631

DATASET_NAME: str = 'qm40_smiles'
DATASET_TYPE: str = 'regression'

DESCRIPTION: str = (
    'QM40 is a collection of 162,954 neutral, drug-like organic molecules screened from '
    'the ZINC database, each annotated with 16 quantum-mechanical properties computed at '
    'the B3LYP/6-31G(2df,p) level of theory in Gaussian16. Molecules contain 10 to 37 '
    'heavy atoms drawn from C, N, O, S, F and Cl, a chemical space chosen to cover '
    'roughly 88% of FDA-approved drugs and to extend the smaller QM9 dataset toward '
    'drug-sized systems. Geometries were pre-optimised with GFN2-xTB before DFT '
    'optimisation and frequency analysis, and structures showing convergence failures, '
    'imaginary frequencies or unphysical local-mode parameters were removed by the '
    'authors. This dataset contains the molecular graphs and scalar properties only; the '
    'corresponding 3D geometries are published separately as qm40_xyz. Targets 0 to 15 '
    'are the properties exactly as the source publishes them. Targets 16 to 20 are '
    'corrected, QM9-compatible versions added by this collection, because three of the '
    'published columns cannot be compared against QM9 as they stand: A, B and C are the '
    'rotational constants in GHz, recovered from the published rotational temperatures in '
    'Kelvin that the publication mislabels as GHz; gap is the positive HOMO-LUMO gap in '
    'Hartree, the sign-flipped counterpart of the published HL_gap; and u0 is the '
    'zero-point-inclusive internal energy at 0 K in Hartree, formed by adding the ZPE '
    'column to the zero-point-exclusive electronic energy the publication reports. The '
    'original columns are retained unchanged, so targets 16 to 20 are exact functions of '
    'earlier targets and should not be used alongside their counterparts as independent '
    'targets. The rotational constants inherit the five-decimal rounding of the published '
    'temperatures, which reaches about half a percent on the smallest values.'
)

# :param METADATA:
#       A dictionary which will be used as the basis for the metadata that will be added
#       as additional information to the file share server.
METADATA: dict = {
    'license': {
        'identifier': 'CC-BY-4.0',
        'url': 'https://creativecommons.org/licenses/by/4.0/',
        'note': (
            'CC BY 4.0 is stated on the Figshare data record itself '
            '(10.6084/m9.figshare.25993060), which is provably the record this script '
            "downloads - the archive MD5 matches the record's. Note the accompanying "
            'Scientific Data article carries the more restrictive CC BY-NC-ND 4.0; '
            'the data record governs the data, not the article. ATTRIBUTION: Mahamada '
            'Kalapuwage, A. M. (2024), QM40, figshare, '
            'doi:10.6084/m9.figshare.25993060.v1; and Madushanka, A., Moura, R. T. & '
            'Kraka, E., Sci Data 11, 1376 (2024), doi:10.1038/s41597-024-04206-y. '
            'MODIFIED, as CC BY requires be indicated: reformatted, and five targets '
            '(A, B, C, gap, u0) added that do not exist upstream. The structures were '
            'screened from ZINC (cite Irwin, J. J. et al., J. Chem. Inf. Model. 52, '
            '1757-1768 (2012)), whose terms permit sharing "the results of a ZINC '
            'search or a screen of molecules from ZINC" and require written '
            'permission only for "major portions". The maintainer determined on '
            '2026-08-19 that this subset (162,954 molecules, ~0.016% of ZINC20) is '
            'not a major portion.'
        ),
    },
    'verbose': 'QM40 SMILES Dataset',
    'tags': [
        'Molecules',
        'SMILES',
        'Quantum Chemistry',
        'DFT',
        'Electronic Properties',
        'Drug Discovery',
        'Multi-Target',
    ],
    # The data record is CC BY 4.0, which requires attribution and an indication that the
    # material was modified - this dataset is a reformatted subset of it. The licence URL
    # is listed here so that requirement travels with the dataset. ZINC, from which the
    # structures were screened, asks to be cited as Irwin et al., J. Chem. Inf. Model.
    # 52, 1757-1768 (2012).
    'sources': [
        'https://doi.org/10.1038/s41597-024-04206-y',
        'https://doi.org/10.6084/m9.figshare.25993060.v1',
        'https://github.com/Ayeshmadu/QM40_dataset_for_ML',
        'https://creativecommons.org/licenses/by/4.0/',
    ],
    'notes': [
        'Targets 16-20 (A, B, C, gap, u0) are exact deterministic functions of targets '
        '0-15. They exist so that QM40 can be compared against QM9, and must not be '
        'used as independent targets alongside their counterparts - a model trained on '
        'all 21 sees five perfectly collinear labels.',
        'For 1,152 of the 162,954 molecules (0.71%), the published rotational values - '
        'targets 8-10 and the derived 16-18 - are inconsistent with the optimised '
        'geometry the same authors deposited, disagreeing by more than 1% and in the '
        'worst case by 265%. The deposit contains no geometry that reproduces the '
        'published numbers, so which side is wrong could not be determined; the other '
        'targets of those molecules behave normally. They are retained here so that '
        'this dataset matches the published 162,954 exactly, but targets 8-10 and 16-18 '
        'should be treated as uncertain for them. The sibling dataset qm40_xyz excludes '
        'these molecules, because for a geometry dataset they are unusable.',
        '5,325 molecules (3.27%) are zwitterions: net-neutral, as the source intends, '
        'but containing formally charged atoms that cancel. They account for the entire '
        'high tail of the dipole moment target, which reaches 59.3 D against 15.6 D for '
        'everything else. A filter on per-atom formal charge, rather than net charge, '
        'removes them.',
    ],
    'target_descriptions': {
        '0': (
            'Internal_E(0K) - electronic (SCF) energy at the optimised geometry in '
            'Hartree, taken from the Gaussian SCF Done line; it follows the '
            'zero-point-exclusive convention, so use target 20 (u0) to compare against '
            'the zero-point-inclusive QM9 U0 target'
        ),
        '1': 'HOMO - energy of the highest occupied molecular orbital in Hartree',
        '2': 'LUMO - energy of the lowest unoccupied molecular orbital in Hartree',
        '3': (
            'HL_gap - HOMO minus LUMO in Hartree as published; in this sign convention it '
            'is always negative (range -0.375 to -0.009), so use target 19 (gap) for the '
            'conventional positive gap'
        ),
        '4': 'Polarizability - isotropic polarizability in cubic Bohr (a0^3)',
        '5': 'spatial extent - electronic spatial extent <R^2> in square Bohr (a0^2)',
        '6': 'dipol_mom - total dipole moment in Debye',
        '7': 'ZPE - zero-point vibrational energy in kcal/mol',
        '8': (
            'rot1 - largest rotational temperature in Kelvin as published; the source '
            'publication labels this a rotational constant in GHz, which it is not, so '
            'use target 16 (A) for the actual constant'
        ),
        '9': (
            'rot2 - second rotational temperature in Kelvin as published; see target 17 '
            '(B) for the actual rotational constant'
        ),
        '10': (
            'rot3 - smallest rotational temperature in Kelvin as published; see target 18 '
            '(C) for the actual rotational constant'
        ),
        '11': (
            'Inter_E(298) - internal energy at 298.15 K in Hartree, including zero-point '
            'and thermal corrections'
        ),
        '12': 'Enthalpy - enthalpy at 298.15 K in Hartree',
        '13': 'Free_E - Gibbs free energy at 298.15 K in Hartree',
        '14': 'CV - constant-volume heat capacity at 298.15 K in cal/(mol K)',
        '15': 'Entropy - total entropy at 298.15 K in cal/(mol K)',
        '16': (
            'A - CORRECTED, QM9-compatible: largest rotational constant in GHz, computed '
            'as rot1 divided by 0.04799243073 K per GHz because the published rot1 is a '
            'rotational temperature in Kelvin despite being labelled GHz. Directly '
            'comparable to the QM9 A target'
        ),
        '17': (
            'B - CORRECTED, QM9-compatible: second rotational constant in GHz, computed '
            'as rot2 divided by 0.04799243073 K per GHz. Directly comparable to the QM9 B '
            'target'
        ),
        '18': (
            'C - CORRECTED, QM9-compatible: smallest rotational constant in GHz, computed '
            'as rot3 divided by 0.04799243073 K per GHz. Directly comparable to the QM9 C '
            'target'
        ),
        '19': (
            'gap - CORRECTED, QM9-compatible: HOMO-LUMO gap in Hartree with the '
            'conventional positive sign, computed as LUMO minus HOMO, i.e. the negation '
            'of the published HL_gap. Directly comparable to the QM9 gap target'
        ),
        '20': (
            'u0 - CORRECTED, QM9-compatible: internal energy at 0 K in Hartree including '
            'the zero-point correction, computed as Internal_E(0K) plus ZPE divided by '
            '627.5094740631. Directly comparable to the QM9 u0 target'
        ),
    },
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
    Stream a large file to disk in chunks.

    Streaming rather than buffering matters here - the archive is 332 MB and would
    otherwise be held in memory in full.

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


@experiment.hook('add_graph_metadata', default=False, replace=True)
def add_graph_metadata(e: Experiment, data: dict, graph: dict) -> None:
    """
    Attach the ZINC registry identifier of the molecule to its graph.

    QM40 does not assign its own identifiers - the first column of the source CSV is the
    ZINC id the molecule was screened under. Keeping it makes each graph traceable back
    to the upstream database entry it came from.

    :param e: The running experiment.
    :param data: The record dict that this graph was built from.
    :param graph: The graph dict to attach the metadata to, mutated in place.

    :return: None
    """
    graph['graph_id'] = data['Zinc_id']


@experiment.hook('load_dataset', default=False, replace=True)
def load_dataset(e: Experiment) -> Dict[int, dict]:
    """
    Download the QM40 archive, extract the properties CSV and build the dataset dict.

    Both the archive and the extracted member are cached inside the experiment folder, so
    a re-run does not repeat the 332 MB download. The archive is streamed rather than
    held in memory, and only the one member this dataset needs is unpacked - the other
    two members are 986 MB and 679 MB and belong to different datasets.

    :param e: The running experiment.

    :return: Mapping of contiguous integer index to record dicts, each carrying at least
        a 'smiles' string and a 'targets' list of 16 floats.
    """
    # -- download the archive (cached) --
    archive_path = os.path.join(e.path, 'QM40_dataset.zip')
    if not os.path.exists(archive_path):
        e.log('downloading QM40 archive (~332 MB), this takes a few minutes...')
        try:
            _stream_download(e, DOWNLOAD_URL, archive_path, verify=True)
        except requests.exceptions.SSLError:
            # Figshare's certificate chain does not validate from every network - a
            # TLS-inspecting proxy presents its own root, which Python does not trust
            # even where curl does. The reference guidance for this collection is to
            # stop fighting it, and the MD5 check below is what keeps that safe.
            e.log('! TLS verification failed for Figshare; retrying unverified')
            e.log('! integrity is enforced by the MD5 check instead')
            _stream_download(e, DOWNLOAD_URL, archive_path, verify=False)
        e.log(f'downloaded {os.path.getsize(archive_path) / 1024**2:.1f} MB')
    else:
        e.log(f'using cached archive @ {archive_path}')

    # -- verify the download against the hash Figshare publishes --
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

    # -- extract only the properties CSV (cached) --
    csv_path = os.path.join(e.path, 'QM40_main.csv')
    if not os.path.exists(csv_path):
        e.log(f'extracting "{ARCHIVE_MEMBER}"...')
        with zipfile.ZipFile(archive_path) as archive:
            with archive.open(ARCHIVE_MEMBER) as source, open(csv_path, 'wb') as file:
                shutil.copyfileobj(source, file)
        e.log(f'extracted {os.path.getsize(csv_path) / 1024**2:.1f} MB')
    else:
        e.log(f'using cached csv @ {csv_path}')

    df = pd.read_csv(csv_path)
    e.log(f'loaded {len(df)} rows with columns {list(df.columns)}')

    dataset: Dict[int, dict] = {}
    skipped: int = 0
    for data in df.to_dict('records'):
        smiles = data[SMILES_COLUMN]

        # == MOLECULE FILTERS ==
        # The source data is already clean - every row parses and none is missing a
        # value - so these guards are a safety net rather than a filter. Anything they
        # do reject would mean the file changed upstream, which is worth seeing in the
        # log rather than silently tolerating.
        if not isinstance(smiles, str) or not smiles:
            skipped += 1
            continue

        if Chem.MolFromSmiles(smiles) is None:
            skipped += 1
            continue

        # == TARGETS ==
        # All 16 source targets are present on every row, so no missing-value sentinel is
        # used here. That matters: several of these targets - the orbital energies, the
        # HOMO-LUMO difference and all three energies in Hartree - take genuinely
        # negative values, and a -1 sentinel would be indistinguishable from a real
        # measurement. Rows missing a target are dropped instead.
        targets = [data[column] for column in SOURCE_TARGET_COLUMNS]
        if any(pd.isna(value) for value in targets):
            skipped += 1
            continue

        # -- corrected, QM9-compatible restatements (targets 16-20) --
        # Exact deterministic functions of the columns above, appended rather than
        # substituted so that what the source actually published stays in the dataset.
        targets.extend(
            [
                data['rot1'] / KELVIN_PER_GHZ,  # A, rotational constant in GHz
                data['rot2'] / KELVIN_PER_GHZ,  # B
                data['rot3'] / KELVIN_PER_GHZ,  # C
                data['LUMO'] - data['HOMO'],  # gap, positive by convention
                data['Internal_E(0K)'] + data['ZPE'] / KCAL_MOL_PER_HARTREE,  # u0
            ]
        )

        data['smiles'] = smiles
        data['targets'] = targets
        dataset[len(dataset)] = data

    e.log(f'built dataset with {len(dataset)} molecules ({skipped} skipped)')
    return dataset


@experiment.hook('save_csv', default=False, replace=True)
def save_csv(e: Experiment, dataset: Dict[int, dict]) -> None:
    """
    Write the raw CSV export with the ZINC identifier retained.

    The default hook writes only ``smiles`` plus the target columns, which would drop the
    ZINC id from the CSV even though the message pack keeps it as ``graph_id``. That
    matters here beyond convenience: the licence approval for this dataset was given on
    the explicit condition that the ZINC identifier ship as provenance, so a CSV without
    it would not honour the terms this dataset is distributed under. It is also the only
    key by which a user can join this dataset against ``qm40_xyz``.

    The ``smiles`` column must stay byte-identical to each graph's ``graph_repr`` - the
    two artifacts are checked for agreement by joining on it.

    :param e: The running experiment.
    :param dataset: The dataset mapping produced by ``load_dataset``.

    :return: None
    """
    path = os.path.join(e.path, f'{e.DATASET_NAME}.csv')
    with open(path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Zinc_id', 'smiles'] + TARGET_COLUMNS)
        for data in dataset.values():
            writer.writerow([data['Zinc_id'], data['smiles']] + list(data['targets']))
    e.log(f'saved csv @ {path}')


experiment.run_if_main()
