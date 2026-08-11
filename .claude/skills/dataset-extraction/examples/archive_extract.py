"""
ARCHETYPE 2 of 4 — data delivered as an archive, or in a custom non-tabular format.

Reach for this when the dataset is not a plain CSV: a ``.zip``/``.tar.gz`` holding the
real file, or a bespoke text format that needs a hand-written parser. Modelled on
``create_graph_datasets__hopv15.py`` (HOPV15, 350 organic photovoltaic molecules in a
custom multi-line format interleaving SMILES, InChI, experimental properties and 3D
conformer geometries).

What this example demonstrates:

- Downloading and extracting an archive, inspecting members before trusting them.
- A standalone parser function for a custom format, kept separate from the hook.
- The ``-1.0`` sentinel for missing values in a multi-target dataset, and documenting
  that choice in ``target_descriptions``.
- Carrying extra provenance fields through to ``add_graph_metadata``.

The archive handling and the custom parser are shown together because they usually
travel together; drop whichever half you do not need.
"""

import os
import zipfile
import tarfile
from typing import Dict, List

import requests
import pandas as pd
import rdkit.Chem as Chem
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

# == SOURCE PARAMETERS ==

# :param DOWNLOAD_URL:
#       Direct download URL. Figshare 'ndownloader' links are per-file ids, not the
#       article id - see references/data-sources.md.
DOWNLOAD_URL: str = 'https://ndownloader.figshare.com/files/4513735'

# :param ARCHIVE_MEMBER:
#       Path of the relevant file inside the archive, or None when the download is a
#       plain file rather than an archive.
ARCHIVE_MEMBER: str = None

# :param DATASET_TYPE:
DATASET_TYPE: str = 'regression'

# :param TARGET_COLUMNS:
#       Documents the target order for the CSV export and for readers of this script.
TARGET_COLUMNS: List[str] = [
    'HOMO',
    'LUMO',
    'electrochemical_gap',
    'optical_gap',
    'PCE',
    'VOC',
    'JSC',
    'fill_factor',
]

DESCRIPTION: str = (
    'The Harvard Organic Photovoltaic 15 (HOPV15) dataset contains 350 small molecules '
    'and polymers used in organic photovoltaic applications, collected from the '
    'experimental literature. Each compound carries up to 8 experimental regression '
    'targets: HOMO energy, LUMO energy, electrochemical gap, optical gap, power '
    'conversion efficiency, open-circuit voltage, short-circuit current density and '
    'fill factor. Not every property was measured for every compound; missing values '
    'are encoded as -1. The dataset is a standard benchmark for models predicting '
    'photovoltaic device performance from molecular structure.'
)

METADATA: dict = {
    'verbose': 'HOPV15 - Harvard Organic Photovoltaic Dataset',
    'tags': [
        'Molecules',
        'SMILES',
        'Organic Electronics',
        'Solar Cells',
        'Electronic Properties',
        'Experimental',
    ],
    'sources': [
        'https://doi.org/10.1038/sdata.2016.86',
        'https://figshare.com/articles/dataset/HOPV15_Dataset/1610063',
    ],
    'target_descriptions': {
        '0': 'HOMO - Highest occupied molecular orbital energy (eV); -1 where not measured',
        '1': 'LUMO - Lowest unoccupied molecular orbital energy (eV); -1 where not measured',
        '2': 'electrochemical_gap - Electrochemical HOMO-LUMO gap (eV); -1 where not measured',
        '3': 'optical_gap - Optical gap (eV); -1 where not measured',
        '4': 'PCE - Power conversion efficiency (%); -1 where not measured',
        '5': 'VOC - Open-circuit voltage (V); -1 where not measured',
        '6': 'JSC - Short-circuit current density (mA/cm^2); -1 where not measured',
        '7': 'fill_factor - Fill factor (%); -1 where not measured',
    },
}

__TESTING__ = False

experiment = Experiment.extend(
    'create_graph_datasets.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)


def _download(url: str, path: str, e: Experiment) -> str:
    """
    Download ``url`` to ``path`` unless the file already exists.

    Uses a streamed request with chunked writes so that large archives do not have to
    be held in memory in full.

    :param url: The direct download URL.
    :param path: Absolute destination path.
    :param e: The running experiment, used for logging.

    :returns: The destination path.
    """
    if os.path.exists(path):
        e.log(f'using cached download @ {path}')
        return path

    e.log(f'downloading from {url} ...')
    with requests.get(url, stream=True, timeout=300) as response:
        response.raise_for_status()
        with open(path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

    e.log(f'downloaded {os.path.getsize(path) / 1024**2:.1f} MB')
    return path


def _extract(archive_path: str, member: str, target_dir: str, e: Experiment) -> str:
    """
    Extract a single member from a zip or tar archive.

    Always logs the member list first: archives routinely nest the real data one or
    two directories deep, and guessing the member path wastes a run attempt.

    :param archive_path: Path to the downloaded archive.
    :param member: Path of the desired file inside the archive.
    :param target_dir: Directory to extract into.
    :param e: The running experiment, used for logging.

    :returns: Absolute path of the extracted file.
    """
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as archive:
            e.log(f'archive members: {archive.namelist()[:20]}')
            archive.extract(member, path=target_dir)

    elif tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path) as archive:
            e.log(f'archive members: {archive.getnames()[:20]}')
            archive.extract(member, path=target_dir)

    else:
        raise ValueError(f'not a recognised archive format: {archive_path}')

    return os.path.join(target_dir, member)


def _parse_hopv15(path: str) -> List[dict]:
    """
    Parse the HOPV15 custom text format into a list of molecule records.

    Each molecule block has the structure::

        <SMILES>
        <InChI>
        <DOI>,<InChIKey>,<type>,<measurement>,<acceptor>,<8 numeric properties>
        <SMILES repeated>
        <number of conformers>
        [conformer blocks, each: header, atom count, coordinates, 4 DFT result lines]

    Only the experimental properties are kept; conformer geometry and the DFT results
    are skipped.

    :param path: Path to the HOPV15 data file.

    :returns: List of record dicts with ``smiles``, ``targets`` and provenance fields.
    """
    with open(path, 'r') as file:
        lines = file.readlines()

    records: List[dict] = []
    index: int = 0
    total: int = len(lines)

    while index < total:
        line = lines[index].strip()
        if not line:
            index += 1
            continue

        smiles = line
        index += 2  # skip the InChI line

        if index >= total:
            break

        fields = lines[index].strip().split(',')
        index += 1

        # DOI, InChIKey, type, measurement, acceptor, then 8 numeric properties
        if len(fields) < 13:
            continue

        targets: List[float] = []
        for value in fields[5:13]:
            value = value.strip()
            try:
                targets.append(float(value))
            except ValueError:
                # 'nan' and empty entries become the missing-value sentinel
                targets.append(-1.0)

        index += 1  # skip the repeated SMILES line

        try:
            num_conformers = int(lines[index].strip())
        except (ValueError, IndexError):
            num_conformers = 0
        index += 1

        # skip the conformer blocks entirely
        for _ in range(num_conformers):
            index += 1  # 'Conformer N' header
            if index >= total:
                break
            try:
                num_atoms = int(lines[index].strip())
            except ValueError:
                num_atoms = 0
            index += 1 + num_atoms + 4  # coordinates plus 4 DFT result lines

        records.append(
            {
                'smiles': smiles,
                'doi': fields[0],
                'inchikey': fields[1],
                'mol_type': fields[2],
                'measurement': fields[3],
                'acceptor': fields[4],
                'targets': targets,
            }
        )

    return records


@experiment.hook('add_graph_metadata', default=False, replace=True)
def add_graph_metadata(e: Experiment, data: dict, graph: dict) -> None:
    """
    Carry the literature provenance through onto each graph.

    :param e: The running experiment.
    :param data: The original record dict from ``load_dataset``.
    :param graph: The graph dict produced by ``MoleculeProcessing``.
    """
    graph['graph_doi'] = data.get('doi', '')
    graph['graph_inchikey'] = data.get('inchikey', '')
    graph['graph_measurement'] = data.get('measurement', '')
    graph['graph_acceptor'] = data.get('acceptor', '')


@experiment.hook('load_dataset', default=False, replace=True)
def load_dataset(e: Experiment) -> Dict[int, dict]:
    """
    Download the dataset (cached), extract it if it is an archive, parse the custom
    format, and return the index → record mapping.

    :param e: The running experiment.

    :returns: Mapping of integer index to record dicts containing ``smiles`` and a
        ``targets`` list of 8 floats.
    """
    download_path = os.path.join(e.path, 'hopv15_raw.data')
    _download(DOWNLOAD_URL, download_path, e)

    # When the download is an archive, pull out the member that holds the data.
    if ARCHIVE_MEMBER:
        data_path = _extract(download_path, ARCHIVE_MEMBER, e.path, e)
    else:
        data_path = download_path

    e.log('parsing the raw data file...')
    records = _parse_hopv15(data_path)
    e.log(f'parsed {len(records)} records')

    dataset: Dict[int, dict] = {}
    skipped_empty: int = 0
    skipped_invalid: int = 0

    for record in records:
        smiles = record['smiles']

        if not smiles or pd.isna(smiles):
            skipped_empty += 1
            continue

        if Chem.MolFromSmiles(smiles) is None:
            skipped_invalid += 1
            continue

        dataset[len(dataset)] = record

    e.log(
        f'built dataset with {len(dataset)} molecules '
        f'({skipped_empty} without SMILES, {skipped_invalid} unparseable)'
    )
    return dataset


experiment.run_if_main()
