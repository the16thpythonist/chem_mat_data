"""
This experiment creates the processed graph dataset for **tmQM** — the foundational
large-scale dataset of ~108,000 mononuclear transition metal complexes with 8 DFT-computed
quantum-mechanical properties.

The raw data is downloaded from the tmQM GitHub repository (2024 release). Each complex
is represented as a whole-complex SMILES string that includes dative bond notation
(``->`` / ``<-``) for metal-ligand coordination bonds. The script decomposes these
into the metal center, individual ligand SMILES, and exact connecting atom indices
using :func:`decompose_complex_smiles`, then uses ``MetalOrganicProcessing`` to build
the molecular graphs.

**Source**: Balcells & Skjelstad, *J. Chem. Inf. Model.* 2020, 60, 6135;
updated 2024 release with SMILES (108k complexes).

**Targets** (8 regression properties):
    Electronic energy, dispersion energy, dipole moment, metal natural charge,
    HOMO-LUMO gap, HOMO energy, LUMO energy, polarizability.

**Usage**::

    python create_graph_datasets__tmqm.py

Results are written to ``results/create_graph_datasets__tmqm/debug/``.
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
from chem_mat_data.tmc_processing import (
    MetalOrganicProcessing,
    TRANSITION_METAL_ATOMIC_NUMBERS,
    decompose_complex_smiles,
)
from chem_mat_data.data import default, ext_hook, save_graphs


# == SOURCE PARAMETERS ==
# These parameters configure where the raw dataset is fetched from and how it is
# interpreted. The tmQM dataset (2024 release) is hosted on GitHub as a semicolon-
# delimited CSV with CSD identifiers, 8 QM properties, and whole-complex SMILES.

# :param DOWNLOAD_URL:
#       The URL from which the raw tmQM CSV file will be downloaded. This points
#       to the 2024 release of the tmQM dataset on GitHub. The file is approximately
#       20 MB and contains ~108,000 rows. Note that ~7,700 rows have missing SMILES
#       and will be skipped.
DOWNLOAD_URL: str = 'https://raw.githubusercontent.com/uiocompcat/tmQM/master/tmQM/tmQM_y.csv'

# :param TARGET_COLUMNS:
#       A list of the 8 column names in the raw CSV that contain the DFT-computed
#       quantum-mechanical properties. These become the regression targets (graph_labels)
#       in the processed graph dataset. The order here determines the order in the
#       target vector.
#
#       Properties are computed at the TPSSh-D3BJ/def2-SVP level of theory, except for
#       polarizability which is at the GFN2-xTB level.
TARGET_COLUMNS: List[str] = [
    'Electronic_E',
    'Dispersion_E',
    'Dipole_M',
    'Metal_q',
    'HL_Gap',
    'HOMO_Energy',
    'LUMO_Energy',
    'Polarizability',
]

# :param DATASET_TYPE:
#       Either 'regression' or 'classification'. All 8 tmQM targets are continuous
#       QM properties, so this is always 'regression'.
DATASET_TYPE: str = 'regression'

# :param DESCRIPTION:
#       A human-readable description of the dataset that will be stored in the
#       metadata.yml file alongside the processed dataset.
DESCRIPTION: str = (
    'Graph dataset of ~100,000 mononuclear transition metal complexes from the tmQM '
    'dataset (Balcells & Skjelstad, JCIM 2020; 2024 release) with 8 DFT-computed '
    'quantum-mechanical properties (TPSSh-D3BJ/def2-SVP). Whole-complex SMILES are '
    'decomposed into metal + ligand fragments using decompose_complex_smiles() and '
    'processed using MetalOrganicProcessing.'
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
        'https://doi.org/10.1021/acs.jcim.0c01041',
        'https://github.com/uiocompcat/tmQM',
    ],
    'verbose': 'tmQM Transition Metal Complex QM Properties',
    'target_descriptions': {
        str(i): col for i, col in enumerate(TARGET_COLUMNS)
    },
}

# == PROCESSING PARAMETERS ==

# :param DATASET_NAME:
#       The base filename for all output files (CSV, mpack, metadata).
DATASET_NAME: str = 'tmqm'

# :param COMPRESS:
#       If True, the mpack file is additionally compressed to gzip format.
COMPRESS: bool = True

# :param MAX_ELEMENTS:
#       Maximum number of complexes to process. Set to None to process the full
#       dataset. Set to a smaller number (e.g. 100) for quick verification runs.
MAX_ELEMENTS: Optional[int] = None

# == EXPERIMENT PARAMETERS ==

# :param __DEBUG__:
#       In debug mode, results overwrite the previous run in the ``debug/`` folder.
__DEBUG__ = True

# :param __TESTING__:
#       If True, the dataset is limited to 50 elements for fast testing.
__TESTING__ = False

experiment = Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)


# ======================================================================================
# PROCESSING WORKER
# ======================================================================================

class TMCProcessingWorker(multiprocessing.Process):
    """
    Multiprocessing worker that converts decomposed TMC data into graph representations
    using ``MetalOrganicProcessing``.
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
    Adds the CSD identifier and metal symbol to each graph dict for traceability.
    """
    if graph is not None:
        graph['graph_id'] = data.get('complex_id', '')
        graph['graph_metal'] = data.get('metal', '')


@experiment.hook('load_dataset', default=True, replace=False)
def load_dataset(e: Experiment) -> Dict[int, dict]:
    """
    Download the tmQM CSV from the tmQM GitHub repository and decompose whole-complex
    SMILES into the decomposed TMC format expected by ``MetalOrganicProcessing``.

    The raw CSV is semicolon-delimited and contains a ``SMILES`` column with whole-complex
    SMILES that include dative bond notation (``->`` / ``<-``). Each SMILES is decomposed
    into metal symbol, individual ligand SMILES, and exact connecting atom indices using
    :func:`decompose_complex_smiles`.

    Rows with missing SMILES (~7,700 of ~108,000) are skipped. Rows where decomposition
    fails (e.g., RDKit parse errors, multinuclear complexes) are also skipped.
    """
    # -- Download the raw CSV --
    csv_path = os.path.join(e.path, 'tmqm_raw.csv')
    if not os.path.exists(csv_path):
        e.log(f'downloading tmQM dataset from {e.DOWNLOAD_URL} ...')
        response = requests.get(e.DOWNLOAD_URL, timeout=300)
        response.raise_for_status()
        with open(csv_path, 'w') as f:
            f.write(response.text)
        e.log(f'downloaded {len(response.content) / 1024 / 1024:.1f} MB')
    else:
        e.log(f'using cached raw CSV at {csv_path}')

    # -- Parse the semicolon-delimited CSV --
    e.log('parsing dataset...')
    df = pd.read_csv(csv_path, sep=';')
    e.log(f'loaded {len(df)} rows from CSV')

    if e.MAX_ELEMENTS is not None:
        df = df.head(e.MAX_ELEMENTS)
        e.log(f'limited to {e.MAX_ELEMENTS} rows')

    # -- Decompose each whole-complex SMILES --
    dataset: Dict[int, dict] = {}
    skipped_no_smiles = 0
    skipped_decompose = 0

    for idx, row in df.iterrows():
        smiles = row.get('SMILES', '')
        if pd.isna(smiles) or str(smiles).strip() == '':
            skipped_no_smiles += 1
            continue

        smiles = str(smiles).strip()

        # Decompose whole-complex SMILES into metal + ligands + connecting atoms
        result = decompose_complex_smiles(smiles)
        if result is None:
            skipped_decompose += 1
            continue

        metal, ligand_smiles, connecting_atom_indices, total_charge = result

        # Collect the 8 QM property values as the regression target vector
        targets = []
        for col in e.TARGET_COLUMNS:
            targets.append(float(row[col]))

        dataset[len(dataset)] = {
            'complex_id': str(row['CSD_code']),
            'metal': metal,
            'ligand_smiles': ligand_smiles,
            'connecting_atom_indices': connecting_atom_indices,
            'total_charge': total_charge,
            'oxidation_state': 0,      # Not available in tmQM
            'spin_multiplicity': 1,    # Not available in tmQM
            'whole_complex_smiles': smiles,
            'targets': targets,
        }

    e.log(f'parsed {len(dataset)} complexes '
          f'({skipped_no_smiles} missing SMILES, '
          f'{skipped_decompose} decomposition failures)')
    return dataset


@experiment.hook('save_csv', default=True, replace=False)
def save_csv(e: Experiment, dataset: Dict[int, dict]) -> None:
    """
    Save the dataset as a CSV file in the decomposed TMC format. The ``ligand_smiles``
    and ``connecting_atom_indices`` columns are JSON-encoded lists, compatible with
    :func:`load_tmc_dataset`.
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
            'whole_complex_smiles': data['whole_complex_smiles'],
        }
        for i, col in enumerate(e.TARGET_COLUMNS):
            row[col] = data['targets'][i]
        rows.append(row)

    df = pd.DataFrame(rows)

    csv_path = os.path.join(e.path, f'{e.DATASET_NAME}.csv')
    df.to_csv(csv_path, index=False)

    # Compressed version for distribution
    gz_path = csv_path + '.gz'
    with open(csv_path, 'rb') as f_in, gzip.open(gz_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

    e.log(f'saved CSV ({len(rows)} rows)')


# ======================================================================================
# MAIN EXPERIMENT
# ======================================================================================

@experiment
def experiment(e: Experiment):

    e.log('starting tmQM processing experiment...')
    e.log_parameters()

    # -- Create worker processes --
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

        # In testing mode, limit to a small subset
        if e.__TESTING__:
            e.log('testing mode: limiting to 50 elements')
            num_elements = min(num_elements, 50)
            dataset = dict(list(dataset.items())[:num_elements])

        # -- Feed data to workers and collect results --
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

    # -- Save processed graphs as msgpack --
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

    # -- Generate and save metadata --
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
