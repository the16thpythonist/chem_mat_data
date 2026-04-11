"""
This experiment creates the processed graph dataset for **tmBIO** — a domain-specific
subset of ~2,800 mononuclear transition metal complexes associated with **biological activity**
applications, with 8 DFT-computed quantum-mechanical regression targets.

The subset is curated via NLP text mining of the scientific literature by Kevlishvili
et al. The CSD refcodes identifying biologically active complexes are extracted from the Zenodo
archive and joined with the tmQM 2024 release for SMILES and QM properties. This enables
domain-specific property prediction benchmarking on biologically active TMCs specifically.

**Source**: Kevlishvili et al., *Faraday Discuss.* 2025. DOI: 10.1039/D4FD00087K;
tmQM: Balcells & Skjelstad, *JCIM* 2020.

**Targets** (8 regression properties, same as tmQM):
    Electronic energy, dispersion energy, dipole moment, metal natural charge,
    HOMO-LUMO gap, HOMO energy, LUMO energy, polarizability.

**Usage**::

    python create_graph_datasets__tm_bio.py

Results are written to ``results/create_graph_datasets__tm_bio/debug/``.
"""
import os
import json
import gzip
import time
import shutil
import zipfile
import datetime
import multiprocessing
from typing import Dict, List, Optional, Set

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
    decompose_complex_smiles,
)
from chem_mat_data.data import default, ext_hook, save_graphs


# == SOURCE PARAMETERS ==

DOWNLOAD_URL_ZENODO: str = 'https://zenodo.org/api/records/11404217/files/Data.zip/content'
DOWNLOAD_URL_TMQM: str = 'https://raw.githubusercontent.com/uiocompcat/tmQM/master/tmQM/tmQM_y.csv'

# :param APPLICATION_CSV_PATH:
#       Path within the Zenodo zip for the tmBIO application CSV.
APPLICATION_CSV_PATH: str = 'Data/Datasets/text_mined/tmBIO.csv'

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

DATASET_TYPE: str = 'regression'

DESCRIPTION: str = (
    'Graph dataset of ~2,800 mononuclear transition metal complexes associated with '
    'biological activity applications, curated from tmQM via NLP text mining (Kevlishvili et al., '
    'Faraday Discuss. 2025). 8 DFT-computed regression targets (TPSSh-D3BJ/def2-SVP). '
    'Processed using MetalOrganicProcessing with decomposed SMILES.'
)

METADATA: dict = {
    'category': 'tmc',
    'min_version': '1.7.0',
    'tags': ['Molecules', 'TransitionMetals', 'TMC', 'QM', 'DFT', 'Biology'],
    'sources': [
        'https://doi.org/10.1039/D4FD00087K',
        'https://doi.org/10.5281/zenodo.11404217',
        'https://github.com/uiocompcat/tmQM',
    ],
    'verbose': 'tmBIO — Biology TMC Subset with QM Properties',
    'target_descriptions': {
        str(i): col for i, col in enumerate(TARGET_COLUMNS)
    },
}

# == PROCESSING PARAMETERS ==

DATASET_NAME: str = 'tm_bio'
COMPRESS: bool = True
MAX_ELEMENTS: Optional[int] = None

# == EXPERIMENT PARAMETERS ==

__DEBUG__ = True
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
    if graph is not None:
        graph['graph_id'] = data.get('complex_id', '')
        graph['graph_metal'] = data.get('metal', '')


@experiment.hook('load_dataset', default=True, replace=False)
def load_dataset(e: Experiment) -> Dict[int, dict]:
    """
    Download the Zenodo archive and tmQM CSV, extract the application-specific refcodes,
    filter tmQM to this subset, decompose SMILES, and collect regression targets.
    """
    # -- Download and cache Zenodo zip --
    zip_path = os.path.join(e.path, 'tmqm_applications_data.zip')
    if not os.path.exists(zip_path):
        e.log(f'downloading Zenodo archive (~588 MB) ...')
        e.log('this may take several minutes on the first run...')
        response = requests.get(e.DOWNLOAD_URL_ZENODO, timeout=1800, stream=True)
        response.raise_for_status()
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        e.log(f'downloaded {os.path.getsize(zip_path) / 1024 / 1024:.1f} MB')
    else:
        e.log(f'using cached Zenodo archive')

    # -- Extract refcodes for this application --
    e.log(f'extracting refcodes from {e.APPLICATION_CSV_PATH} ...')
    with zipfile.ZipFile(zip_path, 'r') as zf:
        with zf.open(e.APPLICATION_CSV_PATH) as csv_file:
            df_app = pd.read_csv(csv_file, usecols=['refcode'])
            app_refcodes: Set[str] = set(df_app['refcode'].dropna().astype(str).str.strip())
    e.log(f'found {len(app_refcodes)} refcodes')

    # -- Download and cache tmQM CSV --
    tmqm_csv_path = os.path.join(e.path, 'tmqm_y.csv')
    if not os.path.exists(tmqm_csv_path):
        e.log(f'downloading tmQM CSV ...')
        response = requests.get(e.DOWNLOAD_URL_TMQM, timeout=300)
        response.raise_for_status()
        with open(tmqm_csv_path, 'w') as f:
            f.write(response.text)
        e.log(f'downloaded {len(response.content) / 1024 / 1024:.1f} MB')
    else:
        e.log(f'using cached tmQM CSV')

    # -- Parse tmQM, filter to application subset, decompose SMILES --
    e.log('parsing tmQM and filtering to application subset...')
    df = pd.read_csv(tmqm_csv_path, sep=';')

    dataset: Dict[int, dict] = {}
    skipped_no_smiles = 0
    skipped_decompose = 0

    for idx, row in df.iterrows():
        csd_code = str(row['CSD_code']).strip()
        if csd_code not in app_refcodes:
            continue

        smiles = row.get('SMILES', '')
        if pd.isna(smiles) or str(smiles).strip() == '':
            skipped_no_smiles += 1
            continue

        smiles = str(smiles).strip()

        result = decompose_complex_smiles(smiles)
        if result is None:
            skipped_decompose += 1
            continue

        metal, ligand_smiles, connecting_atom_indices, total_charge = result

        targets = [float(row[col]) for col in e.TARGET_COLUMNS]

        dataset[len(dataset)] = {
            'complex_id': csd_code,
            'metal': metal,
            'ligand_smiles': ligand_smiles,
            'connecting_atom_indices': connecting_atom_indices,
            'total_charge': total_charge,
            'oxidation_state': 0,
            'spin_multiplicity': 1,
            'whole_complex_smiles': smiles,
            'targets': targets,
        }

        if e.MAX_ELEMENTS is not None and len(dataset) >= e.MAX_ELEMENTS:
            break

    e.log(f'parsed {len(dataset)} complexes '
          f'({skipped_no_smiles} missing SMILES, '
          f'{skipped_decompose} decomposition failures)')
    return dataset


@experiment.hook('save_csv', default=True, replace=False)
def save_csv(e: Experiment, dataset: Dict[int, dict]) -> None:
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

    gz_path = csv_path + '.gz'
    with open(csv_path, 'rb') as f_in, gzip.open(gz_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

    e.log(f'saved CSV ({len(rows)} rows)')


# ======================================================================================
# MAIN EXPERIMENT
# ======================================================================================

@experiment
def experiment(e: Experiment):

    e.log('starting tm_bio processing experiment...')
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
