"""
This experiment creates 4 processed graph datasets for the **application-specific TMC
subsets**: **tmCAT** (catalysis), **tmPHOTO** (photophysics), **tmBIO** (biology), and
**tmSCO** (spin crossover).

These are domain-focused subsets of tmQM curated via NLP text mining of the scientific
literature (Kevlishvili et al., *Faraday Discuss.* 2025). Each subset contains only the
transition metal complexes associated with a specific functional application, enabling
**domain-specific property prediction** — the same 8 tmQM regression targets but
evaluated on a chemically focused subset of complexes.

The script downloads application labels from Zenodo and SMILES + properties from the
tmQM 2024 release, joins them by CSD refcode, and produces 4 separate datasets:

- ``tmcat``: ~21k catalytic TMCs
- ``tmphoto``: ~4.6k photophysical TMCs
- ``tmbio``: ~2.8k biologically active TMCs
- ``tmsco``: ~983 spin-crossover TMCs

Each dataset has the same 8 regression targets as tmQM (electronic energy, dispersion
energy, dipole moment, metal charge, HOMO-LUMO gap, HOMO, LUMO, polarizability).

**Source**: Kevlishvili et al., *Faraday Discuss.* 2025. DOI: 10.1039/D4FD00087K;
tmQM: Balcells & Skjelstad, *JCIM* 2020.

**Usage**::

    python create_graph_datasets__tmqm_applications.py

Results are written to ``results/create_graph_datasets__tmqm_applications/debug/``.
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
    TRANSITION_METAL_ATOMIC_NUMBERS,
    decompose_complex_smiles,
)
from chem_mat_data.data import default, ext_hook, save_graphs


# == SOURCE PARAMETERS ==

# :param DOWNLOAD_URL_ZENODO:
#       The Zenodo URL for the full data archive (~588 MB, cached after first download).
DOWNLOAD_URL_ZENODO: str = 'https://zenodo.org/api/records/11404217/files/Data.zip/content'

# :param DOWNLOAD_URL_TMQM:
#       The tmQM CSV (2024 release) with whole-complex SMILES and QM properties.
DOWNLOAD_URL_TMQM: str = 'https://raw.githubusercontent.com/uiocompcat/tmQM/master/tmQM/tmQM_y.csv'

# :param APPLICATION_CSVS:
#       Paths within the Zenodo zip for each application-labeled CSV.
APPLICATION_CSVS: Dict[str, str] = {
    'tmcat':   'Data/Datasets/text_mined/tmCAT.csv',
    'tmphoto': 'Data/Datasets/text_mined/tmPHOTO.csv',
    'tmbio':   'Data/Datasets/text_mined/tmBIO.csv',
    'tmsco':   'Data/Datasets/text_mined/tmSCO.csv',
}

# :param APPLICATION_DESCRIPTIONS:
#       Human-readable descriptions for each application subset.
APPLICATION_DESCRIPTIONS: Dict[str, str] = {
    'tmcat':   'catalysis',
    'tmphoto': 'photophysics',
    'tmbio':   'biological activity',
    'tmsco':   'spin crossover / magnetism',
}

# :param TARGET_COLUMNS:
#       The 8 tmQM regression target columns. These are the same as the full tmQM
#       dataset — the value of these subsets is domain-specific evaluation.
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
                graph['graph_id'] = data.get('complex_id', '')
                graph['graph_metal'] = data.get('metal', '')

            except Exception as exc:
                metal = data.get('metal', '?')
                cid = data.get('complex_id', '?')
                print(f' ! error processing {cid} ({metal}) - {exc.__class__.__name__}: {exc}')
                graph = None

            graph_encoded = msgpack.packb(graph, default=default)
            self.output_queue.put(graph_encoded)


# ======================================================================================
# HELPER FUNCTIONS
# ======================================================================================

def download_and_cache(url: str, path: str, e: Experiment, timeout: int = 1800,
                       stream: bool = False) -> str:
    """
    Download a file from a URL if not already cached at the given path.
    """
    if not os.path.exists(path):
        e.log(f'downloading {url} ...')
        response = requests.get(url, timeout=timeout, stream=stream)
        response.raise_for_status()
        if stream:
            with open(path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        else:
            with open(path, 'w') as f:
                f.write(response.text)
        file_size = os.path.getsize(path) / 1024 / 1024
        e.log(f'downloaded {file_size:.1f} MB')
    else:
        e.log(f'using cached file at {path}')
    return path


def extract_refcodes(zip_path: str, csv_path_in_zip: str) -> Set[str]:
    """
    Extract the set of CSD refcodes from a CSV inside a zip archive.
    """
    with zipfile.ZipFile(zip_path, 'r') as zf:
        with zf.open(csv_path_in_zip) as csv_file:
            df = pd.read_csv(csv_file, usecols=['refcode'])
            return set(df['refcode'].dropna().astype(str).str.strip())


def process_subset(
    e: Experiment,
    dataset_name: str,
    dataset: Dict[int, dict],
    description: str,
    application: str,
) -> None:
    """
    Process a single application subset through MetalOrganicProcessing and save
    the CSV, mpack, and metadata files.
    """
    num_elements = len(dataset)
    e.log(f'--- Processing {dataset_name} ({num_elements} complexes, {application}) ---')

    if num_elements == 0:
        e.log(f'  WARNING: no complexes for {dataset_name}, skipping')
        return

    # -- Save CSV --
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
    csv_path = os.path.join(e.path, f'{dataset_name}.csv')
    df.to_csv(csv_path, index=False)
    gz_path = csv_path + '.gz'
    with open(csv_path, 'rb') as f_in, gzip.open(gz_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    e.log(f'  saved CSV ({len(rows)} rows)')

    # -- Process graphs --
    if e.__TESTING__:
        subset = dict(list(dataset.items())[:50])
        num_elements = len(subset)
    else:
        subset = dataset

    e.log(f'  creating workers...')
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

    try:
        indices = list(subset.keys())
        num_indices = len(indices)
        graphs = []
        start_time = time.time()
        count = 0
        prev_count = 0

        while count < num_indices:
            while not input_queue.full() and len(indices) != 0:
                index = indices.pop()
                input_queue.put(subset[index])

            while not output_queue.empty():
                graph_encoded = output_queue.get()
                graph = msgpack.unpackb(graph_encoded, ext_hook=ext_hook)
                if graph:
                    graphs.append(graph)
                count += 1

            if count % 1000 == 0 and count != prev_count:
                prev_count = count
                time_passed = time.time() - start_time
                if count > 0:
                    eta_sec = (time_passed / count) * (num_indices - count)
                    eta = datetime.datetime.now() + datetime.timedelta(seconds=eta_sec)
                    e.log(f'  * {count:05d}/{num_indices} - eta: {eta:%H:%M}')

        duration = time.time() - start_time
        e.log(f'  processed {len(graphs)} graphs in {duration:.1f}s')

    finally:
        for worker in workers:
            input_queue.put(None)
            worker.terminate()
            worker.join()
        del input_queue
        del output_queue

    if not graphs:
        e.log(f'  WARNING: no graphs produced for {dataset_name}')
        return

    # -- Save mpack --
    dataset_path = os.path.join(e.path, dataset_name + '.mpack')
    save_graphs(graphs, dataset_path)
    file_size = os.path.getsize(dataset_path)
    e.log(f'  wrote mpack: {file_size / 1024 / 1024:.1f} MB')

    if e.COMPRESS:
        compressed_path = os.path.join(e.path, dataset_name + '.mpack.gz')
        with open(dataset_path, 'rb') as f_in, gzip.open(compressed_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        compressed_size = os.path.getsize(compressed_path)
        e.log(f'  compressed: {compressed_size / 1024 / 1024:.1f} MB')

    # -- Save metadata --
    example_graph = graphs[0]
    metadata: dict = {
        'compounds': len(graphs),
        'targets': len(example_graph['graph_labels']),
        'target_type': [e.DATASET_TYPE],
        'description': description,
        'raw': ['csv'],
        'sources': [
            'https://doi.org/10.1039/D4FD00087K',
            'https://doi.org/10.5281/zenodo.11404217',
            'https://github.com/uiocompcat/tmQM',
        ],
        'category': 'tmc',
        'min_version': '1.7.0',
        'tags': ['Molecules', 'TransitionMetals', 'TMC', 'QM', 'DFT',
                 application.title()],
        'verbose': f'{dataset_name} — tmQM {application} subset',
        'target_descriptions': {
            str(i): col for i, col in enumerate(e.TARGET_COLUMNS)
        },
    }

    metadata_path = os.path.join(e.path, f'{dataset_name}_metadata.yml')
    with open(metadata_path, 'w') as f:
        yaml.dump(metadata, f)

    e.log(f'  saved metadata @ {metadata_path}')


# ======================================================================================
# MAIN EXPERIMENT
# ======================================================================================

@experiment
def experiment(e: Experiment):

    e.log('starting tmQM application subsets processing experiment...')
    e.log_parameters()

    # -- Download and cache data sources --
    zip_path = download_and_cache(
        e.DOWNLOAD_URL_ZENODO,
        os.path.join(e.path, 'tmqm_applications_data.zip'),
        e, timeout=1800, stream=True,
    )
    tmqm_csv_path = download_and_cache(
        e.DOWNLOAD_URL_TMQM,
        os.path.join(e.path, 'tmqm_y.csv'),
        e, timeout=300,
    )

    # -- Extract refcode sets for each application --
    e.log('extracting application labels from Zenodo archive...')
    application_refcodes: Dict[str, Set[str]] = {}
    for app_name, csv_path in e.APPLICATION_CSVS.items():
        try:
            refcodes = extract_refcodes(zip_path, csv_path)
            application_refcodes[app_name] = refcodes
            e.log(f'  {app_name}: {len(refcodes)} refcodes')
        except Exception as exc:
            e.log(f'  WARNING: could not read {csv_path}: {exc}')
            application_refcodes[app_name] = set()

    # -- Parse tmQM CSV --
    e.log('parsing tmQM CSV...')
    df = pd.read_csv(tmqm_csv_path, sep=';')
    e.log(f'loaded {len(df)} tmQM rows')

    # -- Build per-application datasets by filtering tmQM --
    e.log('decomposing SMILES and building per-application datasets...')
    app_datasets: Dict[str, Dict[int, dict]] = {name: {} for name in e.APPLICATION_CSVS}
    total_decomposed = 0
    skipped_no_smiles = 0
    skipped_decompose = 0

    for idx, row in df.iterrows():
        smiles = row.get('SMILES', '')
        if pd.isna(smiles) or str(smiles).strip() == '':
            skipped_no_smiles += 1
            continue

        smiles = str(smiles).strip()
        csd_code = str(row['CSD_code']).strip()

        # Check if this complex belongs to any application subset
        belongs_to = [name for name, refs in application_refcodes.items()
                      if csd_code in refs]
        if not belongs_to:
            continue  # Not in any subset — skip to avoid unnecessary decomposition

        # Decompose SMILES
        result = decompose_complex_smiles(smiles)
        if result is None:
            skipped_decompose += 1
            continue

        metal, ligand_smiles, connecting_atom_indices, total_charge = result
        total_decomposed += 1

        # Collect targets
        targets = [float(row[col]) for col in e.TARGET_COLUMNS]

        data_entry = {
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

        # Add to each matching application dataset
        for app_name in belongs_to:
            app_ds = app_datasets[app_name]
            app_ds[len(app_ds)] = data_entry

        if e.MAX_ELEMENTS is not None and total_decomposed >= e.MAX_ELEMENTS:
            break

    e.log(f'decomposed {total_decomposed} complexes '
          f'({skipped_no_smiles} missing SMILES, {skipped_decompose} decomposition failures)')
    for name, ds in app_datasets.items():
        e.log(f'  {name}: {len(ds)} complexes')

    # -- Process each application subset --
    for app_name, ds in app_datasets.items():
        description = (
            f'Graph dataset of {len(ds)} mononuclear transition metal complexes '
            f'associated with {e.APPLICATION_DESCRIPTIONS[app_name]}, curated from tmQM via '
            f'NLP text mining (Kevlishvili et al., Faraday Discuss. 2025). '
            f'8 DFT-computed regression targets (TPSSh-D3BJ/def2-SVP). '
            f'Processed using MetalOrganicProcessing with decomposed SMILES.'
        )
        process_subset(
            e,
            dataset_name=app_name,
            dataset=ds,
            description=description,
            application=e.APPLICATION_DESCRIPTIONS[app_name],
        )

    e.log('done!')


experiment.run_if_main()
