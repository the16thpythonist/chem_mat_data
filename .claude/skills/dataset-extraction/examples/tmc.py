"""
ARCHETYPE 4 of 4 — transition-metal complexes needing metal-ligand decomposition.

This is the one archetype that does **not** extend the base template. Coordination
complexes cannot go through ``MoleculeProcessing``: a dative metal-ligand bond is not
an ordinary covalent bond, and RDKit will either refuse the SMILES or silently produce
the wrong connectivity. ``MetalOrganicProcessing`` instead takes the complex already
decomposed into a metal centre, its ligands, and the atom indices that connect them.

That different processing class means a different worker, which means a standalone
``Experiment`` with its own multiprocessing loop rather than an ``Experiment.extend``.

**When you actually need this path**: only when the complexes require decomposition -
whole-complex SMILES with dative notation (``->`` / ``<-``), or a source that already
lists metal and ligands separately. A dataset *about* metal complexes that ships plain
organic SMILES is archetype 1.

**Start from the real script.** ``chem_mat_data/scripts/create_graph_datasets__tmqm.py``
is the reference implementation and is ~480 lines; copy it and adapt rather than
retyping the worker loop. This file shows only the parts that differ from the base
template, with the shared machinery elided.

Related real scripts to compare against: ``__tmqmg.py``, ``__tm_bio.py``, ``__tm_cat.py``,
``__tm_sco.py``, ``__kulik_spin.py``.
"""

import os
import multiprocessing
from typing import Dict, List, Optional

import msgpack
import requests
import numpy as np
import pandas as pd
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

from rdkit import RDLogger

# TMC datasets produce a lot of RDKit sanitization noise; silence it.
RDLogger.DisableLog('rdApp.*')

import chem_mat_data._typing as typc
from chem_mat_data.tmc_processing import (
    MetalOrganicProcessing,
    decompose_complex_smiles,
)
from chem_mat_data.data import default

# == SOURCE PARAMETERS ==

# :param DOWNLOAD_URL:
#       The raw tmQM CSV, semicolon-delimited, ~20 MB, ~108k rows.
DOWNLOAD_URL: str = (
    'https://raw.githubusercontent.com/uiocompcat/tmQM/master/tmQM/tmQM_y.csv'
)

# :param TARGET_COLUMNS:
#       The 8 DFT-computed properties, in target-vector order. Computed at
#       TPSSh-D3BJ/def2-SVP except polarizability, which is GFN2-xTB.
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
    'Dataset of approximately 100,000 mononuclear transition metal complexes from the '
    'tmQM dataset, covering 30 different metals across a wide range of ligand '
    'environments. Each complex carries 8 DFT-computed quantum-mechanical properties '
    'calculated at the TPSSh-D3BJ/def2-SVP level of theory (polarizability at GFN2-xTB), '
    'including electronic energy, dispersion energy, dipole moment, metal natural charge '
    'and HOMO-LUMO gap. Structures originate from the Cambridge Structural Database and '
    'are supplied as whole-complex SMILES with dative bond notation.'
)

METADATA: dict = {
    'verbose': 'tmQM Transition Metal Complex QM Properties',
    # 'category' groups TMC datasets; 'min_version' pins the release that introduced
    # MetalOrganicProcessing, so older clients refuse the dataset rather than
    # mis-parsing it.
    'category': 'tmc',
    'min_version': '1.7.0',
    'tags': ['Molecules', 'TransitionMetals', 'TMC', 'QM', 'DFT', 'CSD'],
    'sources': [
        'https://doi.org/10.1021/acs.jcim.0c01041',
        'https://github.com/uiocompcat/tmQM',
    ],
    # Give real descriptions with units. The live tmqm.py currently generates these
    # from the column names alone, which tells a downstream user nothing - do better.
    'target_descriptions': {
        '0': 'Electronic_E - Total electronic energy (Hartree)',
        '1': 'Dispersion_E - Dispersion energy correction (Hartree)',
        '2': 'Dipole_M - Dipole moment (Debye)',
        '3': 'Metal_q - Natural charge on the metal centre (e)',
        '4': 'HL_Gap - HOMO-LUMO gap (Hartree)',
        '5': 'HOMO_Energy - Highest occupied molecular orbital energy (Hartree)',
        '6': 'LUMO_Energy - Lowest unoccupied molecular orbital energy (Hartree)',
        '7': 'Polarizability - Isotropic polarizability at GFN2-xTB (Bohr^3)',
    },
}

# == PROCESSING PARAMETERS ==

DATASET_NAME: str = 'tmqm'
COMPRESS: bool = True

# :param MAX_ELEMENTS:
#       Cap on the number of complexes processed. None processes everything; set a
#       small number for a quick structural check.
MAX_ELEMENTS: Optional[int] = None

__DEBUG__ = True
__TESTING__ = False

# Standalone Experiment - NOT Experiment.extend. The worker below replaces the base
# template's ProcessingWorker, so there is no base implementation to inherit.
experiment = Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)


class TMCProcessingWorker(multiprocessing.Process):
    """
    Multiprocessing worker converting decomposed TMC records into graph dicts.

    Mirrors the base template's ``ProcessingWorker`` but drives
    ``MetalOrganicProcessing``, which builds the graph from an explicit metal centre
    plus ligands rather than from a single SMILES string.
    """

    def __init__(
        self,
        input_queue: multiprocessing.Queue,
        output_queue: multiprocessing.Queue,
    ):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.processing = MetalOrganicProcessing()

    def run(self):
        """
        Consume records until a ``None`` sentinel arrives, emitting exactly one
        (possibly ``None``) encoded graph per input record.
        """
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

                experiment.apply_hook('add_graph_metadata', data=data, graph=graph)

            except Exception as exc:
                complex_id = data.get('complex_id', '?')
                metal = data.get('metal', '?')
                print(
                    f' ! error processing {complex_id} ({metal}) '
                    f'- {exc.__class__.__name__}: {exc}'
                )
                graph = None

            self.output_queue.put(msgpack.packb(graph, default=default))


# In a standalone experiment the hooks are the primary definitions rather than
# overrides, so they use default=True, replace=False - the opposite of a sub-experiment.


@experiment.hook('add_graph_metadata', default=True, replace=False)
def add_graph_metadata(e: Experiment, data: dict, graph: typc.GraphDict) -> None:
    """
    Attach the CSD identifier and metal symbol for traceability.

    :param e: The running experiment.
    :param data: The original record dict.
    :param graph: The graph dict, or None when processing failed.
    """
    if graph is not None:
        graph['graph_id'] = data.get('complex_id', '')
        graph['graph_metal'] = data.get('metal', '')


@experiment.hook('load_dataset', default=True, replace=False)
def load_dataset(e: Experiment) -> Dict[int, dict]:
    """
    Download the tmQM CSV (cached) and decompose each whole-complex SMILES into the
    form ``MetalOrganicProcessing`` expects.

    ``decompose_complex_smiles`` splits a dative-bond SMILES into the metal symbol,
    the individual ligand SMILES, and the ligand atom indices that coordinate to the
    metal. It returns None for complexes it cannot handle - multinuclear species and
    RDKit parse failures - which are skipped and counted.

    :param e: The running experiment.

    :returns: Mapping of integer index to decomposed records, each carrying ``metal``,
        ``ligand_smiles``, ``connecting_atom_indices`` and ``targets``.
    """
    csv_path = os.path.join(e.path, 'tmqm_raw.csv')
    if not os.path.exists(csv_path):
        e.log(f'downloading tmQM dataset from {e.DOWNLOAD_URL} ...')
        response = requests.get(e.DOWNLOAD_URL, timeout=300)
        response.raise_for_status()
        with open(csv_path, 'w') as file:
            file.write(response.text)
        e.log(f'downloaded {len(response.content) / 1024**2:.1f} MB')
    else:
        e.log(f'using cached raw CSV @ {csv_path}')

    # Note the separator: this file is semicolon-delimited.
    df = pd.read_csv(csv_path, sep=';')
    e.log(f'loaded {len(df)} rows')

    if e.MAX_ELEMENTS is not None:
        df = df.head(e.MAX_ELEMENTS)
        e.log(f'limited to {e.MAX_ELEMENTS} rows')

    dataset: Dict[int, dict] = {}
    skipped_no_smiles: int = 0
    skipped_decompose: int = 0

    for _, row in df.iterrows():
        smiles = row.get('SMILES', '')
        if pd.isna(smiles) or not str(smiles).strip():
            skipped_no_smiles += 1
            continue

        result = decompose_complex_smiles(str(smiles).strip())
        if result is None:
            skipped_decompose += 1
            continue

        dataset[len(dataset)] = {
            'complex_id': row.get('CSD_code', ''),
            'metal': result['metal'],
            'ligand_smiles': result['ligand_smiles'],
            'connecting_atom_indices': result['connecting_atom_indices'],
            'whole_complex_smiles': str(smiles).strip(),
            'targets': [float(row[column]) for column in e.TARGET_COLUMNS],
        }

    e.log(
        f'built dataset with {len(dataset)} complexes '
        f'({skipped_no_smiles} without SMILES, {skipped_decompose} decomposition failures)'
    )
    return dataset


@experiment
def experiment(e: Experiment):
    """
    Main experiment body.

    ELIDED. Because this is a standalone experiment it must implement the whole
    pipeline itself: spawn ``TMCProcessingWorker`` instances (before loading the
    dataset, so the data is not copied into every child), feed the queue, collect
    graphs, then save the .mpack, compress it, and write metadata.yml.

    Copy this section verbatim from
    ``chem_mat_data/scripts/create_graph_datasets__tmqm.py`` - it is mechanical and
    identical across every TMC script. Only the worker class and the hooks above
    differ between TMC datasets.
    """
    raise NotImplementedError(
        'Copy the experiment body from create_graph_datasets__tmqm.py'
    )


experiment.run_if_main()
