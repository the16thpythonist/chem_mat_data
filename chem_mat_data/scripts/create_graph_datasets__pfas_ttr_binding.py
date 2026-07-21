"""
This experiment module creates a graph dataset from the PFAS TTR Binding dataset
for regression tasks.

The dataset contains 22 per- and polyfluoroalkyl substances (PFAS) with experimental
transthyretin (TTR) binding percentages. TTR is a key transport protein for thyroid
hormones, and PFAS binding to TTR is an important toxicological endpoint for assessing
endocrine disruption potential.

The data comes from the PFAS-Map framework, which provides a comprehensive database
for rapid screening of structure-function relationships in PFAS chemistry. See
also the ``pfas_classify`` dataset for PFAS structural classification on the full
EPA master list (~6,000 compounds) from the same source.

References:
- Su, A. & Rajan, K., Sci. Data, 8 (2021) 14
  DOI: 10.1038/s41597-021-00798-x
"""

import csv

import rdkit.Chem as Chem
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

from chem_mat_data.connectors import FigshareSource

# == DATASET METADATA ==

DATASET_NAME: str = 'pfas_ttr_binding'

DESCRIPTION: str = (
    'PFAS transthyretin (TTR) binding activity dataset from the PFAS-Map framework. '
    'Contains 22 PFAS compounds with experimental TTR binding percentages measuring '
    'competitive displacement of thyroid hormones. Values near 0% indicate strong TTR '
    'binding (displacing the natural ligand), while values above 100% indicate no binding. '
    'See also the pfas_classify dataset for PFAS structural classification from the same source.'
)

METADATA: dict = {
    'tags': [
        'Molecules',
        'SMILES',
        'PFAS',
        'Bioactivity',
        'Toxicology',
        'TTR',
        'Endocrine Disruption',
        'Regression',
    ],
    'verbose': 'PFAS TTR Binding Activity',
    'sources': [
        'https://doi.org/10.1038/s41597-021-00798-x',
        'https://doi.org/10.6084/m9.figshare.c.5043566',
    ],
    'notes': [
        'Regression task: predict TTR binding percentage.',
        'Values near 0% = strong TTR binding; >100% = no binding.',
        'Part of the PFAS-Map framework (Su & Rajan, 2021).',
        'Related dataset: pfas_classify (structural classification).',
    ],
    'target_descriptions': {
        '0': 'TTR Binding % - Competitive displacement of thyroxine (T4) from transthyretin in vitro; 0% = complete displacement (strong binder), 100% = no displacement (non-binder), >100% = possible enhancement',
    },
}

# == DOWNLOAD SOURCE ==

FIGSHARE_FILE_ID = 25766582
TTR_CSV_PATH = 'PFAS-Map/test_data/PFAS_TTR_Binding.csv'

# == EXPERIMENT PARAMETERS ==

DATASET_TYPE: str = 'regression'

__DEBUG__ = True
__TESTING__ = False

experiment = Experiment.extend(
    'create_graph_datasets.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)


@experiment.hook('load_dataset', default=False, replace=True)
def load_dataset(e: Experiment) -> dict:
    """
    Download and process the PFAS TTR Binding dataset.

    Uses :class:`~chem_mat_data.connectors.FigshareSource` to download the
    PFAS-Map ZIP archive from Figshare and extract the TTR binding CSV.
    Validates SMILES with RDKit and returns compounds with binding percentages
    as the regression target.

    :returns: Dictionary mapping indices to compound data dictionaries.
    """
    e.log('Downloading PFAS TTR binding data from Figshare...')
    with FigshareSource(
        file_id=FIGSHARE_FILE_ID,
        relative_path=TTR_CSV_PATH,
        verbose=True,
    ) as source:
        csv_path = source.fetch()

        dataset: dict = {}
        index = 0

        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                smiles = row['SMILES'].strip()
                if not smiles:
                    continue

                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue

                binding = row['Binding %']
                if not binding:
                    continue

                dataset[index] = {
                    'smiles': smiles,
                    'targets': [float(binding)],
                    'name': row.get('NAME', ''),
                    'full_name': row.get('FULL_NAME', ''),
                }
                index += 1

    e.log(f'Loaded {len(dataset)} PFAS compounds with TTR binding data')
    return dataset


@experiment.hook('add_graph_metadata', default=False, replace=True)
def add_graph_metadata(_e: Experiment, data: dict, graph: dict) -> None:
    """Add compound name metadata to each graph."""
    graph['graph_name'] = data['name']
    graph['graph_full_name'] = data['full_name']


experiment.run_if_main()
