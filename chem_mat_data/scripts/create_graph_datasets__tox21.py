import rdkit.Chem as Chem
from typing import Dict
import pandas as pd

from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

from chem_mat_data import load_smiles_dataset

DATASET_NAME: str = 'tox21'

__TESTING__ = False

# :param METADATA:
#       Mirrors what this dataset already carries in the shared metadata.yml.
#       Without it, "cmmanage metadata collect" - which replaces a dataset's
#       catalogue entry wholesale rather than merging fields - would silently
#       drop these the next time this script is processed.
METADATA: dict = {
    'license': {
        'identifier': 'CC0-1.0',
        'url': 'https://creativecommons.org/publicdomain/zero/1.0/',
        'note': (
            'No party has attached an explicit licence to the distribution route this '
            'dataset comes from: the Tox21 Data Challenge files at tripod.nih.gov '
            'carry no terms (the page\'s only licence-like heading, "Public Domain '
            'Code", covers three code links, not the data), tox21.gov states no terms '
            "for any of its four official routes, PubChem's Tox21 source record has "
            "no licence row, and DeepChem's MIT licence covers its software rather "
            'than its datasets/ directory. The CC0 label is applied because Tox21 is '
            'the work of a four-agency US federal consortium (NCATS/NIH, US EPA, '
            'NIEHS/NTP, US FDA) in which no copyright is asserted, and because the '
            'EPA has attached an explicit CC0 1.0 dedication to its own ToxCast/Tox21 '
            'flat-file release (doi:10.23645/epacomptox.6062479) - a different '
            "distribution route for the same programme's data. The maintainer decided "
            'on 2026-08-19 that public-domain status is the accurate description. '
            'Cite the U.S. Tox21 Program.'
        ),
    },
    'verbose': 'Toxicology',
    'full': True,
    'tags': [
        'Smiles',
        'Molecules',
        'Physiology',
        'Toxicity',
    ],
}

experiment = Experiment.extend(
    'create_graph_datasets.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)


@experiment.hook('add_graph_metadata', default=False, replace=True)
def add_graph_metadata(e: Experiment, data: dict, graph: dict) -> dict:
    """
    We add the mol id fromt he dataset as additional metadata
    """
    graph['graph_id'] = data['mol_id']


@experiment.hook('load_dataset', default=False, replace=True)
def load_dataset(e: Experiment) -> Dict[int, dict]:
    df = load_smiles_dataset('tox21')

    dataset: Dict[int, dict] = {}
    # List of all 12 targets
    columns = [
        'NR-AR',
        'NR-AR-LBD',
        'NR-AhR',
        'NR-Aromatase',
        'NR-ER',
        'NR-ER-LBD',
        'NR-PPAR-gamma',
        'SR-ARE',
        'SR-ATAD5',
        'SR-HSE',
        'SR-MMP',
        'SR-p53',
    ]
    for index, data in enumerate(df.to_dict('records')):
        data['smiles'] = data['smiles']

        # === MOLECULE FILTERS ===
        # We don't want to use compounds with '.' in the smiles (separate molecules)
        if '.' in data['smiles']:
            continue

        # We don't want to use compounds that only consist of a single atom
        mol = Chem.MolFromSmiles(data['smiles'])
        if not mol:
            continue

        # We also don't want to accept "molecules" that are essentially just individual atoms
        if len(mol.GetAtoms()) < 2:
            continue

        # == TARGETS ==
        # Is either 0 (inactive) , 1 (active) or no data given so -1
        data['targets'] = [
            (0 if data[col] == 0 else 1) if pd.notna(data[col]) else -1
            for col in columns
        ]
        dataset[index] = data

    return dataset


experiment.run_if_main()
