import rdkit.Chem as Chem
from typing import Dict
import pandas as pd

from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

from chem_mat_data import load_smiles_dataset

DATASET_NAME: str = 'sider'

__TESTING__ = False

# :param METADATA:
#       Mirrors what this dataset already carries in the shared metadata.yml.
#       Without it, "cmmanage metadata collect" - which replaces a dataset's
#       catalogue entry wholesale rather than merging fields - would silently
#       drop these the next time this script is processed.
METADATA: dict = {
    'license': {
        'identifier': 'CC-BY-SA-4.0',
        'url': 'https://creativecommons.org/licenses/by-sa/4.0/',
        'note': (
            'SIDER states on its download page (sideeffects.embl.de/download/): "The '
            'results of this mapping are available under a Creative Commons '
            'Attribution-Share Alike 4.0 License." ShareAlike, not NonCommercial - '
            'commercial use is permitted, but adaptations must be redistributed under '
            'the same licence, which is why this dataset carries it. Cite Kuhn, M. et '
            'al., the SIDER side effect resource (EMBL). Reached via '
            'MoleculeNet/DeepChem, which adds no licence of its own. This follows the '
            'same ShareAlike policy the maintainer adopted for ChEMBL; see '
            'docs/architecture_decisions/009_dataset_license_field.md.'
        ),
    },
    'verbose': 'Drug Side Effects',
    'full': True,
    'tags': [
        'Smiles',
        'Molecules',
        'Physiology',
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
    No extra metadata
    """
    pass


@experiment.hook('load_dataset', default=False, replace=True)
def load_dataset(e: Experiment) -> Dict[int, dict]:

    df = load_smiles_dataset('sider')
    dataset: Dict[int, dict] = {}
    columns = [
        'Hepatobiliary disorders',
        'Metabolism and nutrition disorders',
        'Product issues',
        'Eye disorders',
        'Investigations',
        'Musculoskeletal and connective tissue disorders',
        'Gastrointestinal disorders',
        'Social circumstances',
        'Immune system disorders',
        'Reproductive system and breast disorders',
        'Neoplasms benign or malignant and unspecified (incl cysts and polyps)',
        'General disorders and administration site conditions',
        'Endocrine disorders',
        'Surgical and medical procedures',
        'Vascular disorders',
        'Blood and lymphatic system disorders',
        'Skin and subcutaneous tissue disorders',
        'Congenital or familial and genetic disorders',
        'Infections and infestations',
        'Respiratory or thoracic and mediastinal disorders',
        'Psychiatric disorders',
        'Renal and urinary disorders',
        'Pregnancy or puerperium and perinatal conditions',
        'Ear and labyrinth disorders',
        'Cardiac disorders',
        'Nervous system disorders',
        'Injury or poisoning and procedural complications',
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
        # We have 17 targets and each one can be active or inactive, so either 0 or 1. For some there is no data for wich use -1.
        data['targets'] = [
            (0 if data[col] == 0 else 1) if pd.notna(data[col]) else -1
            for col in columns
        ]
        dataset[index] = data

    return dataset


experiment.run_if_main()
