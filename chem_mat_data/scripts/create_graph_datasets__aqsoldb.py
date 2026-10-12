from rdkit import Chem
from typing import Dict
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

from chem_mat_data import load_smiles_dataset

DATASET_NAME: str = 'aqsoldb'

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
            'CC0 1.0 is the licence on the Harvard Dataverse record '
            '(doi:10.7910/DVN/OVHAW8), established from the DOI registration metadata '
            'Dataverse publishes to DataCite. UPSTREAM: AqSolDB is a compilation of '
            'nine sources for which its authors state no licence. Three are not '
            'permissive - OECD eChemPortal (6,110 records, its own site terms, the '
            'largest single block), the Beilstein database now Reaxys/Elsevier '
            '(~1,210 records, commercial), and AQUASOL / SRC PHYSPROP (~1,291 '
            'records, commercial). A CC0 dedication waives the rights the depositors '
            'held; it cannot clear rights they never held in those compilations. The '
            "maintainer decided on 2026-08-19 to accept the compiler's licence as "
            'governing, recording the upstream honestly rather than holding the '
            'dataset back.'
        ),
    },
    'verbose': 'Aqueous Solubility',
    'full': True,
    'tags': [
        'Smiles',
        'Molecules',
        'Solubility',
    ],
    'sources': [
        'https://www.nature.com/articles/s41597-019-0151-1',
        'https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OVHAW8',
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
    We add the compound id for identification and the molecular weight
    """
    graph['graph_id'] = data['ID']


@experiment.hook('load_dataset', default=False, replace=True)
def load_dataset(e: Experiment) -> Dict[int, dict]:
    df = load_smiles_dataset('aqsoldb')

    dataset: Dict[int, dict] = {}
    index: int = 0
    for data in df.to_dict('records'):
        data['smiles'] = data['SMILES']

        # == MOLECULE FILTERS ==
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
        # In this dataset we only have one target
        data['targets'] = [data['Solubility']]
        dataset[index] = data

        index += 1

    return dataset


experiment.run_if_main()
