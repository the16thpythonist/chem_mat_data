import pandas as pd
from typing import Dict
import rdkit.Chem as Chem
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

from chem_mat_data import load_smiles_dataset

DATASET_NAME: str = 'beet'

DESCRIPTION = (
    'The toxicity in honey bees (beet) dataset was extract from a study on '
    'the prediction of acute contact toxicity of pesticides in honeybees. '
    'The data set contains 254 compounds with their experimental values. '
    'Each element is associated with two target values (threshold_1, threshold_100) '
    'where the value is the binary label if the compound is toxic at concentration 1 '
    'and the second value is the binary label if the compound is toxic at concentration 100.'
)

__TESTING__ = False

# :param METADATA:
#       Mirrors what this dataset already carries in the shared metadata.yml.
#       Without it, "cmmanage metadata collect" - which replaces a dataset's
#       catalogue entry wholesale rather than merging fields - would silently
#       drop these the next time this script is processed.
METADATA: dict = {
    'license': {
        'identifier': 'CC-BY-4.0',
        'url': 'https://creativecommons.org/licenses/by/4.0/',
        'note': (
            'CC BY 4.0 is stated on the Figshare record this table comes from, '
            'deposited by Taylor & Francis (doi:10.6084/m9.figshare.7093541, the '
            'supplementary table of Venko, Drgan & Novic 2018). Verified three ways - '
            'the DataCite rightsList with SPDX rightsIdentifier cc-by-4.0, the '
            'Figshare v2 API, and the rendered record page - and a sibling Figshare '
            'record was checked as a negative control (it reads CC-BY-NC), confirming '
            'this is a genuine per-record grant rather than a platform default. The '
            'shipped data was verified row-for-row against the record preview. '
            'Attribution required, and CC BY also requires that modification be '
            'indicated: MODIFIED - reformatted into the graph representation. '
            'UPSTREAM: the authors cite the Pesticide Properties DataBase (PPDB, '
            'University of Hertfordshire), which is proprietary, but spot-checks '
            'argue against it being the source - of four values checked only '
            'deltamethrin matched PPDB, dimethoate read 0.16 against 0.1, '
            'parathion-methyl 0.111 against 19.5, and dieldrin has no PPDB contact '
            'value at all. The 254 LD50 values are a multi-source literature and '
            "regulatory compilation published as the authors' own supplementary "
            'table.'
        ),
    },
    'verbose': 'Honey Bee Toxicity',
    'full': True,
    'tags': [
        'Smiles',
        'Molecules',
        'Physiology',
        'Toxicity',
    ],
    'sources': [
        'https://weilab.math.msu.edu/DataLibrary/2D/',
        'https://pubs.acs.org/doi/10.1021/acs.jpclett.1c03058',
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
    We add the id number of the dataset and the cas number as additional metadata
    """
    graph['graph_id'] = data['ID']
    graph['graph_cas'] = data['Cas NUMBER']


@experiment.hook('load_dataset', default=False, replace=True)
def load_dataset(e: Experiment) -> Dict[int, dict]:
    df = load_smiles_dataset('beet')
    dataset: Dict[int, dict] = {}
    columns = ['threshold_1', 'threshold_100']
    for index, data in enumerate(df.to_dict('records')):
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

        # == TARGET VALUES ==
        data['targets'] = [
            (0 if data[col] == 0 else 1) if pd.notna(data[col]) else -1
            for col in columns
        ]
        print(data['threshold_1'], data['threshold_100'], data['targets'])
        dataset[index] = data
    return dataset


experiment.run_if_main()
