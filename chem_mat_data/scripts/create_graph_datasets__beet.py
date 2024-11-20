import pandas as pd
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
def load_dataset(e: Experiment) -> dict[int, dict]:
    df = load_smiles_dataset('beet')
    dataset: dict[int, dict] = {}
    columns = ['threshold_1','threshold_100']
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
        data['targets'] = [(0 if data[col] == 0 else 1) if pd.notna(data[col]) else -1 for col in columns]
        print(data['threshold_1'], data['threshold_100'], data['targets'])
        dataset[index] = data
    return dataset

experiment.run_if_main()


