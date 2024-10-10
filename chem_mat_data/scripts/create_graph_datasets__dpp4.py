import pandas as pd
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

from chem_mat_data import load_smiles_dataset

DATASET_NAME: str = 'dpp4'

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
    We add the molecular formula as additional metadata and wether it is training or testing data
    """
    graph['formula'] = data['MolecularFormula']
    graph['split'] = data['split']

@experiment.hook('load_dataset', default=False, replace=True)
def load_dataset(e: Experiment) -> dict[int, dict]:
    df = load_smiles_dataset('dpp4')
    dataset: dict[int, dict] = {}
    columns = ['Activity(IC50)','Activity(pIC50)']
    for index, data in enumerate(df.to_dict('records')):
        data['smiles'] = data['smiles']
        data['targets'] = [(0 if data[col] == 0 else 1) if pd.notna(data[col]) else -1 for col in columns]
        dataset[index] = data
    return dataset

experiment.run_if_main()

    
