import pandas as pd
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

from chem_mat_data import load_smiles_dataset

DATASET_NAME: str = 'beet'

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
    graph['CAS_Number'] = data['Cas NUMBER']

@experiment.hook('load_dataset', default=False, replace=True)
def load_dataset(e: Experiment) -> dict[int, dict]:
    df = load_smiles_dataset('beet')
    dataset: dict[int, dict] = {}
    columns = ['threshold_1','threshold_100']
    for index, data in enumerate(df.to_dict('records')):
        data['smiles'] = data['SMILES']
        data['targets'] = [(0 if data[col] == 0 else 1) if pd.notna(data[col]) else -1 for col in columns]
        dataset[index] = data
    return dataset

experiment.run_if_main()


