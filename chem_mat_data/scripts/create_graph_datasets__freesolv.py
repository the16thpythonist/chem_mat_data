import os

from rich import print as pprint
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

from chem_mat_data import load_smiles_dataset


DATASET_NAME: str = 'freesolv'
experiment = Experiment.extend(
    'create_graph_datasets.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)

@experiment.hook('add_graph_metadata', default=False, replace=True)
def add_graph_metadata(e: Experiment, data: dict, graph: dict) -> dict:
    """
    In this experiment we add the iupac nomenclature as additional metadata field for possible identification
    """
    graph['graph_id'] = data['iupac']

@experiment.hook('load_dataset', default=False, replace=True)
def load_dataset(e: Experiment) -> dict[int, dict]:
    
    df = load_smiles_dataset('freesolv')
    dataset: dict[int, dict] = {}
    for index, data in enumerate(df.to_dict('records')):

        data['smiles'] = data['smiles']
        # Only one target. We combine the experimental and computational data into a single list 
        # Not sure if the following combines the 2 columns(?)
        data['targets'] = data['expt'] + data['calc']

        dataset[index] = data

    return dataset

experiment.run_if_main()
