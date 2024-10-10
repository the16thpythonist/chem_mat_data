
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

from chem_mat_data import load_smiles_dataset
DATASET_NAME: str = 'BBBP'

__TESTING__ = False

# This is the dataset template.
experiment = Experiment.extend(
    'create_graph_datasets.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)

@experiment.hook('add_graph_metadata', default=False, replace=True)
def add_graph_metadata(e: Experiment, data: dict, graph: dict) -> dict:
    """
    We add the names of the compounds
    """
    graph['name'] = data['name']

@experiment.hook('load_dataset', default=False, replace=True)
def load_dataset(e: Experiment) -> dict[int, dict]:
    df = load_smiles_dataset('BBBP')

    dataset: dict[int, dict] = {}
    for index, data in enumerate(df.to_dict('records')):
        data['smiles'] = data['smiles']
        #p_np is either 0 or 1
        data['targets'] = [data['p_np'] == index for index in range(2)]
        dataset[index] = data
    return dataset

experiment.run_if_main()

