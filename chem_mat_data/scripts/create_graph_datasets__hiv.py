import rdkit.Chem as Chem
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

from chem_mat_data import load_smiles_dataset

DATASET_NAME: str = 'hiv'

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
    The additional metadata is wether the compound is inactive(CI) or modulating(CM)
    """
    graph['graph_activity'] = data['activity']


@experiment.hook('load_dataset', default=False, replace=True)
def load_dataset(e: Experiment) -> dict[int, dict]:
    
    df = load_smiles_dataset('HIV')

    dataset: dict[int, dict] = {}
    index = 0
    for data in df.to_dict('records'):
        
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
        
        # We only have one target, eiter HIV active (1) or not (0)
        data['targets'] = [data['HIV_active'] == index for index in range(2)]
        dataset[index] = data
        index += 1
        
    return dataset

experiment.run_if_main()


