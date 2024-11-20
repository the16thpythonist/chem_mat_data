import rdkit.Chem as Chem
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

from chem_mat_data import load_smiles_dataset


DATASET_NAME: str = 'lipophilicity'
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
    We add the compound id
    """
    graph['graph_chemblid'] = data['CMPD_CHEMBLID']


@experiment.hook('load_dataset', default=False, replace=True)
def load_dataset(e: Experiment) -> dict[int, dict]:
    df = load_smiles_dataset('lipophilicity')

    dataset: dict[int, dict] = {}
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
        # We have only one target
        data['targets'] = [data['exp']]

        dataset[index] = data
    
    return dataset

experiment.run_if_main()
