import rdkit.Chem as Chem
import pandas as pd
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

from chem_mat_data import load_smiles_dataset

DATASET_NAME: str = 'muv'

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
    Add the mol_id for identification
    """
    graph['graph_id'] = data['mol_id']
    

@experiment.hook('load_dataset', default=False, replace=True)
def load_dataset(e: Experiment) -> dict[int, dict]:

    df = load_smiles_dataset('muv')
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
        #We have 17 targets and each one can be active or inactive, so either 0 or 1. For some there is no data for wich use -1.
        assay_columns = ['MUV-466', 'MUV-548', 'MUV-600', 'MUV-644', 'MUV-652', 'MUV-689',
                         'MUV-692', 'MUV-712', 'MUV-713', 'MUV-733', 'MUV-737', 'MUV-810',
                         'MUV-832', 'MUV-846', 'MUV-852', 'MUV-858', 'MUV-859']

        data['targets'] = [(0 if data[assay] == 0 else 1) if pd.notna(data[assay]) else 0 for assay in assay_columns]
        dataset[index] = data

    return dataset

experiment.run_if_main()
