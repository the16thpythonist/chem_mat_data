import rdkit.Chem as Chem
import pandas as pd
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

from chem_mat_data import load_smiles_dataset

DATASET_NAME: str = 'dpp4'
# :param DESCRIPTION:
#       This is a string description of the dataset that will be stored in the experiment
#       metadata.
DESCRIPTION: str = (
    'DPP-4 inhibitors (DPP4) was extract from ChEMBL with DPP-4 target. '
    'The data was processed by removing salt and normalizing molecular structure, '
    'with molecular duplication examination, leaving 3933 molecules.'
    'Each molecule is associated with two target values (Activity(IC50), Activity(pIC50)) '
    'where the value is the binary label if the compound is active for IC50 or not. The '
    'second value is the binary label if the compound is active for pIC50 or not.'
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
    We add the molecular formula as additional metadata and wether it is training or testing data
    """
    graph['graph_molecular_formula'] = data['MolecularFormula']
    graph['graph_split'] = data['split']


@experiment.hook('load_dataset', default=False, replace=True)
def load_dataset(e: Experiment) -> dict[int, dict]:
    df = load_smiles_dataset('dpp4')
    dataset: dict[int, dict] = {}
    columns = ['Activity(IC50)','Activity(pIC50)']
    for index, data in enumerate(df.to_dict('records')):
        data['smiles'] = data['smiles']
        
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
        dataset[index] = data
        
    return dataset

experiment.run_if_main()

    
