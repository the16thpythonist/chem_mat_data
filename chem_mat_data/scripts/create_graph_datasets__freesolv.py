import rdkit.Chem as Chem
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

from chem_mat_data import load_smiles_dataset


DATASET_NAME: str = 'freesolv'
DESCRIPTION: str = (
    'A collection of experimental and calculated hydration free energies for small molecules in water. '
    'The calculated values are derived from alchemical free energy calculations using molecular dynamics simulations. '
    'Each molecule is associated with 2 target values (experimental, calculated) where the first '
    'value is the experimental hydration free energy and the second value is the calculated hydration free energy.'
)

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
        # Only one target. We combine the experimental and computational data into a single list 
        # Not sure if the following combines the 2 columns(?)
        data['targets'] = [data['expt'], data['calc']]

        dataset[index] = data

    return dataset

experiment.run_if_main()
