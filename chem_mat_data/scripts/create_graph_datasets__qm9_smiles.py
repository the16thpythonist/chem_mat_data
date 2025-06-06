from rdkit import Chem
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

from chem_mat_data import load_smiles_dataset

DATASET_NAME: str = 'qm9_smiles'

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
    We add the compound id for identification and the molecular weight
    """
    pass #graph['graph_id'] = data['ID']


@experiment.hook('load_dataset', default=False, replace=True)
def load_dataset(e: Experiment) -> dict[int, dict]:
    df = load_smiles_dataset('qm9_smiles')

    columns = [
        'A',
        'B',
        'C',
        'mu',
        'alpha',
        'homo',
        'lumo',
        'gap',
        'r2',
        'zpve',
        'u0',
        'u298',
        'h298',
        'g298',
        'cv',
        'u0_atom',
        'u298_atom',
        'h298_atom',
        'g298_atom',
    ]

    dataset: dict[int, dict] = {}
    index: int = 0
    for data in df.to_dict('records'):
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
        
        # == TARGETS ==
        # In this dataset we only have one target
        data['targets'] = [data[key] for key in columns]
        dataset[index] = data
        
        index += 1

    return dataset

experiment.run_if_main()
