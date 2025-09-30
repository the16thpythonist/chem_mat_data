from rdkit import Chem
from typing import List, Dict
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

from chem_mat_data import load_smiles_dataset

DATASET_NAME: str = 'qm9_smiles'
DESCRIPTION: str = (
    'The QM9 dataset is a comprehensive collection of approximately 134,000 stable small '
    'organic molecules composed of up to nine heavy atoms (including carbon, oxygen, nitrogen, '
    'and fluorine). Each molecule in the dataset includes detailed geometric, energetic, electronic, '
    'and thermodynamic properties calculated using density functional theory (DFT), '
    'providing quantum-mechanical ground truth data'
)

# :param METADATA:
#       A dictionary which will be used as the basis for the metadata that will be added 
#       as additional information to the file share server.
METADATA: dict = {
    'tags': [
        'Molecules', 
        'SMILES', 
        'Quantum Chemistry',
        'DFT',
        'Electronic Properties',
    ],
    'verbose': 'QM9 SMILES Dataset',
    'sources': [
        'https://www.nature.com/articles/sdata201422',
        'https://moleculenet.org/datasets',
        'https://pytorch-geometric.readthedocs.io/en/2.5.0/generated/torch_geometric.datasets.QM9.html'
    ],
    # TADF: Thermally Activated Delayed Fluorescence
    'target_descriptions': {
        '0': 'A - rotational constant in GHz',
        '1': 'B - rotational constant in GHz',
        '2': 'C - rotational constant in GHz',
        '3': 'mu - dipole moment in Debye',
        '4': 'alpha - isotropic polarizability in Bohr^3',
        '5': 'homo - Highest occupied molecular orbit in eV',
        '6': 'lumo - lowest unoccupied molecular orbit in eV',
        '7': 'gap - Gap between HOMO and LUMO energies in eV',
        '8': 'R^2 - Electronic spatial extent',
        '9': 'ZPVE - Zero Point vibrational energy',
        '10': 'U_0 - Internal energy at 0K in eV',
        '11': 'U - Internal energy at 298.15K',
        '12': 'H - Enthalpy at 298.15K',
        '13': 'G - Free energy at 298.15K',
        '14': 'Cv - Heat capacity at 298.15K',
        '15': 'U_0 atom - Atomization energy at 0K',
    }
}

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
def load_dataset(e: Experiment) -> Dict[int, dict]:
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

    dataset: Dict[int, dict] = {}
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
