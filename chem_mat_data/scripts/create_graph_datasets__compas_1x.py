import io
import requests
import pandas as pd

from rich.pretty import pprint
from rdkit import Chem
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

from chem_mat_data import load_smiles_dataset

# :param DATASET_NAME:
#       This string determines the name of the message pack dataset file that is then 
#       stored into the "results" folder of the experiment as the result of the 
#       processing process. The corresponding file extensions will be added 
#       automatically.
DATASET_NAME: str = 'compas_1x'
# :param METADATA:
#       A dictionary which will be used as the basis for the metadata that will be added 
#       as additional information to the file share server.
METADATA: dict = {
    'description': (
        'The COMPAS-1 dataset is part of the largest freely available collection of geometries and properties of cata-condensed '
        'poly(hetero)cyclic aromatic molecules. It includes quantum chemical properties of 1,000 molecules calculated at the '
        'GFN1-xTB level, representative of a highly diverse chemical space.'
    ),
    'target_type': ['Regression'],
    'tags': ['SMILES', 'Molecules', 'Quantum Chemistry', 'Molecular Properties'],
    'sources': [
        'https://chemrxiv.org/engage/chemrxiv/article-details/64bf8dd7b053dad33ad856cf',
        'https://gitlab.com/porannegroup/compas/-/tree/main/COMPAS-1?ref_type=heads',
    ],
    'target_descriptions': {
        0: 'HOMO_eV_corrected - The corrected energy of the Highest Occupied Molecular Orbital (HOMO) in electron volts (eV).',
        1: 'LUMO_eV_corrected - The corrected energy of the Lowest Unoccupied Molecular Orbital (LUMO) in electron volts (eV).',
        2: 'GAP_eV_corrected - The corrected energy gap between the HOMO and LUMO in electron volts (eV).',
        3: 'aIP_eV_corrected - The corrected adiabatic ionization potential in electron volts (eV).',
        4: 'aEA_eV_corrected - The corrected adiabatic electron affinity in electron volts (eV).',
        5: 'Erel_eV_corrected - The corrected relative energy in electron volts (eV).',
        6: 'Dipmom_Debye - The dipole moment of the molecule in Debye units.',
        7: 'NFOD - The number of free valence electrons in the molecule.',
        8: 'n_rings - The number of ring structures within the molecule.',
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
def load_dataset(e: Experiment) -> dict[int, dict]:
    
    e.log('downloading raw csv file...')
    raw_url = 'https://gitlab.com/porannegroup/compas/-/raw/main/COMPAS-1/compas-1x.csv?ref_type=heads&inline=false'
    response = requests.get(raw_url)
    df = pd.read_csv(io.StringIO(response.text))
    
    print(df.head())
    
    #df = load_smiles_dataset('qm9_smiles')

    columns = [
        'HOMO_eV_corrected',
        'LUMO_eV_corrected',
        'GAP_eV_corrected',
        'aIP_eV_corrected',
        'aEA_eV_corrected',
        'Erel_eV_corrected',
        'Dipmom_Debye',
        'NFOD',
        'n_rings',
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
