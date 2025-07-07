import io
import os
import requests
import pandas as pd
import gzip
import shutil

from rdkit import Chem
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace


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
        'The COMPAS Project provides a computational database of polycyclic aromatic systems, '
        'including optimized ground-state structures and molecular properties of approximately '
        '34k and 9k cata-condensed polybenzenoid hydrocarbons. This dataset supports the investigation '
        'of structure-property relationships and the design of novel functional compounds, with data '
        'generated at the GFN2-xTB and B3LYP-D3BJ/def2-SVP levels.'
    ),
    'target_type': ['Regression'],
    'tags': ['SMILES', 'Molecules', 'Quantum Chemistry', 'Molecular Properties'],
    'sources': [
        'https://pubs.acs.org/doi/full/10.1021/acs.jcim.2c00503',
        'https://gitlab.com/porannegroup/compas/-/tree/main/COMPAS-1?ref_type=heads',
    ],
    'target_descriptions': {
        0: 'HOMO_eV - energy of the highest molecular orbit (HOMO) in electron volt (eV)',
        1: 'LUMO_eV - energy of the lowest unoccupied molecular orbit (LUMO) in electron volt (eV)',
        2: 'GAP_eV - energy gap between HOMO and LUMO in electron volt (eV)',
        3: 'Dipmom_Debye - dipole moment in Debye',
        4: 'Etot_eV - total energy of the molecule in electron volt (eV)',
        5: 'aEA_eV - adiabatic electron affinity in electron volt (eV)',
        6: 'aIP_eV - adiabatic ionization potential in electron volt (eV)',
        7: 'NFOD - fractional occupation density',
        8: 'n_rings - number of rings in the molecule',
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
    response = requests.get(raw_url, verify=False)
    csv_file = io.StringIO(response.text)
    df = pd.read_csv(csv_file)
    
    print(df.head())
    
    compressed_path = os.path.join(e.path, f'{e.DATASET_NAME}.csv.gz')
    with gzip.open(compressed_path, mode='wb') as compressed_file:
        shutil.copyfileobj(csv_file, compressed_file)

    columns = [
        'HOMO_eV',
        'LUMO_eV',
        'GAP_eV',
        'Dipmom_Debye',
        'Etot_eV',
        'aEA_eV',
        'aIP_eV',
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
