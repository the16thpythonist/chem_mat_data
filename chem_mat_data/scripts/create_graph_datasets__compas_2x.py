import io
import os
import gzip
import shutil
import requests
import pandas as pd

from rdkit import Chem
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace


# :param DATASET_NAME:
#       This string determines the name of the message pack dataset file that is then 
#       stored into the "results" folder of the experiment as the result of the 
#       processing process. The corresponding file extensions will be added 
#       automatically.
DATASET_NAME: str = 'compas_2x'
# :param METADATA:
#       A dictionary which will be used as the basis for the metadata that will be added 
#       as additional information to the file share server.
METADATA: dict = {
    'description': (
        'This dataset is part of the COMputational database of Polycyclic Aromatic Systems (COMPAS) Project. '
        'It contains geometries and properties of cata-condensed poly(hetero)cyclic aromatic molecules, '
        'calculated at the GFN1-xTB level. The dataset includes approximately 500k molecules comprising 11 types '
        'of aromatic and antiaromatic building blocks, representing a highly diverse chemical space. '
        'Various electronic properties such as HOMO-LUMO gap, adiabatic ionization potential, and adiabatic electron '
        'affinity are provided. Additionally, the dataset is benchmarked against a ~50k dataset calculated at the '
        'CAM-B3LYP-D3BJ/def2-SVP level, with a fitting scheme developed to correct the xTB values to higher accuracy.'
    ),
    'target_type': ['Regression'],
    'tags': ['SMILES', 'Molecules', 'Quantum Chemistry', 'Molecular Properties'],
    'sources': [
        'https://chemrxiv.org/engage/chemrxiv/article-details/64bf8dd7b053dad33ad856cf',
        'https://gitlab.com/porannegroup/compas/-/tree/main/COMPAS-1?ref_type=heads',
    ],
    'target_descriptions': {
        0: 'homo - energy of the highest molecular orbit (HOMO) in electron volt (eV)',
        1: 'lumo - energy of the lowest unoccupied molecular orbit (LUMO) in electron volt (eV)',
        2: 'gap - energy gap between HOMO and LUMO in electron volt (eV)',
        3: 'energy - total energy of the molecule in electron volt (eV)',
        4: 'nfod - fractional occupation density',
        5: 'rings - number of rings in the molecule',
        6: 'dispersion - dispersion energy in Hartree (Eh)',
        7: 'aip - adiabatic ionization potential in Hartree (Eh)',
        8: 'aea - adiabatic electron affinity in Hartree (Eh)',
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
    
    # For the COMPAS 2 dataset the raw file is provided in the form of a CSV file that is hosted on Gitlab
    # here we download that file and then open the content as a pandas dataframe.
    e.log('downloading raw csv file...')
    raw_url = 'https://gitlab.com/porannegroup/compas/-/raw/main/COMPAS-2/compas-2x.csv'
    response = requests.get(raw_url, verify=False)
    response.raise_for_status()  # Ensure the request was successful
    csv_file = io.StringIO(response.text)
    df = pd.read_csv(csv_file, quotechar='"')
    
    print(df.head())
    
    compressed_path = os.path.join(e.path, f'{e.DATASET_NAME}.csv.gz')
    with gzip.open(compressed_path, mode='wb') as compressed_file:
        shutil.copyfileobj(csv_file, compressed_file)
    
    #df = load_smiles_dataset('qm9_smiles')

    columns = [
        'homo',
        'lumo',
        'gap',
        'energy',
        'nfod',
        'rings',
        'dispersion',
        'aip',
        'aea',
    ]

    dataset: dict[int, dict] = {}
    index: int = 0
    for c, data in enumerate(df.to_dict('records')):
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
        
        if c % 1000 == 0:
            print(f' * processed {c} records')

    return dataset

experiment.run_if_main()
