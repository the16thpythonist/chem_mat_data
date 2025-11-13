import io
from typing import List, Dict
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
    'verbose': 'Cata-Condensed Hetero-Polycyclic Aromatic Systems (COMPAS-2x)',
    'description': (
        'The COMPAS-2x dataset is the second installment of the COMPAS Project, presenting the largest freely '
        'available dataset of geometries and properties of cata-condensed poly(hetero)cyclic aromatic systems. '
        'This dataset contains optimized ground-state structures and molecular properties for approximately '
        '524,392 unique molecules, computed at the GFN1-xTB level of theory. The molecules range in size from '
        '2 to 10 rings and are constructed from a library of 11 diverse building blocks containing heteroatoms '
        '(boron, nitrogen, oxygen, and sulfur) in aromatic and antiaromatic rings of varying sizes (4-6 membered). '
        'This represents a highly diverse chemical space of hetero-polycyclic aromatic systems, expanding beyond '
        'the pure hydrocarbon systems of COMPAS-1. Key quantum chemical properties include HOMO/LUMO energies, '
        'energy gaps, dispersion energies, and electron affinities. The dataset is benchmarked against COMPAS-2D, '
        'a ~52,000-molecule subset calculated at the CAM-B3LYP-D3BJ/def2-SVP level, with a fitting scheme developed '
        'to correct the GFN1-xTB values to higher accuracy. This resource supports high-throughput screening, '
        'generative models, and machine learning applications in organic electronics and optoelectronics.'
    ),
    'target_type': ['regression'],
    'tags': ['SMILES', 'Molecules', 'Quantum Chemistry', 'Molecular Properties', 'Polycyclic Aromatic Systems', 'Heteroaromatic', 'GFN1-xTB', 'Cata-Condensed'],
    'sources': [
        'https://www.nature.com/articles/s41597-024-02927-8',
        'https://pubmed.ncbi.nlm.nih.gov/38242917/',
        'https://pmc.ncbi.nlm.nih.gov/articles/PMC10799083/',
        'https://chemrxiv.org/engage/chemrxiv/article-details/64bf8dd7b053dad33ad856cf',
        'https://figshare.com/articles/dataset/The_b_COMPAS_Project_b_Phase_2_Cata-Condensed_Hetero-Polycyclic_Aromatic_Systems_COMPAS-2_/24347152',
        'https://gitlab.com/porannegroup/compas/-/tree/main/COMPAS-2?ref_type=heads',
    ],
    'target_descriptions': {
        0: 'homo - Energy of the highest occupied molecular orbital (HOMO) in electron volts (eV)',
        1: 'lumo - Energy of the lowest unoccupied molecular orbital (LUMO) in electron volts (eV)',
        2: 'gap - HOMO-LUMO energy gap in electron volts (eV)',
        3: 'energy - Total energy of the molecule in electron volts (eV)',
        4: 'nfod - Fractional occupation density (indicator of polyradical character)',
        5: 'rings - Number of aromatic rings in the molecule',
        6: 'dispersion - Dispersion energy in Hartree (Eh)',
        7: 'aip - Adiabatic ionization potential in Hartree (Eh)',
        8: 'aea - Adiabatic electron affinity in Hartree (Eh)',
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

    dataset: Dict[int, dict] = {}
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
