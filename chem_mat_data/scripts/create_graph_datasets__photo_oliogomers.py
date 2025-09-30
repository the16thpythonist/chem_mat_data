import sys
from rdkit import Chem
from typing import List, Dict
import pandas as pd
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace
from chem_mat_data.connectors import ZenodoSource


DATASET_NAME: str = 'photo_oliogomers'

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
    
    e.log('Downloading dataset from Zenodo...')
    with ZenodoSource(
        url="https://zenodo.org/records/11580890/files/TheJacksonLab/ClosedLoopTransfer-ClosedLoopTransfer.zip?download=1",
        relative_path='TheJacksonLab-ClosedLoopTransfer-84e11e3/OliogomerFeatures.csv'
    ) as source:
        path = source.fetch()
        df = pd.read_csv(path)
        e.log(f'dataset of size {len(df)} loaded from Zenodo')
        e.log(df.head())
    
    sys.exit(1)
    
    df = pd.read_csv('/media/data/Downloads/mcule_stock_210622_prices.csv')
    e.log(f'dataset of size {len(df)} loaded')
    e.log(df.head())
    
    # Subsample 1% of the dataset
    df = df.sample(frac=0.01, random_state=42)

    columns = [
        'price 4 (USD)'
    ]

    dataset: Dict[int, dict] = {}
    index: int = 0
    for c, data in enumerate(df.to_dict('records')):
        data['smiles'] = data['SMILES']
        
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
            print(f' * {c} elements loaded')

    e.log('returning dataset...')
    return dataset

experiment.run_if_main()