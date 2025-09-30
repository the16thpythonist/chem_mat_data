import pandas as pd
import rdkit.Chem as Chem
from typing import List, Dict
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

from chem_mat_data import load_smiles_dataset
from chem_mat_data.connectors import FileDownloadSource

DATASET_NAME: str = 'open_melting_point'
DESCRIPTION: str = (
    'The Jean-Claude Bradley Open Melting Point Dataset contains 28k melting point measurements '
    'for organic compounds. A highly curated subset of 3,041 validated measurements is also '
    'available, containing only data with multiple measurements and temperature ranges between '
    '0.01°C and 5°C. Released as open data under a CC0 license, this dataset is widely used in '
    'machine learning and computational chemistry research for developing models to predict '
    'melting points from molecular structure.'
)

# :param METADATA:
#       A dictionary which will be used as the basis for the metadata that will be added 
#       as additional information to the file share server.
METADATA: dict = {
    'tags': [
        'Molecules', 
        'SMILES', 
        'Physical Property',
        'Melting Point',
    ],
    'verbose': 'Melting Point',
    'sources': [
        'https://www.nature.com/articles/npre.2011.6229.1',
        'http://dx.doi.org/10.6084/m9.figshare.1031638',
        'https://figshare.com/articles/dataset/Jean_Claude_Bradley_Open_Melting_Point_Datset/1031637?file=1503990'
    ],
    # TADF: Thermally Activated Delayed Fluorescence
    'target_descriptions': {
        '0': 'mpC - Experimental melting point in Celsius degrees'
    }
}

experiment = Experiment.extend(
    'create_graph_datasets.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)

@experiment.hook('load_dataset', default=False, replace=True)
def load_dataset(e: Experiment) -> Dict[int, dict]:
    
    # This file contains the dataset in question. It is available as a download link from a 
    # Chemarxiv paper.
    e.log('Downloading dataset...')
    with FileDownloadSource(
        'https://figshare.com/ndownloader/files/1503990',
        verbose=True,
        ssl_verify=False,
    ) as source:
        path = source.fetch()
        df = pd.read_excel(path, sheet_name=0)
        
        e.log(f'Downloaded dataset with {len(df)} entries')
        print(df.head())
        
        # Filter out rows where 'donotuse' is not NaN
        df = df[df['donotuse'].isna()]
        e.log(f'After filtering, dataset has {len(df)} entries')
        print(df.head())
    
    # Now that we have the raw data as a dataframe, we can convert it into the expected 
    # dictionary format where each entry in the dictionary corresponds to a single data point 
    # in the dataset.
    e.log('Processing dataset...')
    dataset: Dict[int, dict] = {}
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
        data['targets'] = [data['mpC']]

        dataset[index] = data

    return dataset

experiment.run_if_main()
