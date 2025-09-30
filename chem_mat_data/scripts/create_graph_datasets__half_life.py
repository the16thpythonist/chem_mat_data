import pandas as pd
import rdkit.Chem as Chem
from typing import List, Dict
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

from chem_mat_data import load_smiles_dataset
from chem_mat_data.connectors import FileDownloadSource

DATASET_NAME: str = 'half_life'
DESCRIPTION: str = (
    'The dataset consists of 6,309 experimental soil biotransformation half-life '
    'values for 893 pesticides and pesticide transformation products, extracted '
    'from publicly available regulatory reports via the EAWAG-SOIL package on enviPath. '
    'Each substance can have multiple reported half-life values obtained from different '
    'experiments conducted under varying environmental conditions (different soils, '
    'temperatures, pH levels, etc.) and calculated using different kinetic models, '
    'resulting in substantial variability spanning several orders of magnitude for '
    'the same chemical. The dataset includes both uncensored half-life measurements '
    'and censored values (311 total) that fall beyond reliable quantification limits, '
    'either as very fast degradation (<0.1 days) or very slow degradation (>1000 days).'
)

# :param METADATA:
#       A dictionary which will be used as the basis for the metadata that will be added 
#       as additional information to the file share server.
METADATA: dict = {
    'tags': [
        'Molecules', 
        'SMILES', 
        'Environment',
        'Half-Life',
    ],
    'verbose': 'Half-Life Biotransformation',
    'sources': [
        'https://chemrxiv.org/engage/chemrxiv/article-details/64be471cb605c6803b425da6'
    ],
    # TADF: Thermally Activated Delayed Fluorescence
    'target_descriptions': {
        '0': 'DT50 - Mean Experimental half-life (DT50) in days under aerobic soil conditions'
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
        'https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/64be48f1ae3d1a7b0d3f5eeb/original/data-set-soil-biotransformation-half-lives.xlsx',
        verbose=True,
        ssl_verify=False,
    ) as source:
        path = source.fetch()
        df = pd.read_excel(path, sheet_name=1)
        
        e.log(f'Downloaded dataset with {len(df)} entries')
        print(df.head())
    
    # Now that we have the raw data as a dataframe, we can convert it into the expected 
    # dictionary format where each entry in the dictionary corresponds to a single data point 
    # in the dataset.
    e.log('Processing dataset...')
    dataset: Dict[int, dict] = {}
    for index, data in enumerate(df.to_dict('records')):

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
        
        # == TARGET VALUES ==
        # Only one target. We combine the experimental and computational data into a single list 
        data['targets'] = [data['DT50_gmean']]

        dataset[index] = data

    return dataset

experiment.run_if_main()
