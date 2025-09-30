import os
from typing import List, Dict
import pandas as pd
import gzip
import shutil
from rdkit import Chem
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

from chem_mat_data.config import Config
from chem_mat_data.web import NextcloudFileShare
from chem_mat_data.main import get_file_share

# :param DATASET_NAME:
#       This is the name of the dataset that will be used to identify the dataset in the
#       file share server. It will also be used to create the folder structure for the dataset
#       on the file share server.
DATASET_NAME: str = 'tadf'
# :param SMILES_COLUMN:
#       This is the string name of the CSV column which contains the SMILES strings of
#       the molecules.
SMILES_COLUMN: str = 'smiles'
# :param TARGET_COLUMNS:
#       This is a list of string names of the CSV columns which contain the target values
#       of the dataset. This can be a single column for regression tasks or multiple columns
#       for multi-target regression or classification tasks. For the final graph dataset
#       the target values will be merged into a single numeric vector that contains the 
#       corresponding values in the same order as the column names are defined here.
TARGET_COLUMNS: List[str] = ['tadf_rate', 'splitting_energy', 'oscillator_strength']
# :param DATASET_TYPE:
#       Either 'regression' or 'classification' to define the type of the dataset. This
#       will also determine how the target values are processed.
DATASET_TYPE: str = 'regression'
# :param DESCRIPTION:
#       This is a string description of the dataset that will be stored in the experiment
#       metadata.
DESCRIPTION: str = (
    'Set of organic molecules used in a high-throughput virtual screening approach of '
    'to find candidates for more efficient molecular organic light-emitting diodes.'
)
# :param METADATA:
#       A dictionary which will be used as the basis for the metadata that will be added 
#       as additional information to the file share server.
METADATA: dict = {
    'tags': [
        'Molecules', 
        'SMILES', 
        'DFT',
        'Electronic Properties',
        'OLED', 
    ],
    'sources': [
        'https://www.nature.com/articles/nmat4717#Sec16',  
    ],
    # TADF: Thermally Activated Delayed Fluorescence
    'target_descriptions': {
        '0': 'tadf_rate - Thermally activated delayed fluorescence (TADF) rate',
        '1': 'splitting_energy - Singlet-triplet splitting energy',
        '2': 'oscillator_strength - Oscillator strength',
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
    #graph['graph_name'] = data['Name']
    #graph['graph_subset'] = data['dataset']


@experiment.hook('load_dataset', default=False, replace=True)
def load_dataset(e: Experiment) -> Dict[int, dict]:
    
    ## -- Load Dataset --
    e.log('Loading the CSV file from the remote file share server...')
    config = Config()
    file_share: NextcloudFileShare = get_file_share(config)
    file_path: str = file_share.download_file('tadf.csv', folder_path=e.path)
    df: pd.DataFrame = pd.read_csv(file_path)
    print(df.head())
    
    ## -- Save Dataset --
    e.log('Saving the dataset as CSV and GZipped CSV file...')
    csv_path = os.path.join(e.path, f'{e.DATASET_NAME}.csv')
    df.to_csv(csv_path, index=False)

    gz_path = csv_path + '.gz'
    with open(csv_path, 'rb') as f_in, gzip.open(gz_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

    ## -- Processing Dataset --
    dataset: Dict[int, dict] = {}
    index: int = 0
    for data in df.to_dict('records'):
        
        data['smiles'] = data[e.SMILES_COLUMN]
        
        ## -- Molecule Filters --
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
        
        ## -- Target Values --
        # In this dataset we only have one target
        data['targets'] = [data[target_key] for target_key in e.TARGET_COLUMNS]
        dataset[index] = data
        
        index += 1

    return dataset

experiment.run_if_main()
