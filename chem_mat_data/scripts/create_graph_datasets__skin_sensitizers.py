import os
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
DATASET_NAME: str = 'skin_sensitizers'
# :param SMILES_COLUMN:
#       This is the string name of the CSV column which contains the SMILES strings of
#       the molecules.
SMILES_COLUMN: str = 'SMILES'
# :param TARGET_COLUMNS:
#       This is a list of string names of the CSV columns which contain the target values
#       of the dataset. This can be a single column for regression tasks or multiple columns
#       for multi-target regression or classification tasks. For the final graph dataset
#       the target values will be merged into a single numeric vector that contains the 
#       corresponding values in the same order as the column names are defined here.
TARGET_COLUMNS: list[str] = ['label']
# :param DATASET_TYPE:
#       Either 'regression' or 'classification' to define the type of the dataset. This
#       will also determine how the target values are processed.
DATASET_TYPE: str = 'classification'
# :param DESCRIPTION:
#       This is a string description of the dataset that will be stored in the experiment
#       metadata.
DESCRIPTION: str = (
    'Curated dataset of roughly 1000 molecules focused on the skin irritation sensitization potential of small organic molecules.'
    'Data were retrieved from the Registration, Evaluation, Authorization and Restriction of Chemicals (REACH) database, and were curated as part of the study by Borba et al.'    
)
# :param METADATA:
#       A dictionary which will be used as the basis for the metadata that will be added 
#       as additional information to the file share server.
METADATA: dict = {
    'tags': [
        'Molecules', 
        'SMILES', 
        'Biology', 
    ],
    'sources': [
        'https://db.chempharos.eu/datasets/Datasets.zul?datasetID=ds15',
        'https://ehp.niehs.nih.gov/doi/10.1289/EHP9341',    
    ],
    'target_descriptions': {
        '0': 'non-sensitizer',
        '1': 'skin-sensitizer',
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
    graph['graph_subset'] = data['dataset']


@experiment.hook('load_dataset', default=False, replace=True)
def load_dataset(e: Experiment) -> dict[int, dict]:
    
    ## -- Load Dataset --
    e.log('Loading the EXCEL file from the remote file share server...')
    config = Config()
    file_share: NextcloudFileShare = get_file_share(config)
    file_path: str = file_share.download_file('skin_irritation_dataset.xlsx', folder_path=e.path)
    df: pd.DataFrame = pd.read_excel(file_path)
    print(df.head())
    
    ## -- Save Dataset --
    e.log('Saving the dataset as CSV and GZipped CSV file...')
    csv_path = os.path.join(e.path, f'{e.DATASET_NAME}.csv')
    df.to_csv(csv_path, index=False)

    gz_path = csv_path + '.gz'
    with open(csv_path, 'rb') as f_in, gzip.open(gz_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

    ## -- Processing Dataset --
    dataset: dict[int, dict] = {}
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
        data['targets'] = [0, 1] if data['Label'] else [1, 0]
        dataset[index] = data
        
        index += 1

    return dataset

experiment.run_if_main()
