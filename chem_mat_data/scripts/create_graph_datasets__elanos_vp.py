from rdkit import Chem
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

from chem_mat_data.config import Config
from chem_mat_data.web import NextcloudFileShare
from chem_mat_data.main import get_file_share
import pandas as pd

# :param DATASET_NAME:
#       This is the name of the dataset that will be used to identify the dataset in the
#       file share server. It will also be used to create the folder structure for the dataset
#       on the file share server.
DATASET_NAME: str = 'elanos_vp'
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
TARGET_COLUMNS: list[str] = ['Log VP']
# :param DATASET_TYPE:
#       Either 'regression' or 'classification' to define the type of the dataset. This
#       will also determine how the target values are processed.
DATASET_TYPE: str = 'regression'
# :param DESCRIPTION:
#       This is a string description of the dataset that will be stored in the experiment
#       metadata.
DESCRIPTION: str = 'A dataset consisting of roughly 2000 molecules annotated with their vapor pressure.'
# :param METADATA:
#       A dictionary which will be used as the basis for the metadata that will be added 
#       as additional information to the file share server.
METADATA: dict = {
    'tags': [
        'Molecules', 
        'SMILES', 
        'Molecular Properties', 
        'Vapor Pressure'
    ],
    'sources': [
        'https://zenodo.org/records/14364265',
        'https://link.springer.com/article/10.1007/s11030-025-11196-5',    
    ],
    'target_descriptions': {
        '0': 'Log VP (Vapor Pressure) in log10(Pa)',
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
    graph['graph_name'] = data['NAME']
    graph['graph_subset'] = data['Subset']


@experiment.hook('load_dataset', default=False, replace=True)
def load_dataset(e: Experiment) -> dict[int, dict]:
    
    ## -- Load Dataset --
    e.log('Loading the CSV file from the remote file share server...')
    config = Config()
    file_share: NextcloudFileShare = get_file_share(config)
    file_path: str = file_share.download_file('elanos_vp.csv', folder_path=e.path)
    df: pd.DataFrame = pd.read_csv(file_path)
    print(df.head())

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
        data['targets'] = [data[col] for col in e.TARGET_COLUMNS]
        dataset[index] = data
        
        index += 1

    return dataset

experiment.run_if_main()
