"""
This module implements the processing of the BACE dataset. It is a sub experiment that 
inherits from "create_graph_datasets.py" base experiment. It overwrites the default 
implementation of the "load_dataset" hook to load the BACE dataset instead from the 
nextcloud data storage.
"""
import rdkit.Chem as Chem
from typing import List, Dict
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

from chem_mat_data import load_smiles_dataset


DATASET_NAME: str = 'bace_cls'
DESCRIPTION: str = (
    'The BACE dataset contains quantitative IC50 and qualitative binary binding '
    'results for 1522 inhibitors of human β-secretase 1 (BACE-1), an important enzyme '
    'target for Alzheimers disease research. The dataset was originally compiled by '
    'Subramanian et al. (2016) from experimental values reported in scientific literature '
    'and integrated into MoleculeNet as a binary classification benchmark. All data represent '
    'experimental values with some compounds having detailed crystal structures available, '
    'making it valuable for drug discovery applications targeting neurodegeneration.'
)

# :param METADATA:
#       A dictionary which will be used as the basis for the metadata that will be added 
#       as additional information to the file share server.
METADATA: dict = {
    'tags': [
        'Molecules', 
        'SMILES', 
        'Protein Target',
        'Protein Binding',
        'Inhibitors',
    ],
    'target_type': ['classification'],
    'verbose': 'Beta Secretase 1 Inhibitors',
    'sources': [
        'https://pubs.acs.org/doi/full/10.1021/acs.jcim.6b00290',
        'https://doi.org/10.1039/C7SC02664A'
    ],
    # TADF: Thermally Activated Delayed Fluorescence
    'target_descriptions': {
        '0': 'Inactive class indicator - 1 if the compound is inactive',
        '1': 'Active class indicator - 1 if the compound is a beta secretase 1 inhibitor',
    }
}

__TESTING__ = False

# This is the dataset template.
experiment = Experiment.extend(
    'create_graph_datasets.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)

@experiment.hook('add_graph_metadata', default=False, replace=True)
def add_graph_metadata(e: Experiment, data: dict, graph: dict) -> dict:
    """
    This hook is invoked in the processing worker after the SMILES code has been converted 
    to the graph dict already. The hook receives the original data dict and the graph dict 
    as arguments and provides the opportunity to add additional metadata to the graph dict.
    
    ---
    
    In this experiment, the CID is added as an additional metadata field to the graph dict 
    as that ID might be useful to identify the molecule later on.
    """
    graph['graph_id'] = data['CID']

@experiment.hook('load_dataset', default=False, replace=True)
def load_dataset(e: Experiment) -> Dict[int, dict]:
    """
    In the experiment, this hook is invoked at the very beginning to obtain the actual 
    raw data of the dataset that should be processed. The output of this function should 
    be a dictionary whose keys are the integer indices of the data elements and the values 
    are in turn dictionary objects that should contain AT LEAST the following keys:
    - 'smiles': The SMILES representation of the molecule
    - 'targets': A list of float target values for the molecule

    ---
    
    In this experiment, the source data is loaded from the nextcloud file share itself 
    and then custom processed based on the original dataset columns contained in the 
    dataset.
    """
    # Instead of loading the dataset from a CSV file that is located in the local file system 
    # we can use the functionality that is already implemented in the chem_mat_data package to 
    # instead load the raw dataset version from the nextcloud data storage as a dataframe.
    df = load_smiles_dataset('bace')
    
    dataset: Dict[int, dict] = {}
    for index, data in enumerate(df.to_dict('records')):
        
        data['smiles'] = data['mol']
        
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
        
        # For this dataset we specifically know that there are only 2 classes. The "Class" column is 
        # either "0" or "1"
        data['targets'] = [data['Class'] == index for index in range(2)]
        
        dataset[index] = data
    
    return dataset


# We need this at the end or otherwise the main experiment code won't actually execute 
# when the python module is started.
experiment.run_if_main()