import rdkit.Chem as Chem
from typing import List, Dict
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

from chem_mat_data import load_smiles_dataset


DATASET_NAME: str = 'lipophilicity'
__TESTING__ = False

DESCRIPTION: str = (
    'The MoleculeNet lipophilicity dataset contains experimental octanol/water '
    'distribution coefficient values ''(logD at pH 7.4) for 4200 compounds curated '
    'from the ChEMBL database. This regression dataset was contributed to MoleculeNet '
    'by Patrick Hop and serves as a benchmark for evaluating machine learning models  '
    'ability to predict molecular lipophilicity from chemical structure. '
    'Lipophilicity is an important physicochemical property that affects drug membrane '
    'permeability and solubility, making this dataset valuable for pharmaceutical research '
    'and computational chemistry applications.'
)

# :param METADATA:
#       A dictionary which will be used as the basis for the metadata that will be added 
#       as additional information to the file share server.
METADATA: dict = {
    'tags': [
        'Molecules', 
        'SMILES', 
        'Experimental',
        'Partition Coefficient',
    ],
    'verbose': 'Octanol-Water Distribution Coefficient',
    'sources': [
        'https://doi.org/10.1039/C7SC02664A',
        'https://moleculenet.org/datasets-1',
    ],
    # TADF: Thermally Activated Delayed Fluorescence
    'target_descriptions': {
        '0': 'logD - Experimental logD (pH 7.4) octanol-water distribution coefficient',
    }
}

experiment = Experiment.extend(
    'create_graph_datasets.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)

@experiment.hook('add_graph_metadata', default=False, replace=True)
def add_graph_metadata(e: Experiment, data: dict, graph: dict) -> dict:
    """
    We add the compound id
    """
    graph['graph_chemblid'] = data['CMPD_CHEMBLID']


@experiment.hook('load_dataset', default=False, replace=True)
def load_dataset(e: Experiment) -> Dict[int, dict]:
    df = load_smiles_dataset('lipophilicity')

    dataset: Dict[int, dict] = {}
    for index, data in enumerate(df.to_dict('records')):
        data['smiles'] = data['smiles']
        
        # === MOLECULE FILTERS ===
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
        # We have only one target
        data['targets'] = [data['exp']]

        dataset[index] = data
    
    return dataset

experiment.run_if_main()
