"""
This experiment module is the base module that handles the creation of a graph dataset from a 
a representation of a TUDataset (https://chrsmrrs.github.io/datasets/docs/datasets/). TUDatasets 
are a collection of datasets for graph-based machine learning tasks.

The problem is that these are not only sourced from chemistry but also different domains and 
as such the representation that is used is very generic. This means that in the first step 
this generic representations needs to be converted back into a SMILES representation before
converting to a graph dataset.
"""
import os
import csv
import rdkit.Chem as Chem
import zipfile
from typing import Dict, List, Optional

import requests
import numpy as np
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

from chem_mat_data.data import TUDatasetParser

# :param SOURCE_PATH:
#       This is the absolute string path of the TU dataset path. This should be a folder which contains 
#       the characteristic files of the TU dataset format - including files that represent the node and 
#       edge features, as well as the graph labels and the connectivity information.
SOURCE_PATH: Optional[str] = None # os.path.join(SCRIPTS_PATH, 'assets', 'qm9.xyz_bundle')
# :param SOURCE_URL:
#       As a fallback option, if no local dataset path is given, this parameter can define the URL at 
#       which the dataset can be downloaded from. This should be a valid URL that points to the TU dataset 
#       format. If given, the experiment will attempt to download and extract before loading the dataset.
SOURCE_URL: Optional[str] = 'https://www.chrsmrrs.com/graphkerneldatasets/MUTAG.zip'
# :param DATASET_NAME:
#       The name of the dataset which will be used as the final file name aka the unique string 
#       identifier on the remote file share server.
DATASET_NAME: str = 'MUTAG'
# :param NODE_LABEL_MAP:
#       This is a dictionary that maps the node label classes to the integer values corresponding to 
#       the actual atomic numbers. This mapping will have to be extracted from the README file of 
#       the TU dataset.
NODE_LABEL_MAP: Dict[str, int] = {
    0: 'C',
    1: 'N',
    2: 'O',
    3: 'F',
    4: 'I',
    5: 'Cl',
    6: 'Br',
}
# :param EDGE_LABEL_MAP:
#       This is a dictionary that maps the edge label classes to the integer values corresponding to
#       the actual bond types. This mapping will have to be extracted from the README file of
#       the TU dataset.
EDGE_LABEL_MAP: Dict[str, int] = {
    0: Chem.BondType.AROMATIC,
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.TRIPLE,
}
# :param GRAPH_LABEL_MAP:
#       This is a dictionary that maps the graph label classes to the integer values corresponding to
#       the actual graph labels. This mapping will have to be extracted from the README file of
#       the TU dataset.
GRAPH_LABEL_MAP: Dict[int, list] = {
    1: [0, 1],
    -1: [1, 0],
}

__DEBUG__ = True
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
    No extra metadata
    """
    pass


@experiment.hook('load_dataset', default=False, replace=True)
def load_dataset(e: Experiment) -> dict[int, dict]:
    """
    This hook should define the loading of the dataset and return the dataset as an index data map
    whose keys are the string indices and the values are the dictionaries that represent the graph 
    structures themselves.
    """
    
    e.log(f'loading TU dataset {e.DATASET_NAME}...')
    
    # This will contain all the converted graph structures.
    index_data_map: Dict[int, dict] = {}
    
    # ~ locating dataset on disk
    # First we need to check if the dataset path already exists on the disk. If that is not the 
    # case then we try to download the dataset from the given URL.
    if e.SOURCE_PATH is None or not os.path.exists(e.SOURCE_PATH):
        
        e.log(f'dataset not found on disk. donwloading @ {e.SOURCE_URL}')
        response = requests.get(e.SOURCE_URL, stream=True)
        zip_filename = os.path.join(e.path, os.path.basename(e.SOURCE_URL))
        
        with open(zip_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(e.path)
        
        e.SOURCE_PATH = os.path.join(e.path, e.DATASET_NAME)  # Update SOURCE_PATH to extracted folder
        
    # ~ loading the dataset
    # Now we can use the TUDatasetParser class to load the dataset from the given path. We can 
    # consume the parser instance as a generator which will yield tuples of (mol, label) for 
    # each graph in the dataset.
    
    e.log(f'loading dataset @ {e.SOURCE_PATH}...')
    parser = TUDatasetParser(
        path=e.SOURCE_PATH, 
        name=e.DATASET_NAME,
        node_label_map=e.NODE_LABEL_MAP,
        edge_label_map=e.EDGE_LABEL_MAP,
        graph_label_map=e.GRAPH_LABEL_MAP,
    )
    parser.initialize()
    parser.load()
    e.log(f' * loaded {len(parser.index_graph_map)} graphs')
    
    for index, (mol, label) in enumerate(parser):
        index_data_map[index] = {
            'smiles': Chem.MolToSmiles(mol),
            'targets': label,
        }
        
    return index_data_map


@experiment.hook('after_save', default=False, replace=False)
def after_save(e: Experiment,
               index_data_map: Dict[int, dict],
               graphs: List[dict],
               **kwargs,
               ) -> None:
    """
    This hook is called after the graphs dataset itself was already saved.
    
    ---
    
    In the case of the xyz bundle datasets, we will create a meta.csv file which contains all the 
    additional information about the dataset.
    """
    e.log('creating meta.csv file...')
    
    # ~ figuring out the keys
    # First of all we need to figure out which of the keys in the "data" dicts we can actually 
    # reasonably export into a csv file. The problem is that this dict also contains the "mol"
    # key which for example is a Chem.Mol instance and therefore not able to export into the 
    # csv format.
    # To do this we loop over all the keys of an example element and check the types.
    example_data: dict = next(iter(index_data_map.values()))
    keys: list[str] = []
    for key, value in example_data.items():
        
        # if the value is a primitive data type we use it for sure.
        if isinstance(value, (str, int, float)):
            keys.append(key)

        # if the value is a one-dimensional xyz file then we also include it.                
        if isinstance(value, (np.ndarray)):
            if value.ndim <= 1:
                keys.append(key)
                
    e.log(f'using keys: {keys}')

    # ~ saving the csv file.
    # Now we can simply iterate over all the elements in our dataset and save the dataset
    
    csv_path = os.path.join(e.path, 'meta.csv')
    with open(csv_path, mode='w') as file:
        dict_writer = csv.DictWriter(file, fieldnames=keys)
        dict_writer.writeheader()
        for index, data in index_data_map.items():
            dict_writer.writerow({key: data[key] for key in keys})
    
    e.log(f'saved meta.csv file @ {csv_path}')


experiment.run_if_main()
