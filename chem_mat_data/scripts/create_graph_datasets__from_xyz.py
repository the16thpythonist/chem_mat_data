"""
This experiment module is the base module that handles the creation of a graph dataset from a 
raw representation as an "xyz bundle". In the most basic form, such an xyz bundle represents a target 
dataset as a folder or an archive consisting of multiple .xyz files where each file represents one 
element of the dataset. These .xyz files already contain additional information about the 3D coordinates 
of the atoms.
"""
import os
import csv
import rdkit.Chem as Chem
import pandas as pd
from typing import Dict, List

import numpy as np
import pandas as pd
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

from chem_mat_data.main import load_xyz_dataset
from chem_mat_data import load_smiles_dataset
from chem_mat_data.utils import SCRIPTS_PATH
from chem_mat_data.data import AbstractXyzParser
from chem_mat_data.data import DefaultXyzParser
from chem_mat_data.data import QM9XyzParser

# :param SOURCE_PATH:
#       This is the absolute string path of the dataset in xyz format. This can either be an archive or 
#       directly a folder which points to an "xyz bundle". This bundle has to ultimately contain a number 
#       of distinct .XYZ files where each file defines one element (molecule) in the target dataset.
#       Optionally, this xyz bundle may also contain a csv file with additional information about each 
#       of the elements in the dataset.
SOURCE_PATH: str = os.path.join(SCRIPTS_PATH, 'assets', '_test.xyz_bundle')
# :param PARSER_CLASS:
#       This has to define the class that is to be used for the parsing of the xyz files. The class that 
#       is defined here needs to be a valid subclass of the AbstractXyzParser interface. There are different 
#       flavors of xyz files for which different parser classes exist.
PARSER_CLASS: type = QM9XyzParser
# :param DATASET_NAME:
#       The name of the dataset which will be used as the final file name aka the unique string 
#       identifier on the remote file share server.
DATASET_NAME: str = 'qm9'

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
    """
    e.log('loading dataset from "xyz_bundle"...')
    
    if os.path.exists(e.SOURCE_PATH):
        e.log(f'the given source path exists @ {e.SOURCE_PATH}')
        source_file = os.path.basename(e.SOURCE_PATH)
        source_folder = os.path.dirname(e.SOURCE_PATH)
        
        df: pd.DataFrame = load_xyz_dataset(
            dataset_name=source_file,
            folder_path=source_folder,
            parser_cls=e.PARSER_CLASS,
            use_cache=True,
        )
    
    else:
        e.log('the given source path does not exist on the local system!')
        e.log(f'attempting to fetch from the remote file share as "{e.SOURCE_PATH}"...')
        
        df: pd.DataFrame = load_xyz_dataset(
            dataset_name=e.SOURCE_PATH,
            parser_cls=e.PARSER_CLASS,
            use_cache=True,
        )
        
    index_data_map: Dict[int, dict] = {}
    for idx, row in df.iterrows():
        index_data_map[idx] = row.to_dict()
    
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
