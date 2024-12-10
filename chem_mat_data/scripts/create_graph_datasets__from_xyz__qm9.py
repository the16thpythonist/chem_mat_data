import os
import rdkit.Chem as Chem
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
# :param DATASET_NAME:
#       The name of the dataset which will be used as the final file name aka the unique string 
#       identifier on the remote file share server.
DATASET_NAME: str = 'xyz'

import os
from typing import Dict

import numpy as np
import pandas as pd
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

from chem_mat_data.main import load_xyz_dataset
from chem_mat_data.utils import SCRIPTS_PATH
from chem_mat_data.data import AbstractXyzParser
from chem_mat_data.data import DefaultXyzParser
from chem_mat_data.data import QM9XyzParser

# == SOURCE PARAMETERS ==

# :param SOURCE_PATH:
#       This is the absolute string path of the dataset in xyz format. This can either be an archive or 
#       directly a folder which points to an "xyz bundle". This bundle has to ultimately contain a number 
#       of distinct .XYZ files where each file defines one element (molecule) in the target dataset.
#       Optionally, this xyz bundle may also contain a csv file with additional information about each 
#       of the elements in the dataset.
SOURCE_PATH: str = '/home/jonas/Downloads/qm9.xyz_bundle'
# :param PARSER_CLASS:
#       This has to define the class that is to be used for the parsing of the xyz files. The class that 
#       is defined here needs to be a valid subclass of the AbstractXyzParser interface. There are different 
#       flavors of xyz files for which different parser classes exist.
PARSER_CLASS: type = QM9XyzParser

experiment = Experiment.extend(
    'create_graph_datasets__from_xyz.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)

experiment.run_if_main()