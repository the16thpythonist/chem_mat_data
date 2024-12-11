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

# == PROCESSING PARAMETERS ==
# These parameters can be used to configure the processing functionality of the script 
# itself. This includes for example whether or not the molecule coordinates should be 
# created by RDKIT as well or not. Also the compression of the final dataset file can
# be configured here.

# :param DATASET_NAME:
#       This string determines the name of the message pack dataset file that is then 
#       stored into the "results" folder of the experiment as the result of the 
#       processing process. The corresponding file extensions will be added 
#       automatically.
DATASET_NAME: str = 'qm9'

experiment = Experiment.extend(
    'create_graph_datasets__from_xyz.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)

experiment.run_if_main()