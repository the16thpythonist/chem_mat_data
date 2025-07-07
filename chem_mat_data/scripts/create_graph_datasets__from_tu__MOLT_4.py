"""
This experiment downloads the MOLT-4 dataset from the TU dataset collection which is a classification 
dataset related to  Leukemia. 
The dataset consists of molecular graphs and binary class labels.
"""
from typing import Optional, Dict

import rdkit.Chem as Chem

from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

# :param SOURCE_URL:
#       As a fallback option, if no local dataset path is given, this parameter can define the URL at 
#       which the dataset can be downloaded from. This should be a valid URL that points to the TU dataset 
#       format. If given, the experiment will attempt to download and extract before loading the dataset.
SOURCE_URL: Optional[str] = 'https://www.chrsmrrs.com/graphkerneldatasets/MOLT-4.zip'
# :param DATASET_NAME:
#       The name of the dataset which will be used as the final file name aka the unique string 
#       identifier on the remote file share server.
DATASET_NAME: str = 'MOLT-4'
# :param NODE_LABEL_MAP:
#       This is a dictionary that maps the node label classes to the integer values corresponding to 
#       the actual atomic numbers. This mapping will have to be extracted from the README file of 
#       the TU dataset.
NODE_LABEL_MAP: Dict[int, str] = {
    0: 'O',
    1: 'N',
    2: 'C',
    3: 'Br',
    4: 'S',
    5: 'Cl',
    6: 'F',
    7: 'Na',
    8: 'Sn',
    9: 'Pt',
    10: 'Ni',
    11: 'Zn',
    12: 'Mn',
    13: 'P',
    14: 'I',
    15: 'Cu',
    16: 'Co',
    17: 'Se',
    18: 'Au',
    19: 'Ge',
    20: 'Fe',
    21: 'Pb',
    22: 'Si',
    23: 'B',
    24: 'Nd',
    25: 'In',
    26: 'Bi',
    27: 'Er',
    28: 'Hg',
    29: 'As',
    30: 'Ga',
    31: 'Ti',
    32: 'Ac',
    33: 'Y',
    34: 'Eu',
    35: 'Tl',
    36: 'Zr',
    37: 'Hf',
    38: 'K',
    39: 'La',
    40: 'Ce',
    41: 'Sm',
    42: 'Gd',
    43: 'Dy',
    44: 'U',
    45: 'Pd',
    46: 'Ir',
    47: 'Re',
    48: 'Li',
    49: 'Sb',
    50: 'W',
    51: 'Mg',
    52: 'Ru',
    53: 'Rh',
    54: 'Os',
    55: 'Th',
    56: 'Mo',
    57: 'Nb',
    58: 'Ta',
    59: 'Ag',
    60: 'Cd',
    61: 'V',
    62: 'Te',
    63: 'Al',
}
# :param EDGE_LABEL_MAP:
#       This is a dictionary that maps the edge label classes to the integer values corresponding to
#       the actual bond types. This mapping will have to be extracted from the README file of
#       the TU dataset.
EDGE_LABEL_MAP: Dict[str, int] = {
    0: Chem.BondType.SINGLE,
    1: Chem.BondType.DOUBLE,
    2: Chem.BondType.TRIPLE,
}
# :param GRAPH_LABEL_MAP:
#       This is a dictionary that maps the graph label classes to the integer values corresponding to
#       the actual graph labels. This mapping will have to be extracted from the README file of
#       the TU dataset.
GRAPH_LABEL_MAP: Dict[int, list] = {
    1: [0, 1],
    0: [1, 0],
}

experiment = Experiment.extend(
    'create_graph_datasets__from_tu.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)

experiment.run_if_main()
