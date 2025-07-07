"""
This experiment downloads the OVCAR-8 dataset from the TU dataset collection which is a classification 
dataset related to ovarian cancer. 
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
SOURCE_URL: Optional[str] = 'https://www.chrsmrrs.com/graphkerneldatasets/OVCAR-8.zip'
# :param DATASET_NAME:
#       The name of the dataset which will be used as the final file name aka the unique string 
#       identifier on the remote file share server.
DATASET_NAME: str = 'OVCAR-8'
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
    6: 'P',
    7: 'F',
    8: 'Na',
    9: 'Sn',
    10: 'Pt',
    11: 'Ni',
    12: 'Zn',
    13: 'Mn',
    14: 'Cu',
    15: 'Co',
    16: 'Se',
    17: 'Au',
    18: 'K',
    19: 'Pb',
    20: 'I',
    21: 'Si',
    22: 'La',
    23: 'Ce',
    24: 'Nd',
    25: 'Fe',
    26: 'Cr',
    27: 'As',
    28: 'B',
    29: 'Ti',
    30: 'Ac',
    31: 'Bi',
    32: 'Y',
    33: 'Eu',
    34: 'Tl',
    35: 'Zr',
    36: 'Hf',
    37: 'In',
    38: 'Ga',
    39: 'Sm',
    40: 'Gd',
    41: 'Dy',
    42: 'U',
    43: 'Pd',
    44: 'Ir',
    45: 'Ge',
    46: 'Re',
    47: 'Li',
    48: 'Sb',
    49: 'W',
    50: 'Hg',
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
    61: 'Er',
    62: 'V',
    63: 'Te',
    64: 'Al',
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