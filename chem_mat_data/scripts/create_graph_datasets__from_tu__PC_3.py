"""
This experiment downloads the PC-3 dataset from the TU dataset collection which is a classification 
dataset related to prostate cancer. 
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
SOURCE_URL: Optional[str] = 'https://www.chrsmrrs.com/graphkerneldatasets/PC-3.zip'
# :param DATASET_NAME:
#       The name of the dataset which will be used as the final file name aka the unique string 
#       identifier on the remote file share server.
DATASET_NAME: str = 'PC-3'
# :param NODE_LABEL_MAP:
#       This is a dictionary that maps the node label classes to the integer values corresponding to 
#       the actual atomic numbers. This mapping will have to be extracted from the README file of 
#       the TU dataset.
NODE_LABEL_MAP: Dict[int, str] = {
    0: 'O',
    1: 'N',
    2: 'C',
    3: 'Cl',
    4: 'S',
    5: 'F',
    6: 'Na',
    7: 'Pt',
    8: 'Zn',
    9: 'Mn',
    10: 'Ni',
    11: 'Br',
    12: 'P',
    13: 'Se',
    14: 'Sn',
    15: 'Pb',
    16: 'I',
    17: 'Si',
    18: 'Cr',
    19: 'As',
    20: 'B',
    21: 'Ac',
    22: 'Bi',
    23: 'K',
    24: 'Cu',
    25: 'Zr',
    26: 'Ir',
    27: 'Ge',
    28: 'Li',
    29: 'Pd',
    30: 'Au',
    31: 'Ga',
    32: 'Fe',
    33: 'Sb',
    34: 'W',
    35: 'Co',
    36: 'Ti',
    37: 'Mg',
    38: 'Hg',
    39: 'Ag',
    40: 'Ru',
    41: 'Cd',
    42: 'Er',
    43: 'V',
    44: 'Tl',
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