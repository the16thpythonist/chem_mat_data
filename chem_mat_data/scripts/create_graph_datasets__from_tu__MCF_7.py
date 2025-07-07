"""
This experiment downloads the MCF-7 dataset from the TU dataset collection which is a classification 
dataset related to breast cancer. 
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
SOURCE_URL: Optional[str] = 'https://www.chrsmrrs.com/graphkerneldatasets/MCF-7.zip'
# :param DATASET_NAME:
#       The name of the dataset which will be used as the final file name aka the unique string 
#       identifier on the remote file share server.
DATASET_NAME: str = 'MCF-7'
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
    8: 'Pt',
    9: 'Zn',
    10: 'Ni',
    11: 'Mn',
    12: 'P',
    13: 'I',
    14: 'Se',
    15: 'Sn',
    16: 'Fe',
    17: 'Pb',
    18: 'Si',
    19: 'Cr',
    20: 'Hg',
    21: 'As',
    22: 'B',
    23: 'Ga',
    24: 'Ti',
    25: 'Bi',
    26: 'K',
    27: 'Cu',
    28: 'Zr',
    29: 'Ir',
    30: 'Li',
    31: 'Pd',
    32: 'Au',
    33: 'W',
    34: 'Sb',
    35: 'Co',
    36: 'Mg',
    37: 'Ag',
    38: 'Rh',
    39: 'Ru',
    40: 'Cd',
    41: 'Er',
    42: 'V',
    43: 'Ac',
    44: 'Tl',
    45: 'Ge',
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
