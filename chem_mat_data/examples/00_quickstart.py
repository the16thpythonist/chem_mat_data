from pandas import DataFrame
from rich import print
from chem_mat_data import load_smiles_dataset, load_graph_dataset
from chem_mat_data import pyg_data_list_from_graphs

# ~ LOADING RAW DATASETS
# Datasets are generally available in the "raw" format and the "processed" format.
# The most common raw format is simply as a CSV file containing the SMILES string 
# representations of hte various molecules in one column and the corresponding 
# target value annotations in another column.
# The "load_smiles_dataset" function can be used to load such a dataset as a 
# pandas data frame.

df: DataFrame = load_smiles_dataset('bace')
print('dataset:\n', df.head())

# ~ LOADING PROCESSED DATASETS
# Alternatively, datasets are available in the "processed" format as well. In this 
# format, every molecule is already represented as a graph structure where nodes 
# represent the atoms and edges represent the bonds.
# A structure like this is especially suited for deep learning applications such as 
# graph neural networks (GNNs). The "load_graph_dataset" function can be used to 
# directly load such graph representations. 

graphs: list[dict] = load_graph_dataset('bace')
# In practice, these graphs are represented as dictionaries with various keys 
# whose values are numpy arrays that represent different aspects of the graph.
print('graph keys:', list(graphs[0].keys()))

# ~ DEEP LEARNING INTEGRATION
# The package also provides convenient functions to easily convert these graphs 
# dictionaries into a PyTorch Geometric (PyG) DataLoader instance which can then 
# be directly employed to train a GNN model!

import torch_geometric.loader
data_list = pyg_data_list_from_graphs(graphs)
data_loader = torch_geometric.loader.DataLoader(
    data_list, 
    batch_size=32, 
    shuffle=False
)