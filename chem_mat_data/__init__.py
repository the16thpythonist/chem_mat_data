"""
As the __init__ module of the package, this determines what functions should be globally 
importable from the package name alone.
"""
# Mainly we want all the important functionality from the "main" module to be easily accessible
from chem_mat_data.main import ensure_dataset
from chem_mat_data.main import load_smiles_dataset
from chem_mat_data.main import load_graph_dataset
from chem_mat_data.main import pyg_data_list_from_graphs
# Also the config and the web related functionality
from chem_mat_data.config import Config
from chem_mat_data.web import NextcloudFileShare