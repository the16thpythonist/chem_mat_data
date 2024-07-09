import os
import warnings
import gzip
import shutil

import pandas as pd
import numpy as np

from chem_mat_data.config import Config
from chem_mat_data.web import NextcloudFileShare
from chem_mat_data.data import load_graphs, save_graphs
from typing import Union
from typing import List

def ensure_dataset(dataset_name: str,
                   extension: str = 'mpack',
                   config: Union[None, Config] = None,
                   file_share: Union[None, NextcloudFileShare] = None,
                   folder_path = os.getcwd(),
                   ) -> str:
    """
    Given the string identifier ``dataset_name`` of a dataset, this function will make sure that 
    the dataset exists on the local system and return the absolute path to the dataset file.
    
    If the dataset already exists in the given ``folder_path``, then that path will be returned.
    Otherwise, the dataset will be downloaded from the remote file share server and saved to the 
    local file system.
    
    :param dataset_name: The unique string identifier of the dataset.
    :param extension: The file extension of the dataset file. Each dataset is available in 
        different file formats, such as a csv or the processed mpack files. This string value 
        should determine the desired extension of the dataset file.
    :param config: An optional Config object which contains the object.
    :param folder_path: The absolute path to the folder where the dataset files should be 
        stored. The default is the current working directory.
    
    :returns: The absolute string path to the dataset file.
    """
    file_name = f'{dataset_name}.{extension}'
    path = os.path.join(folder_path, file_name)
    
    # The easiest case is if the file already exists. In that case we can simply return the path
    # to the file and dont have to interact with the server at all.
    if os.path.exists(path):
        return path
    
    # However, if the file does not exist already, we need to attempt to fetch it from the remote 
    # file share server and download it to the local file system.
    else:
        if not config:
            config = Config()
        
        # 08.07.24
        # There is now also the option to pass a custom file share object to this function that 
        # should be used to download the dataset.
        if not file_share:
            file_share = NextcloudFileShare(url=config.get_fileshare_url())
            
        # This function will download the main metadata yml file from the server to populate the 
        # itnernal metadata dict with all the information about the datasets that are available 
        # on the server.
        file_share.fetch_metadata()
        
        if dataset_name not in file_share['datasets']:
            raise FileNotFoundError(f'The dataset {file_name} could not be found on the server!')
        
        # 04.07.24
        # In the first instance we are going to try and download the compressed (gzip - gz) version 
        # of the dataset because that is usually at least 10x smaller and should therefore be a lot 
        # faster to download and only if that doesn't exist or fails due to some other issue we 
        # attempt to download the uncompressed version.
        try:
            file_name_compressed = f'{file_name}.gz'
            file_path_compressed = file_share.download_file(file_name_compressed, folder_path=folder_path)
            
            # Then we can decompress the file using the gzip module. This may take a while.
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, mode='wb') as file:
                with gzip.open(file_path_compressed, mode='rb') as compressed_file:
                    shutil.copyfileobj(compressed_file, file)
        
        # Otherwise we try to download the file without the compression
        except Exception as exc:
            file_path = file_share.download_file(file_name, folder_path=folder_path)
            
        return file_path



def load_smiles_dataset(dataset_name: str, 
                        folder_path: str = os.getcwd()
                        ) -> pd.DataFrame:
    """
    Loads the SMILES dataset with the unique string identifier ``dataset_name`` and returns it 
    as a pandas data frame which contains at least one column with the SMILES strings of the 
    molecules and additional columns with the target value annotations.
    
    :param dataset_name: The unique string identifier of the dataset.
    :param folder_path: The absolute path to the folder where the dataset files should be 
        stored. The default is the current working directory.
    
    :returns: A data frame containing the SMILES strings of the dataset molecules and 
        the target value annotations.
    """
    # The "ensure_dataset" function is a utility function which will make sure that the dataset 
    # in question just generally exists. To do this, the function first checks if the dataset 
    # file already eixsts in the given folder. If that is not the case it will attempt to download 
    # the dataset from the remote file share server. Either way, the function WILL return a path 
    # towards the requested dataset file in the end.
    file_path = ensure_dataset(
        dataset_name, 
        extension='csv', 
        folder_path=folder_path
    )
    
    # Then we simply have to load that csv file into a pandas DataFrame and return it.
    df = pd.read_csv(file_path)
    return df


def load_graph_dataset(dataset_name: str,
                       folder_path: str = os.getcwd(),
                       ) -> List[dict]:
    """
    Loads the graph dict representations for the dataset with the unique string identifier ``dataset_name`` 
    and returns them as a list of dictionaries. Each dictionary represents a single graph and contains
    the node attributes, edge attributes, edge indices, and optionally node coordinates.
    
    :param dataset_name: The unique string identifier of the dataset.
    :param folder_path: The absolute path to the folder where the dataset files should be
        stored. The default is the current working directory.
        
    :returns: A list of dictionaries where each dictionary represents a single graph.
    """
    # The "ensure_dataset" function is a utility function which will make sure that the dataset 
    # in question just generally exists. To do this, the function first checks if the dataset 
    # file already eixsts in the given folder. If that is not the case it will attempt to download 
    # the dataset from the remote file share server. Either way, the function WILL return a path 
    # towards the requested dataset file in the end.
    file_path = ensure_dataset(
        dataset_name, 
        extension='mpack', 
        folder_path=folder_path
    )
    
    # Then we simply have to load the graphs from that file and return them
    graphs = load_graphs(file_path)
    return graphs


def pyg_data_list_from_graphs(graphs: List[dict]) -> List['Data']:
    """
    Given a list ``graphs`` of graph dict representations, this function will convert them into 
    a list of pytorch geometric "Data" objects which can then be used to train a PyG graph neural 
    network model directly.
    
    :param graphs: A list of graph dict representations of a dataset's molecules.
    
    :returns: A list of PyG Data instances.
    """
    try:
        
        import torch
        import torch_geometric.data 
        
        data_list = []
        for graph in graphs:
            data = torch_geometric.data.Data(
                # standard attributes: These are part of every graph and have to be given
                x=torch.tensor(graph['node_attributes'], dtype=torch.float),
                edge_attr=torch.tensor(graph['edge_attributes'], dtype=torch.float),
                edge_index=torch.tensor(graph['edge_indices'].T, dtype=torch.long),
            )
            
            # optional attributes: These can optionally be part of the graph and are therefore
            # dynamically attached if they are present in the graph dict.
            if 'node_coordinates' in graph:
                data.pos = torch.tensor(graph['node_coordinates'], dtype=torch.float)
            
            data_list.append(data)
            
        return data_list
        
    except ImportError:
        raise ImportError(
            'The PyTorch and PyTorch Geometric packages have not been found in your environment.'
            'To use this functionality please install them with "pip install torch torch-geometric"'
        )
    

# TODO: Implement for KGCNN as well!
