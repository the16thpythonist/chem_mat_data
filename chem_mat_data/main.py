import os
import gzip
import shutil
import tempfile
from typing import Dict, Optional

import pandas as pd

from chem_mat_data.config import Config
from chem_mat_data.web import AbstractFileShare
from chem_mat_data.web import NextcloudFileShare
from chem_mat_data.web import construct_file_share
from chem_mat_data.data import load_graphs
from chem_mat_data._typing import GraphDict
from typing import Union
from typing import List

FILE_SHARE_CLASS_MAP: Dict[str, type] = {
    'nextcloud': NextcloudFileShare,
}


def get_file_share(config: Config) -> AbstractFileShare:
    """
    This function will return a concrete file share object that can be used to interact with a remote file 
    share server. The type of the file share object that is returned depends on the file share type that is 
    configured in the given ``config`` object and by extension in the config TOML file.
    
    :param config: The Config object that contains the configuration data for the application.
    
    :returns: An instance of a AbstractFileShare subclass that can be used to interact with a remote 
        file share server of dynamically determined type.
    """

    # This function will construct the concrete file share object based on the string identifier of the 
    # file share type that is configured in the given config file.
    file_share_type = config.get_fileshare_type()    
    file_share = construct_file_share(
        file_share_type=config.get_fileshare_type(),
        file_share_url=config.get_fileshare_url(),
        file_share_kwargs=config.get_fileshare_parameters(file_share_type),
    )
    
    return file_share


def ensure_dataset(dataset_name: str,
                   extension: str = 'mpack',
                   config: Union[None, Config] = None,
                   file_share: AbstractFileShare = None,
                   use_cache: bool = True,
                   folder_path = tempfile.gettempdir(),
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
        stored. The default is the system's temporary directory.
    
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
            
        # 04.11.24
        # Before attempting to fetch the dataset from the remote server, we first try to see if the 
        # dataset is in the local file system cache.
        if config.cache.contains_dataset(dataset_name, extension) and use_cache:
            
            # If the dataset does exist in the dataset cache, we can simply retrieve it from there 
            # and return the path to dataset file that was copied into the given destination folder_path
            file_path = config.cache.retrieve_dataset(
                name=dataset_name, 
                typ=extension, 
                dest_path=folder_path
            )
            return file_path
        
        else:
        
            # 08.07.24
            # There is now also the option to pass a custom file share object to this function that 
            # should be used to download the dataset.
            if not file_share:
                
                # This function will use the information in the config file to construct the concrete
                # file share instance that should be used to interact with the remote file share server 
                # that is configured in the config file.
                file_share = get_file_share(config=config)
                
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
            except Exception:
                file_path = file_share.download_file(file_name, folder_path=folder_path)
                
            # 04.11.24
            # After the dataset has been downloaded we can then also add the dataset to the cache so that it 
            # does not have to be downloaded the next time.
            if use_cache:
                config.cache.add_dataset(
                    name=dataset_name,
                    typ=extension,
                    path=file_path,
                    metadata=file_share['datasets'][dataset_name],
                )
                
            return file_path


def load_dataset_metadata(dataset_name: str,
                          config: Config = Config(),
                          ) -> Dict:
    file_share: AbstractFileShare = get_file_share(config)
    metadata: Dict = file_share.fetch_metadata()
    return metadata['datasets'][dataset_name]


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
                       config: Optional[Config] = None,
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
        folder_path=folder_path,
        config=config,
    )
    
    # Then we simply have to load the graphs from that file and return them
    graphs = load_graphs(file_path)
    return graphs


def pyg_from_graph(graph: GraphDict) -> 'Data':    # noqa
    """
    Given a graph dict representation ``graph``, this function will convert it into a PyTorch Geometric "Data" 
    object which can then be used to train a PyG graph neural network model directly.
    
    :param graph: A graph dict representation of a dataset's molecule.
    
    :returns: A PyG Data instance.
    """
    try:
        
        import torch                        # noqa
        import torch_geometric.data         # noqa
        
        data = torch_geometric.data.Data(
            # standard attributes: These are part of every graph and have to be given
            x=torch.tensor(graph['node_attributes'], dtype=torch.float),
            edge_attr=torch.tensor(graph['edge_attributes'], dtype=torch.float),
            edge_index=torch.tensor(graph['edge_indices'].T, dtype=torch.long),
        )
        
        # if graph_labels are present, we also add them to the data object
        if 'graph_labels' in graph:
            data.y = torch.tensor(graph['graph_labels'], dtype=torch.float)
        
        # optional attributes: These can optionally be part of the graph and are therefore
        # dynamically attached if they are present in the graph dict.
        if 'node_coordinates' in graph:
            data.pos = torch.tensor(graph['node_coordinates'], dtype=torch.float)
        
        return data
        
    except ImportError:
        raise ImportError('It seems like you are trying to convert GraphDicts to torch_geometric.data.Data objects. '
                          'However, it seems like either TORCH or TORCH_GEOMETRIC are not properly installed in your '
                          'current environment and could not be imported. Please make sure to install them '
                          'properly first!')


def pyg_data_list_from_graphs(graphs: List[dict]) -> List['Data']:    # noqa
    """
    Given a list ``graphs`` of graph dict representations, this function will convert them into 
    a list of pytorch geometric "Data" objects which can then be used to train a PyG graph neural 
    network model directly.
    
    :param graphs: A list of graph dict representations of a dataset's molecules.
    
    :returns: A list of PyG Data instances.
    """
    data_list = []
    for graph in graphs:
        # pyg_from_graph already implements the conversion of a single graph dict to a PyG Data object so 
        # we can just reuse that function here.
        data = pyg_from_graph(graph)
        data_list.append(data)
        
    return data_list
        
        
def jraph_from_graph(graph: GraphDict) -> 'GraphsTuple':  # noqa
    """
    Given a graph dict representation ``graph``, this function will convert it into a Jraph "GraphsTuple"
    object which can then be used to train a Jraph graph neural network model directly.
    
    :param graph: A graph dict representation of a dataset's molecule.
    
    :returns: A Jraph GraphsTuple instance.
    """
    try:
        
        import jax.numpy as jnp
        import jraph
        
        graph_tuple = jraph.GraphsTuple(
            nodes=jnp.array(graph['node_attributes']),
            edges=jnp.array(graph['edge_attributes']),
            senders=jnp.array(graph['edge_indices'][:, 0]),
            receivers=jnp.array(graph['edge_indices'][:, 1]), 
            n_node=jnp.array([len(graph['node_indices'])]),
            n_edge=jnp.array([len(graph['edge_indices'])]),
            globals=None,
        )
        
        return graph_tuple
    
    except ImportError:
        raise ImportError('It seems like you are trying to convert GraphDicts to jraph.GraphTuples. '
                          'However, it seems like either JAY or JRAPH are not properly installed in your '
                          'current environment and could not be imported. Please make sure to install them '
                          'properly first!')
    

def jraph_implicit_batch_from_graphs(graphs: list[GraphDict]) -> list['GraphsTuple']:  # noqa
    """
    This function will convert a list of graph dict representations ``graphs`` into a list of Jraph
    "GraphsTuple" objects which can then be used to train a Jraph graph neural network model directly.
    
    :param graphs: A list of graph dict representations of a dataset's molecules.
    
    :returns: A list of Jraph GraphsTuple instances.
    """
    try: 
        import jraph
        
        graph_tuples = []
        for graph in graphs:
            graph_tuple = jraph_from_graph(graph)
            graph_tuples.append(graph_tuple)
        
        return jraph.batch(graph_tuples)
    
    except ImportError:
        raise ImportError('It seems like you are trying to convert GraphDicts to jraph.GraphTuples. '
                          'However, it seems like either JAY or JRAPH are not properly installed in your '
                          'current environment and could not be imported. Please make sure to install them '
                          'properly first!')

        
    

# TODO: Implement for KGCNN as well!