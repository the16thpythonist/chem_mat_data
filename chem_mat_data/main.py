import os
import gzip
import shutil
import zipfile
import tempfile
from typing import Dict, Optional
from collections import defaultdict

import pandas as pd

from chem_mat_data.config import Config
from chem_mat_data.web import AbstractFileShare
from chem_mat_data.web import NextcloudFileShare
from chem_mat_data.web import construct_file_share
from chem_mat_data.data import load_graphs
from chem_mat_data.data import load_xyz_as_mol
from chem_mat_data.data import DefaultXyzParser
from chem_mat_data.data import QM9XyzParser
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
    if dataset_name.endswith(f'.{extension}'):
        file_name = dataset_name
    else:
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
            
            file_path = os.path.join(folder_path, file_name)
            
            # 04.07.24
            # In the first instance we are going to try and download the compressed (gzip - gz) version 
            # of the dataset because that is usually at least 10x smaller and should therefore be a lot 
            # faster to download and only if that doesn't exist or fails due to some other issue we 
            # attempt to download the uncompressed version.
            try:
                file_name_compressed = f'{file_name}.gz'
                file_path_compressed = file_share.download_file(file_name_compressed, folder_path=folder_path)
                
                # Then we can decompress the file using the gzip module. This may take a while.
                with open(file_path, mode='wb') as file:
                    with gzip.open(file_path_compressed, mode='rb') as compressed_file:
                        shutil.copyfileobj(compressed_file, file)
            
            # Otherwise we try to download the file without the compression
            except Exception as exc:
                #print(exc)
                pass
                
            # 20.11.24
            # In the second instance we are going to try and download the 'zip' compressed version of the 
            # dataset. Some datasets will be available either .gz (files only) or .zip (folders) and we
            # need to be able to handle both cases.
            try:
                file_name_compressed = f'{file_name}.zip'
                file_path_compressed = file_share.download_file(file_name_compressed, folder_path=folder_path)
                
                # Unpack the downloaded zip archive to the "file_path" file path
                folder_path = os.path.dirname(file_path)
                with zipfile.ZipFile(file_path_compressed, 'r') as zip_ref:
                    zip_ref.extractall(folder_path)
                                    
            except Exception as exc:
                #print(exc)
                pass
            
            # Only in the last instance, if the "file_path" does not already exist, we try to download the 
            # uncompressed version of the dataset file.
            if not os.path.exists(file_path):
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
    """
    Load and return the metadata dict of the dataset with the unique string identifier ``dataset_name``.
    Uses the metadata information from the default file share configuration in the given ``config`` object.
    
    :param dataset_name: The unique string identifier of the dataset.
    
    :returns: The metadata dict of the dataset.
    """
    file_share: AbstractFileShare = get_file_share(config)
    metadata: Dict = file_share.fetch_metadata()
    return metadata['datasets'][dataset_name]


def load_xyz_dataset(dataset_name: str,
                     folder_path: str = tempfile.gettempdir(),
                     config: Optional[Config] = None,
                     use_cache: bool = True,
                     parser_cls: Union[type, str] = 'default',
                     ) -> pd.DataFrame:
    """
    Given the string ``dataset_name`` of an existing dataset in the format of an "xyz_bundle", this method 
    will load that dataset either from the remote file share server or from the local file cache. A xyz_bundle 
    dataset is a dataset that also includes information about the atom positions in this case in the form of 
    .xyz files.
    
    :param dataset_name: THe
    :param folder_path: The absolute path of where the dataset should be downloaded to. Default is the 
        system's temp folder.
    :param config: Optional overwrite for the Config instance to be used to retrieve the information about 
        the remote file share server for example.
    :param use_cache: Boolean flag of whether or not to use the local file system cache or force a 
        re-download.
    :param parser_cls: The XzyParser class to be used to load the xyz files of the dataset. There exist 
        different parser implementations for different flavors of xyz file formats.
        
    :returns: A pandas dataframe which contains the two columns "id" and "mol". The id column is a unique 
        string ID for the element and the "mol" column contains a Chem.Mol object that represents the molecule 
        element - including the atom positions in the form of a Conformer. Depending on the dataset, the 
        dataframe may contain additional columns.
    """
    # The "ensure_dataset" function is a utility function which will make sure that the dataset
    # in question just generally exists. To do this, the function first checks if the dataset
    # file already eixsts in the given folder. If that is not the case it will attempt to download
    # the dataset from the remote file share server. Either way, the function WILL return a path
    # towards the requested dataset file in the end.
    
    # In the case of a xyz dataset we know that the result will be a folder path and not a file path!
    folder_path = ensure_dataset(
        dataset_name=dataset_name,
        folder_path=folder_path,
        extension='xyz_bundle',
        config=config,
        use_cache=use_cache,
    )    
    
    # This folder now consists of the following contents:
    # - Multiple .xyz files with the format "{id}.xyz" which contain the actual positional information 
    #   of the atoms in the molecules.
    # - (OPTIONAL) A "meta.csv" file which contains the metadata of the dataset in tabular format. This includes 
    #   most importantly the numeric ID of all the elements and the associated target value annotations.
    
    # We first check if a metadata csv file exists in the folder and if it does we load the metadata contents as 
    # "id_data_map" which is a dict whose keys are the unique string keys of the elements and the values are the 
    # dict elements.
    # TODO: We can more robustly search for a CSV file here...
    meta_path = os.path.join(folder_path, 'meta.csv')
    id_data_map: Dict[str, dict] = defaultdict(dict)
    if os.path.exists(meta_path):
        data_list: List[dict] = pd.read_csv(meta_path).to_dict(orient='records')
        for data in data_list:
            element_id = data['id']
            id_data_map[element_id] = data
    
    # Iterate over the xyz files in the folder and add each one as an entry in the dataframe
    for xyz_file in os.listdir(folder_path):
        if xyz_file.endswith('.xyz'):
            xyz_path = os.path.join(folder_path, xyz_file)
            element_id = os.path.splitext(xyz_file)[0]
            # If possible, we convert the ID into an integer otherwise we leave it as a string
            try:
                element_id = int(element_id)
            except ValueError:
                pass
            
            try:
                mol, info = load_xyz_as_mol(xyz_path, parser_cls=parser_cls)
                data = {
                    'id': element_id, 
                    'mol': mol,
                    # We'll also add all of the information that is included in the additional info dict
                    # for each of the elements as separate columns of the data frame.
                    **info
                }
                id_data_map[element_id].update(data)
            except:
                print("hello")
    
    # Finally, in the end we create a data frame object from the list of data dicts that 
    # we've assembled from the metadata csv and the parsing of the xyz files.
    df = pd.DataFrame([{
        'id': element_id,
        **id_data_map[element_id]
    } for element_id, data in id_data_map.items()])
    
    return df


def load_smiles_dataset(dataset_name: str, 
                        folder_path: str = tempfile.gettempdir(),
                        config: Optional[Config] = None,
                        use_cache: bool = True,
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
        folder_path=folder_path,
        config=config,
        use_cache=use_cache,
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