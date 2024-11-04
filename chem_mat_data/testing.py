"""
This module contains all the additional functionality that is needed for the unittesting.
"""
import os
import tempfile
import random
from typing import Optional, Dict

import numpy as np

from chem_mat_data._typing import GraphDict
from chem_mat_data.config import Config


def create_mock_graph(num_nodes: int = 10,
                      num_node_attributes: int = 3,
                      num_edge_attributes: int = 2,
                      num_graph_labels: int = 1,
                      ) -> GraphDict:
    """
    Creates a mock graph dict representation with the given number of nodes ``num_nodes``. The graph is a
    single cycle that connects all the nodes via directed edges. The nodes have ``num_node_attributes``
    node features and the edges have ``num_edge_attributes`` edge features. The graph has ``num_graph_labels``
    graph labels.
    
    :returns: GraphDict
    """
    node_indices = np.arange(0, num_nodes).astype(int)
    node_attributes = np.random.random((num_nodes, num_node_attributes))
    
    edge_indices = [(i, (i + 1) % num_nodes) for i in range(num_nodes)]
    edge_indices = np.array(edge_indices).astype(int)
    edge_attributes = np.random.random((len(edge_indices), num_edge_attributes))
    
    graph_labels = np.random.random((num_graph_labels, ))
    
    return {
        'node_indices': node_indices,
        'node_attributes': node_attributes,
        'edge_indices': edge_indices,
        'edge_attributes': edge_attributes,
        'graph_labels': graph_labels
    }
    

def sample_mock_graphs(k: int,
                       num_nodes_range: tuple[int, int] = (10, 20),
                       num_node_attributes: int = 10,
                       num_edge_attributes: int = 5,
                       num_graph_labels: int = 2
                       ) -> list[GraphDict]:
    """
    Samples a number of ``k`` mock graphs with a random number of nodes in the given range 
    ``num_nodes_range``. 
    
    :returns: list of GraphDicts
    """
    graphs = []
    for _ in range(k):
        num_nodes = random.randint(*num_nodes_range)
        graph = create_mock_graph(
            num_nodes=num_nodes,
            num_node_attributes=num_node_attributes,
            num_edge_attributes=num_edge_attributes,
            num_graph_labels=num_graph_labels,
        )
        graphs.append(graph)
        
    return graphs


class ConfigIsolation:
    """
    Instances of this class act as context managers that can be used to isolate the state of the "Config" singleton to 
    avoid side effects between tests. The context manager creates a new temp dir and sets the cache and config paths
    of the config object to this temp dir on enter. On exit, the original cache and config paths are restored to 
    their original values.
    """
    def __init__(self,
                 config: Config = Config(),
                 config_data: Dict = {},
                 ):
        
        self.config = config
        self.config_data = config_data
        
        self.temp_dir = tempfile.TemporaryDirectory()
        
        self.path: Optional[str] = None
        self.cache_path: Optional[str] = None
        self.config_path: Optional[str] = None
    
        self.original_cache_path: Optional[str] = None
        self.original_config_path: Optional[str] = None
    
    def __enter__(self):
        
        # By entering the temp dir object we obtain the actual absolute path to the temporary directory.
        # The temporary folder is only created on enter.
        self.path = self.temp_dir.__enter__()
        
        # We obtain the current cache and config paths from the config object. It is important to store these 
        # so that we can later restore them on exit.
        self.original_cache_path = self.config.cache_path
        self.original_config_path = self.config.config_path
        
        # Then we create the cache and config folders inside the temporary directory.
        self.cache_path = os.path.join(self.path, 'cache')
        os.mkdir(self.cache_path)
        
        self.config_path = os.path.join(self.path, 'config')
        os.mkdir(self.config_path)
        
        # The "set" methods on the config object actually handle all the important things such as creating 
        # a new Cache wrapper object and creating a new config TOML file inside the config folder.
        self.config.set_cache_path(self.cache_path)
        self.config.set_config_path(self.config_path)
        
        # If there is some custom overwrites of the config data, we need to apply these updates to the 
        # config data dict.
        self.config.config_data.update(self.config_data)
        self.config.save()
        
        return self
    
    def __exit__(self, *args, **kwargs):
        
        # On exit we need to reset the cache and config paths to the original values.
        self.config.set_cache_path(self.original_cache_path)
        self.config.set_config_path(self.original_config_path)
        
        return self.temp_dir.__exit__(*args, **kwargs)