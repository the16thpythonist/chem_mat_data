"""
This module tests the unittest utilities that in the "testing" module
"""
import os
import numpy as np

from chem_mat_data.config import Config
from chem_mat_data.cache import Cache
from chem_mat_data.testing import create_mock_graph
from chem_mat_data.testing import sample_mock_graphs
from chem_mat_data.testing import ConfigIsolation


def test_create_mock_graph_basically_works():
    """
    ``create_mock_graph`` should create a valid graph dict representation.
    """
    
    num_nodes = 20
    num_node_attributes = 10
    num_edge_attributes = 8
    num_graph_labels = 1
    
    graph = create_mock_graph(
        num_nodes=num_nodes,
        num_node_attributes=num_node_attributes,
        num_edge_attributes=num_edge_attributes,
        num_graph_labels=num_graph_labels
    )
    assert isinstance(graph['node_indices'], np.ndarray)
    assert graph['node_indices'].shape == (num_nodes, )
    
    assert isinstance(graph['node_attributes'], np.ndarray)
    assert graph['node_attributes'].shape == (num_nodes, num_node_attributes)
    
    assert isinstance(graph['edge_indices'], np.ndarray)
    assert graph['edge_indices'].shape == (num_nodes, 2)
    
    assert isinstance(graph['edge_attributes'], np.ndarray)
    assert graph['edge_attributes'].shape == (num_nodes, num_edge_attributes)
    
    assert isinstance(graph['node_indices'], np.ndarray)
    assert graph['graph_labels'].shape == (num_graph_labels, )
    
    
def test_sample_mock_graphs_basically_works():
    """
    ``sample_mock_graphs`` should return a number of mock graphs with different numbers of 
    nodes randomly sampled in a certain range.
    """
    num_graphs = 50
    num_nodes_min = 5
    num_nodes_max = 30
    
    graphs = sample_mock_graphs(
        k=num_graphs,
        num_nodes_range=(num_nodes_min, num_nodes_max)
    )
    
    assert isinstance(graphs, list)
    assert len(graphs) == num_graphs
    
    for graph in graphs:
        assert isinstance(graph, dict)
        num_nodes = len(graph['node_indices'])
        assert num_nodes_min <= num_nodes <= num_nodes_max
        
        
def test_config_isolation_basically_works():
    """
    The ConfigIsolation context manager should create a temporary isolated config and cache folder and 
    temporarily overwrite the values of the Config singleton such that there are no side effects during 
    a test case that relies on the Config object state.
    """
    config = Config()
    cache_path = config.cache_path
    config_path = config.config_path
    
    with ConfigIsolation() as iso:
        
        assert config.cache_path != cache_path
        assert config.config_path != config_path
        
        # Also, the Cache instance should exist and should point to the same cache folder
        assert isinstance(config.cache, Cache)
        assert config.cache.path == iso.cache_path
        
        # Also, the config file should exist in the newly created temp folder
        assert os.path.exists(config.config_file_path)
        
    assert config.cache_path == cache_path
    assert config.config_path == config_path