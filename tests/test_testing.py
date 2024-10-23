"""
This module tests the unittest utilities that in the "testing" module
"""
import numpy as np

from chem_mat_data.testing import create_mock_graph
from chem_mat_data.testing import sample_mock_graphs


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