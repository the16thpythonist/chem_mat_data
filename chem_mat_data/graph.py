from collections import defaultdict

import rdkit.Chem as Chem
import numpy as np

import chem_mat_data._typing as tc
from typing import Dict, List
from scipy.spatial.distance import pdist, squareform


# == UTILITY ==


def to_graph_dict(data: dict) -> tc.GraphDict:
    """
    Ensures that the given dictionary ``data`` passes as a graph dict representation 
    in the sense that this function will convert all the list properties of the given dict 
    into numpy arrays - as it is required by the GraphDict format.
    
    :param data: The dictionary to be converted.
    
    :returns: The orgininal dict, but with all list elements converted into numpy arrays
    """
    for key, value in data.items():
        if isinstance(value, list):
            data[key] = np.array(value)
    
    return data
    
    
def graph_adjacency_list(graph: tc.GraphDict) -> Dict[int, List]:
    """
    Creates an adjacency list / dict from the given GraphDict ``graph``. Returns a dictionary 
    where the keys are the node indices of the given graph and the values are again lists of 
    node indices - containing exactly those node indices which the corresponding node is 
    directly connected to.
    
    :param graph: The graph for which the adjacency list should be created.
    
    :returns: The dict structure which maps "node index -> list of connected node indices"
    """
    adjacency_list: dict = defaultdict(set)
    for i, j in graph['edge_indices']:
        adjacency_list[i].append(j)
    
    return adjacency_list


def _visitation_dfs(node: int, 
                    adjacency_list: dict, 
                    visited: np.ndarray
                    ) -> None:
    """
    Helper function which performs depth first search for the given node and adjacency list.
    """
    visited[node] = 1
    for neighbor in adjacency_list[node]:
        if not visited[neighbor]:
            _visitation_dfs(neighbor, adjacency_list, visited)

    
def is_graph_connected(graph: tc.GraphDict) -> None:
    """
    This function returns the boolean property of whether the ``graph`` is connected or not. 
    This means that from every node of the graph it should be possible to reach every 
    other node of the graph in a finite number of steps.
    
    :param graph: The GraphDict to be checked.
    
    :returns: boolean value
    """
    adjacency_list = graph_adjacency_list(graph)
    visited = np.zeros_like(graph['node_indices'])
    _visitation_dfs(0, adjacency_list, visited)
    
    return (visited > 0).all()


def graph_threshold_edges(graph: tc.GraphDict, 
                          threshold: float,
                          ) -> tc.GraphDict:
    """
    Given a ``graph`` dict representation and a distance ``threshold`` this function will calculate all the 
    pairwise distances between the nodes of the graph and then create new edges by inserting new edges for all 
    nodes whose distance is smaller or equal to the given threshold. The function will also replace the previous 
    edge_attributes of the 
    """
    assert 'node_coordinates' in graph, (
        'graph must contain "node_coordinates" property to calculate distance-based edges. '
    )
    
    node_indices: np.ndarray = graph['node_indices']
    node_coordinates: np.ndarray = graph['node_coordinates']
    # Calculate pairwise distances between all nodes
    distances = squareform(pdist(node_coordinates))

    # Create new edge list based on the threshold
    new_edge_indices = []
    for i in range(len(node_indices)):
        for j in range(i + 1, len(node_indices)):
            if distances[i, j] <= threshold:
                new_edge_indices.append([i, j])

    graph['edge_indices'] = np.array(new_edge_indices, dtye=np.int32)
    graph['edge_attributes'] = np.ones((len(new_edge_indices), 1), dtype=np.float32)
    return graph

# == ASSERTIONS ==

def assert_graph_dict(graph: tc.GraphDict) -> None:
    """
    Wraps a series of assert statements which make sure that the given ``graph`` object is a 
    valid GraphDict representation.
    
    This will for example make sure that the dict contains the required properties, that the 
    values of those properties are proper numpy arrays and that those arrays have the 
    correct shape.
    """
    # The most basic requirement is that the given object is in fact a non-empty dictionary 
    assert isinstance(graph, dict), 'graph dict must be dictionary'
    assert len(graph) != 0, 'graph dict must not be empty'
    
    # Even though the GraphDict representation is somewhat flexible there are some attributes 
    # which are absolutely required and which we will test for here.
    # Additionally, all of those properties need to be numpy arrays or need to be convertible 
    # into numpy arrays.
    required_keys = [
        'node_indices',
        'node_attributes',
        'edge_indices',
        'edge_attributes',
    ]
    for key in required_keys:
        assert key in graph, f'graph dict must contain "{key}" property'
        assert isinstance(graph[key], np.ndarray), f'graph property "{key}" needs to be numpy array'
    
    # ~ specific checks
    
    # The number of nodes needs to be conistent for all node properties
    num_nodes = len(graph['node_indices'])
    for key, value in graph.items():
        if key.startswith('node'):
            assert isinstance(value, np.ndarray), f'node property "{key}" needs to be numpy array'
            assert value.shape[0] == num_nodes, f'node property "{key}" needs to match no nodes {num_nodes}'
    
    # The number of edges needs to be consistent for all edge properties
    num_edges = len(graph['edge_indices'])
    for key, value in graph.items():
        if key.startswith('edge'):
            assert isinstance(value, np.ndarray), f'edge property "{key}" needs to be numpy array'
            assert value.shape[0] == num_edges, f'edge property "{key}" needs to match no edges {num_edges}'
    
