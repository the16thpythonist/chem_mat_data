"""
This module contains all the additional functionality that is needed for the unittesting.
"""
import random
import numpy as np

from chem_mat_data._typing import GraphDict


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