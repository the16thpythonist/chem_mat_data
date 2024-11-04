"""
This module contains the unittests that specifically test the bridges to the major graph 
neural network libraries such as ptorch geometric and jraph.

NOTE: The GNN libraries used here are not part of the packages main dependencies and have 
      to be installed separately.
"""
import time

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn.conv import GCNConv
from torch_geometric.nn.aggr import SumAggregation

import jax
import jraph

from chem_mat_data.testing import create_mock_graph
from chem_mat_data.testing import sample_mock_graphs
from chem_mat_data.main import pyg_from_graph
from chem_mat_data.main import pyg_data_list_from_graphs
from chem_mat_data.main import jraph_from_graph
from chem_mat_data.main import jraph_implicit_batch_from_graphs


# == TORCH/PYTORCH GEOMETRIC ==

class TestPytorchGeometric:
    
    def test_pyg_from_graph_basically_works(self):
        """
        ``pyg_from_graph`` should take a graph dict as input and turn it into a pyg Data object instance
        """
        num_nodes = 15
        num_node_attributes = 20
        num_edge_attributes = 10
        num_graph_labels = 2
        
        graph = create_mock_graph(
            num_nodes=num_nodes,
            num_node_attributes=num_node_attributes,
            num_edge_attributes=num_edge_attributes,
            num_graph_labels=num_graph_labels
        )
        data: Data = pyg_from_graph(graph)
        assert isinstance(data, Data)
        
        assert isinstance(data.x, torch.Tensor)
        assert data.x.shape == (num_nodes, num_node_attributes)
        
        assert isinstance(data.edge_attr, torch.Tensor)
        assert data.edge_attr.shape == (num_nodes, num_edge_attributes)
        
        assert isinstance(data.edge_index, torch.Tensor)
        assert data.edge_index.shape == (2, num_nodes)
        
        # additionally, since the mock graphs actually contain target labels, these target labels should 
        # also be converted to tensors and stored in the data object.
        assert isinstance(data.y, torch.Tensor)
        assert data.y.shape == (num_graph_labels, )
        
    def test_pyg_data_list_from_graphs_basically_works(self):
        """
        ``pyg_data_list_from_graphs`` should take a list of graph dicts as input and turn them into a list of
        pyg Data object instances which should then be usable to train a pyg gnn model.
        """
        num_graphs = 10
        num_node_range = (5, 10)
        num_node_attributes = 20
        num_edge_attributes = 10
        num_graph_labels = 3
        
        graphs = sample_mock_graphs(
            k=num_graphs,
            num_nodes_range=num_node_range,
            num_node_attributes=num_node_attributes,
            num_edge_attributes=num_edge_attributes,
            num_graph_labels=num_graph_labels
        )
        
        data_list = pyg_data_list_from_graphs(graphs)
        assert isinstance(data_list, list)
        assert len(data_list) == num_graphs
        
        for data, graph in zip(data_list, graphs):
            assert isinstance(data, Data)
            assert data.x.shape[0] == len(graph['node_indices'])

    def test_import_computational_overhead(self):
        """
        The function ``pyg_from_graph`` performs a local import for every call of the function 
        to convert an individual graph while the ``pyg_data_list_from_graphs`` function only does 
        this import once and then converts the graphs individually. This test assesses the 
        difference in computational time between the two implementations.
        """
        num_graphs = 1000
        num_nodes_range = (20, 50)
        num_node_attributes = 40
        num_edge_attributes = 20
        num_graph_labels = 1
        
        graphs = sample_mock_graphs(
            k=num_graphs,
            num_nodes_range=num_nodes_range,
            num_node_attributes=num_node_attributes,
            num_edge_attributes=num_edge_attributes,
            num_graph_labels=num_graph_labels,
        )
        
        # ~ individual graphs
        time_start = time.time()
        for graph in graphs:
            pyg_from_graph(graph)
            
        duration_single = time.time() - time_start

        # ~ batch processing
        time_start = time.time()
        pyg_data_list_from_graphs(graphs)
        
        duration_batch = time.time() - time_start
        
        print(f'num graphs: {num_graphs}')
        print(f'single: {duration_single*10e3/num_graphs:.4f}ms/graph - '
              f'batch: {duration_batch*10e3/num_graphs:.4f}ms/graph')
        print(f'single increase: {duration_single/duration_batch:.3f}x')
        
    def test_pyg_model_compatibility(self):
        """
        The converted pyg data object should be possible to be used as input to the forward pass 
        of a simple pyg model.
        """
        num_graphs = 10
        num_nodes_range = (5, 10)
        num_node_attributes = 10
        num_edge_attributes = 5
        num_graph_labels = 2
        
        graphs = sample_mock_graphs(
            k=num_graphs,
            num_nodes_range=num_nodes_range,
            num_node_attributes=num_node_attributes,
            num_edge_attributes=num_edge_attributes,
            num_graph_labels=num_graph_labels,
        )
        
        class Model(nn.Module):
            
            def __init__(self):
                super().__init__()
                self.gcn_1 = GCNConv(num_node_attributes, 16)
                self.gcn_2 = GCNConv(16, 8)
                self.gcn_3 = GCNConv(8, num_graph_labels)
                self.pool = SumAggregation()

            def forward(self, data: Data):
                x = data.x
                x = self.gcn_1(x, data.edge_index)
                x = self.gcn_2(x, data.edge_index)
                x = self.gcn_3(x, data.edge_index)
                x = self.pool(x, data.batch)
                return x
            
        model = Model()
        
        data_list = pyg_data_list_from_graphs(graphs)
        assert len(data_list) == num_graphs
        batch = next(iter(DataLoader(data_list, batch_size=num_graphs)))
        out = model(batch)
        
        assert isinstance(out, torch.Tensor)
        assert out.shape == (num_graphs, num_graph_labels)
        

# == JAX/JRAPH ==

class TestJraph:
    
    def test_jraph_from_graph_basically_works(self):
        """
        ``jraph_from_graph`` should take a graph dict as input and turn it into a jraph.GraphsTuple object.
        """
        num_nodes = 15
        num_node_attributes = 20
        num_edge_attributes = 10
        
        graph = create_mock_graph(
            num_nodes=15,
            num_node_attributes=num_node_attributes,
            num_edge_attributes=num_edge_attributes,
        )
        graph_tuple: jraph.GraphsTuple = jraph_from_graph(graph)

        assert isinstance(graph_tuple, jraph.GraphsTuple)
        
        # Node features
        assert int(graph_tuple.n_node[0]) == num_nodes
        assert graph_tuple.nodes.shape == (num_nodes, num_node_attributes)
        
        # Since the mock graphs are just loops, there are as many edges as nodes
        assert int(graph_tuple.n_edge[0]) == num_nodes
        assert graph_tuple.edges.shape == (num_nodes, num_edge_attributes)
        assert graph_tuple.senders.shape == (num_nodes, )
        assert graph_tuple.receivers.shape == (num_nodes, )
        
    def test_jraph_batch_from_graphs_basically_works(self):
        """
        ``jraph_batch_from_graphs`` should create a jraph batched GraphsTuple instance using a list of graph dicts 
        as the input.
        """
        num_graphs = 10
        num_nodes_range = (5, 10)
        num_node_attributes = 20
        num_edge_attributes = 10
        
        graphs = sample_mock_graphs(
            k=num_graphs,
            num_nodes_range=num_nodes_range,
            num_node_attributes=num_node_attributes,
            num_edge_attributes=num_edge_attributes,
        )
        
        # The aggregated batch also is again a GraphsTuple instance
        batch = jraph_implicit_batch_from_graphs(graphs)
        print(batch)
        assert isinstance(batch, jraph.GraphsTuple)
        assert len(jraph.unbatch(batch)) == num_graphs
        
    def test_jraph_model_compatibility(self):
        """
        It should be possible to use the converted jraph graphs to feed them into a simple jraph graph network.
        """
        num_graphs = 15
        num_nodes_range = (5, 10)
        num_node_attributes = 10
        num_edge_attributes = 6
        num_graph_labels = 3
        
        graphs = sample_mock_graphs(
            k=num_graphs,
            num_nodes_range=num_nodes_range,
            num_node_attributes=num_node_attributes,
            num_edge_attributes=num_edge_attributes,
            num_graph_labels=num_graph_labels,
        )
        batch = jraph_implicit_batch_from_graphs(graphs)
        print(batch)
        
        # Here we just want to construct a very simple graph network
        network = jraph.GraphNetwork(
            update_node_fn=lambda node_features, aggregated_sender_edge_features, *args, **kwargs: node_features,
            update_edge_fn=lambda edge_features, *args, **kwargs: edge_features,
            aggregate_edges_for_nodes_fn=jraph.segment_sum,
        )
        network_jit = jax.jit(network)
        updated_batch = network_jit(batch)
        
        # The result of the graph network should again be a graphs tuple instance with the same 
        # internal make-up but with different features.
        assert isinstance(updated_batch, jraph.GraphsTuple)