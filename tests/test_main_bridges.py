"""
This module contains the unittests that specifically test the bridges to the major graph
neural network libraries such as ptorch geometric and jraph.

NOTE: The GNN libraries used here are not part of the packages main dependencies and have
      to be installed separately.
"""

import time

import pytest

# These GNN libraries are optional dependencies (see module docstring). If any of them is
# not installed, skip this entire test module rather than erroring out during collection.
pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
pytest.importorskip("jax")
pytest.importorskip("jraph")

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


def mock_graph_with_atom_targets(num_nodes: int = 12, missing: int = 4):
    """
    Build a mock graph whose targets are per atom rather than per molecule.

    Mirrors the shape the esnuel_* datasets ship: an (n_nodes, 1) float array of
    node_labels with NaN at the atoms that have no value, alongside the usual
    graph-level label. NaN rather than a numeric sentinel is the convention because
    those datasets contain genuine negative values.
    """
    import numpy as np

    graph = create_mock_graph(num_nodes=num_nodes, num_graph_labels=1)
    labels = np.arange(num_nodes, dtype=float).reshape(-1, 1)
    labels[:missing] = np.nan
    graph['node_labels'] = labels
    graph['edge_labels'] = np.arange(
        len(graph['edge_indices']), dtype=float
    ).reshape(-1, 1)
    return graph


# == TORCH/PYTORCH GEOMETRIC ==


class TestAtomLevelTargets:
    """
    Node- and edge-level targets have to survive conversion.

    These used to be dropped silently: an atom-level dataset converted into a valid
    Data object that was simply missing the values it exists for, with no error and no
    warning. Datasets affected include esnuel_mca, esnuel_maa and qm40_xyz.
    """

    def test_node_labels_become_node_y(self):
        graph = mock_graph_with_atom_targets(num_nodes=12, missing=4)
        data = pyg_from_graph(graph)

        assert hasattr(data, 'node_y'), 'node_labels were dropped by the conversion'
        assert data.node_y.shape == (12, 1)
        # y stays the graph-level target; the two are not interchangeable.
        assert data.y.shape == (1,)

    def test_edge_labels_become_edge_y(self):
        graph = mock_graph_with_atom_targets()
        data = pyg_from_graph(graph)
        assert hasattr(data, 'edge_y')
        assert data.edge_y.shape == (len(graph['edge_indices']), 1)

    def test_missing_entries_stay_nan(self):
        # A numeric sentinel would collide with real values, which reach -248 kJ/mol in
        # esnuel_maa, so absence is encoded as NaN and must survive the round trip.
        graph = mock_graph_with_atom_targets(num_nodes=12, missing=4)
        data = pyg_from_graph(graph)
        assert torch.isnan(data.node_y[:4]).all()
        assert not torch.isnan(data.node_y[4:]).any()

    def test_values_survive_within_float32(self):
        graph = mock_graph_with_atom_targets(num_nodes=12, missing=0)
        data = pyg_from_graph(graph)
        expected = torch.arange(12, dtype=torch.float).reshape(-1, 1)
        assert torch.allclose(data.node_y, expected)

    def test_node_y_batches_alongside_the_nodes(self):
        # The point of the naming: PyG concatenates an unknown attribute along dim 0
        # when its first dimension matches the node count. If that broke, node_y would
        # silently fall out of register with x across a batch.
        graphs = [mock_graph_with_atom_targets(num_nodes=n, missing=1)
                  for n in (5, 8, 11)]
        batch = next(iter(DataLoader(
            [pyg_from_graph(g) for g in graphs], batch_size=3, shuffle=False
        )))
        assert batch.node_y.shape[0] == batch.x.shape[0] == 5 + 8 + 11
        assert batch.y.shape[0] == 3
        # one NaN per graph, and no more
        assert int(torch.isnan(batch.node_y).sum()) == 3

    def test_graphs_without_node_labels_are_unaffected(self):
        # Most of the collection has no node-level targets; they must not grow an
        # empty attribute.
        data = pyg_from_graph(create_mock_graph())
        assert not hasattr(data, 'node_y')
        assert not hasattr(data, 'edge_y')


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
            num_graph_labels=num_graph_labels,
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
        assert data.y.shape == (num_graph_labels,)

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
            num_graph_labels=num_graph_labels,
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
        print(
            f'single: {duration_single * 10e3 / num_graphs:.4f}ms/graph - '
            f'batch: {duration_batch * 10e3 / num_graphs:.4f}ms/graph'
        )
        print(f'single increase: {duration_single / duration_batch:.3f}x')

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
    def test_graph_labels_reach_globals(self):
        """
        ``globals`` used to be hardcoded to None, which dropped the target of *every*
        dataset in the collection - a GraphsTuple has no other field for a graph-level
        quantity, so a converted graph carried no label at all.
        """
        graph = create_mock_graph(num_nodes=9, num_graph_labels=3)
        graph_tuple = jraph_from_graph(graph)

        assert graph_tuple.globals is not None, 'graph_labels were dropped'
        assert jax.numpy.asarray(graph_tuple.globals).shape == (1, 3)
        assert jax.numpy.allclose(
            jax.numpy.asarray(graph_tuple.globals)[0],
            jax.numpy.asarray(graph['graph_labels']),
        )

    def test_graph_without_labels_still_converts(self):
        graph = create_mock_graph()
        del graph['graph_labels']
        graph_tuple = jraph_from_graph(graph)
        assert graph_tuple.globals is None

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
        assert graph_tuple.senders.shape == (num_nodes,)
        assert graph_tuple.receivers.shape == (num_nodes,)

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
            update_node_fn=lambda node_features, aggregated_sender_edge_features, *args, **kwargs: (
                node_features
            ),
            update_edge_fn=lambda edge_features, *args, **kwargs: edge_features,
            aggregate_edges_for_nodes_fn=jraph.segment_sum,
        )
        network_jit = jax.jit(network)
        updated_batch = network_jit(batch)

        # The result of the graph network should again be a graphs tuple instance with the same
        # internal make-up but with different features.
        assert isinstance(updated_batch, jraph.GraphsTuple)
