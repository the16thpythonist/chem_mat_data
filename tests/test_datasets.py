"""
This module tests all the datasets that are availabile on remote file share server 
by obtaining each dataset and attempting to use them for a simple GNN forward pass.
"""
import numpy as np
import pytest
from typing import Dict

from chem_mat_data.testing import ConfigIsolation
from chem_mat_data.testing import assert_graph_dict
from chem_mat_data.web import AbstractFileShare
from chem_mat_data.web import construct_file_share
from chem_mat_data.main import load_graph_dataset
from chem_mat_data.main import pyg_data_list_from_graphs


def fetch_datasets():
    
    with ConfigIsolation() as iso:
        
        file_share_type = iso.config.get_fileshare_type()
        file_share: AbstractFileShare = construct_file_share(
            file_share_type=iso.config.get_fileshare_type(),
            file_share_url=iso.config.get_fileshare_url(),
            file_share_kwargs=iso.config.get_fileshare_parameters(file_share_type),    
        )
        metadata: dict = file_share.fetch_metadata()
        
        datasets: Dict[str, dict] = metadata['datasets']
        
        # Now we also want to be prudent and filter datasets that are too large so we don't significantly slow 
        # down the testing.
        
        dataset_names: list[str] = []
        for dataset_name, dataset_info in datasets.items():
            
            # Skip datasets that are too large and would take too long to process
            if dataset_info['compounds'] >= 200_000:
                continue
            
            # We also want to skip the hidden datasets that start with an underscore
            if dataset_name.startswith('_'):
                continue
            
            dataset_names.append(dataset_name)
            
        # Finally we return the list of the dataset names to be used as the basis of the individual test
        # cases.
        return dataset_names
    
    
# Fetch datasets 
@pytest.mark.parametrize('dataset_name', fetch_datasets())
def test_dataset_torch_forward_pass(dataset_name: str):
    
    with ConfigIsolation() as iso:
        
        file_share_type = iso.config.get_fileshare_type()
        file_share: AbstractFileShare = construct_file_share(
            file_share_type=iso.config.get_fileshare_type(),
            file_share_url=iso.config.get_fileshare_url(),
            file_share_kwargs=iso.config.get_fileshare_parameters(file_share_type),    
        )
        metadata: dict = file_share.fetch_metadata()
        dataset_info = metadata['datasets'][dataset_name]
        
        # ~ loading the graphs from the remote file share
        graphs: list[dict] = load_graph_dataset(
            dataset_name=dataset_name,
            folder_path=iso.config.cache_path,
            config=iso.config,
        )
        assert len(graphs) != 0
        assert len(graphs) == dataset_info['compounds'], (
            f'The number of graphs in the dataset ({len(graphs)}) should match the number of compounds reported '
            f'in the metadata file ({dataset_info["compounds"]}) for dataset {dataset_name}'
        )
        example_graph = graphs[0]
        
        # ~ checking the individual graphs for validity
        for index, graph in enumerate(graphs):
            assert_graph_dict(graph)
            
        # ~ Doing model forward pass
        try:
            from torch import Tensor
            from torch_geometric.data import Data
            from torch_geometric.loader import DataLoader
            from torch_geometric.nn.models import GIN

            data_list = pyg_data_list_from_graphs(graphs)
            data_loader = DataLoader(data_list, batch_size=128, shuffle=True)
            
            model = GIN(
                in_channels=example_graph['node_attributes'].shape[1],
                out_channels=example_graph['graph_labels'].shape[0],
                hidden_channels=64,
                num_layers=3,
            )
            for data in data_loader:
                out_pred: Tensor = model.forward(
                    x=data.x,
                    edge_index=data.edge_index,
                )
                out_pred: np.ndarray = out_pred.cpu().detach().numpy()
                assert not (np.isnan(out_pred).any() or np.isinf(out_pred).any())
            
        except (ImportError, ModuleNotFoundError) as exc:
            
            print('Skipping forward pass test because PyTorch is not installed...')
            
        
        