"""
Testing all the code snippets used in the documentation.
"""

def test_readme_example():
    
    from torch import Tensor
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn.models import GIN
    from rich.pretty import pprint
    
    from chem_mat_data import load_graph_dataset, pyg_data_list_from_graphs
    
    # Load the dataset of graphs
    graphs: list[dict] = load_graph_dataset('clintox')
    example_graph = graphs[0]
    pprint(example_graph)
    
    # Convert the graph dicts into PyG Data objects
    data_list = pyg_data_list_from_graphs(graphs)
    data_loader = DataLoader(data_list, batch_size=32, shuffle=True)
    
    # Construct a GNN model
    model = GIN(
        in_channels=example_graph['node_attributes'].shape[1],
        out_channels=example_graph['graph_labels'].shape[0],
        hidden_channels=32,
        num_layers=3,  
    )
    
    # Perform model forward pass with a batch of graphs
    data: Data = next(iter(data_loader))
    out_pred: Tensor = model.forward(
        x=data.x, 
        edge_index=data.edge_index, 
        batch=data.batch
    )
    pprint(out_pred)

def test_first_steps_smiles():
    
    import pandas as pd
    from chem_mat_data import load_smiles_dataset
    
    df: pd.DataFrame = load_smiles_dataset('clintox')
    print(df.head())
    
    
def test_first_steps_graphs():
    
    from rich.pretty import pprint
    from chem_mat_data import load_graph_dataset

    graphs: list[dict] = load_graph_dataset('clintox')
    example_graph = graphs[0]
    pprint(example_graph)
    

def test_process_new_graphs():
    
    from rich.pretty import pprint
    from chem_mat_data.processing import MoleculeProcessing

    processing = MoleculeProcessing()

    smiles: str = 'C1=CC=CC=C1CCN'
    graph: dict = processing.process(smiles)
    pprint(graph)