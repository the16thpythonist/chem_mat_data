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
    
def test_custom_pre_processing():
    
    from chem_mat_data.processing import MoleculeProcessing
    from chem_mat_data.processing import OneHotEncoder, chem_prop, list_identity

    # Has to inherit from MoleculeProcessing!
    class CustomProcessing(MoleculeProcessing):

        node_attribute_map = {

            'mass': {
                # "chem_prop" is a wrapper function which will call the given 
                # property method on the rdkit.Atom object - in this case the 
                # GetMass() method - and pass the output to the transformation 
                # function given as the second argument. "list_identity" means 
                # that the value is simply converted to a list as it is.
                # Therefore, this configuration will result in outputs such as 
                # [12.08], [9.88] etc. as parts of the overall feature vector.
                'callback': chem_prop('GetMass', list_identity),
                # Provide a human-readable description of what this section of 
                # the node feature vector represents.
                'description': 'The mass of the atom'
            },

            'symbol': {
                # "OneHotEncoder" is a special callable class that can be used 
                # to automatically define one-hot encodings. The object will 
                # accept the output of the given chem prop - in this case the 
                # GetSymbol action on the rdkit.Atom - and create an integer 
                # one-hot vector according to the provided list. In this case, 
                # the encoding will encode a carbon as [1, 0, 0, 0], 
                # a oxygen as [0, 1, 0, 0] etc.
                'callback': chem_prop('GetSymbol', OneHotEncoder(
                    ['C', 'O', 'N', 'S'],
                    add_unknown=False,
                    dtype=str,
                )),
                'description': 'One hot encoding of the atom type',
                'is_type': True,
                'encodes_symbol': True,
            },
        }

        edge_attributes = {
            'type': {
                'callback': chem_prop('GetBondType', OneHotEncoder(
                    [1, 2],
                    add_unknown=False,
                    dtype=int, 
                )),
            }
        }
        
    from rich.pretty import pprint

    processing = CustomProcessing()

    graph: dict = processing.process('CCCC')
    pprint(graph)
    
    assert isinstance(graph, dict)
    
    
def test_custom_pre_processing_2():
    
    import rdkit.Chem as Chem
    from rich.pretty import pprint
    from typing import List
    from chem_mat_data.processing import MoleculeProcessing

    def custom_callback(atom: Chem.Atom) -> List[float]:
        
        # Mass multiplied with the charge
        return [atom.GetMass() * atom.GetCharge()]


    class CustomCallbackProcessing(MoleculeProcessing):

        node_attributes = {
            'mass_times_charge': {
                'callback': custom_callback,
                'description': 'atom mass multiplied with the charge',
            }
        }


    processing = CustomCallbackProcessing()
    graph = processing.process('CCCC')
    pprint(graph)
    
    assert isinstance(graph, dict)