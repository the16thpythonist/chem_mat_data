"""
Testing all the code snippets used in the documentation.
"""


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