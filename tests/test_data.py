import os
import csv

from chem_mat_data.processing import MoleculeProcessing
from chem_mat_data.graph import assert_graph_dict
from chem_mat_data.data import save_graphs, load_graphs

from .utils import ASSETS_PATH, ARTIFACTS_PATH


SMILES_LIST = [
    'C1=CC=CC=C1CCO',
    'CC(=O)O',
    'C1=CC=CC=C1C(N)CC',
]


def test_save_and_load_graphs_basically_works():
    """
    The "save_graphs" method should be able to take a list of graph dict objects and save them 
    as a persistent message pack file to the disk.
    
    Conversely, the "load_graphs" method should be able to load the same file back into memory 
    as a list of graph dict objects.
    """
    path = os.path.join(ARTIFACTS_PATH, 'test_save_graphs_basically_works.mpack')
    
    # First we need to generate some graph objects with which to run the test
    processing = MoleculeProcessing()
    graphs = [processing.process(value) for value in SMILES_LIST]
    
    # Then we need to save these graphs into a file
    save_graphs(graphs, path)
    
    assert os.path.exists(path)
    assert os.path.isfile(path)
    
    # And finally we can attempt to load that file now from memory again
    graphs_loaded = load_graphs(path)
    assert isinstance(graphs_loaded, list)
    assert len(graphs_loaded) == len(graphs)
    for graph in graphs_loaded:
        assert_graph_dict(graph)
        
        
def test_process_test_dataset():
    """
    This is less of a unittest and more of a utility function.
    
    This function will load the _test.csv file from the assets folder and process all the 
    SMILES strings contained in it into graph representations. It will then save these graphs 
    as a message pack file into the same folder.
    """
    name = 'clintox'
    source_path = os.path.join(ASSETS_PATH, f'{name}.csv')
    with open(source_path, mode='r') as file:
        dict_reader = csv.DictReader(file)
        rows = list(dict_reader)
        
    processing = MoleculeProcessing()
    graphs = []
    for row in rows:
        try:
            graph = processing.process(row['smiles'])
            graphs.append(graph)
        except Exception as exc:
            print(exc)
        
    dst_path = os.path.join(ASSETS_PATH, f'{name}.mpack')
    save_graphs(graphs, dst_path)
    assert os.path.exists(dst_path)
    
    graphs_loaded = load_graphs(dst_path)
    assert len(graphs) == len(graphs_loaded)