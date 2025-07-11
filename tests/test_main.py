import os
import tempfile
import pandas as pd
import rdkit.Chem as Chem

from chem_mat_data.main import ensure_dataset
from chem_mat_data.main import load_xyz_dataset
from chem_mat_data.main import load_smiles_dataset

    
    
def test_ensure_dataset_with_folder_works():
    """
    It should be possible to use the "ensure_dataset" function also to download a dataset which is actually a folder.
    On the remote file share we use .zip archives to store such folder-based datasets. This should internally be handled
    by the "ensure_dataset" function and the path that is returned by it should then be the path to the unzipped folder 
    on the local file system.
    """
    
    with tempfile.TemporaryDirectory() as temp_path:
        
        path = ensure_dataset(
            dataset_name='_test2',
            extension='xyz_bundle',
            use_cache=False,
            folder_path=temp_path,
        )
    
        # This should now be a folder path!
        assert os.path.exists(path)
        assert os.path.isdir(path)
        

def test_load_xyz_dataset_works():
    """
    The load_xyz_dataset function should be able to load a "xyz_bundle" dataset from the remote file share 
    server and return it as a dataframe with a column "mol" that contains the corresponding RDKit Mol objects 
    that represent these molecules.
    """
    with tempfile.TemporaryDirectory() as temp_path:
        
        df: pd.DataFrame = load_xyz_dataset(
            dataset_name='_test2',
            folder_path=temp_path,
            use_cache=False
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        print(df.head())
        
        for _, row in df.iterrows():
            assert isinstance(row['mol'], Chem.Mol)
        
        
def test_load_smiles_dataset_works():
    
    with tempfile.TemporaryDirectory() as temp_path:
        
        df: pd.DataFrame = load_smiles_dataset(
            dataset_name='clintox',
            folder_path=temp_path,
            use_cache=False
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        print(df.head())
        