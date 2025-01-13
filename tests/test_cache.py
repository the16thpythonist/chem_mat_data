import os
import tempfile
import yaml
from typing import List, Tuple

from chem_mat_data.cache import Cache
from .utils import ARTIFACTS_PATH, get_mock_dataset_path


class TestCache:
    
    def test_add_dataset_basically_works(self):
        """
        the "add_dataset" method should copy the dataset to the specified cache folder and 
        also create a metadata file.
        """   
        with tempfile.TemporaryDirectory() as path:
            
            assert len(os.listdir(path)) == 0
            
            cache = Cache(path)
            metadata = {
                'num_elements': 100,
            }
            # This function will return the path to one dataset that is used for testing
            # (the mpack file version of that dataset)
            dataset_path = get_mock_dataset_path()
            cache.add_dataset('test', 'mpack', dataset_path, metadata)
            # This should create 2 files in the cache directory: the dataset itself and a metadata file
            assert len(os.listdir(path)) == 2
            
            files = os.listdir(path)
            print('cache files', files)
            assert 'test.mpack.yml' in files
            assert 'test.mpack.zip' in files
            
            # Then we can load that metadata file and inspect the content
            metadata_path = os.path.join(path, 'test.mpack.yml')
            with open(metadata_path) as file:
                metadata = yaml.load(file, Loader=yaml.FullLoader)
                
            print('metadata', metadata)
            assert isinstance(metadata, dict)
            assert 'num_elements' in metadata
            assert '_cache_filename' in metadata
            
    def test_list_datasets_basically_works(self):
        """
        The "list_datasets" method should return a list of tuples where each tuple contains the 
        name of the dataset and the type of the dataset that are currently stored in the cache folder.
        """
        with tempfile.TemporaryDirectory() as path:
            
            cache = Cache(path)
            cache.add_dataset('test_1', 'mpack', get_mock_dataset_path(), {'num_elements': 100})
            cache.add_dataset('test_2', 'mpack', get_mock_dataset_path(), {'num_elements': 200})
            
            datasets: List[Tuple[str, str]] = cache.list_datasets()
            assert len(datasets) == 2
            assert set(datasets) == set([('test_1', 'mpack'), ('test_2', 'mpack')])
            
    def test_contains_dataset_basically_works(self):
        """
        "contains_dataset" should return False at the start and then disable the dataset 
        after it has been added.
        """
        
        with tempfile.TemporaryDirectory() as path:
            
            cache = Cache(path)
            # Since the cache is empty, the contains function should return False
            assert not cache.contains_dataset('test', 'mpack')
            
            # Only after adding the dataset to the cache, the contains function should return True
            cache.add_dataset('test', 'mpack', get_mock_dataset_path(), {'num_elements': 100})
            assert cache.contains_dataset('test', 'mpack')
            
    def test_retrieve_dataset_basically_works(self):
        """
        Retrieve dataset should be able to retrieve a dataset from the cache and put it into the 
        destination folder using the original filename of the dataset when it was stored in the 
        cache.
        """
        # We setup two different temp folders here - one for the cache itself and the other to 
        # then retrieve the cache data into later on.
        with tempfile.TemporaryDirectory() as folder_path:
            with tempfile.TemporaryDirectory() as cache_path:
                
                cache = Cache(cache_path)
                dataset_path = get_mock_dataset_path()
                dataset_file_name = os.path.basename(dataset_path)
                cache.add_dataset('test', 'mpack', dataset_path, {'num_elements': 100})
                
                assert cache.contains_dataset('test', 'mpack')
                # Before retrieval the main folder should be empty
                assert len(os.listdir(folder_path)) == 0
                
                # Now we can retrieve the dataset from the cache
                cache.retrieve_dataset('test', 'mpack', folder_path)
                files = os.listdir(folder_path)
                print(files)
                assert len(files) == 1
                # Now the name of the reconstructed dataset should not actually be the name that 
                # we assigned to it in the cache but the original file name that was put into the 
                # dataset!
                assert dataset_file_name in files
                