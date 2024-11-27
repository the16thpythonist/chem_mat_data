import os
import time
import datetime
import shutil
import zipfile
import yaml
from typing import Dict, List, Tuple, Optional

from chem_mat_data.utils import is_archive


class Cache:
    """
    This class is used to access and interact with a given ``cache_path`` folder. This cache folder 
    can be used to cache datasets that have already been downloaded at some prior point in time, so
    that datasets won't have to be downloaded again if they are needed in the future.
    """
    
    def __init__(self, 
                 cache_path: str
                 ):
        self.path = cache_path
    
    def contains_dataset(self,
                         name: str,
                         typ: str,
                         ) -> bool:
        """
        Returns True if the cache contains a dataset with the given unique string ``name`` and of the 
        given type ``typ``. Otherwise it will return False.
        
        :param name: The unique string name of the dataset that should be checked for in the cache
        :param typ: The type of the dataset that should be checked for in the cache
        
        :returns: True if the cache contains the dataset, otherwise False
        """
        base_name: str = self.construct_base_name(name, typ)
        
        # We'll only check for the metadata path here because only the metadata file will contain the 
        # exact name of the dataset that we are looking for. The dataset file itself may have another 
        # name which will be stored in the metadata file.
        metadata_path = os.path.join(self.path, f'{base_name}.yml')
        contains_metadata = os.path.exists(metadata_path)
        return contains_metadata
    
    def get_dataset_metadata(self,
                             name: str,
                             typ: str,
                             ) -> dict:
        """
        Returns the metadata of the dataset with the given unique string ``name`` and type ``typ`` from the cache.
        
        :param name: The unique string name of the dataset that should be retrieved from the cache
        :param typ: The type of the dataset that should be retrieved from the cache
        
        :returns: A dictionary containing the metadata information about the dataset
        """
        # The base name of the dataset will be a combination of the name and the type.
        base_name: str = self.construct_base_name(name, typ)
        metadata_path = os.path.join(self.path, f'{base_name}.yml')
        with open(metadata_path) as file:
            metadata = yaml.load(file, Loader=yaml.FullLoader)
            
        return metadata
    
    def add_dataset(self,
                    name: str,
                    typ: str,
                    path: str,
                    metadata: dict,
                    ) -> None:
        """
        Given the unique string ``name`` of the dataset of the given ``typ``, 
        the absolute path to the dataset ``path`` and 
        some additional ``metadata`` dict about the dataset, this method will deposit the dataset along 
        with the metadata into the cache folder.
        
        :param name: The unique string name of the dataset that should be added to the cache
        :param typ: The type of the dataset that should be added to the cache
        :param path: The absolute string path to the dataset that should be added to the cache
        :param metadata: A dictionary containing some metadata information about the dataset
        
        :returns: None
        """
        # ~ dataset content
        # To add a dataset we first of all have to move the dataset itself into the cache
        # To save space we want to compress the dataset before saving it there. Of course 
        # we only do that if the dataset is not already a compressed file.
        
        if not os.path.exists(path):
            raise FileNotFoundError(f'The given path "{path}" does not exist! So it cannot be used '
                                    'to add a dataset to the cache.')
        
        # As the base file name we will use a combination between the name of the dataset and the 
        # type of the dataset.
        base_name = self.construct_base_name(name, typ)
        
        # Here we put this into an archive
        dest_path = os.path.join(self.path, f"{base_name}.zip")
        
        # Here we have to differentiate and handle differently if the dataset is a file or a folder
        if os.path.isfile(path):
            with zipfile.ZipFile(dest_path, mode='w') as archive:
                archive.write(path, os.path.basename(path))
                
        elif os.path.isdir(path):
            with zipfile.ZipFile(dest_path, mode='w') as archive:
                for root, dirs, files in os.walk(path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        archive.write(file_path, os.path.relpath(file_path, path))
        
        # ~ dataset metadata
        # Besides the actual dataset data we also want to store some metadata about the dataset 
        # in the cache as well which we can use to perhaps later on decide if we want to update the 
        # cache with a newer version of the dataset or not.
        
        # We'll add some information about the caching process as well
        file_name = os.path.basename(path)
        metadata['_cache_time'] = time.time()
        metadata['_cache_archived'] = not is_archive(path)
        metadata['_cache_filename'] = file_name
        
        metadata_path = os.path.join(self.path, f'{base_name}.yml')
        with open(metadata_path, mode='w') as file:
            yaml.dump(metadata, file)
        
    def retrieve_dataset(self,
                         name: str,
                         typ: str,
                         dest_path: str,
                         ) -> str:
        """
        Retrieves the dataset with the given unique string ``name`` from the cache by copying the dataset 
        to the given destination folder ``dest_path``. The name that will be used for the dataset will be 
        the name of the file/folder that was used when initially storing the dataset in the cache.
        This method will return the path to the dataset that was copied to the destination folder.
        
        :param name: The unique string name of the dataset that should be retrieved from the cache
        :param dest_path: The absolute string path to the destination folder where the dataset should be copied
        
        :returns: The absolute string path to the extracted dataset file/folder
        """
        # the base for the file name will be a combination of the name of the dataset and the type.
        base_name: str = self.construct_base_name(name, typ)
        assert self.contains_dataset(name, typ), f'The dataset "{name}" of type {typ} does not exist in the cache!'
        
        # ~ load metadata
        metadata_path = os.path.join(self.path, f'{base_name}.yml')
        with open(metadata_path) as file:
            metadata = yaml.load(file, Loader=yaml.FullLoader)
            
        # ~ unzip archive
        archive_path = os.path.join(self.path, f'{base_name}.zip')
        with zipfile.ZipFile(archive_path, mode='r') as archive:
            archive.extractall(dest_path)
            
        return os.path.join(dest_path, metadata['_cache_filename'])
    
    def construct_base_name(self, name: str, typ: str) -> str:
        """
        Constructs the base name of a dataset by combining the unique string name of the dataset with the 
        type of the dataset.
        
        :param name: The unique string name of the dataset
        :param typ: The type of the dataset
        
        :returns: The base name of the dataset
        """
        return f'{name}.{typ}'
    
    def list_datasets(self) -> List[Tuple[str, str]]:
        """
        Returns a list containing the string names of all the datasets contained in the cache.
        
        :returns: list of strings
        """
        files = os.listdir(self.path)
        metadata_files = [f for f in files if f.endswith('.yml')]
        datasets = []
        for file_name in metadata_files:
            split_name = file_name.split('.')
            datasets.append((split_name[0], split_name[1]))
        
        return datasets
    
    def iterator_clear_(self) -> str:
        """
        This is a generator function that be used to clear the cache. For each removed file, the generator will 
        yield a tuple of the file name and the file path that are being deleted.
        """
        for file_name in os.listdir(self.path):
            file_path = os.path.join(self.path, file_name)
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
            else:
                os.remove(file_path)
                
            yield file_name, file_path
    
    def clear(self) -> int:
        """
        Clears the cache by deleting all the datasets and metadata files that are stored in the cache.
        
        :returns: The number of deleted elements
        """
        num_elements = 0
        for file_name, file_path in self.iterator_clear_():
            num_elements += 1
            
        return num_elements
    
    def __len__(self):
        return len(os.listdir(self.path))