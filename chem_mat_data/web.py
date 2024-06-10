import os
import tempfile
import typing as t

import io
import yaml
import requests
from rich.progress import Progress


class MockProgress:
    
    def add_task(self, *args, **kwargs):
        return 0
    
    def update(self, *args, **kwargs):
        pass
    
    def start_task(self, *args, **kwargs):
        pass



class NextcloudFileShare:
    
    def __init__(self, 
                 url: str
                 ) -> None:
        # For all the functionality in this class we want to make sure that the URL ends with a 
        # trailing slash
        self.url = url

        # This is the name of the file on the file share server which contains the metadata. This 
        # is a human-readable yml file which not only contains the information about the file share 
        # in general, but also detailed information about the individual datasets that are 
        # available for download.
        self.metadata_name = 'metadata.yml'
        # This is the path where the metadata file will be stored on the local system AFTER it has 
        # been downloaded from the file share server. So this file may or may not exist at the
        # time of the object creation, but it will definitely exist after the first call to the
        # fetch_metadata method.
        self.metadata_path: str = os.path.join(tempfile.gettempdir(), f'chemdata_{self.metadata_name}')
        
        # This is the dictionary where the content of the metadata file will be stored after it has
        # been downloaded from the file share server. This is a dictionary representation of the
        # metadata file. This dictionary will be created by the fetch_metadata method.
        self.metadata: dict[str, t.Any] = None
    
    def fetch_metadata(self, force: bool = False) -> None:
        
        # For the sake of efficiency, we will first check if the metadata file already exists on the
        # local system. If it does, we will simply load it into the metadata attribute of the object
        # and return.
        if not os.path.exists(self.metadata_path) or force:
            # If the metadata file does not exist, we will download it from the file share server.
            content = self.download(self.metadata_name)
            with open(self.metadata_path, mode='wb') as file:
                file.write(content)
                
        # Now that the file definitely exists, we will load it into the metadata attribute of the object.
        with open(self.metadata_path, mode='r') as file:
            self.metadata = yaml.safe_load(file)
    
    def download_dataset(self, 
                         dataset_name: str, 
                         folder_path: str = tempfile.gettempdir(),
                         progress: Progress = MockProgress(),
                         ) -> str:
        
        if not dataset_name.endswith('.mpack'):
            dataset_name = dataset_name + '.mpack'
            
        dataset_path = os.path.join(folder_path, dataset_name)
        with open(dataset_path, mode='wb') as file:
            content = self.download(dataset_name, progress=progress)
            file.write(content)
            
        return dataset_path
            
    def download_file(self,
                      file_name: str,
                      folder_path: str = tempfile.gettempdir(),
                      progress: Progress = MockProgress(),
                      ) -> str:
        
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, mode='wb') as file:
            content = self.download(file_name, progress=progress, folder_path=folder_path)
            file.write(content)
            
        return file_path
    
    # ~ utility methods
    
    def download(self, 
                 file_name: str, 
                 progress: Progress = MockProgress(),
                 folder_path: str | None = None,
                 ) -> bytes:
        """
        Given the full ``file_name`` of a file located in the current file share location, this 
        method will download that file. If the ``folder_path`` argument is given as a valid 
        folder location on the local system, the downloaded content will be written as a local
        file with the same filename into that folder (overwrites existing files). If the ``folder_path``
        is None, the bytes content of the file will instead be directly returned by this method.
        
        :param file_name: The string file name of the file to be downloaded - including the file 
            extension.
        :param progress: An optional rich.progress.Progress instance which will be used to track 
            the download progress in case it is given. If not explicticly provided, will be ignored.
        :param folder_path: An optional string path to a folder on the local system where the
            downloaded file should be stored. If not given, the downloaded content will be returned
            as a bytes object.
            
        :returns: The bytes content of the downloaded file if no folder path is given. Otherwise,
            returns None.
        """
        file_url = self.url + file_name
        response = requests.get(file_url, stream=True)
        
        # Only when we get the correct status code, we can assume that the download was 
        # succesfull and we can proceed with the download.
        if response.status_code == 200:
            
            total_size = int(response.headers.get('Content-Length', 0))
            
            # Optionally, we update the progress of the download with the given Progress 
            # instance so that the progress can be tracked in the command line.
            task = progress.add_task(f'{file_name}', total=total_size)
            
            # 10.06.24
            # If a folder path is given we directly open a new file object in that folder
            # and store the downloaded information there. However, if there is no folder path 
            # given we want to return the downloaded content as a bytes object.
            
            # This distinction is new and we do this here because we want to consider 2 cases:
            # - For very large download sizes we really dont want to have to store the entire 
            #   file content into the memory of the system. Instead we want to write it directly
            #   to the disk to be more efficient.
            # - For very small download sizes there is the opposite argument: To be more efficient 
            #   we directly return the content as a bytes object instead of writing it to the disk
            #   first.
            if folder_path is None:
                file = io.BytesIO()
            else:
                file_path = os.path.join(folder_path, file_name)
                file = open(file_path, mode='wb')    
            
            with file:
                bytes_written = 0
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        file.write(chunk)
                        progress.update(task, advance=bytes_written)
                        bytes_written += len(chunk)
            
                # If there is no file path given, we return the content of the file as a bytes 
                # object.
                if folder_path is None:
                    return file.getvalue()
        
        elif response.status_code == 404:
            raise FileNotFoundError(f'File not found on the server: {file_url}')
        
        else:
            raise Exception(f'Failed to download file from the server: {file_url}')
        
    def check_metadata(self) -> None:
        if self.metadata is None:
            raise LookupError('Metadata has not been fetched yet! Call fetch_metadata first!')
        
    # ~ implement dict-like behaviour
        
    def keys(self) -> t.List[str]:
        return self.metadata.keys()
        
    def __contains__(self, key: str) -> bool:
        
        self.check_metadata()
        return self.metadata[key]
        
    def __getitem__(self, key: str) -> t.Any:
        
        self.check_metadata()
        return self.metadata[key]