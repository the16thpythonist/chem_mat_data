import os
import shutil
import gzip
import tempfile
import typing as t
from typing import Optional, Dict

import io
import yaml
import requests
from requests.auth import HTTPBasicAuth
from rich.progress import Progress
from typing import Union


class MockProgress:
    
    def add_task(self, *args, **kwargs):
        return 0
    
    def update(self, *args, **kwargs):
        pass
    
    def start_task(self, *args, **kwargs):
        pass
    
    
class AbstractFileShare:
    """
    This class defines the abstract interface for a file share server. This interface defines the methods 
    which can be used to interact with the remote file share server solution. The concrete implementation 
    and storage solution that is used for the file share server is not defined here and should be
    implemented in a subclass.
    """
    
    def __init__(self, url: str):
        self.url = url
        self.metadata: Optional[dict] = None
    
    def fetch_metadata(self) -> dict:
        """
        This method should fetch the metadata from the remote file share server and return the metadata 
        as a dictionary that was loaded from the metadata yml file. This method should also populate the 
        "self.metadata" attribute of the object.
        """
        raise NotImplementedError()

    def upload_metadata(self, metadata: dict) -> None:
        """
        This method should take the metadata dictionary and upload that information to the remote file 
        server such that after the function is executed the remote file share information should be 
        updated.
        """
        raise NotImplementedError()

    def download_dataset(self,
                         dataset_name: str,
                         folder_path: str,
                         ) -> None:
        """
        This method should download the dataset which is identified by the unique string name and the 
        downloaded dataset should be placed in the given folder path.
        """
        raise NotImplementedError()
    
    def authenticate(self, auth_info: dict) -> None:
        """
        This method should be implemented so that after calling it the object should be authenticated 
        in the file server - meaning that it should be possible to make modifications to the files there
        This is mostely relevant for developers and not regular users of the package.
        """
        raise NotImplementedError()
    
    # ~ implement dict-like behaviour
    # Here we implement that the file share server instance acts as a dictionary regarding the metadata
        
    def assert_metadata(self):
        """
        This method checks if the metadata attribute of the object has already been loaded and 
        raises an error if that is not the case.
        """
        if self.metadata is None:
            raise LookupError('Metadata has not been fetched from the remote location yet!'
                              'Call the "fetch_metadata" method on the file share object first!')
        
    def keys(self) -> t.List[str]:
        self.assert_metadata()
        return self.metadata.keys()
        
    def __contains__(self, key: str) -> bool:
        self.assert_metadata()
        return self.metadata[key]
        
    def __getitem__(self, key: str) -> t.Any:
        self.assert_metadata()
        return self.metadata[key]


class NextcloudFileShare(AbstractFileShare):
    """
    This specific subblass implements the usage of a Nextcloud server as a file share solution to 
    download the remote datasets from.
    """
    def __init__(self, 
                 url: str,
                 dav_url: Optional[str] = None,
                 dav_username: Optional[str] = None,
                 dav_password: Optional[str] = None,
                 **kwargs,
                 ) -> None:
        
        AbstractFileShare.__init__(self, url)
        
        self.dav_url = dav_url
        self.dav_username = dav_username
        self.dav_password = dav_password

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
    
    def fetch_metadata(self, force: bool = True) -> dict:
        """
        Fetches the "metadata.yml" file from the remote file server, returns the metadata dict and 
        also stores it in the metadata attribute of the object. If the metadata file already exists
        on the local system, it will be loaded from there instead of being downloaded again.
        
        :param force: If set to True, the metadata file will be downloaded again even if it already
        
        :returns: None
        """
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
            
        return self.metadata
    
    def download_dataset(self, 
                         dataset_name: str, 
                         folder_path: str = tempfile.gettempdir(),
                         progress: Progress = MockProgress(),
                         ) -> str:
        """
        Given the string ``dataset_name`` of a dataset on the remote file share server, this method
        will download that dataset to the local system into the given ``folder_path``. Note that 
        the dataset name does NOT have to include the file extension!
        
        :param dataset_name: The string name of the dataset to be downloaded.
        :param folder_path: The string path to the folder where the downloaded dataset should be stored.
        :param progress: An optional Progress instance which will be used to track the download progress.
        
        :returns: The absolute path to the downloaded dataset on the local system.
        """
        
        if not dataset_name.endswith('.mpack'):
            dataset_name = dataset_name + '.mpack'
            
        file_name = dataset_name
            
        # 04.07.24
        # In the first instance we are going to try and download the compressed (gzip - gz) version 
        # of the dataset because that is usually at least 10x smaller and should therefore be a lot 
        # faster to download and only if that doesn't exist or fails due to some other issue we 
        # attempt to download the uncompressed version.
        try:
            file_name_compressed = f'{file_name}.gz'
            file_path_compressed = self.download_file(
                file_name_compressed,
                folder_path=folder_path,
                progress=progress,
            )
            
            # Then we can decompress the file using the gzip module. This may take a while.
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, mode='wb') as file:
                with gzip.open(file_path_compressed, mode='rb') as compressed_file:
                    shutil.copyfileobj(compressed_file, file)
        
        # Otherwise we try to download the file without the compression
        except Exception:
            file_path = self.download_file(
                file_name, 
                folder_path=folder_path,
                progress=progress,
            )
            
        return file_path
            
    def download_file(self,
                      file_name: str,
                      folder_path: str = tempfile.gettempdir(),
                      progress: Progress = MockProgress(),
                      ) -> str:
        """
        Given the string ``file_name`` of a file on the remote file share server, this method 
        will download that file to the local system into the given ``folder_path``. Note that 
        the given file name has to include a file extension!
        
        :param file_name: The string name of the file to be downloaded.
        :param folder_path: The string path to the folder where the downloaded file should be stored.
        :param progress: An optional Progress instance which will be used to track the download 
            progress.
            
        :returns: The absolute path to the downloaded file on the local system.
        """
        file_path = os.path.join(folder_path, file_name)
        self.download(file_name, progress=progress, folder_path=folder_path)
            
        return file_path
    
    # ~ utility methods
    
    def download(self, 
                 file_name: str, 
                 progress: Progress = MockProgress(),
                 folder_path: Union[str, None] = None,
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

    def upload(self,
               file_name: str,
               file_path: str,
               ) -> None:
        
        self.assert_dav(action='upload')
        
        upload_url = f'{self.dav_url}/{file_name}'
        with open(file_path, 'rb') as file:
            response = requests.put(
                upload_url,
                data=file,
                auth=HTTPBasicAuth(
                    self.dav_username,
                    self.dav_password,
                )
            )
            print(response.status_code)
            print(response.text)

    def assert_dav(self, action: str = '') -> None:
        
        assert self.dav_url is not None, (
            f'The requested action "{action}" requires extended access to the nextcloud server and therefore requires the '
            f'dav_url to be given to the NextcloudFileShare object. Please provide the dav_url parameter to the constructor '
            f'or by adding it to the config file.'
        )
        
        assert self.dav_username is not None, (
            f'The requested action "{action}" requires extended access to the nextcloud server and therefore requires the '
            f'dav_username to be given to the NextcloudFileShare object. Please provide the dav_username parameter to the '
            f'constructor or by adding it to the config file.'
        )
        
        assert self.dav_password is not None, (
            f'The requested action "{action}" requires extended access to the nextcloud server and therefore requires the '
            f'dav_password to be given to the NextcloudFileShare object. Please provide the dav_password parameter to the '
            f'constructor or by adding it to the config file.'
        )
        

# This dictionary assigns the string names of the different file share types to the actual classes 
# So given the name of a fileshare type, this map can be used to obtain to actual class to then 
# consruct a fileshare instance from.
FILESHARE_TYPES: Dict[str, type] = {
    'nextcloud': NextcloudFileShare,
}        



def construct_file_share(file_share_type: str,
                         file_share_url: str,
                         file_share_kwargs: dict,
                         ) -> AbstractFileShare:
    """
    Given the string name of a valid ``file_share_type`` and the ``file_share_url`` of the file share
    server, this function will construct an instance of the corresponding file share class which is 
    a subclass implementing the AbstractFileShare interface.
    
    :param file_share_type: The unique string name that identifies the type of file share server that 
        is to be used.
    :param file_share_url: The url that points to the actual file share server. This is a required 
        parameter for the construction of any file share.
    :param file_share_kwargs: A dictionary of additional keyword arguments that may be required for the
        construction of a specific file share subclass.
    """
    fileshare_class: type = FILESHARE_TYPES[file_share_type]
    fileshare: AbstractFileShare = fileshare_class(
        url=file_share_url,
        **file_share_kwargs,
    )
    return fileshare
    