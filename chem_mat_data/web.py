import os
import shutil
import gzip
import tempfile
import typing as t
from typing import Optional, Dict, Tuple

import io
import requests.adapters
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
    
    :param url: The publically shared url from which the datasets can be downloaded.
    :param dav_url: The url to the nextcloud server DAV endpoint of the specific folder that contains 
        the shared datasets.
    :param dav_username: The username that is used to authenticate against the DAV endpoint of the
        nextcloud server.
    :param dav_password: The password that is used to authenticate against the DAV endpoint of the
        nextcloud server.
    :param use_download_url_path: If set to True, the files will be downloaded from the share point 
        using the URL directly instead of using the URL parameters to define the relative path 
        (legacy behaviour). This may be set to False for older versions of nextcloud.
    """
    def __init__(self, 
                 url: str,
                 dav_url: Optional[str] = None,
                 dav_username: Optional[str] = None,
                 dav_password: Optional[str] = None,
                 verify: bool = False,
                 use_download_url_path: bool = True,
                 **kwargs,
                 ) -> None:
        
        AbstractFileShare.__init__(self, url)
        
        self.dav_url = dav_url
        self.dav_username = dav_username
        self.dav_password = dav_password
        self.verify = verify
        self.use_download_url_path = use_download_url_path

        ## -- Derived Attributes --
        
        # The given URL will be in the format of a publically shared link to a folder on the nextcloud 
        # served in the format of "https://nextcloud.example.com/s/abc123xyz456"
        # We want to separate this into the base URL of the nextcloud server and the share token
        # Parse the given URL and reconstruct only the scheme and netloc (base URL)
        base_url, share_token = self.url.split('/s/', 1)
        self.base_url = base_url.rstrip('/')
        self.share_token = share_token.replace('/download', '').rstrip('/')

        ## -- Metadata-related Attributes --
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
    
    def construct_download_url(self, 
                               file_name: str,
                               ) -> str:
        """
        Returns the direct download URL for a file stored on the file share server.

        :param file_name: The name of the file to be downloaded, including any subdirectory structure
            and file extension, relative to the root of the shared folder.
        
        :returns: The fully constructed download URL as a string. This URL can be used in a browser or
            with command-line tools (such as wget or curl) to download the file directly from the server.
            The format is: ``{self.url}/{file_name}?files={file_name}``
        
        Example::
            
            share = NextcloudFileShare(url="https://nextcloud.example.com/s/abc123xyz456")
            url = share.construct_download_url("mydata.mpack.gz")
            # url == 'https://nextcloud.example.com/s/abc123xyz456/mydata.mpack.gz?files=mydata.mpack.gz'
        
        Note:
            The exact format of the download URL may depend on the server's configuration and sharing mechanism.
            This method assumes the server accepts the constructed URL for direct file downloads. If the server
            uses a different API or requires additional parameters, this method may need to be adapted accordingly.
        """
        download_url = (
            f'{self.url}/{file_name}?files={file_name}'
        )
        
        return download_url
    
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
        
        # This method will construct the download URL for the given file name 
        file_url: str = self.construct_download_url(file_name)
        
        # Create a session with no connection pooling
        session = requests.Session()
        session.keep_alive = False
        
        response = session.get(
            file_url, 
            stream=True, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; rv:91.0) Gecko/20100101 Firefox/91.0', 'Connection': 'close'},
            verify=self.verify,
            timeout=5,
        )
        
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
                for chunk in response.iter_content(chunk_size=1024**2):
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

    def exists(self,
               file_name: str,
               ) -> Tuple[bool, dict]:
        """
        Checks whether a file with the given ``file_name`` exists on the remote Nextcloud file share server.

        This method sends a HEAD request to the Nextcloud WebDAV endpoint to determine if the specified file exists. If the file exists, it returns True and a dictionary containing basic metadata about the file (such as size, last modified date, and content type) as obtained from the response headers. If the file does not exist, it returns False and an empty dictionary.

        :param file_name: The name of the file to check for existence on the remote server. This should include any subdirectory structure and file extension, relative to the root of the shared folder.

        :returns: A tuple (exists, metadata) where:
            - exists (bool): True if the file exists on the server, False otherwise.
            - metadata (dict): If the file exists, a dictionary with the following keys:
                - 'size': The size of the file in bytes (as a string), or 'unknown' if not available.
                - 'last_modified': The last modified date of the file (as a string), or 'unknown' if not available.
                - 'content_type': The MIME type of the file (as a string), or 'unknown' if not available.
              If the file does not exist, this will be an empty dictionary.

        :raises AssertionError: If the required DAV credentials (dav_url, dav_username, dav_password) are not set on the object.
        :raises Exception: If the server returns an unexpected status code or there is a network error.

        Example::

            share = NextcloudFileShare(
                url="https://nextcloud.example.com/s/abc123xyz456",
                dav_url="https://nextcloud.example.com/remote.php/dav/files/username/shared_folder",
                dav_username="myuser",
                dav_password="mypassword"
            )
            exists, meta = share.exists("mydata.mpack.gz")
            if exists:
                print(f"File exists! Size: {meta['size']} bytes, Last modified: {meta['last_modified']}")
            else:
                print("File does not exist on the server.")
        """
        # This method will simply check if the file share object instance as it is has enough 
        # information / permission to actually upload files to the remote fileshare. 
        # Will raise an error if that is not the case.
        self.assert_dav(action='upload')
        
        ## -- Making Request --
        # The check for the existence of a file on a remote nextcloud file share server, one 
        # simply needs to make a HEAD request to the DAV endpoint.
        file_url = f'{self.dav_url}/{file_name}'
        response = requests.head(
            file_url,
            auth=HTTPBasicAuth(
                self.dav_username,
                self.dav_password,
            ),
            verify=self.verify,
            timeout=1.0,
        )
        
        ## -- Handling Response --
        # If the HEAD request was successful, that means that the file exists on the server and 
        # we can infer some basic metadata about the file from the response headers. 
        if response.status_code == 200:
            
            size = response.headers.get('Content-Length', 'unknown')
            last_modified = response.headers.get('Last-Modified', 'unknown')
            content_type = response.headers.get('Content-Type', 'unknown')
            
            return True, {
                'size': size,
                'last_modified': last_modified,
                'content_type': content_type,
            }
            
        else:
            
            return False, {}
        

    def upload(self,
               file_name: str,
               file_path: str,
               ) -> None:
        """
        Given the ``file_name`` that a file should have on the server and the ``file_path`` to the 
        local version of the file, this method uploads the file to the remote file share server, 
        using the DAV endpoint of the server.
        
        :param file_name: The string name of the file that the file should have on the server.
        :param file_path: The string path to the local file that should be uploaded
        
        :returns: None
        """
        # This method will simply check if the file share object instance as it is has enough 
        # information / permission to actually upload files to the remote fileshare. 
        # Will raise an error if that is not the case.
        self.assert_dav(action='upload')
        
        upload_url = f'{self.dav_url}/{file_name}'
        with open(file_path, 'rb') as file:
            
            response = requests.put(
                upload_url,
                data=file,
                auth=HTTPBasicAuth(
                    self.dav_username,
                    self.dav_password,
                ),
                verify=self.verify,
            )
            
            # If something went wrong during the upload, we raise an error.
            response.raise_for_status()

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
