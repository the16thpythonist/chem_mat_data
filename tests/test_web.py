"""
Unittests for the "web.py" module that contains the functionality to communicate with the remote 
file share server that hosts the datasets.
"""
import os
import tempfile

from rich.progress import Progress
from chem_mat_data.config import Config
from chem_mat_data.web import NextcloudFileShare

from .utils import ASSETS_PATH


class TestNextcloudFileShare:
    """
    The ``NextcloudFileShare`` class is a wrapper around the Nextcloud API which is one option to 
    use as a file share server to host the datasets.
    """
    
    def test_download_basically_works(self):
        """
        The ``download`` method should be able to download a file from the remote file share server.
        """
        config = Config()
        file_share = NextcloudFileShare(config.get_fileshare_url())
        
        content = file_share.download('metadata.yml')
        assert isinstance(content, bytes)
        assert len(content) != 0
        
    def test_download_progress_works(self):
        """
        The ``download`` method should be able to show a progress bar while downloading a file from
        the remote file share server.
        """
        config = Config()
        file_share = NextcloudFileShare(config.get_fileshare_url())
        
        with Progress() as progress:
            content = file_share.download('metadata.yml', progress=progress)
            
            assert isinstance(content, bytes)
            assert len(content) != 0
        
    def test_download_file_basically_works(self):
        """
        The ``download_file`` method should download a file from the remote file share server and
        store it in the specified folder.
        """
        config = Config()
        file_share = NextcloudFileShare(config.get_fileshare_url())
        
        with tempfile.TemporaryDirectory() as folder_path:
            path = file_share.download_file('metadata.yml', folder_path)
            assert os.path.exists(path)
            
    def test_fetch_metadata_basically_works(self):
        """
        The ``fetch_metadata`` method should fetch the metadata.yml file from the remote file share 
        server and store the metadata then in the file share object as a dictionary.
        """
        config = Config()
        file_share = NextcloudFileShare(config.get_fileshare_url())
        metadata = file_share.fetch_metadata()
        assert isinstance(metadata, dict)
        assert 'datasets' in file_share
        
    def test_upload(self):
        """
        Given the DAV credentials, it should be possible to upload files to the nextcloud file share as well using 
        the ``upload`` method.
        """
        config = Config()
        file_share = NextcloudFileShare(
            config.get_fileshare_url(),
            config.get_fileshare_parameters('nextcloud'),
            verify=False,
        )
        file_path = os.path.join(ASSETS_PATH, 'test.txt')
        file_share.upload('test.txt', file_path)
        
        with tempfile.TemporaryDirectory() as folder_path:
            file_share.download_file('test.txt', folder_path)
            assert os.path.exists(os.path.join(folder_path, 'test.txt'))
        