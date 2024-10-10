import os
import tempfile

from rich.progress import Progress
from chem_mat_data.config import Config
from chem_mat_data.web import NextcloudFileShare


class TestNextcloudFileShare:
    
    def test_download_basically_works(self):
        
        config = Config()
        file_share = NextcloudFileShare(config.get_fileshare_url())
        
        content = file_share.download('metadata.yml')
        assert isinstance(content, bytes)
        assert len(content) != 0
        
    def test_download_progress_works(self):
        
        config = Config()
        file_share = NextcloudFileShare(config.get_fileshare_url())
        
        with Progress() as progress:
            content = file_share.download('metadata.yml', progress=progress)
            
            assert isinstance(content, bytes)
            assert len(content) != 0
        
    def test_download_file_basically_works(self):
        
        config = Config()
        file_share = NextcloudFileShare(config.get_fileshare_url())
        
        with tempfile.TemporaryDirectory() as folder_path:
            path = file_share.download_file('metadata.yml', folder_path)
            assert os.path.exists(path)
            
    def test_fetch_metadata_basically_works(self):
        
        config = Config()
        file_share = NextcloudFileShare(config.get_fileshare_url())
        file_share.fetch_metadata()
        assert 'datasets' in file_share
        
        