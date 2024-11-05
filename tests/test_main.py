import os

from chem_mat_data.config import Config
from chem_mat_data.web import NextcloudFileShare
from chem_mat_data.main import get_file_share

from .utils import get_mock_dataset_path


def test_upload_nextcloud_fileshare_works():
    
    config = Config()
    file_share: NextcloudFileShare = get_file_share(config)
    assert isinstance(file_share, NextcloudFileShare)
    file_share.assert_dav()
    
    file_path = get_mock_dataset_path()
    file_name = os.path.basename(file_path)
    # file_share.upload(file_name, file_path)