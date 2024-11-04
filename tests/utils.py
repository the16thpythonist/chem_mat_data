import os
import pathlib

PATH = pathlib.Path(__file__).parent.absolute()

ASSETS_PATH = os.path.join(PATH, 'assets')
ARTIFACTS_PATH = os.path.join(PATH, 'artifacts')


def get_mock_dataset_path(name: str = 'clintox.mpack') -> str:
    return os.path.join(ASSETS_PATH, name)
