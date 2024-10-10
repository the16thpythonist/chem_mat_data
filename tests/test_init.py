"""
In this file we simply test if importing from the main package name works correctly.
"""


def test_import_ensure_dataset():
    """
    It should be possible to import the ``ensure_dataset`` function directly using the 
    top-level package name.
    """
    from chem_mat_data import ensure_dataset
    assert callable(ensure_dataset)