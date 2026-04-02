"""
This module provides the API function for loading transition metal complex (TMC) datasets
in their decomposed tabular format.

Unlike organic molecule datasets which use a single SMILES column, TMC datasets use a
decomposed representation with separate columns for the metal center, ligand SMILES,
connecting atom indices, and coordination metadata. The :func:`load_tmc_dataset` function
handles downloading, caching, and parsing this richer format.

Example usage:

.. code-block:: python

    from chem_mat_data.tmc import load_tmc_dataset

    df = load_tmc_dataset('tmqmg')
    print(df.columns)
    # ['complex_id', 'metal', 'oxidation_state', 'total_charge',
    #  'spin_multiplicity', 'coordination_number', 'ligand_smiles',
    #  'connecting_atom_indices', 'geometry', ...]
"""
import json
import logging
import tempfile
from typing import Optional, List

import pandas as pd

from chem_mat_data.config import Config
from chem_mat_data.main import ensure_dataset, load_dataset_metadata

logger = logging.getLogger(__name__)


# Columns that contain JSON-encoded data and need to be parsed from strings into
# Python lists after loading the CSV.
JSON_COLUMNS: List[str] = [
    'ligand_smiles',
    'connecting_atom_indices',
]


def load_tmc_dataset(dataset_name: str,
                     folder_path: str = tempfile.gettempdir(),
                     config: Optional[Config] = None,
                     use_cache: bool = True,
                     ) -> pd.DataFrame:
    """
    Load a transition metal complex dataset in decomposed tabular format.

    Downloads (if needed) and returns a pandas DataFrame containing the TMC dataset.
    The decomposed format includes separate columns for the metal center identity,
    ligand SMILES strings, connecting atom indices, and coordination metadata, along
    with target property columns specific to each dataset.

    Columns containing JSON-encoded lists (``ligand_smiles``, ``connecting_atom_indices``)
    are automatically parsed from strings into Python lists.

    :param dataset_name: The unique string identifier of the TMC dataset
        (e.g., ``'tmqmg'``).
    :param folder_path: The absolute path to the folder where the dataset files
        should be stored. Defaults to the system temporary directory.
    :param config: An optional :class:`Config` object for file share configuration.
        If ``None``, the default configuration is used.
    :param use_cache: Whether to use the local cache for previously downloaded datasets.

    :returns: A pandas DataFrame with TMC data in decomposed format. JSON columns
        (``ligand_smiles``, ``connecting_atom_indices``) are parsed into Python lists.
    """
    # Soft validation: warn if the dataset metadata indicates this is not a TMC dataset
    # or if the installed version is too old for the dataset's requirements.
    try:
        metadata = load_dataset_metadata(dataset_name, config=config)
        category = metadata.get('category', 'organic')
        if category != 'tmc':
            logger.warning(
                f'Dataset "{dataset_name}" has category "{category}", not "tmc". '
                f'You may want to use load_smiles_dataset() instead.'
            )

        min_ver = metadata.get('min_version')
        if min_ver:
            from chem_mat_data.utils import check_version_compatible, get_version
            if not check_version_compatible(min_ver):
                logger.warning(
                    f'Dataset "{dataset_name}" requires chem_mat_data >= {min_ver}, '
                    f'but you have {get_version()}. The dataset may not load correctly.'
                )
    except Exception:
        pass  # Metadata may not be available (e.g., local-only datasets)

    file_path = ensure_dataset(
        dataset_name=dataset_name,
        extension='csv',
        config=config,
        use_cache=use_cache,
        folder_path=folder_path,
    )

    df = pd.read_csv(file_path)

    # Parse JSON-encoded columns from strings into Python lists.
    for col in JSON_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(_safe_json_parse)

    return df


def _safe_json_parse(value):
    """
    Parse a JSON string into a Python object. Returns the original value
    if parsing fails or the value is already a non-string type.
    """
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return value
    return value
