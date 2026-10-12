"""
Processing of the experimental chromophore optics database of Joung et al. (Sci. Data 2020) into the graph format.

The database collects experimental optical properties of organic chromophores from the literature: absorption and
emission maxima, their bandwidths, extinction coefficients, photoluminescence quantum yields and fluorescence
lifetimes, each measured in a specific solvent (or in the solid state). Every element of the processed dataset is one
chromophore-solvent combination; the solvent SMILES and the literature DOI are stored as graph metadata. Properties
that were not reported for a combination are NaN.
"""
import csv
import math
from typing import Dict, List

import rdkit.Chem as Chem
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path

from chem_mat_data.connectors import FigshareSource

# :param DATASET_NAME:
#       This string determines the name of the message pack dataset file that is then
#       stored into the "results" folder of the experiment as the result of the
#       processing process. The corresponding file extensions will be added
#       automatically.
DATASET_NAME: str = 'chromophore_db'
# :param FIGSHARE_FILE_ID:
#       Figshare file id of the CSV file ("DB for chromophore", version 3, CC BY 4.0).
FIGSHARE_FILE_ID: int = 49887156
# :param SOURCE_COLUMNS:
#       CSV columns used as targets, in this order.
SOURCE_COLUMNS: List[str] = [
    'Absorption max (nm)',
    'Emission max (nm)',
    'Lifetime (ns)',
    'Quantum yield',
    'log(e/mol-1 dm3 cm-1)',
    'abs FWHM (cm-1)',
    'emi FWHM (cm-1)',
]
# :param TARGET_COLUMNS:
#       Column names of the targets in the saved CSV file.
TARGET_COLUMNS: List[str] = [
    'absorption_max_nm',
    'emission_max_nm',
    'lifetime_ns',
    'quantum_yield',
    'log_extinction_coefficient',
    'absorption_fwhm_cm-1',
    'emission_fwhm_cm-1',
]
# :param METADATA:
#       A dictionary which will be used as the basis for the metadata that will be added
#       as additional information to the file share server.
METADATA: dict = {
    'verbose': 'Experimental Optical Properties of Organic Chromophores',
    'description': (
        'Experimental optical properties of about 7,000 organic chromophores in about 360 solvents or in the solid '
        'state, compiled from the literature by Joung et al. (Sci. Data 2020): first absorption and emission maxima '
        '(nm), absorption and emission bandwidths (FWHM, cm-1), molar extinction coefficients (log10 of '
        'mol-1 dm3 cm-1), photoluminescence quantum yields and fluorescence lifetimes (ns). Each element is one '
        'chromophore-solvent combination (about 20,000), with the solvent SMILES and the source DOI stored as graph '
        'metadata; properties not reported for a combination are NaN. Combinations whose chromophore SMILES contains '
        'several components (e.g. salts) are not included.'
    ),
    'target_type': ['regression'],
    'tags': [
        'SMILES',
        'Molecules',
        'Experimental',
        'Optical Properties',
        'Absorption',
        'Fluorescence',
        'Chromophores',
        'Dyes',
    ],
    'sources': [
        'https://doi.org/10.1038/s41597-020-00634-8',
        'https://figshare.com/articles/dataset/DB_for_chromophore/12045567',
    ],
    'target_descriptions': {
        0: 'absorption_max_nm - wavelength of the first absorption maximum in nm',
        1: 'emission_max_nm - wavelength of the emission maximum in nm',
        2: 'lifetime_ns - fluorescence lifetime in ns',
        3: 'quantum_yield - photoluminescence quantum yield (0-1)',
        4: 'log_extinction_coefficient - log10 of the molar extinction coefficient in mol-1 dm3 cm-1',
        5: 'absorption_fwhm_cm-1 - full width at half maximum of the absorption band in cm-1',
        6: 'emission_fwhm_cm-1 - full width at half maximum of the emission band in cm-1',
    },
}

__TESTING__ = False

experiment = Experiment.extend(
    'create_graph_datasets.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)


def to_float(value: str) -> float:
    """
    Converts a CSV cell to float, returning NaN for empty or non-numeric cells.

    :param value: The raw cell content.

    :returns: The float value or NaN.
    """
    try:
        x = float(value)
    except (TypeError, ValueError):
        return float('nan')
    return x if math.isfinite(x) else float('nan')


@experiment.hook('add_graph_metadata', default=False, replace=True)
def add_graph_metadata(e: Experiment, data: dict, graph: dict) -> dict:
    """
    Stores the row index of the source CSV as graph id, plus the solvent SMILES and the reference DOI.
    """
    graph['graph_id'] = data['tag']
    graph['graph_solvent'] = data['solvent']
    graph['graph_reference'] = data['reference']


@experiment.hook('load_dataset', default=False, replace=True)
def load_dataset(e: Experiment) -> Dict[int, dict]:
    """
    Downloads the chromophore CSV from Figshare and converts every valid chromophore-solvent row into a dataset element.

    :returns: A dict mapping integer indices to dicts with the keys 'smiles', 'targets', 'solvent', 'reference', 'tag'.
    """
    with FigshareSource(file_id=e.FIGSHARE_FILE_ID, verbose=True) as source:
        path = source.fetch()
        with open(path, newline='') as file:
            rows = list(csv.DictReader(file))
    e.log(f'downloaded {len(rows)} chromophore-solvent rows')

    dataset: Dict[int, dict] = {}
    index = 0
    for row in rows:
        smiles = (row.get('Chromophore') or '').strip()
        if not smiles or '.' in smiles:
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms() < 2:
            continue
        targets = [to_float(row.get(col)) for col in e.SOURCE_COLUMNS]
        if all(math.isnan(t) for t in targets):
            continue
        dataset[index] = {
            'smiles': smiles,
            'targets': targets,
            'solvent': row.get('Solvent', ''),
            'reference': row.get('Reference', ''),
            'tag': int(row['Tag']) if row.get('Tag', '').isdigit() else index,
        }
        index += 1

    return dataset


experiment.run_if_main()
