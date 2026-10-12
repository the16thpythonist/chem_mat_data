import pandas as pd
import rdkit.Chem as Chem
from typing import Dict
from rich.pretty import pprint
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

from chem_mat_data.connectors import FigshareSource
from chem_mat_data.data import HOPV15Parser

DATASET_NAME: str = 'hopv15_exp'
DESCRIPTION: str = (
    'The Harvard Organic Photovoltaic Dataset (HOPV15) is a comprehensive collection of '
    'experimental photovoltaic data from the literature for 350 different organic solar '
    'cell donor structures, combined with corresponding quantum-chemical calculations '
    'performed over multiple molecular conformers. The dataset includes both experimental '
    'measurements (such as power conversion efficiency, HOMO/LUMO energies, and other '
    'photovoltaic characteristics) and computational results using various density '
    'functionals and basis sets. This resource is designed to enable calibration of '
    'quantum chemical calculations to experimental observations, support the development '
    'of new computational methodologies, and benchmark current and future model chemistries '
    'for organic electronic applications.'
)

# :param METADATA:
#       A dictionary which will be used as the basis for the metadata that will be added
#       as additional information to the file share server.
METADATA: dict = {
    'tags': [
        'Molecules',
        'SMILES',
        'Organic Electronics',
        'Solar Cells',
    ],
    'verbose': 'Harvard Organic Photovoltaic Dataset Experimental',
    'sources': [
        'https://chemrxiv.org/engage/chemrxiv/article-details/64be471cb605c6803b425da6'
    ],
    # TADF: Thermally Activated Delayed Fluorescence
    'target_descriptions': {
        '0': 'homo - Highest occupied molecular orbit in eV',
        '1': 'lumo - Lowest unoccupied molecular orbit in eV',
        '2': 'gap - Gap between HOMO and LUMO energy levels in eV',
        '3': 'JSC - Closed circuit current in mA/cm2',
        '4': 'VOC - Open circuit voltage in V',
        '5': 'PCE - Power conversion efficience in percent',
    },
    'license': {
        'identifier': 'CC-BY-4.0',
        'url': 'https://creativecommons.org/licenses/by/4.0/',
        'note': (
            'The Figshare data record (doi.org/10.6084/m9.figshare.1610063), which hosts the '
            'HOPV_15_revised_2.data file this script downloads, states CC BY 4.0 on the record '
            'itself; the accompanying Data Descriptor (Lopez et al., Sci. Data 3, 160086, 2016) '
            'is independently CC BY 4.0. Redistribution of a processed derivative is permitted '
            'provided attribution is given and changes are indicated; there is no upstream '
            'database restriction, as HOPV15 is a hand-curated literature compilation rather '
            'than an extract of a structured database.'
        ),
    },
}

experiment = Experiment.extend(
    'create_graph_datasets.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)


@experiment.hook('load_dataset', default=False, replace=True)
def load_dataset(e: Experiment) -> Dict[int, dict]:

    # This file contains the dataset in question. It is available as a download link from a
    # Chemarxiv paper.
    e.log('Downloading dataset...')
    with FigshareSource(file_id=4513735, verbose=True) as source:
        path = source.fetch()

        parser = HOPV15Parser(path=path)
        mol_tuples = parser.parse_all()
        e.log(f'Downloaded dataset with {len(mol_tuples)} entries')

        pprint(mol_tuples[:3], max_depth=3)
        pprint(mol_tuples[0][1]['experimental_properties'], max_depth=3)

        # Convert the mol_tuples into a dataframe format
        df_data = []
        for mol, data in mol_tuples:
            smiles = data.get('smiles')
            # Create a row with SMILES and experimental properties
            if data.get('experimental_properties'):
                row = {
                    'smiles': smiles,
                    **data[
                        'experimental_properties'
                    ],  # Flatten experimental properties into the row
                }
                df_data.append(row)
            else:
                e.log(f'No experimental properties for SMILES: {smiles}')

        df = pd.DataFrame(df_data)
        print(df.head())
        e.log(f'Created dataframe with {len(df)} rows and columns: {list(df.columns)}')

    # Now that we have the raw data as a dataframe, we can convert it into the expected
    # dictionary format where each entry in the dictionary corresponds to a single data point
    # in the dataset.
    e.log('Processing dataset...')
    dataset: Dict[int, dict] = {}
    for index, data in enumerate(df.to_dict('records')):
        data['smiles'] = data['smiles']

        # == MOLECULE FILTERS ==
        # We don't want to use compounds with '.' in the smiles (separate molecules)
        if '.' in data['smiles']:
            continue

        # We don't want to use compounds that only consist of a single atom
        mol = Chem.MolFromSmiles(data['smiles'])
        if not mol:
            continue

        # We also don't want to accept "molecules" that are essentially just individual atoms
        if len(mol.GetAtoms()) < 2:
            continue

        # == TARGET VALUES ==
        # Only one target. We combine the experimental and computational data into a single list
        data['targets'] = [
            data['HOMO'],
            data['LUMO'],
            data['gap'],
            data['JSC'],
            data['VOC'],
            data['PCE'],
        ]

        dataset[index] = data

    return dataset


experiment.run_if_main()
