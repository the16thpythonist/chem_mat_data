import os
from typing import List, Dict
import pandas as pd
import gzip
import shutil
from rdkit import Chem
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

from chem_mat_data.config import Config
from chem_mat_data.web import NextcloudFileShare
from chem_mat_data.main import get_file_share

# :param DATASET_NAME:
#       This is the name of the dataset that will be used to identify the dataset in the
#       file share server. It will also be used to create the folder structure for the dataset
#       on the file share server.
DATASET_NAME: str = 'skin_irritation'
# :param SMILES_COLUMN:
#       This is the string name of the CSV column which contains the SMILES strings of
#       the molecules.
SMILES_COLUMN: str = 'SMILES'
# :param TARGET_COLUMNS:
#       This is a list of string names of the CSV columns which contain the target values
#       of the dataset. This can be a single column for regression tasks or multiple columns
#       for multi-target regression or classification tasks. For the final graph dataset
#       the target values will be merged into a single numeric vector that contains the
#       corresponding values in the same order as the column names are defined here.
TARGET_COLUMNS: List[str] = ['label']
# :param DATASET_TYPE:
#       Either 'regression' or 'classification' to define the type of the dataset. This
#       will also determine how the target values are processed.
DATASET_TYPE: str = 'classification'
# :param DESCRIPTION:
#       This is a string description of the dataset that will be stored in the experiment
#       metadata.
DESCRIPTION: str = (
    'The skin irritation dataset contains 1,012 curated compounds focused on predicting the skin irritation '
    'and corrosion potential of small organic molecules. Experimental animal data were retrieved from the '
    'Registration, Evaluation, Authorization and Restriction of Chemicals (REACH) study results database and '
    'curated as part of the STopTox study by Borba et al. (2022), published in Environmental Health Perspectives. '
    'The dataset comprises 317 skin irritants and 695 non-irritants, providing a binary classification benchmark '
    'for developing in silico alternatives to animal testing. This dataset supports the development of QSAR models '
    'and machine learning approaches for predicting topical toxicity endpoints, contributing to the 3Rs principles '
    '(Replacement, Reduction, and Refinement) in chemical safety assessment.'
)
# :param METADATA:
#       A dictionary which will be used as the basis for the metadata that will be added
#       as additional information to the file share server.
METADATA: dict = {
    'license': {
        'identifier': 'CC-BY-4.0',
        'url': 'https://creativecommons.org/licenses/by/4.0/',
        'note': (
            'CC BY 4.0 is stated on the source publication - Borba, J. V. B. et al., '
            '"STopTox: An in-Silico Alternative to Animal Testing", Environ. Health '
            'Perspect. 130, 027012 (2022), doi:10.1289/EHP9341. The publisher renders '
            '"This publication is licensed under a Creative Commons Attribution 4.0 '
            'International License" in the permissions block immediately above the '
            'Supporting Information listing, and the version of record in PMC '
            '(PMC8863177) carries the same statement as a NISO ALI license_ref. The '
            "datasets are that article's supporting material, distributed via "
            'ChemPharos (db.chempharos.eu), stoptox.mml.unc.edu and '
            'github.com/joyvb/stoptox. Attribution required, and CC BY also requires '
            'that modification be indicated: MODIFIED - reformatted into the graph '
            'representation. UPSTREAM: a substantial part of the underlying study '
            "results comes from ECHA's REACH registered-substances dissemination "
            'portal, whose Legal Notice permits reuse only subject to conditions and '
            'excludes material subject to third-party intellectual property rights. '
            'The STopTox authors did not hold rights over those dossiers, so the '
            'article licence cannot clear that layer. The maintainer decided on '
            '2026-08-19 to treat the article licence as governing for both STopTox '
            'datasets, recording the ECHA position here rather than holding the data '
            'back. Note: ehp.niehs.nih.gov is no longer resolvable - EHP has migrated '
            'onto the ACS platform, so the licence_ref deposited in the article '
            'metadata points at a dead host; use the DOI. This is the skin '
            'irritation/corrosion endpoint, the sibling dataset from the same paper '
            'and the same distribution; its own licence research returned no '
            'statement at any layer, and the label is applied on the basis of the '
            'shared source publication.'
        ),
    },
    'verbose': 'Skin Irritation and Corrosion',
    'tags': [
        'Molecules',
        'SMILES',
        'Biology',
        'Toxicity',
        'Skin Irritation',
        'REACH',
        'QSAR',
        'Alternative Methods',
    ],
    'sources': [
        'https://db.chempharos.eu/datasets/Datasets.zul?datasetID=ds14',
        'https://ehp.niehs.nih.gov/doi/10.1289/EHP9341',
        'https://pubmed.ncbi.nlm.nih.gov/35192406/',
        'https://pmc.ncbi.nlm.nih.gov/articles/PMC8863177/',
        'https://stoptox.mml.unc.edu/',
    ],
    'target_descriptions': {
        '0': 'Non-irritant - Compound does not cause skin irritation or corrosion',
        '1': 'Skin irritant - Compound causes skin irritation or corrosion based on experimental data',
    },
}

__TESTING__ = False

experiment = Experiment.extend(
    'create_graph_datasets.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)


@experiment.hook('add_graph_metadata', default=False, replace=True)
def add_graph_metadata(e: Experiment, data: dict, graph: dict) -> dict:
    """
    We add the compound id for identification and the molecular weight
    """
    # graph['graph_name'] = data['Name']
    graph['graph_subset'] = data['dataset']


@experiment.hook('load_dataset', default=False, replace=True)
def load_dataset(e: Experiment) -> Dict[int, dict]:

    ## -- Load Dataset --
    e.log('Loading the EXCEL file from the remote file share server...')
    config = Config()
    file_share: NextcloudFileShare = get_file_share(config)
    file_path: str = file_share.download_file(
        'skin_irritation_dataset.xlsx', folder_path=e.path
    )
    df: pd.DataFrame = pd.read_excel(file_path)
    print(df.head())

    ## -- Save Dataset --
    e.log('Saving the dataset as CSV and GZipped CSV file...')
    csv_path = os.path.join(e.path, f'{e.DATASET_NAME}.csv')
    df.to_csv(csv_path, index=False)

    gz_path = csv_path + '.gz'
    with open(csv_path, 'rb') as f_in, gzip.open(gz_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

    ## -- Processing Dataset --
    dataset: Dict[int, dict] = {}
    index: int = 0
    for data in df.to_dict('records'):
        data['smiles'] = data[e.SMILES_COLUMN]

        ## -- Molecule Filters --
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

        ## -- Target Values --
        # In this dataset we only have one target
        data['targets'] = [0, 1] if data['Label'] else [1, 0]
        dataset[index] = data

        index += 1

    return dataset


experiment.run_if_main()
