"""
This experiment creates the processed graph dataset for **half_life** - aggregated soil
biotransformation half-lives (DT50) for 892 pesticides and pesticide transformation
products.

**Provenance and licensing**

The underlying measurements come from the EAWAG-SOIL package of enviPath. They were
compiled and aggregated by Hafner, Fenner & Scheidegger, who published the aggregation
as supplementary information to their paper.

That paper exists in two versions with *different* licenses, and this script
deliberately uses the published one:

- ChemRxiv preprint (``10.26434/chemrxiv-2023-604k3``) - **CC-BY-NC-ND-4.0**. The
  NoDerivatives term makes redistributing a processed derivative of it unsafe.
- ACS *Environ. Sci. Technol. Lett.* (``10.1021/acs.estlett.3c00526``) - **CC-BY-4.0**.

The file is fetched from Europe PMC rather than from pubs.acs.org, because the ACS
supplementary URL is behind a bot check that returns 403 to scripted clients, while
Europe PMC serves the identical publisher-deposited file as a plain ZIP.

Both versions carry the same data - the preprint's ``SI-II - By compound`` sheet and
the published ``Table S2`` hold the same 893 compounds with the same values, differing
only in sheet and column naming.

Note that enviPath's own terms restrict use to academic and non-commercial purposes and
forbid systematic redistribution of substantial portions of the database. The
aggregation redistributed here was released under CC-BY-4.0 by authors at Eawag, the
institute that maintains EAWAG-SOIL.

**Source**: Hafner, J.; Fenner, K.; Scheidegger, A. "Systematic Handling of
Environmental Fate Data for Model Development - Illustrated for the Case of
Biodegradation Half-Life Data". *Environ. Sci. Technol. Lett.* 2023, 10 (11),
1039-1044. DOI: 10.1021/acs.estlett.3c00526

**Usage**::

    python create_graph_datasets__half_life.py
"""

import os
import zipfile
from typing import Dict, List

import requests
import pandas as pd
import rdkit.Chem as Chem
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

# == SOURCE PARAMETERS ==

# :param DOWNLOAD_URL:
#       Europe PMC supplementary-files endpoint for the published article
#       (PMC10569042). Returns a ZIP of every publisher-deposited supplementary file.
#       Used in preference to the pubs.acs.org supplementary URL, which returns 403 to
#       scripted clients.
DOWNLOAD_URL: str = (
    'https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10569042/supplementaryFiles'
)

# :param ARCHIVE_MEMBER:
#       The supplementary workbook inside the ZIP. The other members are article
#       figures.
ARCHIVE_MEMBER: str = 'ez3c00526_si_001.xlsx'

# :param SHEET_NAME:
#       Selected by name rather than by index. The workbook has three sheets:
#       'Title page', 'Table S1' (6,309 individual half-life measurements) and
#       'Table S2' (893 compounds, aggregated). The aggregated sheet is the one that
#       corresponds to one row per molecule.
SHEET_NAME: str = 'Table S2'

# :param SMILES_COLUMN:
#       The SMILES column carries a long descriptive header in the published workbook,
#       so it is matched by prefix in the load_dataset hook rather than by literal name.
SMILES_COLUMN: str = 'SMILES'

# :param TARGET_COLUMNS:
#       The single target. 'Geometric mean DT50' is the geometric mean of all reported
#       half-lives for a compound, in days.
#
#       Note that the paper's own contribution is the Bayesian-inferred columns, which
#       handle censored measurements ("<0.1 days" / ">1000 days") properly, whereas
#       this geometric mean carries censoring artefacts at both ends of its range
#       (values down to 0.002 and a hard ceiling at exactly 9999). The geometric mean
#       is retained here because it is what this dataset has always published and
#       changing it would silently alter a dataset that users depend on.
TARGET_COLUMNS: List[str] = ['Geometric mean DT50']

# :param DATASET_TYPE:
DATASET_TYPE: str = 'regression'

# :param DESCRIPTION:
DESCRIPTION: str = (
    'The dataset consists of 6,309 experimental soil biotransformation half-life '
    'values for 893 pesticides and pesticide transformation products, extracted '
    'from publicly available regulatory reports via the EAWAG-SOIL package on enviPath, '
    'aggregated to one geometric mean half-life (DT50) per compound. Each substance can '
    'have multiple reported half-life values obtained from different experiments '
    'conducted under varying environmental conditions (different soils, temperatures, '
    'pH levels, etc.) and calculated using different kinetic models, resulting in '
    'substantial variability spanning several orders of magnitude for the same '
    'chemical. The dataset includes both uncensored half-life measurements and '
    'censored values (311 total) that fall beyond reliable quantification limits, '
    'either as very fast degradation (<0.1 days) or very slow degradation (>1000 '
    'days); the censored entries are reflected in the aggregate as values at the '
    'extremes of its range. The SMILES are canonicalised without stereochemistry and '
    'desalted, so stereoisomers are merged into a single entry.'
)

# :param METADATA:
METADATA: dict = {
    'license': {
        'identifier': 'CC-BY-NC-SA-4.0',
        'url': 'https://creativecommons.org/licenses/by-nc-sa/4.0/',
        'note': (
            'The values come from enviPath, which states: "The enviPath database and '
            'prediction system is provided under a Creative Commons license. '
            'Specifically, we offer the BBD, Soil, Sediment, and Sludge packages '
            'under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 '
            'International (CC BY-NC-SA 4.0) license" (community.envipath.org). This '
            'dataset draws on the EAWAG-SOIL package. NONCOMMERCIAL and SHAREALIKE: '
            'commercial use is not permitted, and adaptations must be redistributed '
            'under the same licence. The maintainer decided on 2026-08-19 that '
            'ChemMatData, as a non-commercial academic distribution, may redistribute '
            'NonCommercial-licensed data provided the condition is carried through to '
            'users via this field.'
        ),
    },
    'verbose': 'Half-Life Biotransformation',
    'tags': [
        'Molecules',
        'SMILES',
        'Environment',
        'Half-Life',
        'Experimental',
    ],
    'sources': [
        'https://doi.org/10.1021/acs.estlett.3c00526',
        'https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10569042/supplementaryFiles',
        'https://envipath.org',
    ],
    'target_descriptions': {
        '0': (
            'DT50 - Geometric mean of the reported experimental half-lives under '
            'aerobic soil conditions, in days'
        ),
    },
    'notes': (
        'Redistributed under CC-BY-4.0 from the published ACS version; cite Hafner, '
        'Fenner & Scheidegger, Environ. Sci. Technol. Lett. 2023, 10 (11), 1039-1044. '
        'The underlying measurements originate from the EAWAG-SOIL package of enviPath '
        '(https://envipath.org), whose terms restrict use to academic and '
        'non-commercial purposes. Values at the extremes of the target range are '
        'censoring artefacts rather than measurements.'
    ),
}

# == EXPERIMENT PARAMETERS ==

__TESTING__ = False

experiment = Experiment.extend(
    'create_graph_datasets.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)


@experiment.hook('load_dataset', default=False, replace=True)
def load_dataset(e: Experiment) -> Dict[int, dict]:
    """
    Download the published supplementary workbook (cached) and build the dataset.

    The download is a ZIP of every supplementary file deposited with the article; only
    the workbook is extracted. The aggregated per-compound sheet is selected by name,
    since the workbook also contains a 6,309-row sheet of individual measurements that
    would otherwise be easy to pick up by accident.

    :param e: The running experiment.

    :returns: Mapping of integer index to record dicts containing a ``smiles`` string
        and a ``targets`` list holding the geometric mean DT50 in days.
    """
    # -- download with caching --
    zip_path = os.path.join(e.path, 'supplementary_files.zip')
    if not os.path.exists(zip_path):
        e.log(f'downloading supplementary files from {DOWNLOAD_URL} ...')
        with requests.get(DOWNLOAD_URL, stream=True, timeout=300) as response:
            response.raise_for_status()
            with open(zip_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
        e.log(f'downloaded {os.path.getsize(zip_path) / 1024**2:.1f} MB')
    else:
        e.log(f'using cached archive @ {zip_path}')

    # -- extract the workbook --
    xlsx_path = os.path.join(e.path, ARCHIVE_MEMBER)
    if not os.path.exists(xlsx_path):
        with zipfile.ZipFile(zip_path) as archive:
            if ARCHIVE_MEMBER not in archive.namelist():
                raise FileNotFoundError(
                    f'{ARCHIVE_MEMBER} not in the archive. Members: {archive.namelist()}'
                )
            archive.extract(ARCHIVE_MEMBER, path=e.path)
        e.log(f'extracted {ARCHIVE_MEMBER}')

    df = pd.read_excel(xlsx_path, sheet_name=SHEET_NAME)
    e.log(f'loaded {len(df)} rows from sheet "{SHEET_NAME}"')

    # The SMILES column header is a long descriptive sentence in the published
    # workbook, so match it by prefix rather than by literal name.
    smiles_column = next(
        (column for column in df.columns if str(column).startswith(SMILES_COLUMN)),
        None,
    )
    if smiles_column is None:
        raise KeyError(
            f'no column starting with "{SMILES_COLUMN}" in sheet "{SHEET_NAME}". '
            f'Columns: {list(df.columns)}'
        )

    # -- build the dataset --
    e.log('processing dataset...')
    dataset: Dict[int, dict] = {}
    skipped_empty: int = 0
    skipped_invalid: int = 0
    skipped_multi: int = 0
    skipped_single_atom: int = 0

    for record in df.to_dict('records'):
        smiles = record.get(smiles_column)

        if not smiles or pd.isna(smiles):
            skipped_empty += 1
            continue

        # NOTE: the two filters below are retained from the original implementation of
        # this dataset so that the published compound count stays stable at 892. They
        # are heuristic rather than data-driven and would not be applied to a dataset
        # added today - see the filtering policy in the dataset-extraction skill.
        if '.' in smiles:
            skipped_multi += 1
            continue

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            skipped_invalid += 1
            continue

        if len(mol.GetAtoms()) < 2:
            skipped_single_atom += 1
            continue

        record['smiles'] = smiles
        record['targets'] = [float(record[column]) for column in TARGET_COLUMNS]

        dataset[len(dataset)] = record

    e.log(
        f'built dataset with {len(dataset)} molecules '
        f'({skipped_empty} without SMILES, {skipped_invalid} unparseable, '
        f'{skipped_multi} multi-component, {skipped_single_atom} single-atom)'
    )
    return dataset


experiment.run_if_main()
