"""
This experiment module creates a graph dataset from the PFAS-Map classification dataset
for multi-class classification tasks.

The dataset contains ~6,000 per- and polyfluoroalkyl substances (PFAS) from the US EPA
master list, each classified into one of 14 structural categories (e.g. PFAAs,
fluorotelomer precursors, side-chain aromatics). The classification scheme follows
established PFAS nomenclature and was developed as part of the PFAS-Map framework for
rapid screening of structure-function relationships.

SMILES are provided as RDKit-canonicalized strings. The target is the ``First_Class``
label encoded as a one-hot vector over 14 classes (two rare error classes are excluded).

See also the ``pfas_ttr_binding`` dataset for TTR binding regression on a subset of
PFAS compounds from the same source.

References:
- Su, A. & Rajan, K., Sci. Data, 8 (2021) 14
  DOI: 10.1038/s41597-021-00798-x
"""

from collections import Counter
from typing import Dict, List

import pandas as pd
import rdkit.Chem as Chem
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

from chem_mat_data.connectors import FigshareSource

# == DATASET METADATA ==

DATASET_NAME: str = 'pfas_classify'

# Ordered list of the 14 meaningful First_Class labels.
# "Not PFAS" (4 entries) and "Unable to open ring(s)" (2 entries) are excluded
# as they represent errors / non-PFAS entries rather than real structural classes.
CLASS_LABELS: List[str] = [
    'Other aliphatics',
    'Side-chain aromatics',
    'Other aliphatics, cyclic',
    'Fluorotelomer PFAA precursors',
    'Silicon PFASs',
    'PFAAs',
    'Non-PFAA perfluoroalkyls',
    'Perfluoro PFAA precursors',
    'Non-PFAA perfluoroalkyls, cyclic',
    'PFAS derivatives',
    'FASA based PFAA precursors',
    'Perfluoro PFAA precursors, cyclic',
    'PFAAs, cyclic',
    'Fluorotelomer PFAA precursors, cyclic',
]

DESCRIPTION: str = (
    'PFAS structural classification dataset from the PFAS-Map framework. Contains ~6,000 '
    'per- and polyfluoroalkyl substances from the US EPA master list, each assigned to one '
    'of 14 structural classes following established PFAS nomenclature. Classes range from '
    'common categories like "Other aliphatics" (~3,000 compounds) to rare ones like '
    '"Fluorotelomer PFAA precursors, cyclic" (2 compounds). '
    'See also the pfas_ttr_binding dataset for TTR binding regression from the same source.'
)

METADATA: dict = {
    'tags': [
        'Molecules',
        'SMILES',
        'PFAS',
        'Classification',
        'Multi-class',
        'Environmental Chemistry',
        'EPA',
    ],
    'verbose': 'PFAS Structural Classification',
    'sources': [
        'https://doi.org/10.1038/s41597-021-00798-x',
        'https://doi.org/10.6084/m9.figshare.c.5043566',
    ],
    'notes': [
        'Multi-class classification: 14 PFAS structural categories.',
        'Target is one-hot encoded over 14 classes.',
        'Derived from US EPA PFAS master list (2020).',
        'Part of the PFAS-Map framework (Su & Rajan, 2021).',
        'Related dataset: pfas_ttr_binding (TTR binding regression).',
    ],
    'target_descriptions': {
        '0': 'Other aliphatics - Non-cyclic PFAS that do not fit specific subclasses (e.g. partially fluorinated alkanes, fluoroethers); largest class (~3,000 compounds)',
        '1': 'Side-chain aromatics - Aromatic compounds with fluorinated side chains (e.g. fluorinated benzenes, biphenyls with perfluoroalkyl substituents)',
        '2': 'Other aliphatics, cyclic - Cyclic PFAS not captured by specific subclasses (e.g. fluorinated cycloalkanes, heterocycles with fluorinated substituents)',
        '3': 'Fluorotelomer PFAA precursors - Telomer-based precursors that can degrade to PFAAs (e.g. n:2 fluorotelomer alcohols (FTOHs), fluorotelomer sulfonates)',
        '4': 'Silicon PFASs - Organosilicon compounds bearing perfluoroalkyl groups (e.g. fluorinated silanes and siloxanes)',
        '5': 'PFAAs - Perfluoroalkyl acids including perfluorocarboxylic acids (PFCAs like PFOA) and perfluorosulfonic acids (PFSAs like PFOS)',
        '6': 'Non-PFAA perfluoroalkyls - Fully fluorinated alkyl compounds that are not acids (e.g. perfluoroalkyl iodides, perfluoroalkanes)',
        '7': 'Perfluoro PFAA precursors - Fully fluorinated precursors that degrade to PFAAs (e.g. perfluoroalkyl sulfonamides (FASAs), perfluoroalkyl sulfonyl fluorides)',
        '8': 'Non-PFAA perfluoroalkyls, cyclic - Cyclic fully fluorinated compounds that are not acids (e.g. perfluorodecalin, cyclic perfluoroethers)',
        '9': 'PFAS derivatives - Modified PFAS with functional group substitutions (e.g. PFAS halogen derivatives, compounds with fluorinated C=C or C=O)',
        '10': 'FASA based PFAA precursors - Perfluoroalkane sulfonamide (FASA) derivatives that can degrade to PFAAs (e.g. N-alkyl FASEs, PASFs)',
        '11': 'Perfluoro PFAA precursors, cyclic - Cyclic fully fluorinated PFAA precursors (e.g. cyclic perfluoroalkyl sulfonamides)',
        '12': 'PFAAs, cyclic - Cyclic perfluoroalkyl acids (e.g. cyclic perfluorocarboxylic acids)',
        '13': 'Fluorotelomer PFAA precursors, cyclic - Cyclic telomer-based PFAA precursors; rarest class (2 compounds)',
    },
    'license': {
        'identifier': 'CC0-1.0',
        'url': 'https://creativecommons.org/publicdomain/zero/1.0/',
        'note': (
            'The Figshare data/software record (10.6084/m9.figshare.12584645, the '
            'PFAS-Map archive these files are downloaded from) is dedicated under '
            'CC0 1.0 Universal, independent of the CC-BY-4.0 license on the '
            'companion Sci. Data article. The underlying PFAS structures trace to '
            'the US EPA CompTox Chemicals Dashboard PFAS master list, which EPA '
            'states is open data free of all copyright restrictions for both '
            'non-commercial and commercial use, so no upstream obligation '
            'restricts redistribution.'
        ),
    },
}

# == DOWNLOAD SOURCE ==

FIGSHARE_FILE_ID = 25766582
CLASSIFICATION_CSV_PATH = 'PFAS-Map/processed_data/Precalculated_EPA_PFAS_pca_classification.csv'

# == EXPERIMENT PARAMETERS ==

DATASET_TYPE: str = 'classification'

__DEBUG__ = True
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
    Download and process the PFAS classification dataset.

    Uses :class:`~chem_mat_data.connectors.FigshareSource` to download the
    PFAS-Map ZIP archive from Figshare and extract the classification CSV.
    Validates SMILES with RDKit and returns compounds with one-hot encoded
    structural class labels as targets.

    Compounds with ``First_Class`` values of ``"Not PFAS"`` or
    ``"Unable to open ring(s)"`` are excluded as these represent data quality
    issues rather than real PFAS classes.

    :returns: Dictionary mapping indices to compound data dictionaries.
    """
    e.log('Downloading PFAS classification data from Figshare...')
    with FigshareSource(
        file_id=FIGSHARE_FILE_ID,
        relative_path=CLASSIFICATION_CSV_PATH,
        verbose=True,
    ) as source:
        csv_path = source.fetch()
        df = pd.read_csv(csv_path)

    e.log(f'Loaded CSV with {len(df)} rows')

    label_to_idx = {label: i for i, label in enumerate(CLASS_LABELS)}
    num_classes = len(CLASS_LABELS)

    dataset: Dict[int, dict] = {}
    index = 0
    skipped_class = 0

    for _, row in df.iterrows():
        smiles = str(row['RDKIT_SMILES']).strip()
        if not smiles:
            continue

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        first_class = row['First_Class']
        if first_class not in label_to_idx:
            skipped_class += 1
            continue

        class_idx = label_to_idx[first_class]
        targets = [int(i == class_idx) for i in range(num_classes)]

        dataset[index] = {
            'smiles': smiles,
            'targets': targets,
            'first_class': first_class,
            'second_class': row.get('Second_Class', ''),
        }
        index += 1

    e.log(f'Processed {len(dataset)} compounds ({skipped_class} skipped for excluded classes)')

    class_counts = Counter(d['first_class'] for d in dataset.values())
    for label in CLASS_LABELS:
        e.log(f'  {label}: {class_counts.get(label, 0)}')

    return dataset


@experiment.hook('add_graph_metadata', default=False, replace=True)
def add_graph_metadata(_e: Experiment, data: dict, graph: dict) -> None:
    """Add PFAS classification metadata to each graph."""
    graph['graph_first_class'] = data['first_class']
    graph['graph_second_class'] = data['second_class']


experiment.run_if_main()
