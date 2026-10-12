"""
This experiment module creates the graph dataset for the curated *acidic* aqueous pKa
endpoint published alongside the open-source OPERA pKa models.

The dataset contains 3260 organic compounds, each with a single experimental
strongest-acidic pKa measured in water. It is the "Option 3" acidic set of Mansouri
et al. (2019) - the variant the paper reports as its final acidic data set - and is
delivered by the publisher as two members of a supplementary archive, a 2445-row
training split and an 815-row test split, which this module concatenates back into
the full endpoint.

**Source**: Mansouri, K., Cariello, N.F., Korotcov, A. et al., J Cheminform 11, 60
(2019). DOI: 10.1186/s13321-019-0384-1

**Targets**: strongest acidic pKa in water (dimensionless)

Two design decisions are worth explaining, because both are easy to get wrong by
doing the apparently obvious thing.

*Which SMILES column.* The source CSVs carry two, and both parse in RDKit without a
single failure across all 3260 rows - so nothing here announces itself as a problem.
``Canonical_QSARr`` holds the KNIME QSAR-ready structures the paper's own models were
trained on, which makes it look like the faithful choice. It is measurably damaged:
across the full set it holds 485 carboxylic acid groups against the original column's
966, and 96 gem-diols against 9. Spot checks show the loss is tautomer corruption
rather than deprotonation - the propanoate of record 12429 becomes the 1,1-enediol
``OC(O)=CCC1NC2C=CC=CC=2N=1``, and the alpha-amino acid of record 17128 becomes an
imine bearing a gem-diol. Formulae balance in both cases, which is exactly why the
damage is silent. This module therefore builds its graphs from ``OriginalSmiles``, the
hand-curated DataWarrior structures, and carries the QSAR-ready SMILES through to
``graph_canonical_qsar_smiles`` so the paper's exact model input remains recoverable.

*Why no missing-value sentinel.* This is a single-target dataset whose values are
genuinely negative for 34 compounds and span -7.15 to 20.50, so ``-1.0`` is an
ordinary measurement here and cannot double as "absent". Rows without a usable pKa
are skipped instead. In practice the source has none to skip.
"""

import io
import os
import zipfile
from typing import Dict, List

import requests
import pandas as pd
import rdkit.Chem as Chem
from rdkit import RDLogger
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

RDLogger.DisableLog('rdApp.*')

# == SOURCE PARAMETERS ==

# :param DATASET_URL:
#       Direct download URL of "Additional file 1" of the publication, a 5.1 MiB ZIP
#       holding the original and curated pKa data. The CSVs exist only inside this
#       archive - the publisher offers no per-file download - so the archive is
#       fetched whole and the two members of interest are read out of it in memory.
DATASET_URL: str = (
    'https://static-content.springer.com/esm/'
    'art%3A10.1186%2Fs13321-019-0384-1/MediaObjects/'
    '13321_2019_384_MOESM1_ESM.zip'
)

# :param ARCHIVE_MEMBERS:
#       The two archive members that together form the curated acidic endpoint, mapped
#       to the split label recorded for each of their rows. The archive also ships the
#       paper's Options 1 and 2 and the corresponding basic-pKa sets; taking the wrong
#       pair would silently yield a differently sized dataset, so the expected row
#       counts are asserted after loading.
ARCHIVE_MEMBERS: Dict[str, str] = {
    'Opt3_acidic_tr.csv': 'train',
    'Opt3_acidic_tst.csv': 'test',
}

# :param EXPECTED_ROWS:
#       Row count the concatenated members must have. Matches the final acidic data
#       set size stated in the publication (3260), which is the cheapest available
#       check that the archive still holds what the paper describes.
EXPECTED_ROWS: int = 3260

# :param SMILES_COLUMN:
#       The hand-curated DataWarrior structure, deliberately preferred over the
#       QSAR-ready column - see the module docstring.
SMILES_COLUMN: str = 'OriginalSmiles'

# :param TARGET_COLUMNS:
#       Names the single target column of the exported raw CSV.
TARGET_COLUMNS: List[str] = ['pKa']

# == DATASET METADATA ==

# :param DATASET_NAME:
DATASET_NAME: str = 'pka_acidic'

# :param DATASET_TYPE:
DATASET_TYPE: str = 'regression'

# :param DESCRIPTION:
DESCRIPTION: str = (
    'Curated set of 3,260 organic compounds with experimental strongest-acidic pKa '
    'values measured in water. The underlying values were compiled from the literature '
    'in the DataWarrior reference dataset pKaInWater.dwar and were curated for QSAR '
    'modelling by Mansouri et al. (2019), who standardised the structures, excluded '
    'inorganics and mixtures, removed salts and counterions, and resolved replicate '
    'measurements; this is their Option 3 acidic set, in which replicates were averaged '
    'when their standard deviation was at most 1 pKa unit and the strongest (minimum) '
    'acidic value was retained otherwise. The measurements are predominantly '
    'potentiometric and spectrophotometric, and the published train/test split is '
    'preserved per compound as graph metadata. Structures are the hand-curated '
    'DataWarrior originals rather than the paper QSAR-ready column, because the latter '
    'converts many carboxylic acids into implausible gem-diol and enediol tautomers; '
    'the QSAR-ready SMILES is carried alongside each graph for reproducibility. Because '
    'pKa controls ionisation and therefore solubility, membrane permeability and '
    'bioavailability, this set is widely used for benchmarking acid dissociation '
    'prediction, and is the training data behind the open-source OPERA pKa models.'
)

# :param METADATA:
METADATA: dict = {
    'license': {
        'identifier': 'CC0-1.0',
        'url': 'https://creativecommons.org/publicdomain/zero/1.0/',
        'note': (
            'The BMC article applies a CC0 waiver to the data in its additional '
            'files. The values themselves come from the DataWarrior reference file '
            'pKaInWater.dwar, which states no licence at all and whose own upstream '
            'cannot be established even in principle - its author records that the '
            'literature references were lost or never collected. Redistribution on '
            'that basis was approved by the maintainer on 2026-08-13; see '
            'docs/rejected_datasets.md.'
        ),
    },
    'verbose': 'Aqueous Acidic pKa (OPERA Curated Set)',
    'tags': [
        'Molecules',
        'SMILES',
        'pKa',
        'Acid Dissociation',
        'Physical Chemistry',
        'Experimental',
        'Curated',
        'QSAR',
    ],
    'sources': [
        'https://doi.org/10.1186/s13321-019-0384-1',
        'https://jcheminf.biomedcentral.com/articles/10.1186/s13321-019-0384-1#MOESM1',
        'https://doi.org/10.1021/ci500588j',
        'https://openmolecules.org/datawarrior/',
    ],
    'target_descriptions': {
        '0': (
            'pKa - Strongest acidic pKa of the compound measured in water '
            '(dimensionless; the negative base-10 logarithm of the acid dissociation '
            'constant). Lower values indicate a stronger acid. Observed range -7.15 '
            'to 20.50.'
        ),
    },
    'notes': [
        'Regression task: predict the strongest acidic pKa in water.',
        'Graphs are built from the hand-curated DataWarrior "OriginalSmiles" column. '
        'The paper QSAR-ready structures damage many carboxyl groups and are carried '
        'as "graph_canonical_qsar_smiles" rather than used.',
        '944 of the source structures carry a formal charge, since a curated "correct '
        'protonation state" for an acid is often the ionised form. The dataset '
        'therefore mixes neutral and ionised representations.',
        'The published 75/25 split is preserved as "graph_split" (train/test); rows '
        'from both splits are included.',
        'Values come from DataWarrior, whose author states the original literature '
        'references were lost, so individual measurements cannot be traced to a '
        'primary source.',
    ],
}

# == EXPERIMENT PARAMETERS ==

__DEBUG__ = True
__TESTING__ = False

experiment = Experiment.extend(
    'create_graph_datasets.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)


def _fetch_archive(e: Experiment) -> str:
    """
    Download the supplementary archive into the experiment folder, unless cached.

    The archive is small enough (5.1 MiB) that a single non-streamed request is
    appropriate. It is cached in the results folder so that repeated runs - in
    particular a smoke run followed by the full run - fetch it only once.

    :param e: The running experiment instance.

    :returns: Absolute path of the downloaded ZIP archive on disk.
    """
    path = os.path.join(e.path, 'additional_file_1.zip')

    if os.path.exists(path):
        e.log(f'using cached archive @ {path}')
        return path

    e.log(f'downloading supplementary archive from {DATASET_URL}')
    response = requests.get(DATASET_URL, timeout=300)
    response.raise_for_status()
    with open(path, mode='wb') as file:
        file.write(response.content)

    e.log(f'downloaded {os.path.getsize(path) / 1024 ** 2:.1f} MB')
    return path


def _read_members(e: Experiment, archive_path: str) -> pd.DataFrame:
    """
    Read the curated acidic train and test members and concatenate them.

    The train/test division is a modelling convenience of the original publication
    rather than a property of the measurements, so the two members are merged back
    into the full endpoint. Split membership is retained in an added ``split`` column
    so that the paper's exact partition can still be reconstructed downstream.

    :param e: The running experiment instance.
    :param archive_path: Absolute path of the downloaded ZIP archive.

    :returns: The concatenated DataFrame of all curated acidic rows.
    """
    frames = []
    with zipfile.ZipFile(archive_path) as archive:
        for member, split in ARCHIVE_MEMBERS.items():
            frame = pd.read_csv(io.BytesIO(archive.read(member)))
            frame['split'] = split
            e.log(f'read {member} - {len(frame)} rows')
            frames.append(frame)

    df = pd.concat(frames, ignore_index=True)

    # The archive also contains the paper's Options 1 and 2 and the basic-pKa sets,
    # all with the same column layout. Reading the wrong pair would produce a valid
    # but differently sized dataset, which no later check would notice.
    if len(df) != EXPECTED_ROWS:
        raise ValueError(
            f'expected {EXPECTED_ROWS} curated acidic rows, got {len(df)} - '
            f'the archive layout may have changed'
        )

    return df


@experiment.hook('load_dataset', default=False, replace=True)
def load_dataset(e: Experiment) -> Dict[int, dict]:
    """
    Download the supplementary archive and build the curated acidic pKa dataset.

    Fetches the publication's Additional file 1, reads the Option 3 acidic train and
    test members, concatenates them, and returns one record per compound with the
    experimental acidic pKa as the single regression target. Rows whose SMILES cannot
    be parsed by RDKit, and rows without a numeric pKa, are skipped - no sentinel is
    used, because -1.0 is a legitimate value for this target.

    :param e: The running experiment instance.

    :returns: Mapping of contiguous integer index to record dicts carrying at least
        the ``smiles`` string and the single-element ``targets`` list.
    """
    archive_path = _fetch_archive(e)
    df = _read_members(e, archive_path)
    e.log(f'loaded {len(df)} curated acidic rows')

    dataset: Dict[int, dict] = {}
    skipped_smiles = 0
    skipped_target = 0

    for row in df.to_dict('records'):

        smiles = str(row[SMILES_COLUMN]).strip()
        if not smiles or smiles == 'nan' or Chem.MolFromSmiles(smiles) is None:
            skipped_smiles += 1
            continue

        pka = row['pKa']
        if pd.isna(pka):
            skipped_target += 1
            continue

        dataset[len(dataset)] = {
            'smiles': smiles,
            'targets': [float(pka)],
            # -- provenance carried through to the graph metadata --
            'canonical_qsar_smiles': str(row['Canonical_QSARr']),
            'split': str(row['split']),
            'record_id': str(row['Molecule name']),
            'method': '' if pd.isna(row['method']) else str(row['method']),
            'measurement_type': '' if pd.isna(row['type']) else str(row['type']),
            'ionizing_group': '' if pd.isna(row['group']) else str(row['group']),
            'temperature': '' if pd.isna(row['temp']) else str(row['temp']),
        }

    e.log(
        f'built dataset with {len(dataset)} molecules '
        f'({skipped_smiles} unparseable SMILES, {skipped_target} missing pKa)'
    )
    return dataset


@experiment.hook('add_graph_metadata', default=False, replace=True)
def add_graph_metadata(_e: Experiment, data: dict, graph: dict) -> None:
    """
    Attach measurement provenance and the alternative structure to each graph.

    :param _e: The running experiment instance, unused.
    :param data: The record dict produced by ``load_dataset``.
    :param graph: The graph dict to be annotated in place.

    :returns: None; ``graph`` is mutated in place.
    """
    graph['graph_canonical_qsar_smiles'] = data['canonical_qsar_smiles']
    graph['graph_split'] = data['split']
    graph['graph_record_id'] = data['record_id']
    graph['graph_method'] = data['method']
    graph['graph_measurement_type'] = data['measurement_type']
    graph['graph_ionizing_group'] = data['ionizing_group']
    graph['graph_temperature'] = data['temperature']


experiment.run_if_main()
