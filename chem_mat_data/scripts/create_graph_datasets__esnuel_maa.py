"""
Atom-level methyl anion affinities (MAAs) from the ESNUEL machine-learning dataset -
a per-atom measure of electrophilicity for ~47k neutral, drug-like molecules.

This dataset is unusual for this collection in that its targets are **per atom, not per
molecule**. The source file lists one row per identified electrophilic site, 534,119 of
them across 47,440 molecules, so a molecule contributes on average 11.3 rows. Those
per-site values are carried through to ``graph['node_labels']``; the graph-level target
is the maximum over a molecule's sites, i.e. the affinity of its most electrophilic site,
which is the reduction the publication itself applies when relating MAA to hydrolysis.

**Source**: Ree, N.; Wollschlaeger, J. M.; Goeller, A. H.; Jensen, J. H. *Atom-based
machine learning for estimating nucleophilicity and electrophilicity with applications
to retrosynthesis and chemical stability.* Chem. Sci. 2025, 16, 5676-5687.
DOI: 10.1039/D4SC07297A

**Targets**: MAA_max - methyl anion affinity of the most electrophilic site, in kJ/mol,
at the r2SCAN-3c SMD(DMSO)//GFN1-xTB ALPB(DMSO) level of theory.

**Licensing note**: the data archive carries no licence of its own, and the molecular
structures derive from ChEMBL (CC BY-SA 3.0). Redistribution here rests on an explicit
maintainer decision; the evidence and the project-wide ChEMBL question are written up in
``docs/chembl_licensing_analysis.md``.

**The companion dataset**: the same archive holds ``data/QMdata4ML/df_nuc_x.csv.gz``, the
methyl *cation* affinities (MCA, nucleophilicity), processed as ``esnuel_mca``. The two
are the same 50,000-molecule ChEMBL pool split by site type, not separate collections.

**Do not confuse the site labels with esnuel_mca's.** ``Ketone``, ``Ester``, ``Amide``,
``Aldehyde``, ``Anhydride``, ``Michael acceptor``, ``Imine`` and ``Iminium`` mark the
electrophilic *carbon* here, where in ``esnuel_mca`` the same names marked the
nucleophilic *oxygen*; and ``double_bond`` marks either end of the bond (C, N, O, S or P),
not only carbon. Any element-based sanity check transplanted from that script will report
tens of thousands of phantom mismatches.
"""

import os
import tarfile
from typing import Dict, List

import requests
import numpy as np
import pandas as pd
import rdkit.Chem as Chem
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

# == SOURCE PARAMETERS ==

# :param DOWNLOAD_URL:
#       Direct download for "data.tar.xz" (1.17 GB) on ERDA, the University of
#       Copenhagen research data store. This is the URL the repository README instructs
#       users to wget; there is no landing page and no per-file URL.
DOWNLOAD_URL: str = 'https://sid.erda.dk/share_redirect/c7LF5NaYvH'

# :param ARCHIVE_MEMBER:
#       The single member extracted from the archive. It is itself gzipped, so the file
#       written to disk is a .csv.gz of 343 MB (1.08 GB uncompressed).
ARCHIVE_MEMBER: str = 'data/QMdata4ML/df_elec_x.csv.gz'

# :param SITE_COLUMN:
#       Column holding the atom index of the electrophilic site. The index refers to
#       ``Chem.MolFromSmiles(smiles)`` atom ordering - verified against the functional
#       group labels, see the note in load_dataset.
SITE_COLUMN: str = 'elec_sites'

# :param SITE_NAME_COLUMN:
#       Column holding the functional group label for the site, e.g.
#       "double_bond" or "Ketone". Note these are electrophile labels - "Amine"
#       and the rest of the nuc_names vocabulary do not occur in this file.
SITE_NAME_COLUMN: str = 'elec_names'

# :param VALUE_COLUMN:
#       Column holding the per-site MAA value in kJ/mol.
VALUE_COLUMN: str = 'MAA_values'

# :param SMILES_COLUMN:
#       Column holding the (canonical, RDKit-generated) SMILES of the parent molecule.
SMILES_COLUMN: str = 'smiles'

# :param TARGET_COLUMNS:
#       Read by the default save_csv hook to name the target columns of the exported
#       CSV. One graph-level target here.
TARGET_COLUMNS: List[str] = ['MAA_max']

# :param DATASET_NAME:
DATASET_NAME: str = 'esnuel_maa'

# :param DATASET_TYPE:
DATASET_TYPE: str = 'regression'

# :param DESCRIPTION:
#       Human-readable summary. Kept verbatim in step with the approved metadata draft.
DESCRIPTION: str = (
    'Atom-level methyl anion affinities (MAAs) for 47,440 neutral, drug-like '
    'molecules drawn from ChEMBL, computed with the ESNUEL quantum-chemistry workflow '
    'at the r2SCAN-3c SMD(DMSO)//GFN1-xTB ALPB(DMSO) level of theory. The MAA '
    'quantifies electrophilicity: the higher the value, the more electrophilic the '
    'site. Every molecule carries per-atom MAA values in node_labels for each '
    'electrophilic site located by SMARTS matching - 534,119 sites in total, a mean '
    'of 11.3 per molecule - with NaN at atoms that are not electrophilic sites. The '
    'single graph-level target is the maximum MAA over the sites a molecule actually '
    'has listed, i.e. the affinity of the most electrophilic site the source workflow '
    'retained - sites whose QM geometry optimisation changed the connectivity were '
    'discarded upstream, so the site list is not exhaustive and 16% of the carbonyl '
    'carbons in these molecules are absent from it. Reducing to the most '
    "electrophilic site mirrors the publication's own use of the model. This is the "
    'electrophilicity counterpart to esnuel_mca, derived from the same '
    '50,000-molecule pool split by site type.'
)

# :param METADATA:
METADATA: dict = {
    'verbose': 'ESNUEL Methyl Anion Affinities (Electrophilicity)',
    'license': {
        'identifier': 'CC-BY-SA-3.0',
        'url': 'https://creativecommons.org/licenses/by-sa/3.0/',
        'note': (
            'The molecular structures derive from ChEMBL, which is licensed CC BY-SA '
            '3.0. The ShareAlike condition propagates, so this dataset is distributed '
            'under the same licence. Note that the ESNUEL data archive states no '
            'licence of its own; this label reflects the ChEMBL obligation flowing '
            'through, not a grant from its authors.'
        ),
    },
    'tags': [
        'Molecules',
        'SMILES',
        'Quantum Chemistry',
        'DFT',
        'Regression',
        'ChEMBL',
        'Drug Discovery',
    ],
    'sources': [
        'https://doi.org/10.1039/D4SC07297A',
        'https://github.com/jensengroup/ESNUEL_ML',
        'https://sid.erda.dk/share_redirect/c7LF5NaYvH',
        'https://www.ebi.ac.uk/chembl/',
    ],
    'target_descriptions': {
        '0': (
            'MAA_max - methyl anion affinity of the most electrophilic site in the '
            'molecule, in kJ/mol, at the r2SCAN-3c SMD(DMSO)//GFN1-xTB ALPB(DMSO) '
            'level; the per-site values for every identified electrophilic atom are '
            'in node_labels, NaN elsewhere'
        ),
    },
    'notes': [
        (
            'Targets are per atom in the source. graph["node_labels"] is an (n_atoms, '
            '1) float array holding the MAA of each electrophilic site in kJ/mol and '
            'NaN at every other atom - mask with ~numpy.isnan(). NaN is used rather '
            'than a numeric sentinel because 41,204 of the 534,119 source values '
            '(7.7%) are negative and the range reaches -248.5 kJ/mol, so any number '
            'chosen as a filler would collide with a legitimate measurement.'
        ),
        (
            'Each graph also carries graph_site_names, a list of the functional-group '
            'labels for its sites, ordered to match the sorted non-NaN indices of '
            'node_labels. Being a list of strings it is not carried into PyTorch '
            'Geometric or jraph; read it from the graph dict directly.'
        ),
        (
            'The exported raw CSV carries only smiles and MAA_max. The 534,119 '
            'per-site values that are the substance of this dataset exist only in the '
            'graph format, so reach for load_graph_dataset rather than '
            'load_smiles_dataset.'
        ),
        (
            'Per-atom targets are carried into PyTorch Geometric by pyg_from_graph as '
            'data.node_y, alongside the graph-level target in data.y, and they batch '
            'correctly under a DataLoader. Older releases of chem_mat_data dropped '
            'node_labels silently - if data.node_y is missing after conversion, '
            'upgrade.'
        ),
        (
            "The publication's own train/test split is deliberately not reproduced "
            'here. It is a per-site split: 37,631 of the 47,440 molecules (79.3%) '
            'have sites in both the training folds and the held-out test set, so it '
            "cannot be expressed as a molecule-level split at all. The paper's "
            'reported accuracies (Pearson r = 0.95, RMSE 22.08 kJ/mol) are site-level '
            'results under that split and are not comparable to a model trained on '
            'the graph-level target.'
        ),
        (
            'The site labels in elec_names do not mean the same atoms as the '
            'like-named labels in esnuel_mca. Here Ketone, Ester, Amide, Aldehyde, '
            'Anhydride, Michael acceptor, Imine and Iminium all mark the '
            'electrophilic carbon; in esnuel_mca those names marked the nucleophilic '
            'oxygen. The double_bond label marks either end of the bond (C, N, O, S '
            'or P), not only carbon.'
        ),
        (
            'Molecular structures in this dataset derive from ChEMBL '
            '(https://www.ebi.ac.uk/chembl/), which is licensed CC BY-SA 3.0. Cite '
            'Mendez et al., Nucleic Acids Res. 47, D930-D940 (2019), '
            'doi:10.1093/nar/gky1075.'
        ),
    ],
}

# == EXPERIMENT PARAMETERS ==

# :param TESTING_ROW_LIMIT:
#       In testing mode only, how many source rows to parse. The base template truncates
#       to 50 *molecules* after load_dataset returns, but that happens too late to save
#       the cost of parsing a 1.08 GB CSV. Reading a prefix instead keeps a smoke run to
#       seconds while still exercising the real download, extraction and grouping path.
TESTING_ROW_LIMIT: int = 20_000

# :param __TESTING__:
__TESTING__ = False

experiment = Experiment.extend(
    'create_graph_datasets.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)


@experiment.hook('add_graph_metadata', default=False, replace=True)
def add_graph_metadata(e: Experiment, data: dict, graph: dict) -> None:
    """
    Attach the per-atom MAA values as ``node_labels``.

    This is where the dataset's real content lives. ``node_labels`` is an
    ``(n_atoms, 1)`` float array carrying the MAA in kJ/mol at each electrophilic site and
    ``NaN`` at every other atom.

    NaN rather than a numeric sentinel is deliberate: the observed MAA range runs from
    -248.5 to 446.1 kJ/mol with 41,204 of the 534,119 values (7.7%) negative, so a filler
    such as -1 or 0 would be indistinguishable from a genuine measurement. Consumers
    select the real entries with ``~numpy.isnan(...)``.

    The alignment between the source's atom indices and the graph's node order holds
    because ``MoleculeProcessing`` builds the graph with ``Chem.MolFromSmiles`` on this
    same SMILES string and takes node order straight from ``atom.GetIdx()``. It is not
    assumed - ``load_dataset`` rejects any site index outside the molecule's atom range.

    :param e: The running experiment.
    :param data: The record dict this graph was built from.
    :param graph: The graph dict, mutated in place.

    :return: None
    """
    num_nodes = len(graph['node_indices'])
    labels = np.full((num_nodes, 1), np.nan, dtype=float)
    for site, value in zip(data['sites'], data['site_values']):
        labels[site, 0] = value

    graph['node_labels'] = labels
    graph['graph_site_names'] = list(data['site_names'])


@experiment.hook('load_dataset', default=False, replace=True)
def load_dataset(e: Experiment) -> Dict[int, dict]:
    """
    Download the archive, extract the MAA table and group its rows into one record per
    molecule.

    The source file is one row per *atomic site*, so the work here is an aggregation:
    534,119 rows collapse into 47,440 molecules, each carrying the list of its site
    indices and their MAA values.

    Two details are load-bearing for the download. The archive is streamed with
    ``tarfile`` in pipe mode (``r|xz``) so that reading stops at the wanted member -
    seek mode would force all 1.17 GB through before the first member became readable,
    where streaming pulls roughly 780 MB - this member sits after df_nuc_x.csv.gz in
    the tar, so the stream must run through that 419 MB member to reach it. And
    ``raw.decode_content`` stays False because the payload is already xz-compressed
    and must not be decoded twice.

    Rows are dropped for exactly two reasons, both genuine corruption rather than
    filtering: a SMILES RDKit cannot parse, and a site index that falls outside the
    molecule's atom range. The latter would silently mislabel a different atom, so it is
    checked rather than trusted.

    :param e: The running experiment.

    :returns: Mapping of integer index to record dicts, each with ``smiles``, a
        ``targets`` list of one float, and the parallel ``sites`` / ``site_values`` /
        ``site_names`` lists consumed by ``add_graph_metadata``.
    """
    # -- download the archive member, with caching --
    csv_path = os.path.join(e.path, 'df_elec_x.csv.gz')
    if not os.path.exists(csv_path):
        e.log(f'streaming {ARCHIVE_MEMBER} from {DOWNLOAD_URL} ...')
        with requests.get(DOWNLOAD_URL, stream=True, timeout=900) as response:
            response.raise_for_status()
            # The payload is xz; urllib3 must not try to decode it on our behalf.
            response.raw.decode_content = False
            with tarfile.open(fileobj=response.raw, mode='r|xz') as tar:
                for member in tar:
                    if member.name != ARCHIVE_MEMBER:
                        continue
                    source = tar.extractfile(member)
                    with open(csv_path, 'wb') as file:
                        while chunk := source.read(8 << 20):
                            file.write(chunk)
                    break
                else:
                    raise RuntimeError(
                        f'{ARCHIVE_MEMBER} not found in the archive at {DOWNLOAD_URL}'
                    )
        e.log(f'extracted {os.path.getsize(csv_path) / 1024 ** 2:.1f} MB')
    else:
        e.log(f'using cached table @ {csv_path}')

    # -- read only the columns that matter --
    # The file also carries six precomputed descriptor columns, each a JSON list inside
    # a string. They are the ML features of the source publication, are not part of this
    # dataset, and together account for almost all of the 1.08 GB.
    columns = [SMILES_COLUMN, SITE_COLUMN, SITE_NAME_COLUMN, VALUE_COLUMN]
    read_kwargs = {'usecols': columns}
    if e.__TESTING__:
        e.log(f' * testing mode: reading only {TESTING_ROW_LIMIT} source rows')
        read_kwargs['nrows'] = TESTING_ROW_LIMIT

    df = pd.read_csv(csv_path, **read_kwargs)
    e.log(f'loaded {len(df)} atomic sites over {df[SMILES_COLUMN].nunique()} molecules')

    # -- group the per-site rows into one record per molecule --
    dataset: Dict[int, dict] = {}
    skipped_invalid: int = 0
    dropped_sites: int = 0
    duplicate_sites: int = 0

    for smiles, group in df.groupby(SMILES_COLUMN, sort=False):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            skipped_invalid += 1
            continue

        num_atoms = mol.GetNumAtoms()
        # A dict keyed by atom index both preserves one value per atom and makes a
        # repeated index visible instead of letting the last row win silently.
        by_site: Dict[int, tuple] = {}
        for site, name, value in zip(
            group[SITE_COLUMN], group[SITE_NAME_COLUMN], group[VALUE_COLUMN]
        ):
            site = int(site)
            if site < 0 or site >= num_atoms:
                dropped_sites += 1
                continue
            if site in by_site:
                duplicate_sites += 1
                # Keep the more electrophilic reading, consistent with the graph target.
                if float(value) <= by_site[site][1]:
                    continue
            by_site[site] = (str(name), float(value))

        if not by_site:
            skipped_invalid += 1
            continue

        sites = sorted(by_site)
        values = [by_site[site][1] for site in sites]

        dataset[len(dataset)] = {
            'smiles': smiles,
            'targets': [max(values)],
            'sites': sites,
            'site_values': values,
            'site_names': [by_site[site][0] for site in sites],
        }

    e.log(
        f'built dataset with {len(dataset)} molecules '
        f'({skipped_invalid} unusable, {dropped_sites} sites out of range, '
        f'{duplicate_sites} duplicate site indices)'
    )
    return dataset


experiment.run_if_main()
