import rdkit.Chem as Chem
from typing import Dict
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

from chem_mat_data import load_smiles_dataset

DATASET_NAME: str = 'ames'

__TESTING__ = False

# :param METADATA:
#       Mirrors what this dataset already carries in the shared metadata.yml.
#       Without it, "cmmanage metadata collect" - which replaces a dataset's
#       catalogue entry wholesale rather than merging fields - would silently
#       drop these the next time this script is processed.
METADATA: dict = {
    'license': {
        'identifier': 'CC-BY-NC-4.0',
        'url': 'https://creativecommons.org/licenses/by-nc/4.0/',
        'note': (
            'CC BY-NC 4.0 is stated on the ACS Figshare record this data comes from '
            '(acs.figshare.com article 2825590, the Supporting Information of Hansen, '
            'K. et al., J. Chem. Inf. Model. 49, 2077-2081 (2009), '
            'doi:10.1021/ci900161g); the Figshare API returns license name "CC BY-NC '
            '4.0" and the file MD5 matches the copy shipped here. NONCOMMERCIAL: '
            'attribution required and commercial use is not permitted. UPSTREAM: the '
            'compiled assay data traces to CCRIS (NLM), GENE-TOX and US EPA - likely '
            'public-domain US government sources - plus Vitic, a commercial toxicity '
            'database from Lhasa Limited whose terms could not be located; Hansen et '
            'al. describe the released compounds only as "nonconfidential". The '
            'maintainer decided on 2026-08-19 that ChemMatData, as a non-commercial '
            'academic distribution, may redistribute NonCommercial-licensed data '
            'provided the condition is carried through to users via this field.'
        ),
    },
    'verbose': 'Ames Mutagenicity Assays',
    'full': True,
    'tags': [
        'Smiles',
        'Molecules',
        'Toxicity',
    ],
}

experiment = Experiment.extend(
    'create_graph_datasets.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)


@experiment.hook('add_graph_metadata', default=False, replace=True)
def add_graph_metadata(e: Experiment, data: dict, graph: dict) -> dict:
    """
    We add the CAS-number as additional metadata
    """
    graph['graph_cas_number'] = data['CAS_NO']


@experiment.hook('load_dataset', default=False, replace=True)
def load_dataset(e: Experiment) -> Dict[int, dict]:
    df = load_smiles_dataset('ames')

    dataset: Dict[int, dict] = {}
    for index, data in enumerate(df.to_dict('records')):
        data['smiles'] = data['Canonical_Smiles']

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

        # == TARGETS ==
        # Either active or inactive
        data['targets'] = [data['Activity'] == index for index in range(2)]
        dataset[index] = data

    return dataset


experiment.run_if_main()
