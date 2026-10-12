from rdkit import Chem
from typing import Dict
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

from chem_mat_data import load_smiles_dataset

DATASET_NAME: str = 'qm9_smiles'
DESCRIPTION: str = (
    'The QM9 dataset is a comprehensive collection of approximately 134,000 stable small '
    'organic molecules composed of up to nine heavy atoms (including carbon, oxygen, nitrogen, '
    'and fluorine). Each molecule in the dataset includes detailed geometric, energetic, electronic, '
    'and thermodynamic properties calculated using density functional theory (DFT), '
    'providing quantum-mechanical ground truth data'
)

# :param METADATA:
#       A dictionary which will be used as the basis for the metadata that will be added
#       as additional information to the file share server.
METADATA: dict = {
    'license': {
        'identifier': 'CC0-1.0',
        'url': 'https://creativecommons.org/publicdomain/zero/1.0/',
        'note': (
            'CC0 1.0 is set on the Figshare record the data comes from - article '
            '1057646, "Data for 133885 GDB-9 molecules" (dsgdb9nsd.xyz.tar.bz2), and '
            'on every article in collection 978904. Cite Ramakrishnan, R., Dral, P. '
            'O., Rupp, M. & von Lilienfeld, O. A., Sci Data 1, 140022 (2014). The '
            "structures derive from the Reymond group's GDB chemical universe "
            "databases. GDB's own download page (gdb.unibe.ch/downloads) carries a "
            'click-through notice that "GDB and large portions thereof must not be '
            'redistributed without the express written permission of Jean-Louis '
            'Reymond", but the Reymond group have themselves deposited the databases '
            '- GDB-11, GDB-13 and GDB-17 - on Zenodo under CC BY 4.0 (record 5172018, '
            'doi:10.5281/zenodo.5172018, Fink, Blum, Ruddigkeit & van Deursen, 2021). '
            'The maintainer determined on 2026-08-19 that the specific, current grant '
            'on the data itself governs over the generic terms page. Cite GDB '
            'accordingly.'
        ),
    },
    'tags': [
        'Molecules',
        'SMILES',
        'Quantum Chemistry',
        'DFT',
        'Electronic Properties',
    ],
    'verbose': 'QM9 SMILES Dataset',
    'sources': [
        'https://www.nature.com/articles/sdata201422',
        'https://moleculenet.org/datasets',
        'https://pytorch-geometric.readthedocs.io/en/2.5.0/generated/torch_geometric.datasets.QM9.html',
    ],
    # Units follow the MoleculeNet QM9 schema and were verified against the shipped
    # values rather than copied from it. Three of these previously read "in eV" - homo,
    # lumo, gap and u0 - which was wrong: the data is in Hartree, and reading it as eV
    # understates those targets by a factor of 27.2114. Cross-checks used: homo spans
    # -0.4286 to -0.1017, which is Hartree (eV would be -11.7 to -2.8); gap equals
    # lumo - homo in the same unit; and the atomization energies reproduce literature
    # values in kcal/mol (NH3 276.86 here against a literature D_0 of 276.7).
    # Indices 16-18 were previously undescribed entirely, although load_dataset has
    # always emitted 19 targets.
    'target_descriptions': {
        '0': 'A - rotational constant in GHz',
        '1': 'B - rotational constant in GHz',
        '2': 'C - rotational constant in GHz',
        '3': 'mu - dipole moment in Debye',
        '4': 'alpha - isotropic polarizability in Bohr^3',
        '5': 'homo - highest occupied molecular orbital energy in Hartree',
        '6': 'lumo - lowest unoccupied molecular orbital energy in Hartree',
        '7': 'gap - LUMO minus HOMO energy difference in Hartree',
        '8': 'R^2 - electronic spatial extent in Bohr^2',
        '9': 'ZPVE - zero point vibrational energy in Hartree',
        '10': 'U_0 - internal energy at 0 K in Hartree, including the zero point energy',
        '11': 'U - internal energy at 298.15 K in Hartree',
        '12': 'H - enthalpy at 298.15 K in Hartree',
        '13': 'G - Gibbs free energy at 298.15 K in Hartree',
        '14': 'Cv - constant-volume heat capacity at 298.15 K in cal/(mol K)',
        '15': 'U_0 atom - atomization energy at 0 K in kcal/mol',
        '16': 'U atom - atomization energy at 298.15 K in kcal/mol',
        '17': 'H atom - atomization enthalpy at 298.15 K in kcal/mol',
        '18': 'G atom - atomization Gibbs free energy at 298.15 K in kcal/mol',
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
    pass  # graph['graph_id'] = data['ID']


@experiment.hook('load_dataset', default=False, replace=True)
def load_dataset(e: Experiment) -> Dict[int, dict]:
    df = load_smiles_dataset('qm9_smiles')

    columns = [
        'A',
        'B',
        'C',
        'mu',
        'alpha',
        'homo',
        'lumo',
        'gap',
        'r2',
        'zpve',
        'u0',
        'u298',
        'h298',
        'g298',
        'cv',
        'u0_atom',
        'u298_atom',
        'h298_atom',
        'g298_atom',
    ]

    dataset: Dict[int, dict] = {}
    index: int = 0
    for data in df.to_dict('records'):
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

        # == TARGETS ==
        # In this dataset we only have one target
        data['targets'] = [data[key] for key in columns]
        dataset[index] = data

        index += 1

    return dataset


experiment.run_if_main()
