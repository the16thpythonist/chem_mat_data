"""
This module implements the ``MetalOrganicProcessing`` class for converting transition metal
complex (TMC) data into graph representations suitable for graph neural networks.

Unlike ``MoleculeProcessing`` which takes a single SMILES string representing an entire molecule,
``MetalOrganicProcessing`` works with a *decomposed* representation: a metal center plus a list
of ligand SMILES strings and their connecting atom indices. This decomposition reflects how
coordination chemistry naturally works — ligands are organic fragments that donate electron
pairs to a central metal ion — and avoids the well-known limitations of RDKit when parsing
whole-complex metal SMILES (valence errors, haptic bonding, canonicalization issues).

The feature encoding is specifically designed for TMCs, including metal-specific node features
(d-electron count, electronegativity, covalent radius) and dative bond edge features. The
design follows insights from ELECTRUM (Orsi & Frei 2025), RACs (Janet & Kulik 2017), and
the tmQMg GNN benchmarks (Kneiding et al. 2023).

Example usage:

.. code-block:: python

    processing = MetalOrganicProcessing()
    graph = processing.process(
        metal='Fe',
        ligand_smiles=['[NH3]', '[NH3]', '[NH3]', '[NH3]', '[NH3]', '[NH3]'],
        connecting_atom_indices=[[0], [0], [0], [0], [0], [0]],
        oxidation_state=2,
        total_charge=2,
        graph_labels=[some_property_value],
    )
"""
from typing import Any, Dict, List, Tuple, Optional, Union

import numpy as np
import rdkit.Chem as Chem
from rdkit.Chem import Descriptors

from chem_mat_data.processing import (
    AbstractProcessing,
    EncoderBase,
    EncodingDescriptionMixin,
    OneHotEncoder,
    chem_prop,
    list_identity,
)
from chem_mat_data._typing import GraphDict
from chem_mat_data.utils import mol_from_smiles


# ======================================================================================
# CONSTANTS AND LOOKUP TABLES
# ======================================================================================

# The 30 transition metals (groups 3-12, periods 4-6) by atomic number.
TRANSITION_METAL_ATOMIC_NUMBERS: set = {
    # 3d series (period 4)
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30,   # Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn
    # 4d series (period 5)
    39, 40, 41, 42, 43, 44, 45, 46, 47, 48,   # Y, Zr, Nb, Mo, Tc, Ru, Rh, Pd, Ag, Cd
    # 5d series (period 6)
    72, 73, 74, 75, 76, 77, 78, 79, 80,        # Hf, Ta, W, Re, Os, Ir, Pt, Au, Hg
    57,                                          # La (included for completeness)
}

# Pauling electronegativity values. Source: standard reference tables.
# Values normalized by dividing by 3.98 (fluorine) during encoding.
PAULING_ELECTRONEGATIVITY: Dict[int, float] = {
    1: 2.20,   # H
    2: 0.00,   # He (noble gas)
    3: 0.98,   # Li
    4: 1.57,   # Be
    5: 2.04,   # B
    6: 2.55,   # C
    7: 3.04,   # N
    8: 3.44,   # O
    9: 3.98,   # F
    11: 0.93,  # Na
    12: 1.31,  # Mg
    13: 1.61,  # Al
    14: 1.90,  # Si
    15: 2.19,  # P
    16: 2.58,  # S
    17: 3.16,  # Cl
    19: 0.82,  # K
    20: 1.00,  # Ca
    21: 1.36,  # Sc
    22: 1.54,  # Ti
    23: 1.63,  # V
    24: 1.66,  # Cr
    25: 1.55,  # Mn
    26: 1.83,  # Fe
    27: 1.88,  # Co
    28: 1.91,  # Ni
    29: 1.90,  # Cu
    30: 1.65,  # Zn
    33: 2.18,  # As
    34: 2.55,  # Se
    35: 2.96,  # Br
    39: 1.22,  # Y
    40: 1.33,  # Zr
    41: 1.60,  # Nb
    42: 2.16,  # Mo
    43: 1.90,  # Tc
    44: 2.20,  # Ru
    45: 2.28,  # Rh
    46: 2.20,  # Pd
    47: 1.93,  # Ag
    48: 1.69,  # Cd
    53: 2.66,  # I
    57: 1.10,  # La
    72: 1.30,  # Hf
    73: 1.50,  # Ta
    74: 2.36,  # W
    75: 1.90,  # Re
    76: 2.20,  # Os
    77: 2.20,  # Ir
    78: 2.28,  # Pt
    79: 2.54,  # Au
    80: 2.00,  # Hg
}

# Group number (1-18) by atomic number. Only elements likely to appear in TMC datasets.
ELEMENT_GROUP: Dict[int, int] = {
    1: 1,      # H
    3: 1,      # Li
    5: 13,     # B
    6: 14,     # C
    7: 15,     # N
    8: 16,     # O
    9: 17,     # F
    11: 1,     # Na
    12: 2,     # Mg
    13: 13,    # Al
    14: 14,    # Si
    15: 15,    # P
    16: 16,    # S
    17: 17,    # Cl
    19: 1,     # K
    20: 2,     # Ca
    21: 3,     # Sc
    22: 4,     # Ti
    23: 5,     # V
    24: 6,     # Cr
    25: 7,     # Mn
    26: 8,     # Fe
    27: 9,     # Co
    28: 10,    # Ni
    29: 11,    # Cu
    30: 12,    # Zn
    33: 15,    # As
    34: 16,    # Se
    35: 17,    # Br
    39: 3,     # Y
    40: 4,     # Zr
    41: 5,     # Nb
    42: 6,     # Mo
    43: 7,     # Tc
    44: 8,     # Ru
    45: 9,     # Rh
    46: 10,    # Pd
    47: 11,    # Ag
    48: 12,    # Cd
    53: 17,    # I
    57: 3,     # La
    72: 4,     # Hf
    73: 5,     # Ta
    74: 6,     # W
    75: 7,     # Re
    76: 8,     # Os
    77: 9,     # Ir
    78: 10,    # Pt
    79: 11,    # Au
    80: 12,    # Hg
}

# The element list for the one-hot encoding. 50 elements covering all common ligand atoms
# and all 30 transition metals, ordered: common organic first, then halogens, then metals
# by period.
TMC_ELEMENTS: List[str] = [
    # Common organic / ligand atoms (20)
    'C', 'N', 'O', 'S', 'P', 'Se', 'B', 'H',
    'F', 'Cl', 'Br', 'I',
    'Si', 'Li', 'Na', 'Mg', 'Al', 'K', 'Ca', 'As',
    # 3d transition metals (10)
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    # 4d transition metals (10)
    'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    # 5d transition metals (10)
    'La', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
]

# Maximum normalization constants
_MAX_ELECTRONEGATIVITY = 3.98  # Fluorine
_MAX_COVALENT_RADIUS = 2.6     # Approximate max (Cs)
_MAX_VDW_RADIUS = 3.0          # Approximate max
_MAX_PERIOD = 7
_MAX_GROUP = 18
_MAX_D_ELECTRONS = 10
_MAX_OUTER_ELECTRONS = 18
_MAX_MASS = 210.0              # Approximate (Hg ~200)

# RDKit periodic table instance for property lookups
_PT = Chem.GetPeriodicTable()


# ======================================================================================
# CUSTOM ENCODERS
# ======================================================================================

class LookupEncoder(EncoderBase, EncodingDescriptionMixin):
    """
    Encoder that looks up a value from a static dictionary keyed by atomic number,
    then normalizes by a maximum value. Returns ``[0.0]`` for unknown elements.

    :param lookup_table: Dictionary mapping atomic number to the raw property value.
    :param max_value: Normalization constant; the returned value is ``raw / max_value``.
    :param description: Human-readable description of the encoded feature.
    """

    requires_molecule: bool = False

    def __init__(self,
                 lookup_table: Dict[int, float],
                 max_value: float,
                 description: str = '',
                 ):
        self.lookup_table = lookup_table
        self.max_value = max_value
        self._description = description

    def __call__(self, atom: Chem.Atom, *args, **kwargs) -> List[float]:
        return self.encode(atom)

    def encode(self, atom: Chem.Atom, *args, **kwargs) -> List[float]:
        z = atom.GetAtomicNum()
        raw = self.lookup_table.get(z, 0.0)
        return [raw / self.max_value]

    def decode(self, encoded: List[float]) -> float:
        return encoded[0] * self.max_value

    def get_description(self, index: int = 0) -> str:
        return self._description

    @property
    def descriptions(self) -> List[str]:
        return [self._description]


class PeriodicTableEncoder(EncoderBase, EncodingDescriptionMixin):
    """
    Encoder that retrieves a property from the RDKit periodic table for a given atom
    and normalizes it.

    :param property_name: Name of the method to call on ``Chem.GetPeriodicTable()``
        (e.g., ``'GetRcovalent'``, ``'GetRvdw'``, ``'GetRow'``, ``'GetNOuterElecs'``).
    :param max_value: Normalization constant.
    :param description: Human-readable description.
    """

    requires_molecule: bool = False

    def __init__(self, property_name: str, max_value: float, description: str = ''):
        self.property_name = property_name
        self.max_value = max_value
        self._description = description

    def __call__(self, atom: Chem.Atom, *args, **kwargs) -> List[float]:
        return self.encode(atom)

    def encode(self, atom: Chem.Atom, *args, **kwargs) -> List[float]:
        z = atom.GetAtomicNum()
        method = getattr(_PT, self.property_name)
        raw = method(z)
        return [raw / self.max_value]

    def decode(self, encoded: List[float]) -> float:
        return encoded[0] * self.max_value

    def get_description(self, index: int = 0) -> str:
        return self._description

    @property
    def descriptions(self) -> List[str]:
        return [self._description]


class MetalFlagEncoder(EncoderBase, EncodingDescriptionMixin):
    """
    Binary encoder that returns ``[1.0]`` if the atom is a transition metal, ``[0.0]`` otherwise.
    """

    requires_molecule: bool = False

    def __call__(self, atom: Chem.Atom, *args, **kwargs) -> List[float]:
        return self.encode(atom)

    def encode(self, atom: Chem.Atom, *args, **kwargs) -> List[float]:
        return [1.0 if atom.GetAtomicNum() in TRANSITION_METAL_ATOMIC_NUMBERS else 0.0]

    def decode(self, encoded: List[float]) -> bool:
        return encoded[0] > 0.5

    def get_description(self, index: int = 0) -> str:
        return 'Is transition metal center'

    @property
    def descriptions(self) -> List[str]:
        return ['Is transition metal center']


class DElectronCountEncoder(EncoderBase, EncodingDescriptionMixin):
    """
    Encodes the d-electron count for transition metals as ``group_number - formal_charge``,
    normalized by 10. Returns ``[0.0]`` for non-metal atoms.

    This is an approximation: formal charge from SMILES is used as a proxy for oxidation
    state, which is exact for simple ions (e.g., ``[Fe+2]``) but may differ for complexes
    with non-innocent ligands.
    """

    requires_molecule: bool = False

    def __call__(self, atom: Chem.Atom, *args, **kwargs) -> List[float]:
        return self.encode(atom)

    def encode(self, atom: Chem.Atom, *args, **kwargs) -> List[float]:
        z = atom.GetAtomicNum()
        if z not in TRANSITION_METAL_ATOMIC_NUMBERS:
            return [0.0]

        group = ELEMENT_GROUP.get(z, 0)
        formal_charge = atom.GetFormalCharge()
        d_count = max(0, min(10, group - formal_charge))
        return [d_count / _MAX_D_ELECTRONS]

    def decode(self, encoded: List[float]) -> int:
        return round(encoded[0] * _MAX_D_ELECTRONS)

    def get_description(self, index: int = 0) -> str:
        return 'd-electron count (normalized, from group - formal charge)'

    @property
    def descriptions(self) -> List[str]:
        return ['d-electron count (normalized, from group - formal charge)']


# ======================================================================================
# METAL-ORGANIC PROCESSING CLASS
# ======================================================================================

class MetalOrganicProcessing(AbstractProcessing):
    """
    Processing class for transition metal complexes (TMCs) using a decomposed representation.

    Instead of a single whole-complex SMILES, this class takes a **metal symbol**, a list of
    **ligand SMILES** (which are standard organic molecules fully compatible with RDKit), and
    the **connecting atom indices** that specify which atoms in each ligand coordinate to the
    metal center. The class assembles these into a unified molecular graph with metal-specific
    node features (91 dims), dative-bond-aware edge features (18 dims), and TMC graph-level
    features (5 dims).

    This decomposed approach mirrors how the most successful TMC ML methods work (ELECTRUM,
    RACs, molSimplify) and avoids RDKit sanitization failures with metal SMILES.

    Example:

    .. code-block:: python

        processing = MetalOrganicProcessing()

        # [Fe(NH3)6]2+ — hexaammineiron(II)
        graph = processing.process(
            metal='Fe',
            ligand_smiles=['[NH3]'] * 6,
            connecting_atom_indices=[[0]] * 6,
            oxidation_state=2,
            total_charge=2,
            graph_labels=[42.0],
        )

        print(graph['node_attributes'].shape)  # (7, 91) — 1 Fe + 6 N
        print(graph['edge_attributes'].shape)  # (6, 18) — 6 dative bonds
    """

    description = (
        'Processing module for transition metal complexes (TMCs). Converts a decomposed '
        'representation (metal + ligand SMILES + connecting atoms) into a graph dict '
        'suitable for graph neural networks.'
    )

    # ==================================================================================
    # ATTRIBUTE MAPS
    # ==================================================================================
    #
    # These follow the same pattern as MoleculeProcessing: each entry has a 'callback'
    # that extracts and encodes one feature, plus metadata for introspection.

    node_attribute_map = {
        'symbol': {
            'callback': chem_prop('GetSymbol', OneHotEncoder(
                TMC_ELEMENTS,
                add_unknown=True,
                dtype=str,
                value_descriptions=[f'{s}' for s in TMC_ELEMENTS],
            )),
            'description': 'One-hot encoding of the atom type (50 elements + unknown)',
            'is_type': True,
            'encodes_symbol': True,
        },
        'hybridization': {
            'callback': chem_prop('GetHybridization', OneHotEncoder(
                [0, 1, 2, 3, 4, 5, 6, 7, 8],
                add_unknown=True,
                dtype=int,
                value_descriptions=[
                    'Unspecified', 'S', 'SP', 'SP2', 'SP3',
                    'SP2D', 'SP3D', 'SP3D2', 'Other',
                ],
            )),
            'description': 'One-hot encoding of atom hybridization (incl. SP3D2 for metals)',
        },
        'total_degree': {
            'callback': chem_prop('GetTotalDegree', OneHotEncoder(
                list(range(13)),
                add_unknown=False,
                dtype=int,
                value_descriptions=[f'{i} neighbors' for i in range(13)],
            )),
            'description': 'One-hot encoding of total degree / coordination number (0-12)',
        },
        'num_hydrogen_atoms': {
            'callback': chem_prop('GetTotalNumHs', OneHotEncoder(
                [0, 1, 2, 3, 4],
                add_unknown=False,
                dtype=int,
                value_descriptions=[f'{i} hydrogens' for i in range(5)],
            )),
            'description': 'One-hot encoding of the total number of attached hydrogen atoms',
        },
        'mass': {
            'callback': chem_prop('GetMass', list_identity),
            'description': 'Atomic mass (unnormalized)',
        },
        'charge': {
            'callback': chem_prop('GetFormalCharge', list_identity),
            'description': 'Formal charge of the atom',
            'encodes_charge': True,
        },
        'is_aromatic': {
            'callback': chem_prop('GetIsAromatic', list_identity),
            'description': 'Is atom aromatic?',
        },
        'is_in_ring': {
            'callback': chem_prop('IsInRing', list_identity),
            'description': 'Is atom in a ring?',
        },
        'is_metal_center': {
            'callback': MetalFlagEncoder(),
            'description': 'Binary flag: is this atom a transition metal?',
        },
        'electronegativity': {
            'callback': LookupEncoder(
                PAULING_ELECTRONEGATIVITY, _MAX_ELECTRONEGATIVITY,
                description='Pauling electronegativity (normalized)',
            ),
            'description': 'Pauling electronegativity (normalized by F=3.98)',
        },
        'covalent_radius': {
            'callback': PeriodicTableEncoder(
                'GetRcovalent', _MAX_COVALENT_RADIUS,
                description='Covalent radius (normalized)',
            ),
            'description': 'Covalent radius from RDKit periodic table (normalized)',
        },
        'vdw_radius': {
            'callback': PeriodicTableEncoder(
                'GetRvdw', _MAX_VDW_RADIUS,
                description='van der Waals radius (normalized)',
            ),
            'description': 'van der Waals radius from RDKit periodic table (normalized)',
        },
        'period': {
            'callback': PeriodicTableEncoder(
                'GetRow', _MAX_PERIOD,
                description='Period / row in periodic table (normalized)',
            ),
            'description': 'Period in the periodic table (normalized by 7)',
        },
        'group_number': {
            'callback': LookupEncoder(
                ELEMENT_GROUP, _MAX_GROUP,
                description='Group number in periodic table (normalized)',
            ),
            'description': 'Group number in the periodic table (normalized by 18)',
        },
        'd_electron_count': {
            'callback': DElectronCountEncoder(),
            'description': 'd-electron count for TMs (group - formal_charge, normalized by 10)',
        },
        'outer_electrons': {
            'callback': PeriodicTableEncoder(
                'GetNOuterElecs', _MAX_OUTER_ELECTRONS,
                description='Number of outer shell electrons (normalized)',
            ),
            'description': 'Number of outer shell electrons (normalized by 18)',
        },
    }

    edge_attribute_map = {
        'bond_type': {
            'callback': chem_prop('GetBondType', OneHotEncoder(
                [1, 2, 3, 12, 17, 21],
                add_unknown=True,
                dtype=int,
                value_descriptions=[
                    'Single Bond', 'Double Bond', 'Triple Bond',
                    'Aromatic Bond', 'Dative Bond', 'Zero-Order Bond',
                ],
                string_values=['S', 'D', 'T', 'A', 'DAT', 'Z'],
            )),
            'description': 'One-hot encoding of bond type (incl. dative)',
            'is_type': True,
            'encodes_bond': True,
        },
        'stereo': {
            'callback': chem_prop('GetStereo', OneHotEncoder(
                [0, 1, 2, 3],
                add_unknown=False,
                dtype=int,
                value_descriptions=[
                    'Stereo None', 'Stereo Any', 'Stereo Z', 'Stereo E',
                ],
            )),
            'description': 'One-hot encoding of stereo configuration',
        },
        'is_aromatic': {
            'callback': chem_prop('GetIsAromatic', list_identity),
            'description': 'Is bond aromatic?',
        },
        'is_in_ring': {
            'callback': chem_prop('IsInRing', list_identity),
            'description': 'Is bond in a ring?',
        },
        'is_conjugated': {
            'callback': chem_prop('GetIsConjugated', list_identity),
            'description': 'Is bond conjugated?',
        },
        'bond_order': {
            'callback': chem_prop('GetBondTypeAsDouble', list_identity),
            'description': 'Numeric bond order (1.0, 1.5, 2.0, 3.0)',
        },
    }

    # Note: is_dative, dative_direction, and is_metal_metal are NOT in the edge_attribute_map
    # because they cannot be computed via chem_prop callbacks on regular RDKit bonds. They are
    # computed directly in the process() method when constructing dative bond edges.
    # The edge_attribute_map is used for intra-ligand bonds; dative bonds are assembled manually.

    graph_attribute_map = {}  # Graph features are computed directly in process()

    # Number of extra edge features appended manually (is_dative, dative_direction, is_metal_metal)
    _EXTRA_EDGE_FEATURES = 3

    # ==================================================================================
    # INITIALIZATION
    # ==================================================================================

    # We need a mock molecule to initialize the attribute map indices, same as MoleculeProcessing.
    MOCK_MOLECULE = mol_from_smiles('CC')
    MOCK_ATOM = MOCK_MOLECULE.GetAtoms()[0]
    MOCK_BOND = MOCK_MOLECULE.GetBonds()[0]

    def __init__(self, *args, ignore_issues: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.ignore_issues = ignore_issues

        # Compute index arrays for type-relevant attributes, symbol encoding, bond encoding
        # following the same pattern as MoleculeProcessing.
        self.node_type_indices = np.array(self._get_attribute_indices(
            self.node_attribute_map,
            self.MOCK_ATOM,
            lambda data: data.get('is_type', False),
        ), dtype=int)

        self.edge_type_indices = np.array(self._get_attribute_indices(
            self.edge_attribute_map,
            self.MOCK_BOND,
            lambda data: data.get('is_type', False),
        ), dtype=int)

        # Extract the symbol encoder for human-readable atom labels
        symbol_data = self._get_attribute_data(
            self.node_attribute_map,
            lambda data: data.get('encodes_symbol', False),
        )
        self.symbol_encoder = symbol_data['callback'].callback if symbol_data else None

        # Extract the bond encoder for human-readable bond labels
        bond_data = self._get_attribute_data(
            self.edge_attribute_map,
            lambda data: data.get('encodes_bond', False),
        )
        self.bond_encoder = bond_data['callback'].callback if bond_data else None

        # Compute the base edge feature dimension from the attribute map (before extra features)
        mock_edge_attrs = []
        for name, data in self.edge_attribute_map.items():
            mock_edge_attrs += self._apply_callback(data['callback'], self.MOCK_MOLECULE, self.MOCK_BOND)
        self._base_edge_dim = len(mock_edge_attrs)

    # ==================================================================================
    # HELPER METHODS
    # ==================================================================================

    def _apply_callback(self, callback, mol, element) -> List[float]:
        """
        Apply a feature extraction callback to an atom or bond element.
        Handles the ``requires_molecule`` protocol for encoders that need the full Mol context.
        """
        if hasattr(callback, 'requires_molecule') and getattr(callback, 'requires_molecule'):
            return callback(mol, element)
        else:
            return callback(element)

    def _get_attribute_data(self, attribute_map, condition):
        for name, data in attribute_map.items():
            if condition(data):
                return data
        return None

    def _get_attribute_indices(self, attribute_map, element, condition) -> List[int]:
        indices = []
        index = 0
        for name, data in attribute_map.items():
            callback = data['callback']
            value = self._apply_callback(callback, self.MOCK_MOLECULE, element)
            for _ in value:
                if condition(data):
                    indices.append(index)
                index += 1
        return indices

    def _encode_atom(self, mol: Chem.Mol, atom: Chem.Atom) -> List[float]:
        """
        Encode a single atom into a flat feature vector using all node_attribute_map callbacks.
        """
        attributes = []
        for name, data in self.node_attribute_map.items():
            callback = data['callback']
            value = self._apply_callback(callback, mol, atom)
            attributes += value
        return attributes

    def _encode_metal_atom(self, metal_symbol: str, oxidation_state: int) -> List[float]:
        """
        Encode the metal center node. Creates a temporary RDKit atom with the correct symbol
        and formal charge (as a proxy for oxidation state) so that all standard callbacks work.
        """
        # Create a single-atom molecule for the metal so that RDKit atom property methods work.
        # We skip valence sanitization to avoid errors with unusual metal valences.
        metal_smiles = f'[{metal_symbol}+{oxidation_state}]' if oxidation_state > 0 else \
                       f'[{metal_symbol}{oxidation_state}]' if oxidation_state < 0 else \
                       f'[{metal_symbol}]'

        metal_mol = Chem.MolFromSmiles(metal_smiles, sanitize=False)
        if metal_mol is None:
            raise ValueError(f'Could not create RDKit atom for metal "{metal_symbol}" '
                             f'with oxidation state {oxidation_state}')

        metal_atom = metal_mol.GetAtoms()[0]
        return self._encode_atom(metal_mol, metal_atom)

    def _encode_bond(self, mol: Chem.Mol, bond: Chem.Bond) -> List[float]:
        """
        Encode a single intra-ligand bond into a flat feature vector, then append the
        three extra TMC-specific features (is_dative=0, dative_direction=0, is_metal_metal=0).
        """
        attributes = []
        for name, data in self.edge_attribute_map.items():
            callback = data['callback']
            value = self._apply_callback(callback, mol, bond)
            attributes += value

        # Append extra features: is_dative=0, dative_direction=0, is_metal_metal=0
        attributes += [0.0, 0.0, 0.0]
        return attributes

    def _make_dative_edge(self, direction: float = 1.0) -> List[float]:
        """
        Create the feature vector for a dative bond (metal-ligand coordination bond).

        :param direction: +1.0 for donor→acceptor (ligand→metal), -1.0 for reverse.
        """
        # Bond type one-hot: DATIVE is at index 4 in [SINGLE, DOUBLE, TRIPLE, AROMATIC, DATIVE, ZERO] + unknown
        bond_type_onehot = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]  # DATIVE
        stereo_onehot = [1.0, 0.0, 0.0, 0.0]  # Stereo None
        is_aromatic = [0.0]
        is_in_ring = [0.0]
        is_conjugated = [0.0]
        bond_order = [1.0]  # Dative bonds have bond order 1.0

        # Extra TMC features
        is_dative = [1.0]
        dative_direction = [direction]
        is_metal_metal = [0.0]

        return (bond_type_onehot + stereo_onehot + is_aromatic + is_in_ring +
                is_conjugated + bond_order + is_dative + dative_direction + is_metal_metal)

    # ==================================================================================
    # MAIN PROCESSING METHOD
    # ==================================================================================

    def process(self,
                metal: str,
                ligand_smiles: List[str],
                connecting_atom_indices: List[List[int]],
                oxidation_state: int = 0,
                total_charge: int = 0,
                spin_multiplicity: int = 1,
                graph_labels: list = [],
                whole_complex_smiles: Optional[str] = None,
                ) -> GraphDict:
        """
        Convert a decomposed TMC representation into a graph dictionary.

        The method assembles a unified molecular graph by:

        1. Creating the metal center as node 0 with TMC-specific features.
        2. Parsing each ligand SMILES and adding its atoms as nodes and its bonds as edges,
           with atom indices offset to avoid collisions.
        3. Adding dative bond edges from each ligand's connecting atoms to the metal node.
        4. Computing graph-level features.

        :param metal: Element symbol of the metal center (e.g., ``'Fe'``, ``'Pt'``).
        :param ligand_smiles: List of SMILES strings, one per ligand.
        :param connecting_atom_indices: List of lists; for each ligand, the atom indices
            (0-based within that ligand's SMILES) of the donor atoms that coordinate to the metal.
        :param oxidation_state: Formal oxidation state of the metal (e.g., 2 for Fe(II)).
        :param total_charge: Total charge of the complex.
        :param spin_multiplicity: Spin multiplicity (2S+1). Default 1 (singlet).
        :param graph_labels: List of target property values to attach to the graph.
        :param whole_complex_smiles: Optional whole-complex SMILES for ``graph_repr``.
            If not provided, a synthetic representation is constructed.

        :returns: A GraphDict with keys ``node_indices``, ``node_attributes``,
            ``edge_indices``, ``edge_attributes``, ``graph_attributes``, ``graph_labels``,
            ``graph_repr``, ``node_atoms``, ``edge_bonds``.
        """
        # --- Metal center (node index 0) ---
        node_indices: List[int] = [0]
        node_attributes: List[List[float]] = [self._encode_metal_atom(metal, oxidation_state)]
        node_atoms: List[str] = [metal]

        edge_indices: List[List[int]] = []
        edge_attributes: List[List[float]] = []
        edge_bonds: List[str] = []

        # Track the current offset for atom indices across ligands.
        # Node 0 is the metal, so ligand atoms start at index 1.
        atom_offset = 1
        coordination_number = 0

        # --- Process each ligand ---
        for lig_idx, (lig_smi, conn_indices) in enumerate(zip(ligand_smiles, connecting_atom_indices)):

            # Some ligand SMILES from TMC datasets contain unusual valences (e.g., B with
            # 4 bonds in tris(pyrazolyl)borate) because they are fragments that were bonded
            # to a metal. We try standard sanitization first, then fall back to partial
            # sanitization that skips only the valence check but still computes ring info,
            # aromaticity, hybridization, etc.
            mol = Chem.MolFromSmiles(lig_smi)
            if mol is None:
                mol = _mol_from_smiles_lenient(lig_smi)
            if mol is None:
                raise ValueError(f'Could not parse ligand SMILES "{lig_smi}" '
                                 f'(ligand index {lig_idx})')

            # Add ligand atoms as nodes
            for atom in mol.GetAtoms():
                global_idx = atom.GetIdx() + atom_offset
                node_indices.append(global_idx)
                node_attributes.append(self._encode_atom(mol, atom))

                if self.symbol_encoder:
                    node_atoms.append(self.symbol_encoder.encode_string(atom.GetSymbol()))

            # Add intra-ligand bonds as edges
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx() + atom_offset
                j = bond.GetEndAtomIdx() + atom_offset
                edge_indices.append([i, j])
                edge_attributes.append(self._encode_bond(mol, bond))

                if self.bond_encoder:
                    edge_bonds.append(self.bond_encoder.encode_string(bond.GetBondType()))

            # Add dative bond edges from connecting atoms to metal (node 0)
            for conn_idx in conn_indices:
                global_conn_idx = conn_idx + atom_offset

                # Edge: donor atom → metal (ligand→metal direction)
                edge_indices.append([global_conn_idx, 0])
                edge_attributes.append(self._make_dative_edge(direction=1.0))
                edge_bonds.append('DAT')

                coordination_number += 1

            atom_offset += mol.GetNumAtoms()

        # --- Graph-level features (5 dims) ---
        # 1. Molecular weight approximation (sum of all atom masses)
        total_mass = sum(
            _PT.GetAtomicWeight(Chem.Atom(s).GetAtomicNum())
            for s in [metal] + [a for lig in ligand_smiles for a in _get_ligand_atoms(lig)]
        ) if False else 0.0  # placeholder, computed below

        # Compute molecular weight by summing individual ligand weights + metal mass
        metal_mass = _PT.GetAtomicWeight(Chem.Atom(metal).GetAtomicNum())
        ligand_masses = []
        for lig_smi in ligand_smiles:
            lig_mol = Chem.MolFromSmiles(lig_smi)
            if lig_mol is None:
                lig_mol = _mol_from_smiles_lenient(lig_smi)
            if lig_mol is not None:
                ligand_masses.append(Descriptors.ExactMolWt(lig_mol))
            else:
                ligand_masses.append(0.0)
        total_mass = metal_mass + sum(ligand_masses)

        metal_z = Chem.Atom(metal).GetAtomicNum()

        graph_attributes = [
            total_mass,             # molecular weight
            float(total_charge),    # total charge
            1.0,                    # num metal centers (mononuclear)
            float(coordination_number),  # coordination number
            float(metal_z),         # metal atomic number
        ]

        # --- Build graph_repr ---
        if whole_complex_smiles:
            graph_repr = whole_complex_smiles
        else:
            # Synthetic representation: "Fe(II)|[NH3].[NH3].[NH3].[NH3].[NH3].[NH3]"
            graph_repr = f'{metal}({_roman_numeral(oxidation_state)})|{".".join(ligand_smiles)}'

        # --- Assemble GraphDict ---
        graph: GraphDict = {
            'node_indices':     np.array(node_indices, dtype=int),
            'node_attributes':  np.array(node_attributes, dtype=float),
            'edge_indices':     np.array(edge_indices, dtype=int) if edge_indices else np.empty((0, 2), dtype=int),
            'edge_attributes':  np.array(edge_attributes, dtype=float) if edge_attributes else np.empty((0, self._base_edge_dim + self._EXTRA_EDGE_FEATURES), dtype=float),
            'graph_attributes': np.array(graph_attributes, dtype=float),
            'graph_labels':     graph_labels,
            'graph_repr':       graph_repr,
            'node_atoms':       np.array(node_atoms, dtype=str),
            'edge_bonds':       np.array(edge_bonds, dtype=str),
        }

        return graph


# ======================================================================================
# UTILITY FUNCTIONS
# ======================================================================================

def _mol_from_smiles_lenient(smiles: str) -> Optional[Chem.Mol]:
    """
    Parse a SMILES string with partial sanitization that skips only the valence check.

    This is needed for ligand fragments from TMC datasets that have unusual valences
    (e.g., B with 4 bonds in tris(pyrazolyl)borate, O with 3 bonds in bridging oxo
    ligands) because they were extracted from a metal complex where the metal satisfied
    the valence. Ring info, aromaticity, hybridization, and conjugation are still computed
    so that downstream property queries work.

    :param smiles: The SMILES string to parse.
    :returns: An RDKit Mol object, or None if parsing fails entirely.
    """
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None

    try:
        Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
    except Exception:
        # If even partial sanitization fails, return the unsanitized mol as a last resort.
        # Some property queries may return incorrect values, but at least we won't crash.
        pass

    return mol


def _roman_numeral(n: int) -> str:
    """
    Convert an integer to a Roman numeral string for oxidation state display.
    Handles 0 and negative values.
    """
    if n == 0:
        return '0'

    negative = n < 0
    n = abs(n)

    val = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    syms = ['M', 'CM', 'D', 'CD', 'C', 'XC', 'L', 'XL', 'X', 'IX', 'V', 'IV', 'I']

    result = ''
    for i, v in enumerate(val):
        while n >= v:
            result += syms[i]
            n -= v

    return f'-{result}' if negative else result
