import numpy as np
import pytest

from chem_mat_data.testing import assert_graph_dict
from chem_mat_data.tmc_processing import (
    MetalOrganicProcessing,
    TRANSITION_METAL_ATOMIC_NUMBERS,
    PAULING_ELECTRONEGATIVITY,
    ELEMENT_GROUP,
    TMC_ELEMENTS,
    LookupEncoder,
    PeriodicTableEncoder,
    MetalFlagEncoder,
    DElectronCountEncoder,
)

# Expected feature dimensions based on the design spec.
EXPECTED_NODE_DIM = 91
EXPECTED_EDGE_DIM = 18
EXPECTED_GRAPH_DIM = 5


class TestLookupTables:
    """Verify that the static lookup tables are consistent and complete."""

    def test_transition_metals_count(self):
        # 29 standard TMs (3d: 10, 4d: 10, 5d: 9) + La = 30
        assert len(TRANSITION_METAL_ATOMIC_NUMBERS) == 30

    def test_tmc_elements_count(self):
        assert len(TMC_ELEMENTS) == 50

    def test_tmc_elements_no_duplicates(self):
        assert len(TMC_ELEMENTS) == len(set(TMC_ELEMENTS))

    def test_all_tms_in_element_list(self):
        """Every transition metal should appear in the TMC_ELEMENTS list."""
        import rdkit.Chem as Chem
        pt = Chem.GetPeriodicTable()
        for z in TRANSITION_METAL_ATOMIC_NUMBERS:
            symbol = pt.GetElementSymbol(z)
            assert symbol in TMC_ELEMENTS, f'{symbol} (Z={z}) not in TMC_ELEMENTS'

    def test_electronegativity_has_all_tms(self):
        for z in TRANSITION_METAL_ATOMIC_NUMBERS:
            assert z in PAULING_ELECTRONEGATIVITY, f'Z={z} missing from PAULING_ELECTRONEGATIVITY'

    def test_group_has_all_tms(self):
        for z in TRANSITION_METAL_ATOMIC_NUMBERS:
            assert z in ELEMENT_GROUP, f'Z={z} missing from ELEMENT_GROUP'


class TestCustomEncoders:
    """Test the TMC-specific encoder classes."""

    def test_metal_flag_encoder_iron(self):
        import rdkit.Chem as Chem
        mol = Chem.MolFromSmiles('[Fe]', sanitize=False)
        atom = mol.GetAtoms()[0]
        encoder = MetalFlagEncoder()
        assert encoder(atom) == [1.0]

    def test_metal_flag_encoder_carbon(self):
        import rdkit.Chem as Chem
        mol = Chem.MolFromSmiles('C')
        atom = mol.GetAtoms()[0]
        encoder = MetalFlagEncoder()
        assert encoder(atom) == [0.0]

    def test_d_electron_count_fe2(self):
        """Fe(II): group 8, formal charge +2 → d6 → 6/10 = 0.6"""
        import rdkit.Chem as Chem
        mol = Chem.MolFromSmiles('[Fe+2]', sanitize=False)
        atom = mol.GetAtoms()[0]
        encoder = DElectronCountEncoder()
        result = encoder(atom)
        assert len(result) == 1
        assert abs(result[0] - 0.6) < 1e-6

    def test_d_electron_count_non_metal(self):
        """Non-metals should return 0.0."""
        import rdkit.Chem as Chem
        mol = Chem.MolFromSmiles('N')
        atom = mol.GetAtoms()[0]
        encoder = DElectronCountEncoder()
        assert encoder(atom) == [0.0]

    def test_lookup_encoder_iron(self):
        import rdkit.Chem as Chem
        mol = Chem.MolFromSmiles('[Fe]', sanitize=False)
        atom = mol.GetAtoms()[0]
        encoder = LookupEncoder(PAULING_ELECTRONEGATIVITY, 3.98)
        result = encoder(atom)
        assert len(result) == 1
        assert abs(result[0] - 1.83 / 3.98) < 1e-6

    def test_periodic_table_encoder(self):
        import rdkit.Chem as Chem
        mol = Chem.MolFromSmiles('[Fe]', sanitize=False)
        atom = mol.GetAtoms()[0]
        encoder = PeriodicTableEncoder('GetRcovalent', 2.6)
        result = encoder(atom)
        assert len(result) == 1
        assert result[0] > 0  # Fe has a positive covalent radius


class TestMetalOrganicProcessing:
    """Test the main MetalOrganicProcessing class."""

    def test_instantiation(self):
        processing = MetalOrganicProcessing()
        assert isinstance(processing, MetalOrganicProcessing)
        assert processing.symbol_encoder is not None
        assert processing.bond_encoder is not None

    def test_octahedral_fe_nh3(self):
        """
        [Fe(NH3)6]2+ — hexaammineiron(II). The simplest octahedral complex.
        Should produce 7 nodes (1 Fe + 6 N) and 6 edges (6 dative bonds).
        """
        processing = MetalOrganicProcessing()
        graph = processing.process(
            metal='Fe',
            ligand_smiles=['[NH3]'] * 6,
            connecting_atom_indices=[[0]] * 6,
            oxidation_state=2,
            total_charge=2,
            graph_labels=[1.0],
        )

        assert_graph_dict(graph)

        # 1 Fe + 6 N = 7 nodes
        assert graph['node_attributes'].shape[0] == 7
        # 6 dative bonds, no intra-ligand bonds (NH3 is a single atom in SMILES)
        assert graph['edge_attributes'].shape[0] == 6

    def test_node_feature_dimensions(self):
        """Node attribute vectors should be 91-dimensional."""
        processing = MetalOrganicProcessing()
        graph = processing.process(
            metal='Fe',
            ligand_smiles=['[NH3]'] * 6,
            connecting_atom_indices=[[0]] * 6,
            oxidation_state=2,
        )
        assert graph['node_attributes'].shape[1] == EXPECTED_NODE_DIM

    def test_edge_feature_dimensions(self):
        """Edge attribute vectors should be 18-dimensional."""
        processing = MetalOrganicProcessing()
        graph = processing.process(
            metal='Fe',
            ligand_smiles=['[NH3]'] * 6,
            connecting_atom_indices=[[0]] * 6,
            oxidation_state=2,
        )
        assert graph['edge_attributes'].shape[1] == EXPECTED_EDGE_DIM

    def test_graph_feature_dimensions(self):
        """Graph attribute vector should be 5-dimensional."""
        processing = MetalOrganicProcessing()
        graph = processing.process(
            metal='Fe',
            ligand_smiles=['[NH3]'] * 6,
            connecting_atom_indices=[[0]] * 6,
            oxidation_state=2,
        )
        assert graph['graph_attributes'].shape[0] == EXPECTED_GRAPH_DIM

    def test_square_planar_pt_cl4(self):
        """
        [PtCl4]2- — tetrachloroplatinate(II). Square planar d8 complex.
        5 nodes (1 Pt + 4 Cl), 4 dative edges.
        """
        processing = MetalOrganicProcessing()
        graph = processing.process(
            metal='Pt',
            ligand_smiles=['[Cl-]'] * 4,
            connecting_atom_indices=[[0]] * 4,
            oxidation_state=2,
            total_charge=-2,
            graph_labels=[2.0],
        )

        assert_graph_dict(graph)
        assert graph['node_attributes'].shape == (5, EXPECTED_NODE_DIM)
        assert graph['edge_attributes'].shape == (4, EXPECTED_EDGE_DIM)

    def test_bidentate_ligand_en(self):
        """
        [Fe(en)3]2+ — tris(ethylenediamine)iron(II).
        Ethylenediamine (en) = H2N-CH2-CH2-NH2, SMILES: NCCN, connecting atoms [0, 3].
        Nodes: 1 Fe + 3*(4 heavy atoms in NCCN) = 13 nodes.
        Intra-ligand edges: 3 bonds per en * 3 = 9.
        Dative edges: 2 per en * 3 = 6.
        Total edges: 15.
        """
        processing = MetalOrganicProcessing()
        graph = processing.process(
            metal='Fe',
            ligand_smiles=['NCCN'] * 3,
            connecting_atom_indices=[[0, 3]] * 3,
            oxidation_state=2,
            total_charge=2,
            graph_labels=[3.0],
        )

        assert_graph_dict(graph)
        assert graph['node_attributes'].shape[0] == 13  # 1 + 3*4
        assert graph['edge_attributes'].shape[0] == 15  # 9 intra + 6 dative
        assert graph['node_attributes'].shape[1] == EXPECTED_NODE_DIM
        assert graph['edge_attributes'].shape[1] == EXPECTED_EDGE_DIM

    def test_dative_edges_have_correct_features(self):
        """Dative bond edges should have is_dative=1.0, dative_direction=1.0, is_metal_metal=0.0."""
        processing = MetalOrganicProcessing()
        graph = processing.process(
            metal='Fe',
            ligand_smiles=['[NH3]'],
            connecting_atom_indices=[[0]],
            oxidation_state=2,
        )

        # Only one edge: the dative bond
        assert graph['edge_attributes'].shape[0] == 1
        edge = graph['edge_attributes'][0]

        # Last 3 features are: is_dative, dative_direction, is_metal_metal
        is_dative = edge[-3]
        dative_direction = edge[-2]
        is_metal_metal = edge[-1]

        assert is_dative == 1.0
        assert dative_direction == 1.0
        assert is_metal_metal == 0.0

    def test_intra_ligand_edges_not_dative(self):
        """Intra-ligand bonds should have is_dative=0.0."""
        processing = MetalOrganicProcessing()
        graph = processing.process(
            metal='Fe',
            ligand_smiles=['NCCN'],
            connecting_atom_indices=[[0, 3]],
            oxidation_state=2,
        )

        # NCCN has 3 intra-ligand bonds + 2 dative bonds = 5 edges
        assert graph['edge_attributes'].shape[0] == 5

        # First 3 edges are intra-ligand (N-C, C-C, C-N)
        for i in range(3):
            assert graph['edge_attributes'][i][-3] == 0.0  # is_dative = 0

        # Last 2 edges are dative
        for i in range(3, 5):
            assert graph['edge_attributes'][i][-3] == 1.0  # is_dative = 1

    def test_metal_node_is_first(self):
        """The metal center should always be node index 0."""
        processing = MetalOrganicProcessing()
        graph = processing.process(
            metal='Ru',
            ligand_smiles=['[NH3]'] * 6,
            connecting_atom_indices=[[0]] * 6,
            oxidation_state=2,
        )

        assert graph['node_indices'][0] == 0
        assert graph['node_atoms'][0] == 'Ru'

    def test_graph_repr_synthetic(self):
        """Without whole_complex_smiles, graph_repr should be a synthetic string."""
        processing = MetalOrganicProcessing()
        graph = processing.process(
            metal='Fe',
            ligand_smiles=['[NH3]', '[NH3]'],
            connecting_atom_indices=[[0], [0]],
            oxidation_state=2,
        )
        assert 'Fe' in graph['graph_repr']
        assert 'II' in graph['graph_repr']

    def test_graph_repr_from_smiles(self):
        """When whole_complex_smiles is provided, it should be used as graph_repr."""
        processing = MetalOrganicProcessing()
        custom_smiles = '[Fe+2](<-[NH3])(<-[NH3])'
        graph = processing.process(
            metal='Fe',
            ligand_smiles=['[NH3]', '[NH3]'],
            connecting_atom_indices=[[0], [0]],
            oxidation_state=2,
            whole_complex_smiles=custom_smiles,
        )
        assert graph['graph_repr'] == custom_smiles

    def test_graph_attributes_content(self):
        """Verify the 5 graph-level features have expected values."""
        processing = MetalOrganicProcessing()
        graph = processing.process(
            metal='Fe',
            ligand_smiles=['[NH3]'] * 6,
            connecting_atom_indices=[[0]] * 6,
            oxidation_state=2,
            total_charge=2,
        )
        attrs = graph['graph_attributes']

        # [mol_weight, total_charge, num_metals, CN, metal_Z]
        assert attrs[1] == 2.0   # total_charge
        assert attrs[2] == 1.0   # num_metals (mononuclear)
        assert attrs[3] == 6.0   # coordination_number
        assert attrs[4] == 26.0  # Fe atomic number

    def test_invalid_ligand_smiles_raises(self):
        """An invalid ligand SMILES should raise a ValueError."""
        processing = MetalOrganicProcessing()
        with pytest.raises(ValueError, match='Could not parse ligand SMILES'):
            processing.process(
                metal='Fe',
                ligand_smiles=['INVALID_SMILES'],
                connecting_atom_indices=[[0]],
                oxidation_state=2,
            )

    def test_invalid_metal_raises(self):
        """An invalid metal symbol should raise a ValueError."""
        processing = MetalOrganicProcessing()
        with pytest.raises(ValueError, match='Could not create RDKit atom'):
            processing.process(
                metal='Xx',
                ligand_smiles=['[NH3]'],
                connecting_atom_indices=[[0]],
            )

    def test_different_metals(self):
        """Processing should work for various common transition metals."""
        processing = MetalOrganicProcessing()
        for metal, ox_state in [('Fe', 2), ('Co', 3), ('Ni', 2), ('Cu', 2),
                                 ('Zn', 2), ('Pt', 2), ('Pd', 2), ('Ru', 2)]:
            graph = processing.process(
                metal=metal,
                ligand_smiles=['[NH3]'] * 4,
                connecting_atom_indices=[[0]] * 4,
                oxidation_state=ox_state,
            )
            assert_graph_dict(graph)
            assert graph['node_atoms'][0] == metal
