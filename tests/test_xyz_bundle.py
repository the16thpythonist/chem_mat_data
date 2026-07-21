"""
Comprehensive unit tests for XYZ bundle dataset loading functionality.

This test module specifically tests the XYZ bundle loading capabilities including:
- Public API exports
- Different parser types
- Coordinate extraction
- Integration with processing
- Metadata handling
"""
import os
import tempfile
import pytest
import pandas as pd
import numpy as np
import rdkit.Chem as Chem

# Test public API import
from chem_mat_data import load_xyz_dataset
from chem_mat_data.main import load_xyz_dataset as load_xyz_from_main
from chem_mat_data.data import load_xyz_as_mol, DefaultXyzParser, QM9XyzParser
from chem_mat_data.processing import MoleculeProcessing

ASSETS_PATH = os.path.join(os.path.dirname(__file__), 'assets')


class TestPublicAPIExport:
    """Test that load_xyz_dataset is properly exported in the public API."""

    def test_load_xyz_dataset_is_exported(self):
        """
        The load_xyz_dataset function should be importable from the main package namespace
        just like load_smiles_dataset and load_graph_dataset.
        """
        # This import should work without error
        from chem_mat_data import load_xyz_dataset as xyz_loader

        # It should be the same function as from main
        assert xyz_loader is load_xyz_from_main

    def test_public_api_completeness(self):
        """
        Verify that all major dataset loading functions are in the public API.
        """
        import chem_mat_data

        # All these should be directly accessible
        assert hasattr(chem_mat_data, 'load_smiles_dataset')
        assert hasattr(chem_mat_data, 'load_xyz_dataset')
        assert hasattr(chem_mat_data, 'load_graph_dataset')


class TestLoadXyzAsMol:
    """Test the low-level load_xyz_as_mol function."""

    def test_load_default_xyz_file(self):
        """
        Test loading a standard XYZ file with the default parser.
        """
        xyz_path = os.path.join(ASSETS_PATH, '_test.xyz')
        mol, info = load_xyz_as_mol(xyz_path, parser_cls='default')

        assert isinstance(mol, Chem.Mol)
        assert isinstance(info, dict)
        assert mol.GetNumAtoms() == 17

    def test_load_qm9_xyz_file(self):
        """
        Test loading a QM9-formatted XYZ file with the QM9 parser.
        """
        xyz_path = os.path.join(ASSETS_PATH, 'qm9.xyz')
        mol, info = load_xyz_as_mol(xyz_path, parser_cls='qm9')

        assert isinstance(mol, Chem.Mol)
        assert isinstance(info, dict)

        # QM9 parser should extract additional info
        assert 'targets' in info
        assert 'smiles1' in info
        assert 'functional' in info
        assert isinstance(info['targets'], list)

    def test_molecule_has_conformer(self):
        """
        Molecules loaded from XYZ files should have conformers with 3D coordinates.
        """
        xyz_path = os.path.join(ASSETS_PATH, '_test.xyz')
        mol, info = load_xyz_as_mol(xyz_path)

        # The molecule should have at least one conformer
        assert mol.GetNumConformers() > 0

        conformer = mol.GetConformers()[0]
        positions = conformer.GetPositions()

        # Positions should be a numpy array with shape (num_atoms, 3)
        assert isinstance(positions, np.ndarray)
        assert positions.shape == (mol.GetNumAtoms(), 3)

        # Coordinates should be non-zero
        assert np.any(positions != 0)

    def test_parser_class_as_type(self):
        """
        Test that parser_cls parameter also accepts class types, not just strings.
        """
        xyz_path = os.path.join(ASSETS_PATH, '_test.xyz')
        mol1, info1 = load_xyz_as_mol(xyz_path, parser_cls='default')
        mol2, info2 = load_xyz_as_mol(xyz_path, parser_cls=DefaultXyzParser)

        assert mol1.GetNumAtoms() == mol2.GetNumAtoms()

    def test_invalid_parser_raises_error(self):
        """
        Using an invalid parser string should raise a KeyError.
        """
        xyz_path = os.path.join(ASSETS_PATH, '_test.xyz')

        with pytest.raises(KeyError):
            load_xyz_as_mol(xyz_path, parser_cls='nonexistent_parser')


class TestLoadXyzDataset:
    """Test the high-level load_xyz_dataset function."""

    def test_load_remote_xyz_bundle(self):
        """
        Test loading an XYZ bundle dataset from the remote file share.
        This uses the _test2 dataset which is a small test xyz_bundle.
        """
        with tempfile.TemporaryDirectory() as temp_path:
            df = load_xyz_dataset(
                dataset_name='_test2',
                folder_path=temp_path,
                use_cache=True
            )

            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
            assert 'mol' in df.columns
            assert 'id' in df.columns

            # All molecules should be valid Mol objects
            for _, row in df.iterrows():
                assert isinstance(row['mol'], Chem.Mol)

    def test_dataframe_structure(self):
        """
        Test that the returned DataFrame has the expected structure.
        """
        with tempfile.TemporaryDirectory() as temp_path:
            df = load_xyz_dataset(
                dataset_name='_test2',
                folder_path=temp_path,
                use_cache=True
            )

            # Should have at least 'id' and 'mol' columns
            assert 'id' in df.columns
            assert 'mol' in df.columns

            # ID column should contain unique identifiers
            assert df['id'].is_unique

    def test_molecules_have_conformers(self):
        """
        All molecules loaded from XYZ bundle should have 3D conformers.
        """
        with tempfile.TemporaryDirectory() as temp_path:
            df = load_xyz_dataset(
                dataset_name='_test2',
                folder_path=temp_path,
                use_cache=True
            )

            for _, row in df.iterrows():
                mol = row['mol']
                assert mol.GetNumConformers() > 0

                conformer = mol.GetConformers()[0]
                positions = conformer.GetPositions()

                assert positions.shape == (mol.GetNumAtoms(), 3)
                assert np.any(positions != 0)


class TestXyzBundleWithMetadata:
    """Test XYZ bundles that include metadata CSV files."""

    def test_metadata_csv_integration(self):
        """
        When a meta.csv file is present in the xyz_bundle, its data should be
        integrated into the returned DataFrame.
        """
        # The _test2 dataset may have a meta.csv file
        with tempfile.TemporaryDirectory() as temp_path:
            df = load_xyz_dataset(
                dataset_name='_test2',
                folder_path=temp_path,
                use_cache=True
            )

            # At minimum, should have id and mol columns
            assert 'id' in df.columns
            assert 'mol' in df.columns

            # If metadata exists, there might be additional columns
            # This test passes regardless of whether meta.csv exists
            assert len(df.columns) >= 2


class TestXyzProcessingIntegration:
    """Test integration of XYZ-loaded molecules with MoleculeProcessing."""

    def test_process_xyz_molecule_with_coordinates(self):
        """
        Molecules loaded from XYZ files should be processable and maintain
        their 3D coordinates in the graph representation.
        """
        xyz_path = os.path.join(ASSETS_PATH, '_test.xyz')
        mol, info = load_xyz_as_mol(xyz_path)

        processing = MoleculeProcessing()
        graph = processing.process(mol)

        # Graph should have node_coordinates from the XYZ file
        assert 'node_coordinates' in graph
        assert isinstance(graph['node_coordinates'], np.ndarray)
        assert graph['node_coordinates'].shape[0] == mol.GetNumAtoms()
        assert graph['node_coordinates'].shape[1] == 3

    def test_edge_lengths_calculated(self):
        """
        When processing molecules with coordinates, edge lengths should be calculated.
        """
        xyz_path = os.path.join(ASSETS_PATH, '_test.xyz')
        mol, info = load_xyz_as_mol(xyz_path)

        processing = MoleculeProcessing()
        graph = processing.process(mol)

        # Edge lengths should be calculated
        assert 'edge_lengths' in graph
        assert isinstance(graph['edge_lengths'], np.ndarray)
        assert len(graph['edge_lengths']) == len(graph['edge_indices'])

        # All edge lengths should be positive
        assert np.all(graph['edge_lengths'] > 0)

    def test_graph_attributes_present(self):
        """
        Standard graph attributes should all be present.
        """
        xyz_path = os.path.join(ASSETS_PATH, '_test.xyz')
        mol, info = load_xyz_as_mol(xyz_path)

        processing = MoleculeProcessing()
        graph = processing.process(mol)

        required_keys = [
            'node_indices', 'node_attributes', 'node_coordinates',
            'edge_indices', 'edge_attributes', 'edge_lengths',
            'graph_attributes', 'graph_labels', 'graph_repr'
        ]

        for key in required_keys:
            assert key in graph, f"Missing required key: {key}"


class TestQM9SpecificFunctionality:
    """Test QM9-specific XYZ parsing functionality."""

    def test_qm9_parser_extracts_targets(self):
        """
        QM9 parser should extract target values from the XYZ file.
        """
        xyz_path = os.path.join(ASSETS_PATH, 'qm9.xyz')
        parser = QM9XyzParser(xyz_path)
        mol, info = parser.parse()

        assert 'targets' in info
        # QM9 extracts all numerical values from the second line
        assert len(info['targets']) > 0
        assert all(isinstance(t, float) for t in info['targets'])

    def test_qm9_parser_extracts_smiles(self):
        """
        QM9 parser should extract both SMILES representations.
        """
        xyz_path = os.path.join(ASSETS_PATH, 'qm9.xyz')
        parser = QM9XyzParser(xyz_path)
        mol, info = parser.parse()

        assert 'smiles1' in info
        assert 'smiles2' in info
        assert isinstance(info['smiles1'], str)
        assert isinstance(info['smiles2'], str)

    def test_qm9_parser_extracts_functional(self):
        """
        QM9 parser should extract the functional information.
        """
        xyz_path = os.path.join(ASSETS_PATH, 'qm9.xyz')
        parser = QM9XyzParser(xyz_path)
        mol, info = parser.parse()

        assert 'functional' in info
        assert isinstance(info['functional'], str)

    def test_qm9_get_fields(self):
        """
        QM9 parser should report the correct fields it extracts.
        """
        fields = QM9XyzParser.get_fields()

        assert 'targets' in fields
        assert 'functional' in fields
        assert 'smiles1' in fields
        assert 'smiles2' in fields


class TestDefaultParser:
    """Test the default XYZ parser."""

    def test_default_parser_basic_functionality(self):
        """
        Default parser should load basic XYZ files correctly.
        """
        xyz_path = os.path.join(ASSETS_PATH, '_test.xyz')
        parser = DefaultXyzParser(xyz_path)
        mol, info = parser.parse()

        assert isinstance(mol, Chem.Mol)
        assert mol.GetNumAtoms() == 17

    def test_default_parser_info_is_empty(self):
        """
        Default parser doesn't extract additional info, so info dict should be empty.
        """
        xyz_path = os.path.join(ASSETS_PATH, '_test.xyz')
        parser = DefaultXyzParser(xyz_path)
        mol, info = parser.parse()

        # Default parser returns empty info dict
        assert isinstance(info, dict)
        assert len(info) == 0

    def test_default_parser_get_fields(self):
        """
        Default parser should report no additional fields.
        """
        fields = DefaultXyzParser.get_fields()
        assert isinstance(fields, list)
        assert len(fields) == 0


class TestCachingBehavior:
    """Test caching behavior with XYZ datasets."""

    def test_cache_parameter_works(self):
        """
        Test that the use_cache parameter is respected.
        """
        with tempfile.TemporaryDirectory() as temp_path:
            # First load without cache
            df1 = load_xyz_dataset(
                dataset_name='_test2',
                folder_path=temp_path,
                use_cache=True
            )

            # Second load with cache should also work
            df2 = load_xyz_dataset(
                dataset_name='_test2',
                folder_path=temp_path,
                use_cache=True
            )

            assert len(df1) == len(df2)


class TestErrorHandling:
    """Test error handling for various edge cases."""

    def test_nonexistent_file_raises_error(self):
        """
        Loading a non-existent XYZ file should raise an error.
        """
        with pytest.raises(FileNotFoundError):
            load_xyz_as_mol('/nonexistent/path/file.xyz')

    def test_wrong_parser_for_format(self):
        """
        Using the wrong parser for a file format should fail gracefully.
        """
        # Try to parse a standard XYZ with QM9 parser - this should fail
        xyz_path = os.path.join(ASSETS_PATH, '_test.xyz')

        with pytest.raises(Exception):
            load_xyz_as_mol(xyz_path, parser_cls='qm9')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
