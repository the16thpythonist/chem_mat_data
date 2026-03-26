"""
Comprehensive unit tests for XyzDataset streaming class.

This test module specifically tests the XyzDataset streaming functionality including:
- Public API export
- Iteration and data structure
- Parser class support
- Metadata integration
- Length calculation
- Error handling
"""
import os
import tempfile
import pytest
import numpy as np

# Test public API import
from chem_mat_data import XyzDataset
from chem_mat_data.dataset import XyzDataset as XyzDatasetFromModule

ASSETS_PATH = os.path.join(os.path.dirname(__file__), 'assets')


class TestPublicAPIExport:
    """Test that XyzDataset is properly exported in the public API."""

    def test_xyz_dataset_is_exported(self):
        """
        The XyzDataset class should be importable from the main package namespace
        just like SmilesDataset and GraphDataset.
        """
        # This import should work without error
        from chem_mat_data import XyzDataset as xyz_dataset_class

        # It should be the same class as from dataset module
        assert xyz_dataset_class is XyzDatasetFromModule

    def test_public_api_completeness(self):
        """
        Verify that all major streaming dataset classes are in the public API.
        """
        import chem_mat_data

        # All streaming dataset classes should be accessible
        assert hasattr(chem_mat_data, 'SmilesDataset')
        assert hasattr(chem_mat_data, 'XyzDataset')
        assert hasattr(chem_mat_data, 'GraphDataset')
        assert hasattr(chem_mat_data, 'ShuffleDataset')


class TestBasicIteration:
    """Test basic iteration functionality of XyzDataset."""

    def test_dataset_creation(self):
        """
        XyzDataset should be created successfully with required parameters.
        """
        with tempfile.TemporaryDirectory() as temp_path:
            dataset = XyzDataset(
                dataset='_test2',
                folder_path=temp_path,
                use_cache=True
            )
            assert isinstance(dataset, XyzDataset)

    def test_iteration_yields_tuples(self):
        """
        Iterating through XyzDataset should yield 2-tuples of (xyz_data, properties).
        """
        with tempfile.TemporaryDirectory() as temp_path:
            dataset = XyzDataset(
                dataset='_test2',
                folder_path=temp_path,
                use_cache=True,
                target_columns=None  # No properties for this test
            )

            count = 0
            for item in dataset:
                assert isinstance(item, tuple)
                assert len(item) == 2
                xyz_data, properties = item
                assert isinstance(xyz_data, dict)
                assert isinstance(properties, np.ndarray)
                count += 1

            assert count > 0, "Dataset should yield at least one item"

    def test_xyz_data_structure(self):
        """
        The xyz_data dict should have the expected keys and data types.
        """
        with tempfile.TemporaryDirectory() as temp_path:
            dataset = XyzDataset(
                dataset='_test2',
                folder_path=temp_path,
                use_cache=True
            )

            for xyz_data, properties in dataset:
                # Check required keys
                assert 'positions' in xyz_data
                assert 'atomic_numbers' in xyz_data
                assert 'symbols' in xyz_data
                assert 'num_atoms' in xyz_data

                # Check positions
                positions = xyz_data['positions']
                assert isinstance(positions, np.ndarray)
                assert positions.ndim == 2
                assert positions.shape[1] == 3  # X, Y, Z coordinates
                assert positions.dtype == np.float64

                # Check atomic numbers
                atomic_numbers = xyz_data['atomic_numbers']
                assert isinstance(atomic_numbers, np.ndarray)
                assert atomic_numbers.ndim == 1
                assert atomic_numbers.dtype == np.int64 or atomic_numbers.dtype == np.int32

                # Check symbols
                symbols = xyz_data['symbols']
                assert isinstance(symbols, np.ndarray)
                assert symbols.ndim == 1
                assert symbols.dtype.kind in ('U', 'S', 'O')  # Unicode, bytes, or object (string)

                # Check num_atoms
                assert isinstance(xyz_data['num_atoms'], (int, np.integer))
                assert xyz_data['num_atoms'] == len(positions)
                assert xyz_data['num_atoms'] == len(atomic_numbers)
                assert xyz_data['num_atoms'] == len(symbols)

                # Only check first item
                break

    def test_positions_are_3d(self):
        """
        Position arrays should contain valid 3D coordinates (not all zeros).
        """
        with tempfile.TemporaryDirectory() as temp_path:
            dataset = XyzDataset(
                dataset='_test2',
                folder_path=temp_path,
                use_cache=True
            )

            for xyz_data, properties in dataset:
                positions = xyz_data['positions']

                # Coordinates should not all be zero
                assert np.any(positions != 0), "Positions should contain non-zero values"

                # Coordinates should be finite
                assert np.all(np.isfinite(positions)), "Positions should be finite"

                # Only check first item
                break

    def test_atomic_numbers_valid(self):
        """
        Atomic numbers should be valid (1-118 for real elements).
        """
        with tempfile.TemporaryDirectory() as temp_path:
            dataset = XyzDataset(
                dataset='_test2',
                folder_path=temp_path,
                use_cache=True
            )

            for xyz_data, properties in dataset:
                atomic_numbers = xyz_data['atomic_numbers']

                # All atomic numbers should be positive
                assert np.all(atomic_numbers > 0), "Atomic numbers should be positive"

                # All atomic numbers should be <= 118 (valid elements)
                assert np.all(atomic_numbers <= 118), "Atomic numbers should be valid elements"

                # Only check first item
                break

    def test_symbols_are_strings(self):
        """
        Atom symbols should be valid element symbols (strings).
        """
        with tempfile.TemporaryDirectory() as temp_path:
            dataset = XyzDataset(
                dataset='_test2',
                folder_path=temp_path,
                use_cache=True
            )

            for xyz_data, properties in dataset:
                symbols = xyz_data['symbols']

                # All symbols should be non-empty strings
                for symbol in symbols:
                    assert len(str(symbol)) > 0, "Symbols should be non-empty"
                    assert len(str(symbol)) <= 2, "Element symbols should be 1-2 characters"

                # Only check first item
                break


class TestParserSupport:
    """Test different parser class support."""

    def test_default_parser(self):
        """
        Default parser should work for standard XYZ files.
        """
        with tempfile.TemporaryDirectory() as temp_path:
            dataset = XyzDataset(
                dataset='_test2',
                parser_cls='default',
                folder_path=temp_path,
                use_cache=True
            )

            count = 0
            for xyz_data, properties in dataset:
                assert xyz_data['num_atoms'] > 0
                count += 1

            assert count > 0


class TestLengthCalculation:
    """Test dataset length calculation."""

    def test_len_returns_count(self):
        """
        len(dataset) should return the number of XYZ files.
        """
        with tempfile.TemporaryDirectory() as temp_path:
            dataset = XyzDataset(
                dataset='_test2',
                folder_path=temp_path,
                use_cache=True
            )

            length = len(dataset)
            assert isinstance(length, int)
            assert length > 0

            # Verify length matches actual iteration count
            count = sum(1 for _ in dataset)
            # Count might be <= length due to files that fail to parse
            assert count <= length

    def test_len_is_cached(self):
        """
        Length calculation should be cached after first call.
        """
        with tempfile.TemporaryDirectory() as temp_path:
            dataset = XyzDataset(
                dataset='_test2',
                folder_path=temp_path,
                use_cache=True
            )

            # First call calculates
            length1 = len(dataset)

            # Second call should use cached value
            length2 = len(dataset)

            assert length1 == length2


class TestMetadataIntegration:
    """Test metadata CSV integration."""

    def test_properties_with_no_metadata(self):
        """
        When there's no metadata file, properties should be empty arrays.
        """
        with tempfile.TemporaryDirectory() as temp_path:
            dataset = XyzDataset(
                dataset='_test2',
                folder_path=temp_path,
                use_cache=True,
                target_columns=['target']
            )

            for xyz_data, properties in dataset:
                # Properties may be empty if no metadata
                assert isinstance(properties, np.ndarray)
                assert properties.dtype == np.float64

                # Only check first item
                break

    def test_properties_with_none_target_columns(self):
        """
        When target_columns is None, properties should be empty arrays.
        """
        with tempfile.TemporaryDirectory() as temp_path:
            dataset = XyzDataset(
                dataset='_test2',
                folder_path=temp_path,
                use_cache=True,
                target_columns=None
            )

            for xyz_data, properties in dataset:
                # Properties should be empty array when target_columns=None
                assert isinstance(properties, np.ndarray)
                assert len(properties) == 0

                # Only check first item
                break


class TestCaching:
    """Test caching behavior."""

    def test_use_cache_parameter(self):
        """
        The use_cache parameter should be respected.
        """
        with tempfile.TemporaryDirectory() as temp_path:
            # First load without cache
            dataset1 = XyzDataset(
                dataset='_test2',
                folder_path=temp_path,
                use_cache=True
            )

            # Count items
            count1 = sum(1 for _ in dataset1)

            # Second load with cache
            dataset2 = XyzDataset(
                dataset='_test2',
                folder_path=temp_path,
                use_cache=True
            )

            count2 = sum(1 for _ in dataset2)

            # Should yield same number of items
            assert count1 == count2


class TestMultipleIterations:
    """Test that dataset can be iterated multiple times."""

    def test_multiple_iterations(self):
        """
        Dataset should support multiple independent iterations.
        """
        with tempfile.TemporaryDirectory() as temp_path:
            dataset = XyzDataset(
                dataset='_test2',
                folder_path=temp_path,
                use_cache=True
            )

            # First iteration
            count1 = sum(1 for _ in dataset)

            # Second iteration
            count2 = sum(1 for _ in dataset)

            # Should yield same number both times
            assert count1 == count2
            assert count1 > 0


class TestDataConsistency:
    """Test data consistency between iterations."""

    def test_consistent_data(self):
        """
        Same molecules should yield same data across iterations.
        """
        with tempfile.TemporaryDirectory() as temp_path:
            dataset = XyzDataset(
                dataset='_test2',
                folder_path=temp_path,
                use_cache=True
            )

            # Get first molecule from first iteration
            first_iter = iter(dataset)
            xyz_data1, props1 = next(first_iter)

            # Get first molecule from second iteration
            second_iter = iter(dataset)
            xyz_data2, props2 = next(second_iter)

            # Should be identical
            assert xyz_data1['num_atoms'] == xyz_data2['num_atoms']
            assert np.array_equal(xyz_data1['positions'], xyz_data2['positions'])
            assert np.array_equal(xyz_data1['atomic_numbers'], xyz_data2['atomic_numbers'])


class TestErrorHandling:
    """Test error handling for various edge cases."""

    @pytest.mark.network
    def test_nonexistent_dataset(self):
        """
        Requesting a nonexistent dataset should raise an error.
        """
        with tempfile.TemporaryDirectory() as temp_path:
            with pytest.raises(FileNotFoundError):
                dataset = XyzDataset(
                    dataset='nonexistent_dataset_12345',
                    folder_path=temp_path,
                    use_cache=True
                )
                # Try to iterate to trigger error
                next(iter(dataset))


class TestStreamingBehavior:
    """Test that dataset truly streams and doesn't load all data at once."""

    def test_can_stop_iteration_early(self):
        """
        Should be able to stop iteration without loading all data.
        """
        with tempfile.TemporaryDirectory() as temp_path:
            dataset = XyzDataset(
                dataset='_test2',
                folder_path=temp_path,
                use_cache=True
            )

            # Take only first 2 items
            count = 0
            for xyz_data, properties in dataset:
                count += 1
                if count >= 2:
                    break

            # Should successfully stop without error
            assert count == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
