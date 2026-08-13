"""
Unit tests for streaming dataset functionality.
"""

import os
import tempfile
import pytest
import numpy as np
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem

from chem_mat_data.dataset import (
    SmilesDataset,
    XyzDataset,
    GraphDataset,
    ShuffleDataset,
    identity_mol_transform,
)
from chem_mat_data.processing import MoleculeProcessing


class TestSmilesDataset:
    """Test suite for SmilesDataset class."""

    def test_initialization(self):
        """Test that SmilesDataset initializes correctly."""
        with tempfile.TemporaryDirectory() as temp_path:
            dataset = SmilesDataset(
                dataset='_test',
                smiles_column='smiles',
                target_columns=['target'],
                folder_path=temp_path,
                use_cache=True,
            )

            assert dataset.dataset == '_test'
            assert dataset.smiles_column == 'smiles'
            assert dataset.target_columns == ['target']
            assert os.path.exists(dataset.file_path)

    def test_iteration(self):
        """Test that iteration yields correct tuples."""
        with tempfile.TemporaryDirectory() as temp_path:
            dataset = SmilesDataset(
                dataset='_test', folder_path=temp_path, use_cache=True
            )

            count = 0
            for smiles, properties in dataset:
                assert isinstance(smiles, str)
                assert isinstance(properties, np.ndarray)
                assert properties.dtype == np.float64
                assert len(properties) == 1  # _test has 1 target column
                count += 1

            assert count > 0

    def test_length(self):
        """Test that __len__ returns correct count."""
        with tempfile.TemporaryDirectory() as temp_path:
            dataset = SmilesDataset(
                dataset='_test', folder_path=temp_path, use_cache=True
            )

            length = len(dataset)
            assert length > 0
            assert isinstance(length, int)

            # Test that length is cached
            length2 = len(dataset)
            assert length == length2

    def test_length_matches_iteration(self):
        """Test that __len__ matches actual iteration count."""
        with tempfile.TemporaryDirectory() as temp_path:
            dataset = SmilesDataset(
                dataset='_test', folder_path=temp_path, use_cache=True
            )

            expected_length = len(dataset)
            actual_count = sum(1 for _ in dataset)

            assert expected_length == actual_count

    def test_multiple_iterations(self):
        """Test that dataset can be iterated multiple times."""
        with tempfile.TemporaryDirectory() as temp_path:
            dataset = SmilesDataset(
                dataset='_test', folder_path=temp_path, use_cache=True
            )

            # First iteration
            count1 = sum(1 for _ in dataset)

            # Second iteration
            count2 = sum(1 for _ in dataset)

            assert count1 == count2
            assert count1 > 0

    def test_multiple_target_columns(self):
        """Test dataset with multiple target columns."""
        with tempfile.TemporaryDirectory() as temp_path:
            dataset = SmilesDataset(
                dataset='bace_cls',
                smiles_column='smiles',
                target_columns=['target', 'target_1'],
                folder_path=temp_path,
                use_cache=True,
            )

            for smiles, properties in dataset:
                assert len(properties) == 2  # Two target columns
                # Should handle boolean values converted to floats
                assert all(isinstance(p, (float, np.floating)) for p in properties)
                break  # Only check first row

    def test_boolean_conversion(self):
        """Test that boolean strings are converted to floats."""
        with tempfile.TemporaryDirectory() as temp_path:
            dataset = SmilesDataset(
                dataset='bace_cls',
                smiles_column='smiles',
                target_columns=['target'],
                folder_path=temp_path,
                use_cache=True,
            )

            for smiles, properties in dataset:
                # Boolean columns should be converted to 0.0 or 1.0
                assert properties[0] in [0.0, 1.0]
                break


class TestGraphDataset:
    """Test suite for GraphDataset class."""

    def test_initialization_sequential(self):
        """Test GraphDataset initialization in sequential mode."""
        with tempfile.TemporaryDirectory() as temp_path:
            dataset = GraphDataset(
                dataset='_test', num_workers=0, folder_path=temp_path, use_cache=True
            )

            assert dataset.dataset == '_test'
            assert dataset.num_workers == 0
            assert dataset.processing_class == MoleculeProcessing
            # Check that raw dataset was created and type detected
            assert hasattr(dataset, 'raw_dataset')
            assert dataset.dataset_type == 'smiles'
            assert isinstance(dataset.raw_dataset, SmilesDataset)

    def test_initialization_parallel(self):
        """Test GraphDataset initialization in parallel mode."""
        with tempfile.TemporaryDirectory() as temp_path:
            dataset = GraphDataset(
                dataset='_test', num_workers=2, folder_path=temp_path, use_cache=True
            )

            assert dataset.num_workers == 2
            assert dataset.buffer_size == 100  # Default

    def test_sequential_iteration(self):
        """Test that sequential mode yields valid graph dicts."""
        with tempfile.TemporaryDirectory() as temp_path:
            dataset = GraphDataset(
                dataset='_test', num_workers=0, folder_path=temp_path, use_cache=True
            )

            count = 0
            for graph in dataset:
                # Check graph dict structure
                assert isinstance(graph, dict)
                assert 'node_indices' in graph
                assert 'node_attributes' in graph
                assert 'edge_indices' in graph
                assert 'edge_attributes' in graph
                assert 'graph_labels' in graph

                # Check types
                assert isinstance(graph['node_indices'], np.ndarray)
                assert isinstance(graph['node_attributes'], np.ndarray)
                assert isinstance(graph['edge_indices'], np.ndarray)
                assert isinstance(graph['edge_attributes'], np.ndarray)

                count += 1

            assert count > 0

    def test_parallel_iteration(self):
        """Test that parallel mode yields valid graph dicts."""
        with tempfile.TemporaryDirectory() as temp_path:
            dataset = GraphDataset(
                dataset='_test', num_workers=2, folder_path=temp_path, use_cache=True
            )

            count = 0
            for graph in dataset:
                # Check graph dict structure
                assert isinstance(graph, dict)
                assert 'node_indices' in graph
                assert 'node_attributes' in graph
                assert 'edge_indices' in graph
                assert 'edge_attributes' in graph
                assert 'graph_labels' in graph

                count += 1

            assert count > 0

    def test_order_preservation(self):
        """Test that parallel processing preserves order."""
        with tempfile.TemporaryDirectory() as temp_path:
            # Sequential results
            dataset_seq = GraphDataset(
                dataset='_test', num_workers=0, folder_path=temp_path, use_cache=True
            )
            results_seq = list(dataset_seq)

            # Parallel results
            dataset_par = GraphDataset(
                dataset='_test', num_workers=2, folder_path=temp_path, use_cache=True
            )
            results_par = list(dataset_par)

            # Check same length
            assert len(results_seq) == len(results_par)

            # Check order is preserved by comparing graph_labels
            for i, (g_seq, g_par) in enumerate(zip(results_seq, results_par)):
                assert g_seq['graph_labels'] == g_par['graph_labels'], (
                    f'Order mismatch at index {i}'
                )

    def test_length(self):
        """Test that __len__ works correctly."""
        with tempfile.TemporaryDirectory() as temp_path:
            dataset = GraphDataset(
                dataset='_test', num_workers=0, folder_path=temp_path, use_cache=True
            )

            length = len(dataset)
            assert length > 0
            assert isinstance(length, int)

    def test_multiple_iterations(self):
        """Test that GraphDataset can be iterated multiple times."""
        with tempfile.TemporaryDirectory() as temp_path:
            dataset = GraphDataset(
                dataset='_test', num_workers=0, folder_path=temp_path, use_cache=True
            )

            # First iteration
            count1 = sum(1 for _ in dataset)

            # Second iteration
            count2 = sum(1 for _ in dataset)

            assert count1 == count2
            assert count1 > 0

    def test_manual_cleanup(self):
        """Test that manual cleanup works."""
        with tempfile.TemporaryDirectory() as temp_path:
            dataset = GraphDataset(
                dataset='_test', num_workers=2, folder_path=temp_path, use_cache=True
            )

            # Process one item
            for _ in dataset:
                break

            # Manual cleanup should not raise
            dataset.close()
            dataset.close()  # Should be safe to call twice

    def test_graph_labels_match_targets(self):
        """Test that graph_labels contain the correct target values."""
        with tempfile.TemporaryDirectory() as temp_path:
            # Get SMILES data
            smiles_dataset = SmilesDataset(
                dataset='_test', folder_path=temp_path, use_cache=True
            )
            smiles_data = list(smiles_dataset)

            # Get graph data
            graph_dataset = GraphDataset(
                dataset='_test', num_workers=0, folder_path=temp_path, use_cache=True
            )
            graph_data = list(graph_dataset)

            # Check that labels match
            for (_, props), graph in zip(smiles_data, graph_data):
                expected = props.tolist()
                actual = graph['graph_labels']
                assert actual == expected

    def test_custom_processing_class(self):
        """Test that custom processing class can be used."""
        # This test just verifies the parameter works
        # We use the default MoleculeProcessing since we don't have another
        with tempfile.TemporaryDirectory() as temp_path:
            dataset = GraphDataset(
                dataset='_test',
                num_workers=0,
                processing_class=MoleculeProcessing,
                folder_path=temp_path,
                use_cache=True,
            )

            # Should work fine
            count = sum(1 for _ in dataset)
            assert count > 0

    def test_buffer_size_parameter(self):
        """Test that buffer_size parameter is accepted."""
        with tempfile.TemporaryDirectory() as temp_path:
            dataset = GraphDataset(
                dataset='_test',
                num_workers=2,
                buffer_size=50,
                folder_path=temp_path,
                use_cache=True,
            )

            assert dataset.buffer_size == 50

            # Should work fine
            count = sum(1 for _ in dataset)
            assert count > 0

    def test_different_worker_counts(self):
        """Test that different worker counts all produce valid results."""
        with tempfile.TemporaryDirectory() as temp_path:
            for num_workers in [0, 1, 2, 4]:
                dataset = GraphDataset(
                    dataset='_test',
                    num_workers=num_workers,
                    folder_path=temp_path,
                    use_cache=True,
                )

                graphs = list(dataset)
                assert len(graphs) > 0

                # All should produce same number of graphs
                if num_workers == 0:
                    expected_count = len(graphs)
                else:
                    assert len(graphs) == expected_count


class TestGraphDatasetMolTransform:
    """Test suite for mol_transform functionality in GraphDataset."""

    def test_identity_transform_default(self):
        """Test that default identity transform works correctly."""
        with tempfile.TemporaryDirectory() as temp_path:
            dataset = GraphDataset(
                dataset='_test', num_workers=0, folder_path=temp_path, use_cache=True
            )

            # Should use identity_mol_transform by default
            assert dataset.mol_transform == identity_mol_transform

            # Should produce valid graphs
            for graph in dataset:
                assert isinstance(graph, dict)
                assert 'node_indices' in graph
                assert len(graph['node_indices']) > 0
                break

    def test_custom_transform_sequential(self):
        """Test that custom transform works in sequential mode."""

        def add_hydrogens(mol):
            return Chem.AddHs(mol)

        with tempfile.TemporaryDirectory() as temp_path:
            dataset = GraphDataset(
                dataset='_test',
                num_workers=0,
                mol_transform=add_hydrogens,
                folder_path=temp_path,
                use_cache=True,
            )

            # Should produce valid graphs with more nodes (due to H atoms)
            count = 0
            for graph in dataset:
                assert isinstance(graph, dict)
                assert len(graph['node_indices']) > 0
                count += 1

            assert count > 0

    def test_custom_transform_parallel(self):
        """Test that custom transform works in parallel mode."""

        def add_hydrogens(mol):
            return Chem.AddHs(mol)

        with tempfile.TemporaryDirectory() as temp_path:
            dataset = GraphDataset(
                dataset='_test',
                num_workers=2,
                mol_transform=add_hydrogens,
                folder_path=temp_path,
                use_cache=True,
            )

            # Should produce valid graphs - test first 10
            count = 0
            for graph in dataset:
                assert isinstance(graph, dict)
                assert len(graph['node_indices']) > 0
                count += 1
                if count >= 10:
                    break

            assert count > 0

    def test_add_explicit_hydrogens(self):
        """Test adding explicit hydrogens increases node count."""
        with tempfile.TemporaryDirectory() as temp_path:
            # Without hydrogens (default)
            dataset_no_h = GraphDataset(
                dataset='_test', num_workers=0, folder_path=temp_path, use_cache=True
            )
            graph_no_h = next(iter(dataset_no_h))
            num_atoms_no_h = len(graph_no_h['node_indices'])

            # With explicit hydrogens
            dataset_with_h = GraphDataset(
                dataset='_test',
                num_workers=0,
                mol_transform=lambda mol: Chem.AddHs(mol),
                folder_path=temp_path,
                use_cache=True,
            )
            graph_with_h = next(iter(dataset_with_h))
            num_atoms_with_h = len(graph_with_h['node_indices'])

            # Should have more atoms with explicit H
            assert num_atoms_with_h > num_atoms_no_h

    def test_remove_hydrogens(self):
        """Test removing hydrogens decreases node count."""
        with tempfile.TemporaryDirectory() as temp_path:
            # First add hydrogens
            dataset_with_h = GraphDataset(
                dataset='_test',
                num_workers=0,
                mol_transform=lambda mol: Chem.AddHs(mol),
                folder_path=temp_path,
                use_cache=True,
            )
            graph_with_h = next(iter(dataset_with_h))
            num_atoms_with_h = len(graph_with_h['node_indices'])

            # Then remove them
            dataset_no_h = GraphDataset(
                dataset='_test',
                num_workers=0,
                mol_transform=lambda mol: Chem.RemoveHs(mol),
                folder_path=temp_path,
                use_cache=True,
            )
            graph_no_h = next(iter(dataset_no_h))
            num_atoms_no_h = len(graph_no_h['node_indices'])

            # Should have fewer atoms after removing H
            assert num_atoms_no_h < num_atoms_with_h

    def test_3d_conformer_generation(self):
        """Test generating 3D conformers for molecules."""

        def add_3d_coords(mol):
            """Add 3D coordinates via conformer generation."""
            mol = Chem.AddHs(mol)
            # EmbedMolecule returns -1 on failure, we can handle that
            result = AllChem.EmbedMolecule(mol, randomSeed=42)
            if result == -1:
                # If embedding fails, just return the molecule as-is
                return mol
            return mol

        with tempfile.TemporaryDirectory() as temp_path:
            dataset = GraphDataset(
                dataset='_test',
                num_workers=0,
                mol_transform=add_3d_coords,
                folder_path=temp_path,
                use_cache=True,
            )

            # Should produce valid graphs - test first 2 (3D is computationally expensive)
            count = 0
            for graph in dataset:
                assert isinstance(graph, dict)
                assert len(graph['node_indices']) > 0
                count += 1
                if count >= 2:
                    break

            assert count > 0

    def test_properties_preserved(self):
        """Test that graph_labels are preserved through transformation."""
        with tempfile.TemporaryDirectory() as temp_path:
            # Get first 10 results without transform
            dataset_no_transform = GraphDataset(
                dataset='_test', num_workers=0, folder_path=temp_path, use_cache=True
            )
            labels_no_transform = []
            for i, g in enumerate(dataset_no_transform):
                labels_no_transform.append(g['graph_labels'])
                if i >= 9:  # Get first 10
                    break

            # Get first 10 results with transform
            dataset_with_transform = GraphDataset(
                dataset='_test',
                num_workers=0,
                mol_transform=lambda mol: Chem.AddHs(mol),
                folder_path=temp_path,
                use_cache=True,
            )
            labels_with_transform = []
            for i, g in enumerate(dataset_with_transform):
                labels_with_transform.append(g['graph_labels'])
                if i >= 9:  # Get first 10
                    break

            # Labels should be identical
            assert labels_no_transform == labels_with_transform

    def test_sequential_parallel_consistency(self):
        """Test that sequential and parallel modes produce same results with transform."""

        def add_hydrogens(mol):
            return Chem.AddHs(mol)

        with tempfile.TemporaryDirectory() as temp_path:
            # Sequential mode - get first 10
            dataset_seq = GraphDataset(
                dataset='_test',
                num_workers=0,
                mol_transform=add_hydrogens,
                folder_path=temp_path,
                use_cache=True,
            )
            results_seq = []
            for i, g in enumerate(dataset_seq):
                results_seq.append(g)
                if i >= 9:  # Get first 10
                    break

            # Parallel mode - get first 10
            dataset_par = GraphDataset(
                dataset='_test',
                num_workers=2,
                mol_transform=add_hydrogens,
                folder_path=temp_path,
                use_cache=True,
            )
            results_par = []
            for i, g in enumerate(dataset_par):
                results_par.append(g)
                if i >= 9:  # Get first 10
                    break

            # Should produce same number of results
            assert len(results_seq) == len(results_par)

            # Should have same node counts and labels
            for g_seq, g_par in zip(results_seq, results_par):
                assert len(g_seq['node_indices']) == len(g_par['node_indices'])
                assert g_seq['graph_labels'] == g_par['graph_labels']


class TestShuffleDataset:
    """Test suite for ShuffleDataset class."""

    def test_initialization(self):
        """Test ShuffleDataset initialization."""
        with tempfile.TemporaryDirectory() as temp_path:
            base_dataset = SmilesDataset(
                dataset='_test', folder_path=temp_path, use_cache=True
            )

            shuffled_dataset = ShuffleDataset(
                dataset=base_dataset, buffer_size=100, seed=42
            )

            assert shuffled_dataset.dataset is base_dataset
            assert shuffled_dataset.buffer_size == 100
            assert shuffled_dataset.seed == 42

    def test_length(self):
        """Test that __len__ returns same length as base dataset."""
        with tempfile.TemporaryDirectory() as temp_path:
            base_dataset = SmilesDataset(
                dataset='_test', folder_path=temp_path, use_cache=True
            )

            shuffled_dataset = ShuffleDataset(dataset=base_dataset, buffer_size=10)

            assert len(shuffled_dataset) == len(base_dataset)

    def test_all_items_yielded(self):
        """Test that all items are yielded exactly once."""
        with tempfile.TemporaryDirectory() as temp_path:
            base_dataset = SmilesDataset(
                dataset='_test', folder_path=temp_path, use_cache=True
            )

            # Get original items
            original_smiles = [smiles for smiles, _ in base_dataset]

            # Get shuffled items
            shuffled_dataset = ShuffleDataset(
                dataset=base_dataset, buffer_size=10, seed=42
            )
            shuffled_smiles = [smiles for smiles, _ in shuffled_dataset]

            # Check same length
            assert len(shuffled_smiles) == len(original_smiles)

            # Check same items (as sets)
            assert set(shuffled_smiles) == set(original_smiles)

            # Check no duplicates
            assert len(shuffled_smiles) == len(set(shuffled_smiles))

    def test_shuffling_actually_shuffles(self):
        """Test that items are actually shuffled (order changes)."""
        with tempfile.TemporaryDirectory() as temp_path:
            base_dataset = SmilesDataset(
                dataset='_test', folder_path=temp_path, use_cache=True
            )

            # Get original order
            original_smiles = [smiles for smiles, _ in base_dataset]

            # Get shuffled order (with seed for reproducibility)
            shuffled_dataset = ShuffleDataset(
                dataset=base_dataset, buffer_size=100, seed=42
            )
            shuffled_smiles = [smiles for smiles, _ in shuffled_dataset]

            # For datasets with more than 2 items, order should likely be different
            # With seed=42 and buffer_size=100, this should shuffle
            if len(original_smiles) > 2:
                # Check that at least some positions changed
                # Note: there's a small probability they could be the same, but very unlikely
                differences = sum(
                    1
                    for i in range(len(original_smiles))
                    if original_smiles[i] != shuffled_smiles[i]
                )
                assert differences > 0, 'Shuffling should change the order'

    def test_seed_reproducibility(self):
        """Test that same seed produces same shuffle order."""
        with tempfile.TemporaryDirectory() as temp_path:
            base_dataset = SmilesDataset(
                dataset='_test', folder_path=temp_path, use_cache=True
            )

            # First shuffle with seed=42
            shuffled1 = ShuffleDataset(dataset=base_dataset, buffer_size=10, seed=42)
            order1 = [smiles for smiles, _ in shuffled1]

            # Second shuffle with same seed
            shuffled2 = ShuffleDataset(dataset=base_dataset, buffer_size=10, seed=42)
            order2 = [smiles for smiles, _ in shuffled2]

            # Should be identical
            assert order1 == order2

    def test_different_seeds_different_order(self):
        """Test that different seeds produce different orders."""
        with tempfile.TemporaryDirectory() as temp_path:
            base_dataset = SmilesDataset(
                dataset='_test', folder_path=temp_path, use_cache=True
            )

            # Shuffle with seed=42
            shuffled1 = ShuffleDataset(dataset=base_dataset, buffer_size=10, seed=42)
            order1 = [smiles for smiles, _ in shuffled1]

            # Shuffle with seed=123
            shuffled2 = ShuffleDataset(dataset=base_dataset, buffer_size=10, seed=123)
            order2 = [smiles for smiles, _ in shuffled2]

            # Should be different (with very high probability for different seeds)
            if len(order1) > 2:
                assert order1 != order2

    def test_works_with_graph_dataset(self):
        """Test that ShuffleDataset works with GraphDataset."""
        with tempfile.TemporaryDirectory() as temp_path:
            base_dataset = GraphDataset(
                dataset='_test', num_workers=0, folder_path=temp_path, use_cache=True
            )

            # Get original labels
            original_labels = [graph['graph_labels'] for graph in base_dataset]

            # Shuffle
            shuffled_dataset = ShuffleDataset(
                dataset=base_dataset, buffer_size=10, seed=42
            )
            shuffled_labels = [graph['graph_labels'] for graph in shuffled_dataset]

            # Check all items present
            assert len(shuffled_labels) == len(original_labels)
            assert set(tuple(l) for l in shuffled_labels) == set(
                tuple(l) for l in original_labels
            )

    def test_works_with_parallel_graph_dataset(self):
        """Test that ShuffleDataset works with parallel GraphDataset."""
        with tempfile.TemporaryDirectory() as temp_path:
            base_dataset = GraphDataset(
                dataset='_test', num_workers=2, folder_path=temp_path, use_cache=True
            )

            shuffled_dataset = ShuffleDataset(
                dataset=base_dataset, buffer_size=10, seed=42
            )

            # Should iterate successfully
            count = sum(1 for _ in shuffled_dataset)
            assert count == len(base_dataset)

    def test_multiple_iterations(self):
        """Test that ShuffleDataset can be iterated multiple times."""
        with tempfile.TemporaryDirectory() as temp_path:
            base_dataset = SmilesDataset(
                dataset='_test', folder_path=temp_path, use_cache=True
            )

            shuffled_dataset = ShuffleDataset(
                dataset=base_dataset, buffer_size=10, seed=42
            )

            # First iteration
            order1 = [smiles for smiles, _ in shuffled_dataset]

            # Second iteration
            order2 = [smiles for smiles, _ in shuffled_dataset]

            # With same seed, should produce same order
            assert order1 == order2
            assert len(order1) > 0

    def test_different_buffer_sizes(self):
        """Test that different buffer sizes work correctly."""
        with tempfile.TemporaryDirectory() as temp_path:
            base_dataset = SmilesDataset(
                dataset='_test', folder_path=temp_path, use_cache=True
            )

            original_smiles = set(smiles for smiles, _ in base_dataset)

            # Test various buffer sizes
            for buffer_size in [1, 5, 100, 1000]:
                shuffled_dataset = ShuffleDataset(
                    dataset=base_dataset, buffer_size=buffer_size, seed=42
                )

                shuffled_smiles = set(smiles for smiles, _ in shuffled_dataset)

                # All items should be present regardless of buffer size
                assert shuffled_smiles == original_smiles

    def test_buffer_size_larger_than_dataset(self):
        """Test that buffer_size larger than dataset works correctly."""
        with tempfile.TemporaryDirectory() as temp_path:
            base_dataset = SmilesDataset(
                dataset='_test', folder_path=temp_path, use_cache=True
            )

            # Buffer size much larger than dataset
            shuffled_dataset = ShuffleDataset(
                dataset=base_dataset, buffer_size=10000, seed=42
            )

            original_smiles = [smiles for smiles, _ in base_dataset]
            shuffled_smiles = [smiles for smiles, _ in shuffled_dataset]

            # Should still work correctly
            assert len(shuffled_smiles) == len(original_smiles)
            assert set(shuffled_smiles) == set(original_smiles)

    def test_properties_preserved(self):
        """Test that properties are preserved during shuffling."""
        with tempfile.TemporaryDirectory() as temp_path:
            base_dataset = SmilesDataset(
                dataset='_test', folder_path=temp_path, use_cache=True
            )

            # Get original smiles -> properties mapping
            original_map = {smiles: props.tolist() for smiles, props in base_dataset}

            # Shuffle
            shuffled_dataset = ShuffleDataset(
                dataset=base_dataset, buffer_size=10, seed=42
            )

            # Check that each SMILES has its correct properties
            for smiles, props in shuffled_dataset:
                assert props.tolist() == original_map[smiles]

    def test_no_seed_gives_random_order(self):
        """
        Without a seed, shuffling must not always produce the same order.

        Two shuffles of a small dataset genuinely coincide sometimes, so comparing one
        pair and asserting they differ fails on correct code - measured at roughly one
        run in five here. The property being tested is that the order is *not fixed*,
        which needs more than one sample: several shuffles are taken and the test asks
        whether they are all identical. For that to fail spuriously, every attempt would
        have to collide, which is vanishingly unlikely while a genuinely deterministic
        shuffle still fails every time.
        """
        attempts = 8

        with tempfile.TemporaryDirectory() as temp_path:
            base_dataset = SmilesDataset(
                dataset='_test', folder_path=temp_path, use_cache=True
            )

            orders = [
                [smiles for smiles, _ in ShuffleDataset(dataset=base_dataset, buffer_size=10)]
                for _ in range(attempts)
            ]

            # A single-element (or empty) dataset has only one possible order, so there
            # is nothing to assert about randomness.
            if len(orders[0]) < 2:
                pytest.skip('dataset too small for order to vary')

            assert any(order != orders[0] for order in orders[1:]), (
                f'all {attempts} shuffles produced an identical order - the shuffle '
                'appears to be deterministic without a seed'
            )


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_missing_column_name(self):
        """Test that missing column raises appropriate error."""
        with tempfile.TemporaryDirectory() as temp_path:
            dataset = SmilesDataset(
                dataset='_test',
                smiles_column='nonexistent_column',
                folder_path=temp_path,
                use_cache=True,
            )

            # Should raise KeyError when iterating
            with pytest.raises(KeyError):
                for _ in dataset:
                    pass

    def test_empty_target_columns(self):
        """Test behavior with empty target columns list."""
        with tempfile.TemporaryDirectory() as temp_path:
            dataset = SmilesDataset(
                dataset='_test',
                target_columns=[],
                folder_path=temp_path,
                use_cache=True,
            )

            for smiles, properties in dataset:
                assert len(properties) == 0
                break


class TestGraphDatasetAutoDetection:
    """Test suite for GraphDataset auto-detection and XYZ dataset support."""

    def test_auto_detect_smiles_dataset(self):
        """Test that GraphDataset auto-detects SMILES datasets (CSV format)."""
        with tempfile.TemporaryDirectory() as temp_path:
            dataset = GraphDataset(
                dataset='_test',  # This is a CSV-based SMILES dataset
                num_workers=0,
                folder_path=temp_path,
                use_cache=True,
            )

            # Should detect and use SmilesDataset
            assert dataset.dataset_type == 'smiles'
            assert hasattr(dataset, 'raw_dataset')
            assert isinstance(dataset.raw_dataset, SmilesDataset)

            # Should produce valid graphs
            count = 0
            for graph in dataset:
                assert isinstance(graph, dict)
                assert 'node_indices' in graph
                assert 'graph_labels' in graph
                count += 1

            assert count > 0

    def test_auto_detect_xyz_dataset(self):
        """Test that GraphDataset auto-detects XYZ datasets (xyz_bundle format)."""
        with tempfile.TemporaryDirectory() as temp_path:
            dataset = GraphDataset(
                dataset='_test2',  # This is an xyz_bundle dataset
                num_workers=0,
                folder_path=temp_path,
                use_cache=True,
            )

            # Should detect and use XyzDataset
            assert dataset.dataset_type == 'xyz'
            assert hasattr(dataset, 'raw_dataset')
            assert isinstance(dataset.raw_dataset, XyzDataset)

            # Should produce valid graphs
            count = 0
            for graph in dataset:
                assert isinstance(graph, dict)
                assert 'node_indices' in graph
                assert 'node_attributes' in graph
                assert 'edge_indices' in graph
                assert 'graph_labels' in graph
                count += 1

            assert count > 0

    def test_xyz_sequential_processing(self):
        """Test XYZ dataset with sequential processing (num_workers=0)."""
        with tempfile.TemporaryDirectory() as temp_path:
            dataset = GraphDataset(
                dataset='_test2', num_workers=0, folder_path=temp_path, use_cache=True
            )

            graphs = list(dataset)
            assert len(graphs) > 0

            # Verify graph structure
            for graph in graphs:
                assert 'node_indices' in graph
                assert 'node_attributes' in graph
                assert isinstance(graph['node_indices'], np.ndarray)
                assert isinstance(graph['node_attributes'], np.ndarray)

    def test_xyz_parallel_processing(self):
        """Test XYZ dataset with parallel processing (num_workers>0)."""
        with tempfile.TemporaryDirectory() as temp_path:
            dataset = GraphDataset(
                dataset='_test2', num_workers=2, folder_path=temp_path, use_cache=True
            )

            graphs = list(dataset)
            assert len(graphs) > 0

            # Verify graph structure
            for graph in graphs:
                assert 'node_indices' in graph
                assert 'node_attributes' in graph

    def test_xyz_with_parser_class(self):
        """Test XYZ dataset with custom parser class."""
        with tempfile.TemporaryDirectory() as temp_path:
            dataset = GraphDataset(
                dataset='_test2',
                num_workers=0,
                parser_cls='default',  # Specify parser explicitly
                folder_path=temp_path,
                use_cache=True,
            )

            assert dataset.dataset_type == 'xyz'
            graphs = list(dataset)
            assert len(graphs) > 0

    def test_xyz_with_mol_transform(self):
        """Test that mol_transform works with XYZ datasets."""

        def add_hydrogens(mol):
            return Chem.AddHs(mol)

        with tempfile.TemporaryDirectory() as temp_path:
            # Without hydrogens
            dataset_no_h = GraphDataset(
                dataset='_test2', num_workers=0, folder_path=temp_path, use_cache=True
            )
            graph_no_h = next(iter(dataset_no_h))
            num_atoms_no_h = len(graph_no_h['node_indices'])

            # With explicit hydrogens
            dataset_with_h = GraphDataset(
                dataset='_test2',
                num_workers=0,
                mol_transform=add_hydrogens,
                folder_path=temp_path,
                use_cache=True,
            )
            graph_with_h = next(iter(dataset_with_h))
            num_atoms_with_h = len(graph_with_h['node_indices'])

            # Should have more atoms with explicit H
            assert num_atoms_with_h >= num_atoms_no_h

    def test_xyz_order_preservation_parallel(self):
        """Test that parallel processing preserves order for XYZ datasets."""
        with tempfile.TemporaryDirectory() as temp_path:
            # Sequential results
            dataset_seq = GraphDataset(
                dataset='_test2', num_workers=0, folder_path=temp_path, use_cache=True
            )
            results_seq = list(dataset_seq)

            # Parallel results
            dataset_par = GraphDataset(
                dataset='_test2', num_workers=2, folder_path=temp_path, use_cache=True
            )
            results_par = list(dataset_par)

            # Check same length
            assert len(results_seq) == len(results_par)

            # Check that graph structures match (order preserved)
            for g_seq, g_par in zip(results_seq, results_par):
                # Should have same number of atoms
                assert len(g_seq['node_indices']) == len(g_par['node_indices'])

    def test_nonexistent_dataset_error(self):
        """Test that nonexistent dataset raises appropriate error."""
        with tempfile.TemporaryDirectory() as temp_path:
            with pytest.raises(FileNotFoundError) as exc_info:
                dataset = GraphDataset(
                    dataset='nonexistent_dataset_xyz_12345',
                    num_workers=0,
                    folder_path=temp_path,
                    use_cache=True,
                )

            # Error message should mention both formats tried
            error_msg = str(exc_info.value)
            assert 'CSV' in error_msg or 'SmilesDataset' in error_msg
            assert 'xyz_bundle' in error_msg or 'XyzDataset' in error_msg

    def test_xyz_dataset_length(self):
        """Test that __len__ works correctly for XYZ datasets."""
        with tempfile.TemporaryDirectory() as temp_path:
            dataset = GraphDataset(
                dataset='_test2', num_workers=0, folder_path=temp_path, use_cache=True
            )

            length = len(dataset)
            assert isinstance(length, int)
            assert length > 0

            # Verify length matches iteration count
            actual_count = sum(1 for _ in dataset)
            assert actual_count <= length  # May be less if some files fail to parse

    def test_xyz_multiple_iterations(self):
        """Test that XYZ GraphDataset can be iterated multiple times."""
        with tempfile.TemporaryDirectory() as temp_path:
            dataset = GraphDataset(
                dataset='_test2', num_workers=0, folder_path=temp_path, use_cache=True
            )

            # First iteration
            count1 = sum(1 for _ in dataset)

            # Second iteration
            count2 = sum(1 for _ in dataset)

            assert count1 == count2
            assert count1 > 0

    def test_adapter_interface(self):
        """Test that the adapter interface works correctly."""
        from chem_mat_data.dataset import (
            DatasetAdapter,
            SmilesDatasetAdapter,
            XyzDatasetAdapter,
        )

        with tempfile.TemporaryDirectory() as temp_path:
            # Test SmilesDatasetAdapter
            smiles_ds = SmilesDataset(
                dataset='_test', folder_path=temp_path, use_cache=True
            )
            smiles_adapter = SmilesDatasetAdapter(smiles_ds)

            assert isinstance(smiles_adapter, DatasetAdapter)
            assert len(smiles_adapter) == len(smiles_ds)

            # Test iteration
            for mol, props in smiles_adapter:
                assert mol is not None  # Should be RDKit Mol object
                assert isinstance(props, np.ndarray)
                break

            # Test XyzDatasetAdapter
            xyz_ds = XyzDataset(dataset='_test2', folder_path=temp_path, use_cache=True)
            xyz_adapter = XyzDatasetAdapter(xyz_ds)

            assert isinstance(xyz_adapter, DatasetAdapter)
            assert len(xyz_adapter) == len(xyz_ds)

            # Test iteration
            for mol, props in xyz_adapter:
                assert mol is not None  # Should be RDKit Mol object
                assert isinstance(props, np.ndarray)
                break


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
