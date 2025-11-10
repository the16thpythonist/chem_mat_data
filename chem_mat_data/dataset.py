"""
Streaming Dataset Module for ChemMatData

This module provides memory-efficient streaming datasets for molecular data:
- SmilesDataset: Streams raw SMILES strings and properties from CSV
- XyzDataset: Streams 3D molecular structures from XYZ files
- GraphDataset: Streams graph representations with optional parallel processing

PARALLEL PROCESSING DESIGN NOTES:
=================================

Performance Characteristics (typical molecules on modern hardware):
- Sequential (num_workers=0): ~2000 mol/s
- Parallel (num_workers=4): ~500 mol/s (but scales with complexity)

Why is parallel slower for simple molecules?
- Overhead per molecule: ~0.6ms (pickling + queue ops + coordination)
- Processing time per molecule: ~0.5ms (RDKit graph generation)
- Overhead > actual work = slower overall

When to use parallel mode:
1. Complex molecules (processing time > 2ms)
2. Custom processing_class with heavy computation
3. Large datasets where total throughput matters
4. When you can use 4+ workers for better amortization

Overhead breakdown:
- Pickling/unpickling: ~0.05ms per graph dict (~10KB pickled)
- Queue operations: ~0.1ms per item (get + put through multiprocessing queues)
- Thread coordination: GIL contention between 3 threads
- Context switching: Between producer, collector, and main threads

The parallel implementation uses a sophisticated architecture to prevent deadlock
when the iterator suspends at yield. See detailed comments in GraphDataset.__iter__().
"""

import csv
import os
import tempfile
import heapq
import queue
import threading
import multiprocessing as mp
from typing import Tuple, Iterator, Optional, List, Any, Dict, Callable
from collections import defaultdict
import numpy as np
import pandas as pd

from chem_mat_data.processing import MoleculeProcessing
from chem_mat_data.main import ensure_dataset
from chem_mat_data.data import load_xyz_as_mol


# === MODULE-LEVEL HELPER FUNCTIONS ===

def identity_mol_transform(mol: 'Chem.Mol') -> 'Chem.Mol':
    """
    Default mol_transform that returns the molecule unchanged.

    This is the identity function used when no custom transformation is needed.
    It serves as the default value for the mol_transform parameter in GraphDataset.

    :param mol: RDKit molecule object

    :returns: The same molecule object unchanged
    """
    return mol

def _convert_value(value: Any) -> float:
    """
    Convert a value to float, handling boolean strings.

    :param value: The value to convert (can be string, bool, int, float)

    :returns: Float representation of the value
    """
    if isinstance(value, str):
        # Handle boolean strings
        if value.lower() in ('true', 'false'):
            return 1.0 if value.lower() == 'true' else 0.0
        else:
            return float(value)
    else:
        return float(value)


def _convert_raw_to_mol(raw_data: Any, dataset_type: str) -> 'Chem.Mol':
    """
    Convert raw dataset output to RDKit Mol object.

    This function provides a unified interface for converting different raw data formats
    into RDKit Mol objects. It's used by both sequential and parallel processing modes
    to convert the raw data (SMILES strings or XYZ data dictionaries) into Mol objects
    that can be processed into graph representations.

    **Design Rationale:**
    This conversion is performed in the worker processes (for parallel mode) or just
    before processing (for sequential mode) rather than in the main process. This approach:
    - Parallelizes the conversion work across workers
    - Reduces pickling overhead (SMILES strings are tiny, Mol objects are large)
    - Keeps the producer thread lightweight and fast

    :param raw_data: Either a SMILES string (for 'smiles' type) or xyz_data dict (for 'xyz' type)
    :param dataset_type: Dataset type identifier ('smiles' or 'xyz')

    :returns: RDKit Mol object, or None if the SMILES string is invalid (with warning logged)

    :raises ValueError: If dataset_type is unknown
    """
    import rdkit.Chem as Chem

    if dataset_type == 'smiles':
        # Convert SMILES string to Mol object
        mol = Chem.MolFromSmiles(raw_data)
        if mol is None:
            import logging
            logging.warning(f"Skipping invalid SMILES string: {raw_data}")
            return None
        return mol

    elif dataset_type == 'xyz':
        # Reconstruct Mol object from xyz_data dictionary
        # xyz_data contains: atomic_numbers, positions, num_atoms, symbols

        # Create read-write molecule
        mol = Chem.RWMol()

        # Add atoms based on atomic numbers
        for atomic_num in raw_data['atomic_numbers']:
            atom = Chem.Atom(int(atomic_num))
            mol.AddAtom(atom)

        # Create conformer with 3D positions
        num_atoms = raw_data['num_atoms']
        conf = Chem.Conformer(num_atoms)

        for i, position in enumerate(raw_data['positions']):
            conf.SetAtomPosition(i, tuple(position))

        mol.AddConformer(conf)

        # Convert to read-only molecule and update property cache
        mol = mol.GetMol()
        mol.UpdatePropertyCache()

        return mol

    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Expected 'smiles' or 'xyz'.")


def _graph_worker(
    worker_id: int,
    processing_class: type,
    mol_transform: Callable,
    input_queue: mp.Queue,
    result_queue: mp.Queue,
    dataset_type: str,
) -> None:
    """
    Worker process that processes individual molecules from input queue.

    Each worker:
    - Gets (row_idx, raw_data, properties) from input_queue (blocking)
    - Converts raw_data to Mol object using dataset_type
    - Applies mol_transform to the Mol object
    - Processes transformed Mol into graph representation
    - Puts (row_idx, graph) to result_queue
    - Exits when receiving None sentinel

    **Performance Design:**
    The conversion from raw data to Mol objects happens in the worker processes,
    not in the main process. This parallelizes the conversion work and reduces
    pickling overhead (SMILES strings and xyz_data are smaller than Mol objects).

    :param worker_id: Unique ID for this worker (0 to num_workers-1)
    :param processing_class: Class to instantiate for graph processing
    :param mol_transform: Callable to transform Mol objects before processing
    :param input_queue: Queue to receive work items from main process
    :param result_queue: Queue to send (row_index, graph) results back
    :param dataset_type: Type of dataset ('smiles' or 'xyz') for conversion
    """
    try:
        # Create processing instance once per worker (not per molecule)
        # This amortizes any initialization cost across all molecules
        processing = processing_class()

        while True:
            # BLOCKING get: Efficient - worker sleeps until work arrives
            # No busy-waiting, minimal CPU usage when idle
            item = input_queue.get()

            # Shutdown protocol: None signals worker to exit
            # Worker sends its own sentinel to result_queue so collector
            # knows when all workers are done
            if item is None:
                result_queue.put((worker_id, None))
                break

            # Unpack work item - receives raw data (SMILES string or xyz_data dict)
            row_idx, raw_data, properties = item

            try:
                # Step 1: Convert raw data to Mol object
                # This happens in parallel across workers, not in main process
                mol = _convert_raw_to_mol(raw_data, dataset_type)

                # Skip invalid molecules (mol is None)
                if mol is None:
                    # Send a special sentinel to indicate this molecule should be skipped
                    result_queue.put((row_idx, 'SKIP'))
                    continue

                # Step 2: Apply mol_transform
                mol = mol_transform(mol)

                # Step 3: CORE COMPUTATION - Process transformed Mol into graph
                # Processing time: ~0.5ms for typical molecules
                # This is where parallel processing provides benefit
                graph = processing.process(mol, graph_labels=properties)

                # Send result with original row index for reordering
                # The row_idx is crucial - it allows the collector thread
                # to maintain the original dataset order even though
                # workers may finish molecules out of order
                result_queue.put((row_idx, graph))

            except Exception as e:
                # Per-molecule error handling: Send the exception back
                # with the row index so the main process knows which
                # molecule failed
                result_queue.put((row_idx, e))

    except Exception as e:
        # Worker-level catastrophic error (e.g., queue corruption)
        # Send worker_id to help with debugging
        result_queue.put((worker_id, e))


# === DATASET IMPLEMENTATION ===

class StreamingDataset:
    """
    Abstract base class for streaming datasets that load data lazily rather than
    loading the entire dataset into memory at once.
    """

    def __init__(self):
        pass

    def __iter__(self):
        raise NotImplementedError()


class SmilesDataset(StreamingDataset):
    """
    Streaming dataset for SMILES-based molecular datasets.

    This dataset loads SMILES strings and their associated properties from a CSV file
    in a streaming manner, meaning that only one row is loaded into memory at a time.
    This is particularly useful for large datasets that don't fit into memory.

    The dataset first ensures that the dataset file exists locally by downloading it
    from the remote file share server if necessary. Then it iterates through the CSV
    file line by line, yielding tuples of (smiles_string, properties_array).

    Example:

    .. code-block:: python

        dataset = SmilesDataset(
            dataset='esol',
            smiles_column='smiles',
            target_columns=['measured log solubility in mols per litre']
        )

        for smiles, properties in dataset:
            print(f"SMILES: {smiles}, Properties: {properties}")

    :param dataset: The unique string identifier of the dataset to load
    :param smiles_column: The name of the column containing SMILES strings (default: 'smiles')
    :param target_columns: List of column names containing target properties (default: ['target'])
    :param folder_path: The absolute path where dataset files should be stored (default: system temp dir)
    :param use_cache: Whether to use the local file system cache (default: True)
    """

    def __init__(
        self,
        dataset: str,
        smiles_column: str = 'smiles',
        target_columns: list = ['target'],
        folder_path: str = tempfile.gettempdir(),
        use_cache: bool = True,
    ) -> None:
        super().__init__()

        self.dataset = dataset
        self.smiles_column = smiles_column
        self.target_columns = target_columns
        self.folder_path = folder_path
        self.use_cache = use_cache

        # Ensure the dataset exists locally
        self.file_path = ensure_dataset(
            dataset_name=dataset,
            extension='csv',
            folder_path=folder_path,
            use_cache=use_cache,
        )

        # Cache for dataset length
        self._length: Optional[int] = None

    def __iter__(self) -> Iterator[Tuple[str, np.ndarray]]:
        """
        Iterate through the dataset, yielding SMILES strings and their properties.

        :yields: Tuples of (smiles_string, properties_array)
        """
        with open(self.file_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)

            for row in reader:
                smiles = row[self.smiles_column]
                # Convert properties to floats, handling boolean strings
                properties = []
                for col in self.target_columns:
                    value = row[col]
                    # Handle boolean strings
                    if value.lower() in ('true', 'false'):
                        properties.append(1.0 if value.lower() == 'true' else 0.0)
                    else:
                        properties.append(float(value))
                properties = np.array(properties, dtype=float)
                yield smiles, properties

    def __len__(self) -> int:
        """
        Return the number of elements in the dataset.

        The count is cached after the first call for efficiency.

        :returns: The number of rows in the CSV file (excluding header)
        """
        if self._length is None:
            self._length = self._count_lines()
        return self._length

    def _count_lines(self) -> int:
        """
        Efficiently count the number of lines in the CSV file.

        :returns: The number of data rows (excluding header)
        """
        with open(self.file_path, mode='r', encoding='utf-8') as file:
            # Skip header
            next(file)
            # Count remaining lines
            return sum(1 for _ in file)


class XyzDataset(StreamingDataset):
    """
    Streaming dataset for XYZ-based molecular datasets with 3D coordinates.

    This dataset loads molecular structures from XYZ files in a streaming manner,
    meaning that only one molecule is loaded into memory at a time. This is particularly
    useful for large datasets that don't fit into memory.

    XYZ datasets are stored as "xyz_bundle" folders containing:
    - Multiple .xyz files (one per molecule), named by their ID (e.g., "0.xyz", "1.xyz")
    - Optional meta.csv file with target properties and metadata

    The dataset first ensures that the xyz_bundle exists locally by downloading it
    from the remote file share server if necessary. Then it iterates through the XYZ
    files, yielding tuples of (xyz_data, properties).

    Example:

    .. code-block:: python

        dataset = XyzDataset(
            dataset='qm9',
            parser_cls='qm9',
            target_columns=['targets']
        )

        for xyz_data, properties in dataset:
            positions = xyz_data['positions']  # (N, 3) coordinates
            atoms = xyz_data['atomic_numbers']  # (N,) atomic numbers
            symbols = xyz_data['symbols']  # (N,) atom symbols
            print(f"Molecule with {xyz_data['num_atoms']} atoms")

    :param dataset: The unique string identifier of the dataset to load
    :param parser_cls: The parser class to use for loading XYZ files (default: 'default').
                      Options: 'default', 'qm9', 'hopv15'
    :param target_columns: List of column names containing target properties in meta.csv
                          (default: ['target']). Use None to skip property loading.
    :param folder_path: The absolute path where dataset files should be stored
                       (default: system temp dir)
    :param use_cache: Whether to use the local file system cache (default: True)
    """

    def __init__(
        self,
        dataset: str,
        parser_cls: str = 'default',
        target_columns: Optional[List[str]] = ['target'],
        folder_path: str = tempfile.gettempdir(),
        use_cache: bool = True,
    ) -> None:
        super().__init__()

        self.dataset = dataset
        self.parser_cls = parser_cls
        self.target_columns = target_columns
        self.folder_path = folder_path
        self.use_cache = use_cache

        # Ensure the xyz_bundle exists locally
        self.bundle_path = ensure_dataset(
            dataset_name=dataset,
            extension='xyz_bundle',
            folder_path=folder_path,
            use_cache=use_cache,
        )

        # Load metadata CSV if it exists
        self.metadata_map: Dict[Any, Dict] = defaultdict(dict)
        meta_path = os.path.join(self.bundle_path, 'meta.csv')
        if os.path.exists(meta_path):
            data_list = pd.read_csv(meta_path).to_dict(orient='records')
            for data in data_list:
                element_id = data['id']
                # Try to convert to int if possible (for consistency)
                try:
                    element_id = int(element_id)
                except (ValueError, TypeError):
                    pass
                self.metadata_map[element_id] = data

        # Get list of xyz files, sorted for reproducibility
        self.xyz_files = sorted([
            f for f in os.listdir(self.bundle_path)
            if f.endswith('.xyz')
        ])

        # Cache for dataset length
        self._length: Optional[int] = None

    def __iter__(self) -> Iterator[Tuple[Dict[str, Any], np.ndarray]]:
        """
        Iterate through the dataset, yielding XYZ data and properties.

        Each iteration yields a tuple (xyz_data, properties) where:

        - xyz_data is a dictionary containing:
          - 'positions': np.ndarray of shape (num_atoms, 3) with 3D coordinates
          - 'atomic_numbers': np.ndarray of shape (num_atoms,) with atomic numbers (int)
          - 'symbols': np.ndarray of shape (num_atoms,) with atom symbols (str)
          - 'num_atoms': int, number of atoms in the molecule

        - properties is a np.ndarray of target values from the metadata CSV

        :yields: Tuples of (xyz_data_dict, properties_array)
        """
        for xyz_file in self.xyz_files:
            xyz_path = os.path.join(self.bundle_path, xyz_file)

            # Extract ID from filename
            element_id = os.path.splitext(xyz_file)[0]
            try:
                element_id = int(element_id)
            except (ValueError, TypeError):
                pass

            try:
                # Parse the XYZ file
                mol, info = load_xyz_as_mol(xyz_path, parser_cls=self.parser_cls)

                # Extract 3D coordinates from conformer
                if mol.GetNumConformers() == 0:
                    # Skip molecules without conformers
                    continue

                conformer = mol.GetConformers()[0]
                positions = conformer.GetPositions()

                # Extract atom information
                atoms = mol.GetAtoms()
                atomic_numbers = np.array([atom.GetAtomicNum() for atom in atoms], dtype=int)
                symbols = np.array([atom.GetSymbol() for atom in atoms], dtype=str)

                # Construct xyz_data dict
                xyz_data = {
                    'positions': positions,
                    'atomic_numbers': atomic_numbers,
                    'symbols': symbols,
                    'num_atoms': mol.GetNumAtoms(),
                }

                # Extract properties from metadata
                properties = []
                if self.target_columns is not None and element_id in self.metadata_map:
                    metadata = self.metadata_map[element_id]
                    for col in self.target_columns:
                        if col in metadata:
                            value = metadata[col]
                            properties.append(_convert_value(value))
                        else:
                            # Column not found, append NaN
                            properties.append(float('nan'))

                # If no target columns specified or no metadata, return empty array
                properties = np.array(properties, dtype=float) if properties else np.array([], dtype=float)

                yield xyz_data, properties

            except Exception as e:
                # Skip molecules that fail to load
                # In production, you might want to log this
                continue

    def __len__(self) -> int:
        """
        Return the number of XYZ files in the dataset.

        The count is cached after the first call for efficiency.

        :returns: The number of XYZ files in the bundle
        """
        if self._length is None:
            self._length = len(self.xyz_files)
        return self._length


# === DATASET ADAPTERS ===
#
# NOTE: These adapter classes are currently not used internally by GraphDataset.
# GraphDataset directly iterates raw datasets and converts data in worker processes
# for better performance. These classes are kept for potential external use or future
# features.

class DatasetAdapter:
    """
    Abstract base class for dataset adapters.

    Dataset adapters provide a unified interface for converting different dataset types
    into a common format that yields (mol, properties) tuples. This abstraction can be
    useful for external code that wants to work with multiple dataset types.

    **Note:** GraphDataset does not use these adapters internally. It directly iterates
    raw datasets and performs conversions in worker processes for better performance
    (parallelized conversion, reduced pickling overhead).

    The adapter pattern makes it easy to work with different dataset types:
    simply create a new adapter class that inherits from DatasetAdapter
    and implements the __iter__ method to yield (mol, properties) tuples.

    :param dataset: The underlying StreamingDataset to adapt
    """

    def __init__(self, dataset: StreamingDataset):
        self.dataset = dataset

    def __iter__(self) -> Iterator[Tuple['Chem.Mol', np.ndarray]]:
        """
        Iterate through the dataset, yielding (mol, properties) tuples.

        This method must be implemented by concrete adapter classes.

        :yields: Tuples of (RDKit Mol object, properties array)
        """
        raise NotImplementedError("Subclasses must implement __iter__")

    def __len__(self) -> int:
        """
        Return the number of elements in the dataset.

        :returns: The number of items in the underlying dataset
        """
        return len(self.dataset)


class SmilesDatasetAdapter(DatasetAdapter):
    """
    Adapter for SmilesDataset that converts SMILES strings to Mol objects.

    This adapter takes a SmilesDataset (which yields SMILES strings and properties)
    and converts it to yield (mol, properties) tuples by parsing each SMILES string
    into an RDKit Mol object.

    Example:

    .. code-block:: python

        smiles_dataset = SmilesDataset(dataset='esol')
        adapter = SmilesDatasetAdapter(smiles_dataset)

        for mol, properties in adapter:
            # mol is an RDKit Mol object
            print(f"Molecule with {mol.GetNumAtoms()} atoms")

    :param dataset: A SmilesDataset instance to adapt
    """

    def __iter__(self) -> Iterator[Tuple['Chem.Mol', np.ndarray]]:
        """
        Iterate through the SmilesDataset, converting SMILES to Mol objects.

        :yields: Tuples of (RDKit Mol object, properties array)
        :raises ValueError: If a SMILES string cannot be parsed
        """
        import rdkit.Chem as Chem

        for smiles, properties in self.dataset:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES string: {smiles}")
            yield mol, properties


class XyzDatasetAdapter(DatasetAdapter):
    """
    Adapter for XyzDataset that reconstructs Mol objects from xyz_data.

    This adapter takes an XyzDataset (which yields xyz_data dicts with positions,
    atomic_numbers, symbols, etc.) and reconstructs RDKit Mol objects from this
    information. The reconstructed molecules include 3D conformer data.

    The reconstruction process:
    1. Creates a new RWMol (read-write molecule)
    2. Adds atoms based on atomic numbers from xyz_data
    3. Creates a conformer with 3D positions from xyz_data
    4. Converts to read-only Mol object

    Example:

    .. code-block:: python

        xyz_dataset = XyzDataset(dataset='qm9', parser_cls='qm9')
        adapter = XyzDatasetAdapter(xyz_dataset)

        for mol, properties in adapter:
            # mol is an RDKit Mol object with 3D coordinates
            conformer = mol.GetConformers()[0]
            print(f"Molecule with {mol.GetNumAtoms()} atoms")

    :param dataset: An XyzDataset instance to adapt
    """

    def __iter__(self) -> Iterator[Tuple['Chem.Mol', np.ndarray]]:
        """
        Iterate through the XyzDataset, reconstructing Mol objects from xyz_data.

        :yields: Tuples of (RDKit Mol object with 3D conformer, properties array)
        """
        import rdkit.Chem as Chem

        for xyz_data, properties in self.dataset:
            mol = self._mol_from_xyz_data(xyz_data)
            yield mol, properties

    @staticmethod
    def _mol_from_xyz_data(xyz_data: Dict[str, Any]) -> 'Chem.Mol':
        """
        Reconstruct an RDKit Mol object from xyz_data dictionary.

        The xyz_data dict must contain:
        - 'atomic_numbers': Array of atomic numbers
        - 'positions': Array of 3D coordinates (N, 3)
        - 'num_atoms': Number of atoms

        :param xyz_data: Dictionary containing atomic structure information

        :returns: RDKit Mol object with 3D conformer
        """
        import rdkit.Chem as Chem

        # Create read-write molecule
        mol = Chem.RWMol()

        # Add atoms based on atomic numbers
        for atomic_num in xyz_data['atomic_numbers']:
            atom = Chem.Atom(int(atomic_num))
            mol.AddAtom(atom)

        # Create conformer with 3D positions
        num_atoms = xyz_data['num_atoms']
        conf = Chem.Conformer(num_atoms)

        for i, position in enumerate(xyz_data['positions']):
            conf.SetAtomPosition(i, tuple(position))

        mol.AddConformer(conf)

        # Convert to read-only molecule and update property cache
        mol = mol.GetMol()
        mol.UpdatePropertyCache()

        return mol


class GraphDataset(StreamingDataset):
    """
    Streaming dataset that processes molecules into graph dict representations.

    This dataset automatically detects whether the underlying data is in SMILES format
    (CSV files) or XYZ format (xyz_bundle folders) and processes molecules accordingly.
    It applies molecular graph processing to each molecule as it is streamed. The processing
    can be done either sequentially or in parallel using multiple worker processes.

    The dataset uses a main process coordination strategy with automatic order preservation.
    When num_workers > 0, the main process reads from the dataset and distributes work
    to worker processes, then collects and reorders results. This approach eliminates
    unnecessary threading overhead and provides better performance scaling.

    **Auto-Detection:**
    GraphDataset automatically detects the dataset format:
    1. First tries to load as CSV (SmilesDataset)
    2. If not available, tries XYZ bundle format (XyzDataset)
    3. Creates appropriate adapter to normalize the interface

    **Extensibility:**
    The adapter pattern makes it easy to add support for new dataset types in the future.
    Simply create a new DatasetAdapter subclass and add detection logic.

    Example:

    .. code-block:: python

        # Works with SMILES datasets (CSV)
        dataset = GraphDataset(
            dataset='esol',  # CSV-based SMILES dataset
            num_workers=0,
            processing_class=MoleculeProcessing
        )

        # Also works with XYZ datasets (xyz_bundle)
        dataset = GraphDataset(
            dataset='qm9',  # XYZ-based 3D structure dataset
            num_workers=4,
            parser_cls='qm9'
        )

        for graph in dataset:
            print(f"Graph with {len(graph['node_indices'])} atoms")

        # Manual cleanup if needed
        dataset.close()

    :param dataset: The unique string identifier of the dataset to load
    :param num_workers: Number of parallel worker processes (0 = sequential, default: 2)
    :param processing_class: The processing class to use for graph generation (default: MoleculeProcessing)
    :param smiles_column: The name of the column containing SMILES strings (default: 'smiles').
                         Only used for SMILES datasets.
    :param target_columns: List of column names containing target properties (default: ['target'])
    :param parser_cls: Parser class for XYZ datasets (default: 'default').
                      Only used for XYZ datasets. Options: 'default', 'qm9', 'hopv15'
    :param folder_path: The absolute path where dataset files should be stored (default: system temp dir)
    :param use_cache: Whether to use the local file system cache (default: True)
    :param buffer_size: Number of items to queue ahead for processing (default: 100)
    :param mol_transform: Callable to transform Mol objects before processing (default: identity_mol_transform)
    """

    def __init__(
        self,
        dataset: str,
        num_workers: int = 2,
        processing_class: type = MoleculeProcessing,
        smiles_column: str = 'smiles',
        target_columns: list = ['target'],
        parser_cls: str = 'default',
        folder_path: str = tempfile.gettempdir(),
        use_cache: bool = True,
        buffer_size: int = 100,
        mol_transform: Callable = identity_mol_transform,
    ) -> None:
        super().__init__()

        self.dataset = dataset
        self.num_workers = num_workers
        self.processing_class = processing_class
        self.buffer_size = buffer_size
        self.mol_transform = mol_transform

        # Auto-detect dataset type and store raw dataset
        # Try SMILES dataset (CSV) first, then XYZ dataset (xyz_bundle)
        self.raw_dataset, self.dataset_type = self._detect_dataset_type(
            dataset=dataset,
            smiles_column=smiles_column,
            target_columns=target_columns,
            parser_cls=parser_cls,
            folder_path=folder_path,
            use_cache=use_cache,
        )

        # For sequential processing
        self._processing: Optional[MoleculeProcessing] = None

    def _detect_dataset_type(
        self,
        dataset: str,
        smiles_column: str,
        target_columns: list,
        parser_cls: str,
        folder_path: str,
        use_cache: bool,
    ) -> Tuple[StreamingDataset, str]:
        """
        Auto-detect dataset type and return raw dataset with type identifier.

        This method attempts to load the dataset in the following order:
        1. Try to load as SmilesDataset (CSV format)
        2. If that fails, try to load as XyzDataset (xyz_bundle format)
        3. If both fail, raise an informative error

        The detection is based on attempting to locate the dataset files - if a CSV
        file exists for the dataset, it's treated as a SmilesDataset. Otherwise, if
        an xyz_bundle folder exists, it's treated as an XyzDataset.

        **Design Rationale:**
        We return the raw dataset (not an adapter) so that the producer thread can
        iterate it directly and send raw data (SMILES strings or xyz_data dicts)
        to workers. This keeps the producer lightweight and parallelizes the
        conversion work across workers.

        :param dataset: Dataset name/identifier
        :param smiles_column: Column name for SMILES (SmilesDataset only)
        :param target_columns: List of target column names
        :param parser_cls: Parser class for XYZ files (XyzDataset only)
        :param folder_path: Path to dataset storage folder
        :param use_cache: Whether to use cached datasets

        :returns: Tuple of (dataset instance, type string 'smiles' or 'xyz')

        :raises FileNotFoundError: If dataset cannot be found in any supported format
        """
        # Try SmilesDataset first (CSV format)
        try:
            smiles_dataset = SmilesDataset(
                dataset=dataset,
                smiles_column=smiles_column,
                target_columns=target_columns,
                folder_path=folder_path,
                use_cache=use_cache,
            )
            return smiles_dataset, 'smiles'
        except FileNotFoundError:
            pass  # Try next format

        # Try XyzDataset (xyz_bundle format)
        try:
            xyz_dataset = XyzDataset(
                dataset=dataset,
                parser_cls=parser_cls,
                target_columns=target_columns,
                folder_path=folder_path,
                use_cache=use_cache,
            )
            return xyz_dataset, 'xyz'
        except FileNotFoundError:
            pass  # Both formats failed

        # If we get here, dataset not found in any supported format
        raise FileNotFoundError(
            f"Dataset '{dataset}' not found in any supported format. "
            f"Tried: CSV (SmilesDataset), xyz_bundle (XyzDataset). "
            f"Please check the dataset name and ensure it's available in the remote file share."
        )

    def __iter__(self) -> Iterator[dict]:
        """
        Iterate through the dataset, yielding graph dict representations.

        **Sequential Mode (num_workers=0)**:
        Processes one molecule at a time in a truly streaming manner.
        Minimal memory usage - only one row loaded at a time.
        Best performance for small molecules (~2000 mol/s on typical hardware).

        **Parallel Mode (num_workers>0)**:
        Uses producer-collector thread architecture to prevent deadlock while yielding:

        1. **Producer thread** (I/O bound):
           - Reads molecules from dataset adapter
           - Feeds (row_idx, mol, properties) to bounded input_queue

        2. **Worker processes** (CPU bound):
           - Get items from input_queue (blocking)
           - Process Mol objects into graph representations
           - Put results to unbounded result_queue with row index

        3. **Collector thread** (coordination):
           - Drains result_queue continuously (blocking - no busy-wait)
           - Maintains priority queue for reordering
           - Puts ordered results to thread-safe results_buffer

        4. **Main iterator**:
           - Yields from results_buffer
           - Can suspend without blocking workers (prevents deadlock)

        **Performance Characteristics**:
        - Processing time: ~0.5ms per molecule (RDKit graph generation)
        - Pickle overhead: ~0.05ms per graph dict (~10KB pickled size)
        - Queue operations: ~0.1ms per item (get + put)
        - Scaling: 4 workers ≈ 3x faster than 1 worker
        - Overhead: Base overhead makes it slower than sequential for small molecules,
          but parallelism pays off for complex molecules or CPU-bound processing

        Use parallel mode when:
        - Processing complex molecules (longer processing time)
        - Using custom processing_class with heavy computation
        - Processing large datasets where throughput matters more than latency

        Parallelism: True CPU parallelism (multiprocessing bypasses GIL)

        :yields: Graph dict representations with node/edge attributes and graph_labels
        """
        if self.num_workers == 0:
            # Sequential processing - true streaming, one molecule at a time
            if self._processing is None:
                self._processing = self.processing_class()

            # Iterate raw dataset and convert to Mol objects
            for raw_data, properties in self.raw_dataset:
                # Convert raw data to Mol object
                mol = _convert_raw_to_mol(raw_data, self.dataset_type)
                # Skip invalid molecules (mol is None)
                if mol is None:
                    continue
                yield self._process_one(mol, properties)

        else:
            # ================================================================
            # PARALLEL PROCESSING WITH DEADLOCK PREVENTION
            # ================================================================
            #
            # ARCHITECTURE OVERVIEW:
            # ┌─────────────┐      ┌──────────┐      ┌───────────┐      ┌──────────┐
            # │   Producer  │──→   │  input   │──→   │  Workers  │──→   │  result  │
            # │   Thread    │      │  _queue  │      │ (N procs) │      │  _queue  │
            # └─────────────┘      └──────────┘      └───────────┘      └──────────┘
            #                                                                   │
            #                                                                   ↓
            # ┌─────────────┐      ┌──────────┐      ┌───────────────────────────┐
            # │    Main     │←──   │ results  │←──   │    Collector Thread       │
            # │  Iterator   │      │ _buffer  │      │ (reorders with heap)      │
            # └─────────────┘      └──────────┘      └───────────────────────────┘
            #
            # WHY THIS ARCHITECTURE?
            #
            # The Problem: Iterator Suspension Deadlock
            # -----------------------------------------
            # When the main iterator yields a value, it SUSPENDS execution until
            # the user calls next(). During this suspension:
            #   1. Workers keep processing and putting to result_queue
            #   2. result_queue fills up (if bounded)
            #   3. Workers block on put(), waiting for space
            #   4. Main iterator is suspended, not draining the queue
            #   5. DEADLOCK! Nothing moves forward.
            #
            # The Solution: Collector Thread
            # ------------------------------
            # A separate collector thread continuously drains result_queue with
            # BLOCKING gets (no busy-wait), regardless of iterator state. It:
            #   - Prevents result_queue from filling up (no worker blocking)
            #   - Maintains order using a priority queue (heap)
            #   - Feeds ordered results to a thread-safe buffer
            #   - Main iterator reads from buffer (can suspend safely)
            #
            # QUEUE DESIGN DECISIONS:
            # ----------------------
            # input_queue:    BOUNDED (buffer_size)
            #   - Backpressure: Prevents producer from getting too far ahead
            #   - Memory control: Limits number of items in flight
            #
            # result_queue:   UNBOUNDED
            #   - Critical: Must never block workers (prevents deadlock)
            #   - Safe: Collector thread drains continuously
            #   - Memory: Bounded by processing rate (workers can't produce 
            #     faster than they process)
            #
            # results_buffer: UNBOUNDED (thread queue, not multiprocessing)
            #   - Low overhead: Thread-local, no pickling
            #   - Safe: Main iterator drains when resumed
            #   - Memory: Bounded by how far ahead collector gets
            #
            # ================================================================

            # Setup queues with appropriate bounds
            input_queue = mp.Queue(maxsize=self.buffer_size)
            result_queue = mp.Queue()  # UNBOUNDED - critical for deadlock prevention

            # Thread-safe results buffer (regular queue, not mp.Queue)
            import queue as thread_queue
            results_buffer = thread_queue.Queue()

            # ----------------------------------------------------------------
            # PRODUCER THREAD: Dataset Reading
            # ----------------------------------------------------------------
            def producer():
                """
                Reads raw data from dataset and distributes to workers.

                Why a separate thread?
                - Decouples I/O (dataset reading) from computation (graph processing)
                - Allows workers to start processing while more data is loaded
                - Python's GIL allows I/O operations to release the lock

                Sends raw data (SMILES strings or xyz_data dicts) instead of Mol
                objects to minimize pickling overhead and parallelize conversion.
                """
                row_idx = 0
                for raw_data, properties in self.raw_dataset:
                    # Will block if input_queue is full (backpressure)
                    # Send raw data directly - conversion happens in workers
                    # properties.tolist(): Convert numpy array for pickling efficiency
                    input_queue.put((row_idx, raw_data, properties.tolist()))
                    row_idx += 1

                # Shutdown protocol: Send one None per worker
                # Each worker will consume one sentinel and exit
                for _ in range(self.num_workers):
                    input_queue.put(None)

            # ----------------------------------------------------------------
            # COLLECTOR THREAD: Result Reordering
            # ----------------------------------------------------------------
            def collector():
                """
                Collects results from workers, maintains order, feeds main iterator.

                Why a separate thread?
                - CRITICAL: Prevents deadlock by continuously draining result_queue
                - Workers can finish molecules out of order (different complexity)
                - Must reorder to maintain dataset sequence
                - Can't be in main thread (yields would block collection)
                """
                # Priority queue (min-heap) for reordering
                # Stores tuples: (row_idx, graph)
                # heap[0] is always the smallest row_idx not yet yielded
                heap = []
                next_to_yield = 0  # Track which row should be yielded next
                workers_done = 0   # Count worker shutdown sentinels

                while workers_done < self.num_workers:
                    # BLOCKING get: Efficient - thread sleeps until result arrives
                    # No timeout, no busy-wait. This is key to good performance.
                    row_idx, result = result_queue.get()

                    # Handle worker shutdown sentinel
                    if result is None:
                        workers_done += 1
                        continue

                    # Handle skipped molecules (invalid SMILES, etc.)
                    if result == 'SKIP':
                        # Don't add to heap, just continue - this skips the molecule
                        continue

                    # Handle errors (propagate to main iterator)
                    if isinstance(result, Exception):
                        results_buffer.put(('error', result))
                        return

                    # Add result to heap for reordering
                    heapq.heappush(heap, (row_idx, result))

                    # Push all sequential results to buffer
                    # Example: If next_to_yield=5 and heap has [5,6,7,9,10]
                    # This will push 5,6,7 and stop (waiting for 8)
                    while heap and heap[0][0] == next_to_yield:
                        _, graph = heapq.heappop(heap)
                        results_buffer.put(('result', graph))
                        next_to_yield += 1

                # All workers done - push any remaining results
                # (Shouldn't normally happen unless workers finish out of order)
                while heap:
                    _, graph = heapq.heappop(heap)
                    results_buffer.put(('result', graph))

                # Signal completion to main iterator
                results_buffer.put(('done', None))

            # ----------------------------------------------------------------
            # START WORKERS AND THREADS
            # ----------------------------------------------------------------

            # Start worker processes (these do the actual computation)
            workers = []
            for worker_id in range(self.num_workers):
                p = mp.Process(
                    target=_graph_worker,
                    args=(
                        worker_id,
                        self.processing_class,
                        self.mol_transform,
                        input_queue,
                        result_queue,
                        self.dataset_type,  # Pass dataset type for raw data conversion
                    )
                )
                p.start()
                workers.append(p)

            # Start coordination threads (daemon=True: exit when main exits)
            producer_thread = threading.Thread(target=producer, daemon=True)
            collector_thread = threading.Thread(target=collector, daemon=True)
            producer_thread.start()
            collector_thread.start()

            try:
                # ----------------------------------------------------------------
                # MAIN ITERATOR: Yield Results
                # ----------------------------------------------------------------
                # Simply reads from results_buffer and yields
                # Can suspend at yield without blocking the pipeline
                # Collector thread keeps draining result_queue regardless
                while True:
                    msg_type, value = results_buffer.get()

                    if msg_type == 'error':
                        raise value  # Propagate worker error
                    elif msg_type == 'done':
                        break  # All results yielded
                    else:  # 'result'
                        yield value

            finally:
                # ----------------------------------------------------------------
                # CLEANUP
                # ----------------------------------------------------------------
                # Wait for threads to finish gracefully
                producer_thread.join(timeout=2.0)
                collector_thread.join(timeout=2.0)

                # Drain queues to unblock any waiting threads/processes
                # (Important if iterator is stopped early via break/exception)
                try:
                    while not input_queue.empty():
                        input_queue.get_nowait()
                except:
                    pass

                try:
                    while not result_queue.empty():
                        result_queue.get_nowait()
                except:
                    pass

                # Terminate worker processes
                for p in workers:
                    p.join(timeout=1.0)
                    if p.is_alive():
                        p.terminate()  # Force kill if not responding
                        p.join()

    def _process_one(self, mol: 'Chem.Mol', properties: np.ndarray) -> dict:
        """
        Process a single Mol object into a graph dict (sequential mode).

        The mol object comes from the dataset adapter, which has already converted
        the raw data (SMILES string or XYZ data) into an RDKit Mol object.

        :param mol: The RDKit Mol object to process
        :param properties: The target properties array

        :returns: Graph dict representation
        """
        # Step 1: Apply mol_transform
        mol = self.mol_transform(mol)

        # Step 2: Process transformed Mol into graph
        graph = self._processing.process(mol, graph_labels=properties.tolist())
        return graph

    def __len__(self) -> int:
        """
        Return the number of elements in the dataset.

        :returns: The number of molecules in the dataset
        """
        return len(self.raw_dataset)

    def close(self) -> None:
        """
        No-op method for compatibility.

        Worker processes are automatically cleaned up when iteration completes.
        This method exists for API compatibility but does nothing.
        """
        pass


class ShuffleDataset(StreamingDataset):
    """
    Wrapper dataset that shuffles a streaming dataset using a shuffle buffer.

    This class provides approximate shuffling of streaming datasets without loading
    the entire dataset into memory. It uses a shuffle buffer approach where items are
    randomly selected from a fixed-size buffer, providing good shuffling performance
    with controlled memory usage.

    The quality of shuffling depends on the buffer size:
    - buffer_size >= dataset size: Perfect shuffling (equivalent to random.shuffle)
    - Smaller buffer: Approximate shuffling (items far apart in original order are
      less likely to be shuffled together)

    This is particularly useful for training machine learning models where shuffled
    data is desired but the dataset is too large to fit entirely in memory.

    Algorithm Design:
    The shuffle buffer algorithm ensures all items are yielded exactly once while
    maintaining approximate randomization:

    1. **Fill Phase**: Load first `buffer_size` items into buffer
    2. **Stream Phase**: For each remaining item:
       - Randomly select index from buffer
       - Yield the item at that index
       - Replace with new item from stream
    3. **Drain Phase**: Shuffle and yield all remaining buffer items

    This approach differs from classic reservoir sampling (which samples k items from n)
    by yielding ALL items in shuffled order, making it ideal for ML training where you
    need every example but in random order.

    Example:

    .. code-block:: python

        # Shuffle a graph dataset for training
        base_dataset = GraphDataset(
            dataset='esol',
            num_workers=4
        )

        shuffled = ShuffleDataset(
            dataset=base_dataset,
            buffer_size=5000,
            seed=42
        )

        # Training loop with shuffled data
        for epoch in range(num_epochs):
            for graph in shuffled:
                # Each epoch uses same shuffle order (deterministic with seed)
                train_step(graph)

    :param dataset: The underlying StreamingDataset to wrap (SmilesDataset or GraphDataset)
    :param buffer_size: Size of the shuffle buffer (default: 10000). Larger values
                       provide better shuffling but use more memory. Set to dataset
                       size or larger for perfect shuffling.
    :param seed: Random seed for reproducibility (default: None). If provided, each
                iteration will produce the same shuffle order.
    """

    def __init__(
        self,
        dataset: StreamingDataset,
        buffer_size: int = 10000,
        seed: Optional[int] = None
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.buffer_size = buffer_size
        self.seed = seed

    def __iter__(self) -> Iterator:
        """
        Iterate through the dataset in shuffled order using a shuffle buffer.

        The shuffle buffer algorithm works as follows:

        1. **Fill Phase**: Fill buffer with first `buffer_size` items from stream
        2. **Stream Phase**: For each subsequent item:
           - Generate random index in [0, buffer_size)
           - Yield item at that index
           - Replace it with new item from stream
        3. **Drain Phase**: Shuffle remaining buffer and yield all items

        This ensures all items are yielded exactly once, in approximately random order.
        The randomness quality depends on buffer_size: larger buffers provide better
        global shuffling, while smaller buffers only shuffle locally.

        Memory Complexity: O(buffer_size)
        Time Complexity: O(n) where n is dataset size

        :yields: Items from the underlying dataset in shuffled order
        """
        # Create RNG for this iteration (same seed gives same shuffle)
        rng = np.random.RandomState(self.seed)

        buffer = []
        dataset_iter = iter(self.dataset)

        # Phase 1: Fill the initial buffer
        for item in dataset_iter:
            buffer.append(item)
            if len(buffer) >= self.buffer_size:
                break

        # Phase 2: For each new item, yield random item from buffer and replace
        for item in dataset_iter:
            # Randomly select index to yield
            idx = rng.randint(0, len(buffer))
            yield buffer[idx]
            # Replace yielded item with new item from stream
            buffer[idx] = item

        # Phase 3: Shuffle and yield all remaining items in buffer
        rng.shuffle(buffer)
        for item in buffer:
            yield item

    def __len__(self) -> int:
        """
        Return the number of elements in the dataset.

        This is the same as the underlying dataset length since all items are yielded.

        :returns: The number of items that will be yielded
        """
        return len(self.dataset)
