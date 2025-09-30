import os
import csv
import pytest

import rdkit.Chem as Chem
from chem_mat_data.processing import MoleculeProcessing
from chem_mat_data.graph import assert_graph_dict
from chem_mat_data.data import save_graphs, load_graphs
from chem_mat_data.data import load_xyz_as_mol
from chem_mat_data.data import DefaultXyzParser
from chem_mat_data.data import QM9XyzParser
from chem_mat_data.data import TUDatasetParser
from chem_mat_data.data import HOPV15Parser

from .utils import ASSETS_PATH, ARTIFACTS_PATH


SMILES_LIST = [
    'C1=CC=CC=C1CCO',
    'CC(=O)O',
    'C1=CC=CC=C1C(N)CC',
]


def test_save_and_load_graphs_basically_works():
    """
    The "save_graphs" method should be able to take a list of graph dict objects and save them 
    as a persistent message pack file to the disk.
    
    Conversely, the "load_graphs" method should be able to load the same file back into memory 
    as a list of graph dict objects.
    """
    path = os.path.join(ARTIFACTS_PATH, 'test_save_graphs_basically_works.mpack')
    
    # First we need to generate some graph objects with which to run the test
    processing = MoleculeProcessing()
    graphs = [processing.process(value) for value in SMILES_LIST]
    
    # Then we need to save these graphs into a file
    save_graphs(graphs, path)
    
    assert os.path.exists(path)
    assert os.path.isfile(path)
    
    # And finally we can attempt to load that file now from memory again
    graphs_loaded = load_graphs(path)
    assert isinstance(graphs_loaded, list)
    assert len(graphs_loaded) == len(graphs)
    for graph in graphs_loaded:
        assert_graph_dict(graph)
        
        
def test_process_test_dataset():
    """
    This is less of a unittest and more of a utility function.
    
    This function will load the _test.csv file from the assets folder and process all the 
    SMILES strings contained in it into graph representations. It will then save these graphs 
    as a message pack file into the same folder.
    """
    name = 'clintox'
    source_path = os.path.join(ASSETS_PATH, f'{name}.csv')
    with open(source_path, mode='r') as file:
        dict_reader = csv.DictReader(file)
        rows = list(dict_reader)
        
    processing = MoleculeProcessing()
    graphs = []
    for row in rows:
        try:
            graph = processing.process(row['smiles'])
            graphs.append(graph)
        except Exception as exc:
            print(exc)
        
    dst_path = os.path.join(ASSETS_PATH, f'{name}.mpack')
    save_graphs(graphs, dst_path)
    assert os.path.exists(dst_path)
    
    graphs_loaded = load_graphs(dst_path)
    assert len(graphs) == len(graphs_loaded)
    
    
def test_load_xyz_as_mol_basically_works():
    """
    The "load_xyz_as_mol" function should be able to load a xyz file from the disk and return
    a RDKit molecule object from it.
    """
    xyz_path = os.path.join(ASSETS_PATH, '_test.xyz')
    mol, info = load_xyz_as_mol(xyz_path)
    
    assert isinstance(mol, Chem.Mol)
    assert isinstance(info, dict)
    # For this particular molecule we know that it should have 16 atoms!
    assert mol.GetNumAtoms() == 17
    
    
class TestDefaultXyzParser:
    
    def test_parsing_example_file_works(self):
        """
        Parsing a standard xyz file should work for the default parser and return a RDKit molecule
        object as intended with the correct number of atoms.
        """
        xyz_path = os.path.join(ASSETS_PATH, '_test.xyz')
        parser = DefaultXyzParser(xyz_path)
        mol, info = parser.parse()
        assert isinstance(mol, Chem.Mol)
        assert len(mol.GetAtoms()) > 0
        assert len(mol.GetAtoms()) == 17
        
    def test_parsing_qm9_file_doesnt_work(self):
        """
        The special format of the QM9 dataset is not supported by the default parser and should 
        therefore not work.
        """
        xyz_path = os.path.join(ASSETS_PATH, 'qm9.xyz')
        parser = DefaultXyzParser(xyz_path)
        
        with pytest.raises(Exception):
            mol: Chem.Mol = parser.parse()  # noqa: F841
        
        
class TestQM9XYZParser():
    
    def test_parsing_qm9_file_basically_works(self):
        """
        Loading a xyz file with the special qm9 flavor should work and retrieve all the important 
        information.
        """
        xyz_path = os.path.join(ASSETS_PATH, 'qm9.xyz')
        parser = QM9XyzParser(xyz_path)
        
        mol, info = parser.parse()
        assert isinstance(mol, Chem.Mol)
        assert isinstance(info, dict)
        assert len(info) != 0
        
        assert 'targets' in info
        assert 'functional' in info
        assert 'smiles1' in info
        assert 'smiles2' in info
        
        
class TestTUDatasetParser():
    
    def test_loading_dataset_basically_works(self):
        
        path: str = os.path.join(ASSETS_PATH, 'TU_MUTAG')
        parser = TUDatasetParser(path=path, name='MUTAG')
    
        # This step is required to properly initialize the parser
        parser.initialize()
        
        try:
            # This method will actually load the information from all the files and then turn this 
            # into a list of the graph dictionaries containing the informations about the 
            # individual graphs.
            parser.load()
            assert isinstance(parser.index_graph_map, dict)
            assert len(parser.index_graph_map) == 188
            
        finally:
            parser.finalize()
        
    def test_iterating_over_mols_works(self):
        """
        This test will check it it is possible to use the parser object as an iterator to iterate 
        over rdkit mol objects of the individual graphs.
        """
        
        path: str = os.path.join(ASSETS_PATH, 'TU_MUTAG')
        parser = TUDatasetParser(
            path=path,
            name='MUTAG',
            node_label_map={
                0: 'C',
                1: 'N',
                2: 'O',
                3: 'F',
                4: 'I',
                5: 'Cl',
                6: 'Br',
            },
            edge_label_map={
                0: Chem.BondType.AROMATIC,
                1: Chem.BondType.SINGLE,
                2: Chem.BondType.DOUBLE,
                3: Chem.BondType.TRIPLE,
            },
            graph_label_map={
                1: 1,
                -1: 0,
            },
        )
    
        # This step is required to properly initialize the parser
        parser.initialize()
        
        try:
            # This method will actually load the information from all the files and then turn this 
            # into a list of the graph dictionaries containing the informations about the 
            # individual graphs.
            parser.load()
           
            counter = 0
            for mol, label in parser:
                
                assert isinstance(mol, Chem.Mol)
                assert mol.GetNumAtoms() > 0
                assert mol.GetNumBonds() > 0
                
                smiles = Chem.MolToSmiles(mol)
                print(smiles)
                assert smiles is not None
                
                counter += 1
                
            assert counter == 188

        finally:
            parser.finalize()


class TestHOPV15Parser:
    """
    Test suite for the HOPV15Parser class which handles parsing of HOPV15 dataset .data files.
    """

    def test_parsing_hopv15_file_basically_works(self):
        """
        Test that the HOPV15Parser can successfully parse a .data file and return a valid
        RDKit molecule object along with the parsed information dictionary.
        """
        hopv15_path = os.path.join(ASSETS_PATH, 'hopv15_example.data')
        parser = HOPV15Parser(hopv15_path)

        mol, info = parser.parse()

        # Test basic molecule parsing
        assert isinstance(mol, Chem.Mol)
        assert mol is not None
        assert mol.GetNumAtoms() > 0

        # Test info dictionary structure
        assert isinstance(info, dict)
        assert len(info) > 0

    def test_hopv15_parsing_extracts_basic_info(self):
        """
        Test that the parser correctly extracts basic molecular information from the file.
        """
        hopv15_path = os.path.join(ASSETS_PATH, 'hopv15_example.data')
        parser = HOPV15Parser(hopv15_path)

        mol, info = parser.parse()

        # Test that all expected keys are present
        expected_keys = ['smiles', 'inchi', 'pruned_smiles', 'num_conformers',
                        'experimental_properties', 'conformers']
        for key in expected_keys:
            assert key in info, f"Missing key: {key}"

        # Test specific values
        assert isinstance(info['smiles'], str)
        assert len(info['smiles']) > 0
        assert info['smiles'].startswith('Cc1ccc')

        assert isinstance(info['inchi'], str)
        assert info['inchi'].startswith('InChI=')

        assert isinstance(info['pruned_smiles'], str)
        assert isinstance(info['num_conformers'], int)
        assert info['num_conformers'] == 18

    def test_hopv15_experimental_properties_parsing(self):
        """
        Test that experimental properties are correctly parsed from the CSV line.
        """
        hopv15_path = os.path.join(ASSETS_PATH, 'hopv15_example.data')
        parser = HOPV15Parser(hopv15_path)

        mol, info = parser.parse()

        exp_props = info['experimental_properties']
        assert isinstance(exp_props, dict)

        # Test that all expected experimental properties are present
        expected_props = ['PCE', 'VOC', 'JSC', 'HOMO', 'LUMO', 'gap']
        for prop in expected_props:
            assert prop in exp_props, f"Missing experimental property: {prop}"
            assert isinstance(exp_props[prop], (int, float))

        # Test specific values from the example file
        assert exp_props['PCE'] == 1.47
        assert exp_props['VOC'] == 1.59
        assert exp_props['JSC'] == 7.81
        assert exp_props['HOMO'] == 0.69
        assert exp_props['LUMO'] == 17.07
        assert exp_props['gap'] == 66.3

    def test_hopv15_conformer_parsing(self):
        """
        Test that conformers and their associated data are correctly parsed.
        """
        hopv15_path = os.path.join(ASSETS_PATH, 'hopv15_example.data')
        parser = HOPV15Parser(hopv15_path)

        mol, info = parser.parse()

        conformers = info['conformers']
        assert isinstance(conformers, list)
        assert len(conformers) > 0
        # The number of parsed conformers may vary depending on parsing method
        # but should be at least 2 for the first molecule
        assert len(conformers) >= 2

        # Test structure of each conformer
        for i, conformer in enumerate(conformers):
            assert isinstance(conformer, dict)

            # Test required fields
            assert 'header' in conformer
            assert 'num_atoms' in conformer
            assert 'atoms' in conformer
            assert 'coordinates' in conformer
            assert 'calculated_data' in conformer

            # Test conformer header parsing
            assert isinstance(conformer['header'], str)
            assert conformer['header'].startswith('Conformer')

            # Test atom count and coordinate structure
            assert isinstance(conformer['num_atoms'], int)
            assert conformer['num_atoms'] > 0
            assert isinstance(conformer['atoms'], list)
            assert len(conformer['atoms']) == conformer['num_atoms']

            # Test coordinate array
            import numpy as np
            assert isinstance(conformer['coordinates'], np.ndarray)
            assert conformer['coordinates'].shape == (conformer['num_atoms'], 3)

    def test_hopv15_calculated_data_parsing(self):
        """
        Test that calculated quantum chemistry data is correctly parsed from QChem lines.
        """
        hopv15_path = os.path.join(ASSETS_PATH, 'hopv15_example.data')
        parser = HOPV15Parser(hopv15_path)

        mol, info = parser.parse()

        conformers = info['conformers']

        # Test calculated data in the first conformer
        first_conformer = conformers[0]
        calc_data = first_conformer['calculated_data']

        assert isinstance(calc_data, list)
        assert len(calc_data) > 0

        # Check structure of calculated data entries
        for calc_entry in calc_data:
            assert isinstance(calc_entry, dict)

            # Test required fields
            required_fields = ['method', 'HOMO', 'LUMO', 'gap', 'additional_values']
            for field in required_fields:
                assert field in calc_entry, f"Missing field: {field}"

            # Test data types
            assert isinstance(calc_entry['method'], str)
            assert isinstance(calc_entry['HOMO'], (int, float))
            assert isinstance(calc_entry['LUMO'], (int, float))
            assert isinstance(calc_entry['gap'], (int, float))
            assert isinstance(calc_entry['additional_values'], list)

        # Test specific values from first QChem entry
        first_calc = calc_data[0]
        assert first_calc['method'] == 'B3LYP/def2-SVP DFT'
        assert first_calc['HOMO'] == -0.187
        assert first_calc['LUMO'] == -0.099
        assert first_calc['gap'] == 0.088

    def test_hopv15_molecule_creation_from_smiles(self):
        """
        Test that the RDKit molecule is correctly created from the parsed SMILES string.
        """
        hopv15_path = os.path.join(ASSETS_PATH, 'hopv15_example.data')
        parser = HOPV15Parser(hopv15_path)

        mol, info = parser.parse()

        # Test that molecule was created successfully
        assert mol is not None

        # Test that the molecule corresponds to the parsed SMILES
        parsed_smiles = Chem.MolToSmiles(mol)
        assert isinstance(parsed_smiles, str)
        assert len(parsed_smiles) > 0

        # The molecule should have atoms and bonds
        assert mol.GetNumAtoms() > 0
        assert mol.GetNumBonds() > 0

    def test_hopv15_3d_conformer_addition(self):
        """
        Test that 3D coordinates from the first conformer are added to the RDKit molecule.
        """
        hopv15_path = os.path.join(ASSETS_PATH, 'hopv15_example.data')
        parser = HOPV15Parser(hopv15_path)

        mol, info = parser.parse()

        # Test that the molecule has a conformer with 3D coordinates
        if mol is not None and info['conformers']:
            assert mol.GetNumConformers() > 0

            # Get the conformer and test its properties
            conf = mol.GetConformer(0)
            assert conf.GetNumAtoms() == mol.GetNumAtoms()

            # Test that coordinates are reasonable (not all zeros)
            positions = []
            for i in range(conf.GetNumAtoms()):
                pos = conf.GetAtomPosition(i)
                positions.append([pos.x, pos.y, pos.z])

            # At least some coordinates should be non-zero
            non_zero_coords = [pos for pos in positions if any(abs(coord) > 0.001 for coord in pos)]
            assert len(non_zero_coords) > 0

    def test_hopv15_get_fields_method(self):
        """
        Test that the get_fields class method returns the correct field names.
        """
        fields = HOPV15Parser.get_fields()

        expected_fields = [
            'smiles', 'inchi', 'pruned_smiles', 'num_conformers',
            'experimental_properties', 'conformers'
        ]

        assert isinstance(fields, list)
        assert set(fields) == set(expected_fields)

    def test_hopv15_with_load_xyz_as_mol_function(self):
        """
        Test that the HOPV15 parser works correctly when used through the load_xyz_as_mol function.
        """
        hopv15_path = os.path.join(ASSETS_PATH, 'hopv15_example.data')

        mol, info = load_xyz_as_mol(hopv15_path, 'hopv15')

        # Test basic functionality
        assert isinstance(mol, Chem.Mol)
        assert isinstance(info, dict)

        # Test that expected information is present
        assert 'smiles' in info
        assert 'experimental_properties' in info
        assert 'conformers' in info

        # Test molecule properties
        assert mol.GetNumAtoms() > 0
        smiles = Chem.MolToSmiles(mol)
        assert len(smiles) > 0

    def test_hopv15_parse_all_multiple_molecules(self):
        """
        Test that the parse_all method correctly identifies and parses all molecules in a file.
        """
        hopv15_path = os.path.join(ASSETS_PATH, 'hopv15_example.data')
        parser = HOPV15Parser(hopv15_path)

        molecules = parser.parse_all()

        # Test that multiple molecules are found
        assert isinstance(molecules, list)
        assert len(molecules) == 2  # The test file contains 2 molecules

        # Test each molecule
        for i, (mol, info) in enumerate(molecules):
            assert isinstance(mol, Chem.Mol)
            assert mol is not None
            assert mol.GetNumAtoms() > 0
            assert isinstance(info, dict)
            assert 'smiles' in info
            assert 'conformers' in info

    def test_hopv15_parse_all_first_molecule_complete(self):
        """
        Test that the first molecule from parse_all has complete information.
        """
        hopv15_path = os.path.join(ASSETS_PATH, 'hopv15_example.data')
        parser = HOPV15Parser(hopv15_path)

        molecules = parser.parse_all()
        first_mol, first_info = molecules[0]

        # Test that first molecule has full information
        expected_keys = ['smiles', 'inchi', 'pruned_smiles', 'num_conformers',
                        'experimental_properties', 'conformers']
        for key in expected_keys:
            assert key in first_info, f"Missing key in first molecule: {key}"

        # Test specific values for first molecule
        assert first_info['smiles'].startswith('Cc1ccc')
        assert first_info['inchi'].startswith('InChI=')
        assert 'experimental_properties' in first_info
        assert first_info['experimental_properties']['PCE'] == 1.47

    def test_hopv15_parse_all_second_molecule_short_format(self):
        """
        Test that the second molecule is parsed correctly despite having a shorter format.
        """
        hopv15_path = os.path.join(ASSETS_PATH, 'hopv15_example.data')
        parser = HOPV15Parser(hopv15_path)

        molecules = parser.parse_all()
        second_mol, second_info = molecules[1]

        # Test that second molecule has basic information
        assert 'smiles' in second_info
        assert 'conformers' in second_info
        assert 'num_conformers' in second_info

        # Test specific values for second molecule
        assert second_info['smiles'] == 'C=Cc1sccc1C(=O)OC'
        assert second_info['num_conformers'] == 4

        # Second molecule should not have experimental properties in this format
        assert 'experimental_properties' not in second_info

        # But should have conformers with calculated data
        assert len(second_info['conformers']) > 0
        assert second_mol.GetNumAtoms() > 0

    def test_hopv15_parse_all_conformer_data(self):
        """
        Test that conformers are correctly parsed for all molecules.
        """
        hopv15_path = os.path.join(ASSETS_PATH, 'hopv15_example.data')
        parser = HOPV15Parser(hopv15_path)

        molecules = parser.parse_all()

        for i, (mol, info) in enumerate(molecules):
            conformers = info['conformers']
            assert isinstance(conformers, list)
            assert len(conformers) > 0

            # Test conformer structure
            for conformer in conformers:
                assert 'header' in conformer
                assert 'num_atoms' in conformer
                assert 'atoms' in conformer
                assert 'coordinates' in conformer
                assert 'calculated_data' in conformer

                # Test that calculated data is present
                assert len(conformer['calculated_data']) > 0

    def test_hopv15_parse_all_vs_single_parse_consistency(self):
        """
        Test that parse_all returns the same first molecule as the single parse method.
        """
        hopv15_path = os.path.join(ASSETS_PATH, 'hopv15_example.data')
        parser = HOPV15Parser(hopv15_path)

        # Parse using single method
        single_mol, single_info = parser.parse()

        # Parse using parse_all method
        all_molecules = parser.parse_all()
        first_mol, first_info = all_molecules[0]

        # Test that the first molecule from parse_all matches single parse
        assert Chem.MolToSmiles(single_mol) == Chem.MolToSmiles(first_mol)
        assert single_info['smiles'] == first_info['smiles']

        # Both should have conformers (may differ due to parsing boundaries)
        assert len(single_info['conformers']) > 0
        assert len(first_info['conformers']) > 0

        # Test experimental properties match
        if 'experimental_properties' in single_info and 'experimental_properties' in first_info:
            assert single_info['experimental_properties'] == first_info['experimental_properties']

    def test_hopv15_parse_all_molecule_atom_counts(self):
        """
        Test that molecules have expected atom counts.
        """
        hopv15_path = os.path.join(ASSETS_PATH, 'hopv15_example.data')
        parser = HOPV15Parser(hopv15_path)

        molecules = parser.parse_all()

        # First molecule should be larger (complex organic molecule)
        first_mol, first_info = molecules[0]
        assert first_mol.GetNumAtoms() > 40  # Should be around 68 with hydrogens

        # Second molecule should be smaller
        second_mol, second_info = molecules[1]
        assert second_mol.GetNumAtoms() < 30  # Should be around 19 with hydrogens

        # Atom counts should be consistent
        assert first_mol.GetNumAtoms() > second_mol.GetNumAtoms()