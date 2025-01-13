import os

import lorem
import matplotlib.pyplot as plt
import chem_mat_data._typing as tc
import rdkit.Chem as Chem
from rich.console import Console
from rich.pretty import pprint
from chem_mat_data.testing import assert_graph_dict
#from chem_mat_data.graph import assert_graph_dict
from chem_mat_data.processing import OneHotEncoder
from chem_mat_data.processing import RichProcessingSummary
from chem_mat_data.processing import CrippenEncoder
from chem_mat_data.processing import MoleculeProcessing
from chem_mat_data.data import load_xyz_as_mol

from .utils import ARTIFACTS_PATH, ASSETS_PATH


class TestMoleculeProcessing:
    
    DEFAULT_SMILES = 'C1=CC=CC=C1CCO'
    
    def test_basically_works(self):
        """
        Constructs a new instance of the class. "process" method should turn a SMILES into a 
        GraphDict representation
        """
        processing = MoleculeProcessing()
        assert isinstance(processing, MoleculeProcessing)
        
        smiles = self.DEFAULT_SMILES
        graph: tc.GraphDict = processing.process(smiles)
        
        # This function will run a series of assert statements that checks the validity of the graph dict
        # formatting. So if it contains the correct keys and the correct array shapes etc.
        assert_graph_dict(graph)
        pprint(graph)
        
    def test_process_includes_node_atom_symbols(self):
        """
        When processing a smiles string the resulting graph dict should also contian the "node_atoms" entry which 
        is essentially a list of string representations of the atom symbols.
        """
        processing = MoleculeProcessing()
        
        smiles = self.DEFAULT_SMILES
        graph: tc.GraphDict = processing.process(smiles)
        
        assert 'node_atoms' in graph
        print(graph['node_atoms'])
        assert len(graph['node_indices']) == len(graph['node_atoms'])
        assert isinstance(graph['node_atoms'][0], str)
        
    def test_process_includes_edge_bond_types(self):
        """
        When processing a smiles string the resulting graph dict should also contain the "edge_bonds" entry which
        is essentially a list of string representations of the bond types.
        """
        processing = MoleculeProcessing()
        
        smiles = self.DEFAULT_SMILES
        graph: tc.GraphDict = processing.process(smiles)
        
        assert 'edge_bonds' in graph
        print(graph['edge_bonds'])
        assert len(graph['edge_bonds']) == len(graph['edge_indices'])
        assert isinstance(graph['edge_bonds'][0], str)
        
    def test_process_includes_graph_representation(self):
        """
        When processing a smiles string the resulting graph dict should also contain the "graph_repr" entry which 
        is essentially a string representation of the original SMILES string.
        """
        processing = MoleculeProcessing()
        
        smiles = self.DEFAULT_SMILES
        graph: tc.GraphDict = processing.process(smiles)
        
        assert 'graph_repr' in graph
        assert isinstance(graph['graph_repr'], str)
        assert graph['graph_repr'] == smiles
        
    def test_visualize_as_figure(self):
        """
        visualize_as_figure should use the rdkit SVG engine to draw the given molecule and 
        and project it into a matplotlib Figure which we can then save to the disk as PNG
        """
        processing = MoleculeProcessing()
        smiles = self.DEFAULT_SMILES
        fig, _ = processing.visualize_as_figure(smiles, width=1000, height=1000)

        assert isinstance(fig, plt.Figure)
        
        fig_path = os.path.join(ARTIFACTS_PATH, 'test_molecule_processing_visualize_as_figure.png')
        fig.savefig(fig_path)
        assert os.path.exists(fig_path)
        
    def test_summary(self):
        """
        The summary method should return / print a summary and description of the molecule processing 
        steps aka descriptions of the node and edge attributes that were extracted for the molecular 
        graph.
        """
        processing = MoleculeProcessing()
        summary = processing.summary(echo=False)
        st = str(summary)
        print('\n', st)
        
        assert isinstance(st, str)
        assert len(st) != 0

    def test_process_mol_object_works(self):
        """
        In principle, the process function is designed to work on the SMILES representation of a molecule 
        but it should also work with the rdkit Mol object itself as converting to a Mol is the first 
        thing that happens in the process function anyway.
        """
        smiles = self.DEFAULT_SMILES
        mol = Chem.MolFromSmiles(smiles)
        
        processing = MoleculeProcessing()
        graph = processing.process(mol)
        assert isinstance(graph, dict)
        assert_graph_dict(graph)
        
    def test_process_mol_object_from_xyz_file_works(self):
        """
        As a special case of processing a RDKit molecule object, we can also load a molecule from a
        xyz file and then process it into a graph representation. This is perhaps a bit different 
        because by default the Mol object that is loaded from the xyz file does not contain any 
        bonds/edges.
        """
        xyz_path = os.path.join(ASSETS_PATH, '_test.xyz')
        mol, info = load_xyz_as_mol(xyz_path)
        print(mol, mol.GetNumAtoms())
        assert isinstance(mol, Chem.Mol)
        assert isinstance(info, dict)
        
        processing = MoleculeProcessing()
        graph = processing.process(mol)
        assert isinstance(graph, dict)
        assert_graph_dict(graph)
        
        # Another specialty of processing a mol from an xyz file is that the graph representation
        # should also contain the 3D node coordinates.
        assert 'node_coordinates' in graph


class TestCrippenEncoder:
    
    DEFAULT_SMILES = 'C1=CC=CC=C1CCO'
    
    def test_basically_works(self):
        
        encoder = CrippenEncoder()
        
        mol = Chem.MolFromSmiles(self.DEFAULT_SMILES)
        for atom in mol.GetAtoms():
            value: list[float] = encoder.encode(mol, atom)
            print(value)
            
            assert isinstance(value, list)
            assert len(value) > 0
            # We know that the crippen contribution consists of two values.
            assert len(value) == 2 

    def test_description_works(self):
        """
        The descriptions property should return a list of strings which describe the encoding of 
        the two values that are created by the encoder.
        """
        encoder = CrippenEncoder()
        descriptions = encoder.descriptions
        
        assert isinstance(descriptions, list)
        assert len(descriptions) == 2
        for desc in descriptions:
            assert isinstance(desc, str)
            assert len(desc) > 0
            

class TestRichProcessingSummary:
    
    DEFAULT_SECTIONS = {
            'section 1': {
                0: lorem.sentence(),
                1: lorem.paragraph(),
            },
            'section 2': {
                0: lorem.sentence(),
                1: lorem.sentence(),
                2: lorem.sentence(), 
            }
        }
    
    def test_basically_works(self):
            
        rich_summary = RichProcessingSummary(self.DEFAULT_SECTIONS)
        assert isinstance(rich_summary, RichProcessingSummary)
        
        console = Console()
        with console.capture() as capture:
            console.print(rich_summary)
            
        st = capture.get()
        print('')
        print(st)
        
        # Checking for some sections whether they are in the string
        assert 'section 1' in st
        assert 'section 2' in st
        assert '00' in st
        
    def test_str_conversion_works(self):
        """
        RichProcessingSummary extends RichMixin, which provides a default implementation for 
        the __str__ method. Therefore it should be possible to call str() on the object and 
        get a plain text representation of the string.
        """
        rich_summary = RichProcessingSummary(self.DEFAULT_SECTIONS)
        st = str(rich_summary)
        print(st)
        
        assert isinstance(st, str)
        assert len(st) > 0
        
        # Checking for some sections whether they are in the string
        assert 'section 1' in st
        assert 'section 2' in st
        assert '00' in st


class TestOneHotEncoder:
    
    def test_basically_works(self):
        """
        The OneHotEncoder should be able to encode and decode values from a list of categories.
        """
        encoder = OneHotEncoder(['a', 'b', 'c'], add_unknown=False, dtype=str)
        
        # the most simple test cases
        assert encoder.encode('a') == [1., 0., 0.]
        assert encoder.encode('b') == [0., 1., 0.]
        assert encoder.encode('c') == [0., 0., 1.]
        
        # Without an unknown this should simply not be encoded at all
        assert encoder.encode('x') == [0., 0., 0.]
        
        # decoding should also work for the basic cases
        assert encoder.decode([1., 0., 0.]) == 'a'
        assert encoder.decode([0., 1., 0.]) == 'b'
        assert encoder.decode([0., 0., 1.]) == 'c'
        
    def test_unknown_works(self):
        """
        When construction the encoder, we can specify that we want to add an "unknown" category
        in which case all values not in the original list will be encoded into a additional 
        category "unknown".
        """
        encoder = OneHotEncoder(
            ['a', 'b', 'c'], 
            add_unknown=True, 
            dtype=str,
            unknown='?'    
        )
        
        assert encoder.encode('a') == [1., 0., 0., 0.]
        assert encoder.encode('x') == [0., 0., 0., 1.]
        
        # Also the decoding operation should now return the default value for the unkown category
        # when decoding such an unknown vector.
        assert encoder.decode([0., 0., 0., 1.]) == '?'
        
    def test_descriptions_basically_work(self):
        """
        The "descriptions" property should return a list of strings which describe the encoding.
        """
        encoder = OneHotEncoder(['a', 'b', 'c'], add_unknown=False, dtype=str)
        
        # "descriptions" is a property which should create a list of string descriptions
        descriptions = encoder.descriptions
        assert isinstance(descriptions, list)
        assert descriptions == ['is a?', 'is b?', 'is c?']
        
    def test_custom_descriptions_work(self):
        """
        It should be possible to provide custom descriptions for the encoder categories in the 
        constructor. These descriptions will then be used instead of the actual values to generate 
        the descriptions list.
        """
        encoder = OneHotEncoder(
            ['a', 'b', 'c'], 
            add_unknown=False, 
            dtype=str,
            value_descriptions=['letter A', 'letter B', 'letter C']
        )
        
        descriptions = encoder.descriptions
        assert descriptions == ['is letter A?', 'is letter B?', 'is letter C?']