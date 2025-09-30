"""
This module implements the saving and loading of the datasets from and to the persistent 
file storage representations.
"""
import re
import os
import ase.io
import rdkit.Chem as Chem
import msgpack
import numpy as np
import tempfile
import zipfile
import pandas as pd

from typing import Dict, List, Tuple, Union, Optional, Any, Generator
from chem_mat_data._typing import GraphDict

from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor


def default(obj):
    
    if isinstance(obj, np.ndarray):
        return msgpack.ExtType(1, msgpack.packb({
            'data': obj.tobytes(),
            'dtype': str(obj.dtype),
            'shape': obj.shape,
        }))
    
    return obj


def ext_hook(code, data):
    
    if code == 1:
        d = msgpack.unpackb(data)
        return np.frombuffer(d['data'], dtype=d['dtype']).reshape(d['shape'])
    
    return data



def save_graphs(graphs: List[GraphDict],
                path: str,
                ) -> None:

    with open(path, mode='wb') as file:
        packed = msgpack.packb(graphs, default=default)
        file.write(packed)


def load_graphs(path: str) -> List[GraphDict]:
    
    with open(path, mode='rb') as file:
        content: bytes = file.read()
        return msgpack.unpackb(content, ext_hook=ext_hook)


# == MOLECULE UTILS ==

def fix_nitro_groups(mol: Chem.Mol) -> Chem.Mol:
    """
    Identify and fix neutral nitro groups by assigning charges to the nitrogen and oxygen atoms.
    
    :param mol: The RDKit molecule object to be fixed.
    
    :returns: The modified RDKit molecule object with the nitro groups fixed.
    """
    # Pattern for neutral nitro group (N single-bonded to two O)
    nitro_pattern = Chem.MolFromSmarts('[N](=O)[O]')  # N (3 bonds) with two O

    # Find all matches
    matches = mol.GetSubstructMatches(nitro_pattern)

    for match in matches:
        n_idx, o1_idx, o2_idx = match
        n_atom = mol.GetAtomWithIdx(n_idx)
        o1_atom = mol.GetAtomWithIdx(o1_idx)
        o2_atom = mol.GetAtomWithIdx(o2_idx)

        # Assign charges: N gets +1, one O gets -1
        n_atom.SetFormalCharge(1)
        # Pick the O atom that has only one bond!
        if not (o1_atom.GetDegree() == 1):
            o1_atom.SetFormalCharge(-1)
            o1_atom.SetNumExplicitHs(0)
        else:
            o2_atom.SetFormalCharge(-1)
            o2_atom.SetNumExplicitHs(0)

    return mol


# == XYZ FILES ==

class AbstractXyzParser:
    """
    Abstract base class for xyz file parsers. This class defines the interface that all different 
    implementations of xyz file parsers should adhere to. The main method that needs to be implemented 
    is the ``parse`` method which should return a tuple of a RDKit molecule object and a dictionary 
    with additional information (including the target values).
    """
    def __init__(self, path: str, **kwargs):
        self.path = path

    def parse(self) -> Tuple[Chem.Mol, dict]:
        """
        This method should be implemented by the concrete implementations of the XYZ parser. It should
        actually load the content of the xyz file and return the corresponding RDKit molecule object
        as well as a dictionary with additional information. 
        
        :returns: A tuple (mol, info) where mol is the Chem.Mol object representing the loaded molecule 
            and info is a dictionary object containing additional information from the xyz file which 
            cannot be attached to the mol object such as potentially information about target property 
            annotations.
        """
        raise NotImplementedError()
    
    @classmethod
    def get_fields(cls) -> List[str]:
        """
        This class method should return a list containing the string key names which will be included in 
        the additional "info" dicts that are returned by the "parse" method for the particular flavor 
        of xyz file.
        
        :returns: a list of string keys
        """
        raise NotImplementedError()
    
    
class DefaultXyzParser(AbstractXyzParser):
    """
    This is the default implementation of the XYZ parser which uses the loading capabilities that are 
    already present in the ASE library to load the information in the xyz file. This parser should be 
    sufficient for most use cases.
    """
    
    def __init__(self, path: str, **kwargs):
        AbstractXyzParser.__init__(self, path)
    
    def parse(self) -> Tuple[Chem.Mol, dict]:
        # This dict will store the additional information about the molecule, which in the default case
        # there are None.
        info: dict = {}
        
        # first we initialize a read-write molecule which we can then populate with the atoms
        # loaded from the xyz file
        mol = Chem.RWMol()
        
        # The "read" function will parse the xyz file and return an Atoms object which itself 
        # is an iterable of Atom objects. We can then iterate over these atoms and add them
        # to the molecule.
        atoms: ase.atoms.Atoms = ase.io.read(self.path, format='xyz')
        for atom in atoms:
            mol.AddAtom(Chem.Atom(atom.symbol))
        
        # We then need to add the positions of the atoms to the molecule's conformer object
        # which is a container for the 3D coordinates of the atoms.
        conf = Chem.Conformer(len(atoms))
        for i, atom in enumerate(atoms):
            pos = atom.position
            conf.SetAtomPosition(i, pos)
            
        mol.AddConformer(conf)
        # Finally we need to convert the read-write molecule to a read-only molecule and return
        mol: Chem.Mol = mol.GetMol()
        mol.UpdatePropertyCache()
        
        return mol, info

    @classmethod
    def get_fields(cls) -> List[str]:
        """
        Since the default xyz file format does not contain any additional info besides the atom positions,
        there are no additional keys in the "info" dicts and this method returns an empty list.
        
        :returns: empty list
        """
        return []

    
class QM9XyzParser(AbstractXyzParser):
    """
    This is the specific parser class for the QM9 flavor of xyz files. Unlike the standard format, the QM9 XYZ files 
    contain additional information about the target values that have been calculated as well as the SMILES strings of 
    the corresponding molecules.
    """
    def __init__(self, path: str, **kwargs):
        AbstractXyzParser.__init__(self, path)
        
    def parse(self) -> Tuple[Chem.Mol, dict]:
        """
        This method will parse the content of the xyz file and return the corresponding RDKit molecule object as well
        as a dictionary with additional information about the molecule.
        
        :returns: A tuple (mol, info) where mol is the Chem.Mol object representing the loaded molecule and info is a
            dictionary object containing additional information from the xyz file which cannot be attached to the mol 
            object such as potentially information about target property annotations.
        """
        # This dict will serve to hold all of the additional information that is loaded 
        # from the file which cannot be directly attached to the mol object.
        info: dict = {}
        
        with open(self.path, mode='r') as file:
            content = file.read()
        
        # ~ header information
        pattern_header = re.compile(
            r'^(?P<num_atoms>\d+)\s?\n'
            r'(?P<functional>[\w\d]*)\s?'
            r'(?P<target_values>(?:-?\d+(?:\.\d+)?\s+)+)\n'
        )
        
        match = pattern_header.match(content)
        
        num_atoms = int(match.group('num_atoms'))
        functional = match.group('functional')
        target_values = match.group('target_values')
        target_values = [
            float(value.replace(' ', '')) 
            for value in target_values.split() 
            if value != ''
        ]
        info['functional'] = functional
        info['targets'] = target_values
        
        # ~ atom information
        pattern_atoms = re.compile(
            r'\n(?P<symbol>\w)[\s\t]+'
            r'(?P<x>-?[\d\.]*)[\s\t]+'
            r'(?P<y>-?[\d\.]*)[\s\t]+'
            r'(?P<z>-?[\d\.]*)[\s\t]+'
            r'(?P<charge>-?[\d\.]*)'
        )
        
        matches = pattern_atoms.finditer(content)
        
        mol = Chem.RWMol()
        conf = Chem.Conformer(num_atoms)
        
        for i, match in enumerate(matches):
            symbol = match.group('symbol')
            x = float(match.group('x'))
            y = float(match.group('y'))
            z = float(match.group('z'))
            
            atom = Chem.Atom(symbol)
            mol.AddAtom(atom)
            conf.SetAtomPosition(i, (x, y, z))
        
        mol.AddConformer(conf)
        mol.UpdatePropertyCache()
        
        # ~ smiles information
        pattern_smiles = re.compile(
            r'\n(?P<smiles1>[-\#\.\+\w\d\\(\)[\]\@\=]*)[\s\t]+'
            r'(?P<smiles2>[-\#\.\+\w\d\\(\)[\]\@\=]*)[\s\t]+\n'
        )
        match = re.search(pattern_smiles, content)
        smiles1 = match.group('smiles1')
        smiles2 = match.group('smiles2')
        info['smiles1'] = smiles1
        info['smiles2'] = smiles2
        
        return mol, info
    
    @classmethod
    def get_fields(cls):
        """
        Returns the list of string keys which are included in the additional "info" dict that is
        returned by the "parse" method.

        For the QM9 flavor of xyz files, this includes the following keys:
        - targets: A list of float values representing the 12 target values that have been calculated for
          each molecule element using the QM calculations.
        - functional: The string name of the functional that was used for the calculations.
        - smiles1: The smiles string that represents the molecule that is described by the xyz file.
        """
        return ['targets', 'functional', 'smiles1', 'smiles2']


class HOPV15Parser(AbstractXyzParser):
    """
    Parser for the HOPV15 dataset format using a 2-step approach:
    1. Split the file into individual molecule chunks based on recognizable patterns
    2. Parse each chunk separately using pattern matching and grammar parsing

    The HOPV15 dataset consists of .data files containing molecular information including
    SMILES, InChI, experimental properties, and multiple conformers with XYZ coordinates
    and calculated properties.

    The file format can have two variations:
    1. Full format: SMILES -> InChI -> experimental data -> pruned SMILES -> conformers
    2. Short format: SMILES -> conformers
    """

    # Define the single molecule grammar (simpler than multi-molecule grammar)
    SINGLE_MOLECULE_GRAMMAR = r"""
    molecule = full_molecule / short_molecule

    full_molecule = smiles_line newline inchi_line newline experimental_line newline
                   pruned_smiles_line newline conformers

    short_molecule = smiles_line newline conformers

    smiles_line = smiles
    inchi_line = inchi
    experimental_line = experimental_data
    pruned_smiles_line = smiles

    conformers = num_conformers_line newline conformer+

    num_conformers_line = number

    conformer = conformer_header newline num_atoms_line newline xyz_coords calculated_data*

    conformer_header = "Conformer" ws+ number
    num_atoms_line = number

    xyz_coords = xyz_coord+
    xyz_coord = element ws+ coordinate ws+ coordinate ws+ coordinate newline

    calculated_data = qchem_line newline?
    qchem_line = "QChem" ws+ method "," coordinate "," coordinate "," coordinate "," additional_values

    smiles = ~r"[A-Za-z0-9@+\-\[\]()=#\\.\\/:]+(?=\n)"
    inchi = ~r"InChI=[^\n]+(?=\n)"
    experimental_data = ~r"[^,\n]*(?:,[^,\n]*)*,-?\d+(?:\.\d+)?,-?\d+(?:\.\d+)?,-?\d+(?:\.\d+)?,-?\d+(?:\.\d+)?,-?\d+(?:\.\d+)?,-?\d+(?:\.\d+)?(?=\n)"

    method = ~r"[^,]+(?=,)"
    element = ~r"[A-Z][a-z]?(?=\s)"
    coordinate = ~r"-?\d+\.\d+(?:[eE][+-]?\d+)?"
    number = ~r"\d+(?=\n)"
    additional_values = ~r"[^\n\r]*"

    ws = ~r"[ \t]+"
    newline = ~r"\n"
    """

    def __init__(self, path: str, **kwargs):
        AbstractXyzParser.__init__(self, path)
        self.grammar = Grammar(self.SINGLE_MOLECULE_GRAMMAR)

    class HOPV15Visitor(NodeVisitor):
        """Custom visitor to transform the parsed tree into structured data"""

        def visit_molecule(self, node, visited_children):
            return visited_children[0]  # full_molecule or short_molecule

        def visit_full_molecule(self, node, visited_children):
            smiles, _, inchi, _, exp_data, _, pruned_smiles, _, conformers = visited_children
            return {
                'smiles': smiles,
                'inchi': inchi,
                'experimental_properties': exp_data,
                'pruned_smiles': pruned_smiles,
                'num_conformers': conformers['num_conformers'],
                'conformers': conformers['conformers']
            }

        def visit_short_molecule(self, node, visited_children):
            smiles, _, conformers = visited_children
            return {
                'smiles': smiles,
                'num_conformers': conformers['num_conformers'],
                'conformers': conformers['conformers']
            }

        def visit_smiles_line(self, node, visited_children):
            return visited_children[0]

        def visit_inchi_line(self, node, visited_children):
            return visited_children[0]

        def visit_experimental_line(self, node, visited_children):
            return visited_children[0]

        def visit_pruned_smiles_line(self, node, visited_children):
            return visited_children[0]

        def visit_conformers(self, node, visited_children):
            num_conformers, _, conformer_list = visited_children
            return {
                'num_conformers': num_conformers,
                'conformers': conformer_list
            }

        def visit_num_conformers_line(self, node, visited_children):
            return visited_children[0]

        def visit_conformer(self, node, visited_children):
            header, _, num_atoms, _, xyz_coords, calc_data = visited_children
            conformer_info = {
                'header': header['header'],
                'conformer_id': header['conformer_id'],
                'num_atoms': num_atoms,
                'atoms': [coord['element'] for coord in xyz_coords],
                'coordinates': np.array([[coord['x'], coord['y'], coord['z']] for coord in xyz_coords]),
                'calculated_data': calc_data
            }
            return conformer_info

        def visit_conformer_header(self, node, visited_children):
            _, _, number = visited_children
            return {
                'header': node.text.strip(),
                'conformer_id': number
            }

        def visit_num_atoms_line(self, node, visited_children):
            return visited_children[0]

        def visit_xyz_coords(self, node, visited_children):
            return visited_children

        def visit_xyz_coord(self, node, visited_children):
            element, _, x, _, y, _, z, _ = visited_children
            return {
                'element': element,
                'x': x,
                'y': y,
                'z': z
            }

        def visit_calculated_data(self, node, visited_children):
            qchem_line, _ = visited_children
            return qchem_line

        def visit_qchem_line(self, node, visited_children):
            _, _, method, _, homo, _, lumo, _, gap, _, additional = visited_children

            # Parse additional values
            additional_values = []
            for val in additional.split(','):
                try:
                    additional_values.append(float(val.strip()))
                except ValueError:
                    continue

            return {
                'method': method,
                'HOMO': homo,
                'LUMO': lumo,
                'gap': gap,
                'additional_values': additional_values
            }

        def visit_smiles(self, node, visited_children):
            return node.text.strip()

        def visit_inchi(self, node, visited_children):
            return node.text.strip()

        def visit_experimental_data(self, node, visited_children):
            exp_line = node.text.strip()
            # Extract the last 6 numerical values
            parts = exp_line.split(',')
            values = []
            for part in reversed(parts):
                try:
                    values.append(float(part.strip()))
                    if len(values) == 6:
                        break
                except ValueError:
                    continue

            if len(values) == 6:
                values.reverse()  # Reverse to get correct order
                return {
                    'PCE': values[0],    # Power conversion efficiency
                    'VOC': values[1],    # Open circuit potential
                    'JSC': values[2],    # Short circuit current density
                    'HOMO': values[3],   # HOMO energy
                    'LUMO': values[4],   # LUMO energy
                    'gap': values[5]     # HOMO-LUMO gap
                }
            return None

        def visit_method(self, node, visited_children):
            return node.text.strip()

        def visit_element(self, node, visited_children):
            return node.text.strip()

        def visit_coordinate(self, node, visited_children):
            return float(node.text.strip())

        def visit_number(self, node, visited_children):
            return int(node.text.strip())

        def visit_additional_values(self, node, visited_children):
            return node.text.strip()

        def generic_visit(self, node, visited_children):
            return visited_children or node

    def _split_file_into_molecules(self, content: str) -> List[str]:
        """
        Split the HOPV15 file content into individual molecule chunks.

        This is the first step of the 2-step parsing approach.

        :param content: The complete file content as a string
        :returns: List of strings, each containing one molecule's data
        """
        lines = content.split('\n')
        molecule_chunks = []
        current_chunk_lines = []
        in_qchem_section = False
        qchem_count = 0

        for i, line in enumerate(lines):
            line = line.strip()

            # Skip empty lines at the start of a potential new molecule
            if not line and not current_chunk_lines:
                continue

            current_chunk_lines.append(line)

            # Track QChem sections to identify molecule boundaries
            if line.startswith('QChem '):
                in_qchem_section = True
                qchem_count += 1

            # After being in a QChem section, if we encounter a line that looks like a new SMILES
            # and we've seen some QChem data, this is likely a new molecule
            elif in_qchem_section and qchem_count >= 1:
                # Check if this line looks like a SMILES string starting a new molecule
                if (len(line) > 20 and
                    not line.startswith('InChI=') and
                    not line.startswith('QChem ') and
                    not line.startswith('Conformer ') and
                    not line.replace('.', '').replace(',', '').replace('-', '').replace('nan', '').replace(' ', '').isdigit() and
                    ('c' in line.lower() or 'C' in line) and  # Contains carbon (likely organic chemistry)
                    '(' in line and ')' in line):  # Has typical SMILES parentheses

                    # This is likely the start of a new molecule
                    # Remove this line from current chunk and save the current chunk
                    current_chunk_lines.pop()

                    if current_chunk_lines:
                        molecule_content = '\n'.join(current_chunk_lines)
                        if molecule_content.strip():
                            molecule_chunks.append(molecule_content)

                    # Start new chunk with this line
                    current_chunk_lines = [line]
                    in_qchem_section = False
                    qchem_count = 0

        # Add the last chunk if it exists
        if current_chunk_lines:
            molecule_content = '\n'.join(current_chunk_lines)
            if molecule_content.strip():
                molecule_chunks.append(molecule_content)

        return molecule_chunks

    def _parse_single_molecule(self, molecule_content: str) -> dict:
        """
        Parse a single molecule chunk using the grammar.

        This is the second step of the 2-step parsing approach.

        :param molecule_content: String containing one molecule's data
        :returns: Dictionary containing parsed molecule data
        """
        try:
            # Parse using grammar
            tree = self.grammar.parse(molecule_content)
            visitor = self.HOPV15Visitor()
            molecule_data = visitor.visit(tree)
            return molecule_data
        except Exception as e:
            # If grammar parsing fails, try to extract basic information manually
            lines = molecule_content.split('\n')
            basic_data = {'smiles': None, 'conformers': []}

            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue

                # First substantial line is likely SMILES
                if basic_data['smiles'] is None and len(line) > 10 and not line.startswith('InChI='):
                    basic_data['smiles'] = line

            # If we at least have a SMILES, return basic data
            if basic_data['smiles']:
                basic_data['num_conformers'] = 0
                return basic_data
            else:
                raise e

    def parse(self) -> Tuple[Chem.Mol, dict]:
        """
        Parse the HOPV15 .data file and extract molecular information and properties.

        Uses the 2-step approach: first splits the file into molecule chunks,
        then parses the first molecule chunk.

        :returns: A tuple (mol, info) where mol is the primary Chem.Mol object and info contains
            all additional information including experimental data, conformers, and calculated properties.
        """
        with open(self.path, mode='r') as file:
            content = file.read()

        # Step 1: Split file into molecule chunks
        molecule_chunks = self._split_file_into_molecules(content)

        if not molecule_chunks:
            raise ValueError("No molecules found in the file")

        # Step 2: Parse the first molecule chunk
        molecule_data = self._parse_single_molecule(molecule_chunks[0])
        mol = self._create_molecule_from_data(molecule_data)

        return mol, molecule_data

    def _create_molecule_from_data(self, molecule_data: dict) -> Chem.Mol:
        """
        Create an RDKit molecule from parsed data and add 3D conformer if available.

        :param molecule_data: Dictionary containing parsed molecule data
        :returns: RDKit molecule object
        """
        # Create RDKit molecule from SMILES
        smiles = molecule_data['smiles']
        mol = Chem.MolFromSmiles(smiles)

        # Fallback to pruned SMILES if available and main SMILES fails
        if mol is None and 'pruned_smiles' in molecule_data:
            mol = Chem.MolFromSmiles(molecule_data['pruned_smiles'])

        if mol is not None and molecule_data.get('conformers'):
            # Add 3D conformer from first conformer if available
            first_conformer = molecule_data['conformers'][0]
            if 'coordinates' in first_conformer:
                mol_atoms = mol.GetNumAtoms()
                xyz_atoms = first_conformer['coordinates'].shape[0]

                if mol_atoms == xyz_atoms:
                    conf = Chem.Conformer(mol.GetNumAtoms())
                    for i, coord in enumerate(first_conformer['coordinates']):
                        conf.SetAtomPosition(i, coord)
                    mol.AddConformer(conf)
                else:
                    # Try to add hydrogens to match XYZ structure
                    mol_with_h = Chem.AddHs(mol)
                    if mol_with_h.GetNumAtoms() == xyz_atoms:
                        conf = Chem.Conformer(mol_with_h.GetNumAtoms())
                        for i, coord in enumerate(first_conformer['coordinates']):
                            conf.SetAtomPosition(i, coord)
                        mol_with_h.AddConformer(conf)
                        mol = mol_with_h

        return mol

    def parse_all(self) -> List[Tuple[Chem.Mol, dict]]:
        """
        Parse all molecules from a HOPV15 .data file that contains multiple molecules.

        Uses the 2-step approach: first splits the file into molecule chunks,
        then parses each chunk separately.

        :returns: List of tuples (mol, info) for each molecule in the file
        """
        with open(self.path, mode='r') as file:
            content = file.read()

        # Step 1: Split file into molecule chunks
        molecule_chunks = self._split_file_into_molecules(content)

        # Step 2: Parse each molecule chunk separately
        results = []
        for i, chunk in enumerate(molecule_chunks):
            try:
                molecule_data = self._parse_single_molecule(chunk)
                mol = self._create_molecule_from_data(molecule_data)
                if mol is not None:
                    results.append((mol, molecule_data))
            except Exception as e:
                # Log the error but continue with other molecules
                print(f"Warning: Failed to parse molecule {i+1}: {e}")
                continue

        return results

    @classmethod
    def get_fields(cls) -> List[str]:
        """
        Returns the list of string keys included in the info dict for HOPV15 files.

        :returns: List of field names available in the parsed info dictionary
        """
        return [
            'smiles', 'inchi', 'pruned_smiles', 'num_conformers',
            'experimental_properties', 'conformers'
        ]


# This dictionary maps the string keys to the corresponding parser classes. This way we can dynamically
# select the correct parser class based on the string key that is provided by the user.
XYZ_PARSER_MAP = {
    'default': DefaultXyzParser,
    'qm9': QM9XyzParser,
    'hopv15': HOPV15Parser,
}


def load_xyz_as_mol(file_path: str,
                    parser_cls: Union[str, type] = 'default',
                    ) -> Tuple[Chem.Mol, dict]:
    """
    Given the absolute string ``file_path`` to a .xyz file, this function will load the corresponding 
    molecule/atom constallation into an RDKit molecule object and return it. This means that all the atom 
    types and their positions are extracted from the xyz file.
    
    :param file_path: The absolute path to the .xyz file that should be loaded.
    
    :returns: A tuple (mol, info) where the first element is the RDKit Mol object that represents 
        the molecule from the xyz file and the second value is the additional info dict.
    """
    if isinstance(parser_cls, str): 
        try:
            parser_cls = XYZ_PARSER_MAP[parser_cls]
        except KeyError:
            raise KeyError(f'The given string "{parser_cls}" is not a valid identifier for an xyz parser class. '
                           f'Has to be one of the following: {", ".join(XYZ_PARSER_MAP.keys())}')
    
    # We will use the dynamically injected parser class to construct a new parser instance which we can 
    # then use to actually parse the molecule information from the file. Since all the parsers have to 
    # implement the AbstractXyzParser interface, we know that the first argument of the constructor is 
    # the file path to be parsed.
    parser = parser_cls(path=file_path)
    mol, info = parser.parse()
    
    return mol, info


# == TUDATASET ==

class TUDatasetParser:
    
    def __init__(self, 
                 path: str,
                 name: Optional[str] = None,
                 node_label_map: Dict[int, int] = {},
                 edge_label_map: Dict[int, int] = {},
                 graph_label_map: Dict[int, Any] = {},
                 **kwargs,
                 ) -> None:
        
        self.path: str = path
        # A dict that maps the node labels of the dataset to the actual atomic numbers in the periodic table.
        self.node_label_map: Dict[int, int] = node_label_map
        # A dict that maps the edge labels of the dataset to the actual bond type identifiers in the RDKit library.
        self.edge_label_map: Dict[int, int] = edge_label_map
        # A dict that maps the graph labels of the dataset to the actual target value vectors
        self.graph_label_map: Dict[int, int] = graph_label_map
        
        # We'll use this temporary directory to perform any intermediate file operations that are needed
        # during the parsing process. Alternatively if the given path is a ZIP file, we will extract the 
        # contents into this temporary directory and then parse the files from there.
        self.temp_dir: tempfile.TemporaryDirectory = tempfile.TemporaryDirectory()
        # This will be the absolute path of the temporary directory once it is initialized.
        self.temp_path: Optional[None] = None

        # If the name is not explicitly given, we will try to extract it from the given path and assume 
        # that the name of the path folder/file corresponds to the dataset name. Although ideally, the 
        # name would be given explicitly.
        if name is None:
            base_name: str = os.path.basename(self.path)
            if '.' in base_name:
                self.name = base_name.split('.')[0]
            else:
                self.name = base_name
                
        else:
            self.name = name
            
        # ~ derived properties
        
        # These are the internal variables which will contain the absolute paths to the individual 
        # files that make up a TU dataset.
        self.graph_indicator_path: Optional[str] = None
        self.graph_labels_path: Optional[str] = None
        self.node_labels_path: Optional[str] = None
        self.edge_labels_path: Optional[str] = None
        self.adjencency_path: Optional[str] = None
        
    def initialize(self) -> None:
        """
        This method should be called before the actual parsing of the dataset. It can be used to 
        initialize any additional information that is needed for the parsing process.
        """
        # Create the temporary directory
        self.temp_path = self.temp_dir.__enter__()
        
        # If the given path is a ZIP file, we will extract the contents into the temporary directory
        # and then parse the files from there.
        if self.path.endswith('.zip'):
            
            with zipfile.ZipFile(self.path, 'r') as zip_ref:
                
                zip_ref.extractall(self.temp_path)
                # We will assume that the ZIP file contains a folder with the same name as the dataset
                # and we will use this folder as the base path for the parsing process.
                self.path = os.path.join(self.temp_path, self.name)
                
        # At the end of this process, the self.path variables will have to contain the absolute path to 
        # a FOLDER that then contains the individual files that make up the dataset. 
        assert os.path.isdir(self.path), (
            'The given path does not point to a valid directory / is not a folder!'
        )
                
        # ~ discover the individual files that make up the dataset
        self.graph_indicator_path = os.path.join(self.path, f'{self.name}_graph_indicator.txt')
        self.graph_labels_path = os.path.join(self.path, f'{self.name}_graph_labels.txt')
        self.node_labels_path = os.path.join(self.path, f'{self.name}_node_labels.txt')
        self.edge_labels_path = os.path.join(self.path, f'{self.name}_edge_labels.txt')
        self.adjencency_path = os.path.join(self.path, f'{self.name}_A.txt')
        
        # After the loading of the dataset, this map will contain all the graph elements of the dataset
        # where the keys are the element indices and the values are the dictionaries containing all the 
        # relevant information about the individual graphs.
        self.index_graph_map: Dict[int, dict] = {}
    
    def finalize(self, exc: Optional[Any] = None, value: Optional[Any] = None, tb: Optional[Any] = None) -> None:
        
        # Destroy the temporary directory and all its contents.
        self.temp_dir.__exit__(exc, value, tb)

    def load(self) -> None:
        """
        This method will actually load the dataset from the individual files that make up the TUdataset folder.
        Each individual graph structure will be re-assembled from the information that is contained in the 
        files and the resulting graphs are stored in the "self.graphs" attribute for further processing.
        
        This function will most importantly NOT perform the node/edge label mapping or the SMILES conversion, 
        it will only load the information as it is contained in the files.
        """
        
        # first of all we need to load the "graph_indicator" file which contains the mapping of which 
        # nodes belong to which graph.
        with open(self.graph_indicator_path, mode='r') as file:
            # This is vector where the indices are the node indices and the values are the graph 
            # indices. We subtract 1 from the values since they annoyingly start counting at 1 
            batch_index: np.ndarray = np.array([int(line) - 1 for line in file.readlines()])
            num_graphs: int = batch_index.max() + 1

        # Then we load the node labels and the edge labels tensors.
        with open(self.node_labels_path, mode='r') as file:
            # This is a vector where the indices are the node indices and the values are the node labels 
            # aka categories which in this case represent different atom types and which will have 
            # to be mapped back to those later on.
            node_labels: np.ndarray = np.array([int(line) for line in file.readlines()], ndmin=1)
            # num_nodes: int = len(node_labels) + 1
        
        with open(self.edge_labels_path, mode='r') as file:
            # This is a vector where the indices are the edge indices and the values are the edge labels 
            # aka categories which in this case represent different bond types and which will have 
            # to be mapped back to those later on.
            edge_labels: np.ndarray = np.array([int(line) for line in file.readlines()], ndmin=1)
            
        # Now we load the structure information of the graphs which is given as a sparse adjacency matrix 
        # which is more like a list of edges. The file contains one edge per line and each line contains 
        # two node indices which are comma separated. The indices are 1-based and we need to convert them to
        # 0-based indices.
        with open(self.adjencency_path, mode='r') as file:
            edge_indices: List[Tuple[int, int]] = []
            for line in file.readlines():
                # Annoyingly they start indexing at 1 and not at 0, so we need to subtract 1 from the indices
                # to get the correct indices for the adjacency matrix.
                src, dst = tuple([int(value) - 1 for value in line.replace(' ', '').split(',')])
                edge_indices.append((src, dst))
                
            # This is the (m, 2) edge index list
            edge_indices: np.ndarray = np.array(edge_indices)
            
        # Finally we load the graph labels which is a file that contains one label per line - one line per graph 
        # These labels also have to be mapped to the actual class names later on.
        with open(self.graph_labels_path, mode='r') as file:
            # This is a vector where the indices are the graph indices and the values are the graph labels 
            # aka categories which in this case represent different graph types and which will have 
            # to be mapped back to those later on.
            graph_labels: np.ndarray = np.array([int(line) for line in file.readlines()], ndmin=1)

        # Then we can start constructing the actual graphs as dictionaries combining all the necessary 
        # information from the different files on a per-graph basis.
        for index in range(num_graphs):

            node_indices_graph = np.where(batch_index == index)
            node_labels_graph = node_labels[node_indices_graph]

            self.index_graph_map[index] = {
                'index': index,
                'node_indices': node_indices_graph,
                'node_labels': node_labels_graph,
                'edge_indices': [],
                'edge_labels': [],
                'graph_label': graph_labels[index],
            }
            
        for (src, dst), edge_label in zip(edge_indices, edge_labels):
            # We need to find the graph index for the source and destination node
            src_graph_index = batch_index[src]
            dst_graph_index = batch_index[dst]
            
            # We only add the edge if both nodes belong to the same graph
            if src_graph_index == dst_graph_index:
                self.index_graph_map[src_graph_index]['edge_indices'].append((src, dst))
                self.index_graph_map[src_graph_index]['edge_labels'].append(edge_label)
                
        # Convert the edge indices and labels to numpy arrays
        for index, graph in self.index_graph_map.items():
            graph['edge_indices'] = np.array(graph['edge_indices'])
            graph['edge_labels'] = np.array(graph['edge_labels'])
        
        assert len(self.index_graph_map) == num_graphs, (
            'The number of graphs that have been loaded does not match the number of graphs that '
            'have been found in the dataset!'
        )
        
    def mol_from_graph(self, graph: dict) -> Chem.Mol:
        """
        Given a ``graph`` structure from a TU dataset, this method will create the corresponding 
        RDKit molecule object from the information contained in that graph dictionary.
        
        :returns: The RDKit molecule object that corresponds to the given graph structure.
        """
        
        # We need to create the new RDKit molecule object based on the information contained within 
        # the graph dictionary.
        mol: Chem.Mol = Chem.RWMol()
    
        # First we add all the atoms to the molecule by iterating over the nodes contained in the 
        # graph dictionary information, mapping the node labels to the actual atom types and then 
        # adding those atoms to the molecule.
        for node_label in graph['node_labels']:
            atom_type: int = self.node_label_map[node_label]
            atom = Chem.Atom(atom_type)
            mol.AddAtom(atom)
            
        min_index = np.min(graph['node_indices'])
        # Then we need to add the edges corresponding to the edges contained in the graph.
        for e, (i, j) in enumerate(graph['edge_indices']):
            src, dst = int(i - min_index), int(j - min_index)
            bond_type: int = self.edge_label_map[graph['edge_labels'][e]]
            if not mol.GetBondBetweenAtoms(src, dst):
                mol.AddBond(src, dst, bond_type)

        mol.UpdatePropertyCache(strict=False)
        # Since the graph information does not contain any information about the charge state of the 
        # atoms, we need to manually fix certain functional groups that are based on charged atoms.
        # The most important example of this is the nitro group.
        mol = fix_nitro_groups(mol)
        
        mol.UpdatePropertyCache(strict=False)
        
        # Finally we need to convert the read-write molecule to a read-only molecule and return
        mol: Chem.Mol = mol.GetMol()
        
        return mol
            
        
    def generate_mol(self) -> Generator[Tuple[Chem.Mol, np.ndarray], None, None]:
        """
        Returns a generator that yields the individual RDKit molecule objects and the corrsponding 
        graph label numpy array for each graph in the dataset. This method will iterate over the 
        individual graphs contained in the dataset and create the corresponding RDKit molecule
        object from the information contained in the graph dictionary.
        
        :returns: A generator that yields tuples of the form (mol, graph_label) where mol is the
            RDKit molecule object that corresponds to the graph and graph_label is the numpy array 
            containing the graph label information.
        """
        for index, graph in self.index_graph_map.items():
            
            # This method will create the RDKit molecule object from the graph information contained 
            # in the graph dictionary. The 
            mol: Chem.Mol = self.mol_from_graph(graph)
            
            label = graph['graph_label']
            label = self.graph_label_map[label]
            
            yield mol, label
            
    # ~ object acts as an iterable
        
    def __iter__(self):
        return self.generate_mol() 


def load_tu_dataset() -> Tuple[dict, dict]:
    pass


