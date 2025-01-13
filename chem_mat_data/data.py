"""
This module implements the saving and loading of the datasets from and to the persistent 
file storage representations.
"""
import re
import ase.io
import rdkit.Chem as Chem
import msgpack
import numpy as np

from typing import List, Tuple, Union
from chem_mat_data._typing import GraphDict


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


# This dictionary maps the string keys to the corresponding parser classes. This way we can dynamically
# select the correct parser class based on the string key that is provided by the user.
XYZ_PARSER_MAP = {
    'default': DefaultXyzParser,
    'qm9': QM9XyzParser,
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