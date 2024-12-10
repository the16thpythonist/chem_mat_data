"""
This module implements the saving and loading of the datasets from and to the persistent 
file storage representations.
"""
import re
import ase.io
import rdkit.Chem as Chem
import msgpack
import numpy as np

from typing import List, Tuple
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
    
    def parse(self) -> Chem.Mol:
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
        
        return mol

    
    
class QM9XyzParser(AbstractXyzParser):

    def __init__(self, path: str, **kwargs):
        AbstractXyzParser.__init__(self, path)
        
    def parse(self) -> Chem.Mol:
        
        with open(self.path, mode='r') as file:
            content = file.read()
        
        # ~ header information
        pattern_header = re.compile(
            r'^(?P<num_atoms>\d+)\n'
            r'+(?P<functional>[\w\d]*)\s?'
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
        
        # ~ atom information
        pattern_atoms = re.compile(
            r'^(?P<symbol>\w)\s+'
            r'(?P<x>-?[\d\.]*)\s+'
            r'(?P<y>-?[\d\.]*)\s+'
            r'(?P<z>-?[\d\.]*)\s+'
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
        
        return mol
        


def load_xyz_as_mol(file_path: str
                    ) -> Chem.Mol:
    """
    Given the absolute string ``file_path`` to a .xyz file, this function will load the corresponding 
    molecule/atom constallation into an RDKit molecule object and return it. This means that all the atom 
    types and their positions are extracted from the xyz file.
    
    :param file_path: The absolute path to the .xyz file that should be loaded.
    
    :returns: The RDKit Mol object that represents the molecule from the xyz file.
    """
    # first we initialize a read-write molecule which we can then populate with the atoms
    # loaded from the xyz file
    mol = Chem.RWMol()
    
    # The "read" function will parse the xyz file and return an Atoms object which itself 
    # is an iterable of Atom objects. We can then iterate over these atoms and add them
    # to the molecule.
    atoms: ase.atoms.Atoms = ase.io.read(file_path, format='xyz')
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
    
    return mol
    