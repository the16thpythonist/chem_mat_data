"""
This module implements the saving and loading of the datasets from and to the persistent 
file storage representations.
"""
import ase.io
import rdkit.Chem as Chem
import msgpack
import numpy as np

from typing import List
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
    