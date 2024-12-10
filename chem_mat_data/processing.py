from typing import Any, Dict, List, Tuple, Callable, Optional, Union

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Descriptors
from rich.panel import Panel
from rich.table import Table
from rich.style import Style

import chem_mat_data._typing as tv
import chem_mat_data._typing as tc
from chem_mat_data.utils import mol_from_smiles
from chem_mat_data.utils import RichMixin
from chem_mat_data.visualization import create_frameless_figure
from chem_mat_data.visualization import visualize_molecular_graph_from_mol
from typing import List, Dict


def identity(value: Any) -> Any:
    """
    Simple implementation of the identity function. Returns the given ``value`` as it is.
    
    :param value: The value to be returned
    
    :returns: Any
    """
    return value


def list_identity(value: Any, dtype: type = float) -> Any:
    """
    Returns the given ``value`` as the singular element inside a list. Converts the given 
    value into the given ``dtype``.
    
    :param value: The value to be returned as the list element.
    :param dtype: The type to which the value should be converted. Default is float.
    
    :returns: A list containing the given value as the single element.
    """
    return [dtype(value)]


def chem_prop(property_name: str,
              callback: Callable[[Any], Any],
              ) -> Callable:
    """
    This function can be used to construct a callback function to encode a property of either a Atom or a
    Bond object belonging to an RDKit Mol. The returned function will query the ``property_name`` from the
    atom or bond object and apply the additional ``callback`` function to it and return the result.

    :param property_name: The string name of the method(!) of the atom or bond object to use to get the
        property value
    :param callback: An additional function that can be used to encode the extracted property value into
        the correct format of a list of floats.

    :returns: A function with the signature [Union[Chem.Atom, Chem.Bond]] -> List[float]
    """
    def func(element: Union[Chem.Atom, Chem.Bond], data: dict = {}):
        method = getattr(element, property_name)
        value = method()
        value = callback(value)
        return value
    
    # 11.06.23 - We are attaching the callback object itself here as a property of the decorated function here. 
    # We do that because in some advanced functionality it will actually be necessary to retrieve that object 
    # from the decorated function again. Specifically, when using a OneHotEncoder object as the callback 
    # we want to be able to access that original encoder object to also be able to make use of it's "decode" 
    # method.
    setattr(func, 'callback', callback)

    return func


def chem_descriptor(descriptor_func: Callable[[Chem.Mol], Any],
                    callback: Callable[[Any], Any],
                    ) -> Callable[[Chem.Mol], Any]:
    """
    Given a ``descripter_func`` callable which transforms a Mol object into some other ``callback`` 
    function, this function returns a callable which combines both of these functions. The callable 
    which is returned first uses the descripter_func on the Mol input and then applies the callback 
    on the result of that descriptor.
    
    :returns: A callable
    """
    def func(mol: Chem.Mol, data: dict = {}):
        value = descriptor_func(mol)
        value = callback(value)
        return value

    return func


class EncodingDescriptionMixin:
    """
    This is a mixin/interface which can be implemented in addition to the ``EncoderBase`` class.
    
    Implementing this mixing means to expose certain functionality which can be used to generate 
    human-readable descriptions of the encoded values. This is useful for the command line interface,
    for example, where the user can get a description of the encoded values.
    
    Specifically, two methods have to be implemented:

    ``get_descriptions(index: int) -> str`` should return the string description for the encoded 
    value at the given index position within the encoded vector.
    
    ``descriptions() -> List[str]`` should be a property which represents a list of string 
    descriptions where each string describes the corresponding encoded value.
    """
    
    def get_description(self, index: int = 0) -> str:
        raise NotImplementedError()
    
    def descriptions() -> List[str]:
        raise NotImplementedError()

class EncoderBase: 
    """
    This is the abstract base class for the implementation of attribute processing encoder 
    objects.

    Such encoder objects have the function of effectively encoding some kind of non-numeric 
    property into a vector of numeric values, which unlike the orignal format is suitable as 
    part of the input for a machine learning model.
    
    Any encoder implementation has to implement the following two methods:
    
    - ``encode``: This method should take the original value to be encoded as the argument and 
      then return a list of float values, which represents the encoded vector.
    - ``decode``: This method is the exact inverse functionality. It should take the list 
      of numeric values as the input and return the equivalent original value whatever that 
      may be.
      
    Each encoder object is callable by default through an implementation of the __call__ method, 
    which internally uses the implementation of the ``encode`` method.
    """
    
    def __call__(self, value: Any, *args, **kwargs) -> List[float]:
        return self.encode(value, *args, **kwargs)
    
    def encode(self, value: Any, *args, **kwargs) -> List[float]:
        raise NotImplementedError()
    
    def decode(self, encoded: List[float]) -> Any:
        raise NotImplementedError()
    
    
class StringEncoderMixin:
    """
    This is an interface which can optionally be implemented by an EncoderBase subclass to provide the 
    additional functionality of encoding and decoding string representations of the domain values.
    
    Subclasses need to implement the following methods:
    - ``encode_string``: Given the domain value to be encoded, this method will return a human-readable
        string representation of that value.
    - ``decode_string``: Given the string representation of a domain value, this method will return the
        original domain value.
    """
    def encode_string(self, value: Any) -> str:
        raise NotImplementedError()
    
    def decode_string(self, string: str) -> Any:
        raise NotImplementedError()


class OneHotEncoder(EncoderBase, StringEncoderMixin):
    """
    This is the specific implementation of an attribute Encoder class for the process of 
    OneHotEncoding elements of different types.
    
    The one-hot encoder is constructed by supplying a list of elements which should be encoded. 
    This may be any data type or structure, which implements the __eq__ method, such as strings 
    for example. The encoded vector representation of a single element will have as many elements 
    as the provided list of elements, where all values are zero except for the position of the 
    that matches the given element through an equality check.
    
    :param values: A list of elements which each will be checked when encoding an element
    :param add_unknown: Boolean flag which determines whether an additional one-hot encoding 
        element is added to the end of the list. This element will be used as the encoding for
        any element which is not part of the original list of values. If this flag is False 
        and an unkown element is otherwise encountered, the encoder will silently ignore it 
        and return a vector of zeros.
    :param unknown: The value which will be used as the encoding for any element which is not part
        of the original list of values. This parameter is only relevant if the add_unknown flag is
        set to True.
    :param dtype: a type callable that defines the type of the elements in the ``values`` list
    :param string_values: Optionally a list of string which provide human-readable string representations 
        for each of the elements in the ``values`` parameter. Therefore, this list needs to be the same 
        length as the ``values`` list. This parameter can optionally be None, in which case simply the 
        str() transformation of the elements in the ``values`` list will be used as the string 
        representations.
    :param use_soft_decode: If this flag is set to True, instead of matching the given encoded vector 
        exactly, the decoder will return the value which has the highest value in the encoded vector. 
        This is useful when the encoded vector is not exactly one-hot encoded, but rather a probability 
        distribution over the possible values.
    """
    def __init__(self,
                 values: List[Any],
                 add_unknown: bool = False,
                 unknown: Any = 'H',
                 dtype: type = float,
                 string_values: Optional[List[str]] = None,
                 use_soft_decode: bool = False,
                 value_descriptions: List[str] = [],
                 ):
        EncoderBase.__init__(self)
        StringEncoderMixin.__init__(self)
        
        self.values = values
        self.add_unknown = add_unknown
        self.unknown = unknown
        self.dtype = dtype
        self.use_soft_decode = use_soft_decode
        self.value_descriptions = value_descriptions
        
        # We want the "string_values" to always be a list of strings. If the parameter is None that means 
        # it is unnecessary to define a separate list and we can just use the "values" list as the string 
        # representation as well.
        if string_values is None:
            self.string_values: List[str] = [str(v) for v in values]
        else:
            self.string_values: List[str] = string_values

    def __call__(self, value: Any, *args, **kwargs) -> List[float]:
        return self.encode(value)
    
    # implement "EncoderBase"

    def encode(self, value: Any, *args, **kwargs) -> List[float]:
        """
        Given the domain ``value`` to be encoded, this method will return a list of float values that 
        represents the one-hot encoding corresponding to that exact value as defined by the list of possible 
        values given to the constructor.
        
        :param value: The domain value to be encoded. Must be part of the list of values given to the
            constructor - otherwise if add_unknown is True, the unknown one-hot encoding will be returned.
        
        :returns: list of float values which are either 1. or 0. (one-hot encoded)
        """
        one_hot = [1. if v == self.dtype(value) else 0. for v in self.values]
        if self.add_unknown:
            one_hot += [0. if 1 in one_hot else 1.]

        return one_hot
        
    def decode(self, encoded: List[float]) -> Any:
        """
        Given the one-hot encoded representation ``encoded`` of a domain value, this method will return the
        original domain value.
        
        Note that this method will try to do an exact match of the one-hot position. If the one-hot encoding
        is not exact, then the "unknown" value will be returned.
        
        :returns: The domain value which corresponds to the given one-hot encoding. This will have whatever 
            type the original domain values have.
        """
        if self.use_soft_decode:
            return self.decode_soft(encoded)
        else:
            return self.decode_hard(encoded)
    
    def decode_soft(self, encoded: List[float]) -> Any:
        """
        Given the one-hot encoded representation ``encoded`` of a domain value, this method will return the
        original domain value. This method is a soft decoding method, which means that it will return the
        value which has the highest value in the encoded vector. This is useful when the encoded vector is
        not exactly one-hot encoded, but rather a probability distribution over the possible values.
        
        :param encoded: The one-hot encoded representation of the domain value
        
        :returns: The domain value which corresponds to the given one-hot encoding. This will have whatever
            type the original domain values have.
        """
        max_index = np.argmax(encoded)
        if max_index < len(self.values):
            return self.values[max_index]
        else:
            return self.unknown
    
    def decode_hard(self, encoded: List[float]) -> Any:
        """
        Given the one-hot encoded representation ``encoded`` of a domain value, this method will return the
        original domain value. This method is a hard decoding method, which means that it will return the
        value which has the exact one-hot encoding as the given encoded vector.
        
        :param encoded: The one-hot encoded representation of the domain value
        
        :returns: The domain value which corresponds to the given one-hot encoding. This will have whatever
            type the original domain values have.
        """
        for one_hot, value in zip(encoded, self.values):
            if one_hot:
                return value
            
        # If the previous loop has failed to return anything then we can assume that the 
        # value is the unknown and we will instead return the "unkown" value provided in 
        # the constructor.
        return self.unknown
    
    @property
    def descriptions(self) -> Dict[str, str]:
        if self.value_descriptions:
            descriptions = self.value_descriptions
        else:
            descriptions = [str(v) for v in self.values]
            
        return [f'is {desc}?' for desc in descriptions]

    # implement "StringEncoderMixin"

    def encode_string(self, value: Any) -> str:
        """
        Given the domain ``value`` to be encoded, this method will return a human-readable string representation 
        of that value.
        
        :param value: The domain value to be encoded. Must be part of the list of values given to the
            constructor - otherwise if add_unknown is True, returns the string "unknown".
        
        :returns: A single string value which represents the given domain value
        """
        for v, s in zip(self.values, self.string_values):
            if v == self.dtype(value):
                return s
    
        return 'unknown'
    
    def decode_string(self, string: str) -> Any:
        """
        Given the string representation ``string`` of a domain value, this method will return the original domain 
        value.
        
        Note that this method will try to do an exact match of the string. If the string is not exact, then the 
        "unknown" value will be returned.
        
        :returns: The domain value which corresponds to the given string representation. This will have whatever 
            type the original domain values have.
        """
        for v, s in zip(self.values, self.string_values):
            if s == string:
                return v
        
        return self.unknown
    

class CrippenEncoder(EncoderBase, EncodingDescriptionMixin):
    
    # We have to set this attribute here to True to signal to the Processing class
    # that this encoder actually needs the molecule object to be passed to the encode 
    # method as well.
    requires_molecule: bool = True
    
    def encode(self, mol: Chem.Mol, atom: Chem.Atom) -> List[float]:
        # First of all we need to calculate the crippen contributions with an external method IF it does
        # not already exist!
        if not hasattr(mol, 'crippen_contributions'):
            contributions = rdMolDescriptors._CalcCrippenContribs(mol)
            setattr(mol, 'crippen_contributions', contributions)

        # At this point we can be certain, that every atom has been updated with the corresponding
        # attribute that contains it's crippen contribution values
        contributions = getattr(mol, 'crippen_contributions')
        return list(contributions[atom.GetIdx()])
    
    def get_description(self, index: int) -> str:
        return {
            0: 'crippen logP contribution',
            1: 'crippen MR contribution',
        }[index]

    @property
    def descriptions(self) -> List[str]:
        return [
            self.get_description(0),
            self.get_description(1),
        ]


class RichProcessingSummary(RichMixin):
    
    INDEX_STYLE = Style(bold=True, color='blue')
    
    def __init__(self, 
                 sections: Dict[str, Dict]
                 ):
        self.sections = sections
    
    def __rich_console__(self, console, options) -> Any:
        
        for name, descriptions in self.sections.items():
            
            table = Table('Index', 'Description', show_lines=False, show_header=False, box=None)
            
            # "descriptions" is a dictionary whose keys are the integer indices of the encoded 
            # vector elements nad the values are the corresponding descriptions.
            for index, description in descriptions.items():
                table.add_row(
                    # Segment(text=f'{index:02d}', style=self.INDEX_STYLE), 
                    f'{index:02d}',
                    description,
                )
            
            yield Panel(
                table,
                title=name,
                title_align='left',
            )


class MoleculeProcessing():
    """
    This class is used for the processing of the special "Molecule Graph" graph type. A molecule graph consists of 
    mutliple atoms (nodes) which are connected by different types of bonds (edges). The class provides methods to process 
    the domain string representation of molecule graphs (SMILES strings) into the graph dict representation. The class also 
    provides methods to create a new visual graph dataset element based on the given molecule graph.
    
    **Domain Representation**
    
    The domain specific representation of a molecule graph is called SMILES. It is a string representation of the
    molecular graph. SMILES is an established format for representing molecules and is used by many cheminformatics
    software packages. Characters in this string represent the different atom types while special characters such as 
    brackets and numbers represent the connections between the atoms. The SMILES string can be processed into a graph
    dict representation by using the ``process`` method.
    
    Examples:
    
    The string "C1=CC=C(N)C=C1" is a benzene ring to which a amine group is attached.
    
    **Graph Dict Representation**
    
    A domain-specific SMILES representation of a color graph can be processed into a graph dict representation 
    by using the ``process`` method. The graph dict representation is a dictionary which contains the full graph 
    data of the molecular graph.
    
    .. code-block:: python
    
        smiles = "C1=CC=C(N)C=C1"
        processing = MoleculeProcessing()
        graph = processing.process(smiles)
        print(graph)
    
    **Visualization**
    
    A molecular graph can be visualized using the ``visualize`` method (numpy array) or alternatively the 
    ``visualize_as_figure`` method (matplotlib figure). The visualization will result in an image with the given 
    width and height in pixels.
    
    .. code-block:: python
    
        smiles = "C1=CC=C(N)C=C1"
        processing = MoleculeProcessing()
        fig = processing.visualize_as_figure(smiles, width=1000, height=1000)
        plt.show()
        
    All the visualizations are created with the RDKit cheminformatics library - specificall the RDKit MolDraw2DSVG 
    functionality. The visualizations are created as SVG strings which are then converted into a numpy array or a
    matplotlib figure.
    
    **Customization**
    
    Generally, the most basic information that is encoded in a molecular graph is a one-hot encoding of the atom type 
    as part of the node attributes and an encoding of the bond type as part of the bond attributes. However, there are 
    many more possible bits of information that can be included for both the nodes and the edges. One might want to 
    customize the exact information that is encoded for a specific application. This can be easily done by subclassing
    the MoleculeProcessing class and overwriting the ``node_attribute_map`` and ``edge_attribute_map`` class variables.
    
    These class variables are dictionaries which map the names of the node and edge attributes to a dictionary which
    contains the callback function which is used to extract the attribute value from the molecule graph. The callback
    function is a function which takes the molecule graph and the atom or bond object as input and returns the value
    of the attribute. The callback function can be any function which returns a list of floats. The list of floats
    will be the attribute vector of the node or edge in the graph dict representation.    
    """

    # This is the descriptive string which will be used for the --help option if the command line interface 
    # for this class is invoked.
    description = (
        'This module exposes commands, which can be used to process domain-specific input data into valid '
        'elements of a visual graph dataset\n\n'
        'In this case, a SMILES string is used as the domain-specific representation of a molecular graph. '
        'Using the commands provided by this module, this smiles string can be converted into a JSON '
        'graph representation or be visualized as a PNG.\n\n'
        'Use the --help options for the various individual commands for more information.'
    )

    node_attribute_map = {
        'symbol': {
            'callback': chem_prop('GetSymbol', OneHotEncoder(
                ['C', 'O', 'N', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'Si', 'B', 'Na', 'Mg', 'Ca', 'Fe', 'Al', 'Cu', 'Zn', 'K'],
                add_unknown=True,
                dtype=str,
                value_descriptions=[
                    'Carbon (C)', 'Oxygen (O)', 'Nitrogen (N)', 'Sulfur (S)', 'Phosphorus (P)', 'Fluorine (F)',
                    'Chlorine (Cl)', 'Bromine (Br)', 'Iodine (I)', 'Silicon (Si)', 'Boron (B)', 'Sodium (Na)',
                    'Magnesium (Mg)', 'Calcium (Ca)', 'Iron (Fe)', 'Aluminium (Al)', 'Copper (Cu)', 'Zinc (Zn)',
                    'Potassium (K)',
                ]
            )),
            'description': 'One hot encoding of the atom type',
            'is_type': True,
            'encodes_symbol': True,
        },
        'hybridization': {
            'callback': chem_prop('GetHybridization', OneHotEncoder(
                [1, 2, 3, 4, 5, 6],
                add_unknown=True,
                dtype=int,
                value_descriptions=[
                    'S hybrid', 'SP hybrid', 'SP2 hybrid', 
                    'SP3 hybrid', 'SP2D hybrid', 'SP3D hybrid',
                ]
                
            )),
            'description': 'one-hot encoding of atom hybridization',
        },
        'total_degree': {
            'callback': chem_prop('GetTotalDegree', OneHotEncoder(
                [0, 1, 2, 3, 4, 5],
                add_unknown=False,
                dtype=int,
                value_descriptions=[
                    '0 Neighbors', '1 Neighbor', '2 Neighbors', '3 Neighbors', '4 Neighbors', '5 Neighbors'
                ]
            )),
            'description': 'one-hot encoding of the degree of the atom'
        },
        'num_hydrogen_atoms': {
            'callback': chem_prop('GetTotalNumHs', OneHotEncoder(
                [0, 1, 2, 3, 4],
                add_unknown=False,
                dtype=int,
                value_descriptions=[
                    '0 Hydrogens', '1 Hydrogens', '2 Hydrogens', '3 Hydrogens', '4 Hydrogens',
                ]
            )),
            'description': 'one-hot encoding of the total number of attached hydrogen atoms'
        },
        'mass': {
            'callback': chem_prop('GetMass', list_identity),
            'description': 'The mass of the atom'
        },
        'charge': {
            'callback': chem_prop('GetFormalCharge', list_identity),
            'description': 'electrical charge of the atom',
        },
        'is_aromatic': {
            'callback': chem_prop('GetIsAromatic', list_identity),
            'description': 'is atom aromatic?',
        },
        'is_in_ring': {
            'callback': chem_prop('IsInRing', list_identity),
            'description': 'is atom in ring?'
        },
        'crippen_contributions': {
            'callback': CrippenEncoder(),
            'description': 'The crippen logP contributions of the atom as computed by RDKit'
        },
    }

    edge_attribute_map = {
        'bond_type': {
            'callback': chem_prop('GetBondType', OneHotEncoder(
                [1, 2, 3, 12, 13, 14],
                add_unknown=True,
                dtype=int,
                value_descriptions=[
                    'Single Bond', 'Double Bond', 'Triple Bond', 
                    'Aromatic Bond', 'Ionic Bond', 'Hydrogen Bond',
                ],
                string_values=['S', 'D', 'T', 'A', 'I', 'H'],
            )),
            'description': 'One hot encoding of the bond type',
            'is_type': True,
            'encodes_bond': True 
        },
        'stereo': {
            'callback': chem_prop('GetStereo', OneHotEncoder(
                [0, 1, 2, 3],
                add_unknown=False,
                dtype=int,
                value_descriptions=[
                    'Stereo None', 'Stereo Any', 'Stereo Z', 'Stereo E'
                ]
            )),
            'description': 'one-hot encoding of the stereo property',
        },
        'is_aromatic': {
            'callback': chem_prop('GetIsAromatic', list_identity),
            'description': 'is aromatic?',
        },
        'is_in_ring': {
            'callback': chem_prop('IsInRing', list_identity),
            'description': 'is part of ring?',
        },
        'is_conjugated': {
            'callback': chem_prop('GetIsConjugated', list_identity),
            'description': 'is conjungated?'
        }
    }

    graph_attribute_map = {
        'molecular_weight': {
            'callback': chem_descriptor(Descriptors.ExactMolWt, list_identity),
            'description': 'The molecular weight of the entire molecule'
        }
    }

    # These are simply utility variables. These object will be needed to query the various callbacks which
    # are defined in the dictionaries above. They obviously won't result in a real value, but they are only
    # needed to get *any* result from the callback, because for the purpose of constructing the description
    # map we only need to know the length of the lists which are returned by them, not the content.
    MOCK_MOLECULE = mol_from_smiles('CC')
    MOCK_ATOM = MOCK_MOLECULE.GetAtoms()[0]
    MOCK_BOND = MOCK_MOLECULE.GetBonds()[0]
    
    def __init__(self, *args, ignore_issues: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.ignore_issues = ignore_issues
        
        # This will be an array of node attribute vector indices of all those elements in 
        # the node attribute vector which have been annotated with the special "is_type" flag. 
        # This flag determines that these elements are relevant to match node types.
        self.node_type_indices = np.array(self.get_attribute_indices(
            self.node_attribute_map,
            self.MOCK_ATOM,
            lambda data: 'is_type' in data and data['is_type']
        ), dtype=int)
        
        # This will be an array of edge attribute vector indices of all those elements in the 
        # edge attribute vector which have been annotated with the special "is_type" flag. 
        # This flag determines that these elements are relevant to match edge types.
        self.edge_type_indices = np.array(self.get_attribute_indices(
            self.edge_attribute_map,
            self.MOCK_BOND,
            lambda data: 'is_type' in data and data['is_type']
        ), dtype=int)
        
        # Here we search for the entry in the node attribute map which implements the "encodes_symbol" 
        # flag. This signals it being the Encoder which is responsible for the main atom symbol type 
        # encoding. 
        # Ultimately we want to extract that very Encoder object for future use where we want to 
        # use it for decoding numeric vectors back into symbols as well.
        data = self.get_attribute_data(
            self.node_attribute_map,
            lambda data: 'encodes_symbol' in data and data['encodes_symbol']
        )
        try:
            self.symbol_encoder: Optional[Any] = data['callback'].callback
        except TypeError:
            if not self.ignore_issues:
                raise AssertionError('None of elements defined in node_attribute_map implement the flag'
                                     '"encodes_symbol". Please make sure to add the flag to identify the '
                                     'the main symbol Encoder which will be required for the processing functions.')

        # This is an array which contains all the node attribute vector indices which can be used 
        # to extract the sub vector responsible for encoding the symbol.
        self.symbol_indices = np.array(self.get_attribute_indices(
            self.node_attribute_map,
            self.MOCK_ATOM,
            lambda data: 'encodes_symbol' in data and data['encodes_symbol']
        ), dtype=int)
        
        # 29.01.24
        self.charge_indices: List[int] = np.array(self.get_attribute_indices(
            self.node_attribute_map,
            self.MOCK_ATOM,
            lambda data: 'encodes_charge' in data and data['encodes_charge'],
        ))
        
        data = self.get_attribute_data(
            self.edge_attribute_map,
            lambda data: 'encodes_bond' in data and data['encodes_bond'],
        )
        try:
            self.bond_encoder: Optional[Any] = data['callback'].callback
        except TypeError:
            if not self.ignore_issues:
                raise AssertionError('None of the elements defined in edge_attribute_map implement the flag '
                                     '"encodes_bond". Please make sure to add the flag to identify the '
                                     'the main bond type Encoder which will be required for the processing functions')
                
        self.bond_indices = np.array(self.get_attribute_indices(
            self.edge_attribute_map,
            self.MOCK_BOND,
            lambda data: 'encodes_bond' in data and data['encodes_bond'],
        ))
        
        
    def get_attribute_data(self,
                           attribute_map: Dict[str, dict],
                           condition: Callable,
                           ) -> dict:
        for name, data in attribute_map.items():
            if condition(data):
                return data
        
    def get_attribute_indices(self,
                              attribute_map: Dict[str, dict],
                              element: Any,
                              condition: Callable
                              ) -> List[int]:
        """
        Given given an ``attribute_map`` which describes the attribute extraction from a molecule, an 
        atom object ``element`` and a callable ``condition``, this method returns a list which contains 
        the integer indices of a node attributes vector / elements of the node attribute map for which 
        the given condition is true.
        
        If the given condition cannot be found at all, this method will return an empty list.
        
        :param attribute_map: The node attribute map which to search.
        :param element: An Atom or Bond ojbect for which the search be executed. Usually MOCK_ATOM
        :param condition: A callable function which receives the ``data`` dict element of the attribute map 
            as a parameter and is supposed to return a boolean value that determines whether that dict 
            fullfills the desired condition.
        
        :returns: A list of integer indices
        """
        indices = []
        index = 0
        for name, data in attribute_map.items():
            callback = data['callback']
            value: list = self.apply_callback(callback, self.MOCK_MOLECULE, element)
            for _ in value:
                if condition(data):
                    indices.append(index)
                index += 1
                
        return indices
    
    def node_match(self, node_attributes_1, node_attributes_2):
        return np.isclose(
            node_attributes_1[self.node_type_indices], 
            node_attributes_2[self.node_type_indices],
        ).all()

    def edge_match(self, edge_attributes_1, edge_attributes_2):
        return np.isclose(
            edge_attributes_1[self.edge_type_indices],
            edge_attributes_2[self.edge_type_indices],
        ).all()
    
    def extract(self,
                graph: tv.GraphDict,
                mask: np.ndarray,
                clear_aromaticity: bool = True,
                process_kwargs: Dict[str, Any] = {},
                unprocess_kwargs: Dict[str, Any] = {},
                ) -> Tuple[str, tc.GraphDict]:
        
        return super().extract(
            graph=graph,
            mask=mask,
            process_kwargs={
                **process_kwargs,
            },
            unprocess_kwargs={
                'clear_aromaticity': clear_aromaticity,
                **unprocess_kwargs,
            }
        )
    
    def unprocess(self,
                  graph: tv.GraphDict,
                  clear_aromaticity: bool = False,
                  **kwargs
                  ) -> str:
        """
        Given the ``graph`` dict representation of a molecular graph, this method will transform that graph 
        back into it's domain representation which in this case is a SMILES string.
        
        the aromaticity problem for fragments
        -------------------------------------
        
        This method should also work for molecular fragements, so graph dicts which don't necessarily describe 
        a complete molecule but rather only parts of it that were extracted from other larger molecules. This 
        can cause a problem when atoms were extracted from an aromatic ring but now in the extracted form they 
        are no longer part of a valid aromatic ring. These kinds of smiles cannot be turned back into a Mol 
        object successfully.
        In such cases the ``clear_aromaticity`` flag of this method has to be used, which will erase the 
        aromatic flags such that the resulting SMILES is still somewhat valid, although the molecule which 
        can then be reconstructed from that smiles is not in itself valid!
        
        :param graph: The graph dict to be converted to smiles
        :param clear_aromaticity: If set, this will forcefully clear the aromaticity flags of the molecule  
            which will result in a valid molecular representation as far as RDKit is concerned but not  
            in the chemical sense.
        
        :returns: The SMILES string corresponding to the graph dict
        """
        
        # For molecular graphs the domain representation is SMILES strings. For SMILES strings we need 
        # to rely on the conversion functionality implemented in RDKit and that in turn can perform the 
        # conversion when provided with a Mol object. So what we have to do here is to iteratively 
        # construct such a Mol object from the given graph dict.
        
        mol = Chem.RWMol()
        for node_index in graph['node_indices']:
            node_attributes = graph['node_attributes'][node_index]
            symbol = self.symbol_encoder.decode(node_attributes[self.symbol_indices])
            atom = Chem.AtomFromSmarts(symbol)
            
            if self.charge_indices:
                charge = int(node_attributes[self.charge_indices][0])
                atom.SetFormalCharge(charge)
            
            mol.AddAtom(atom)
        
        for edge_index, (i, j) in enumerate(graph['edge_indices']):
            i, j = int(i), int(j)
            edge_attributes = graph['edge_attributes'][edge_index]
            bond_type = self.bond_encoder.decode(edge_attributes[self.bond_indices])
            # The decoder here only returns the integer representation of the bond type, but the signature 
            # of the AddBond method REALLY wants that to be wrapped as a BondType object...
            bond_type = Chem.BondType(bond_type)
            if not mol.GetBondBetweenAtoms(i, j):
                mol.AddBond(i, j, bond_type)
            
        # If we only extract a sub graph, aromatic bond types will probably cause a problem since at that 
        # point they are no longer part of a valid ring. In that case we need to manually set all the aromatic 
        # flags to false here to fix that.
        # Note: This does not result in a canonical SMILES then in the end!
        if clear_aromaticity:
            for atom in mol.GetAtoms():
                atom.SetIsAromatic(False)

        # The previous RWMol object is a special kind of *mutable* mol object and here we need to 
        # convert that into the regular mol object.
        mol = mol.GetMol()
        return Chem.MolToSmiles(mol)
    
    def apply_callback(self, 
                       callback: Callable, 
                       mol: Chem.Mol, 
                       element: Any
                       ) -> List[float]:
        """
        Given a ``callback`` function, the base ``mol`` molecule object and the ``element`` - an atom or a bond 
        object - on which to apply that callback, this method will apply the callback in the correct manner and 
        return the return value of that callback. 

        The seemingly unnecessary wrapping of this functionality in it's own method here is necessary because the 
        application of the callback is not necessary due to some special rules whic should not be scattered 
        throughout the rest of the code in this class.

        :param callback: _description_
        :type callback: t.Callable
        :param mol: _description_
        :type mol: Chem.Mol
        :param element: _description_
        :type element: t.Any
        
        :return: _description_
        :rtype: t.List[float]
        """
        # Most callbacks are rather simple and only operate on the basis of the element (atom / bond) itself 
        # but there are other - more advanced - callbacks which need the context of the whole molecule for 
        # some more fancy computations. These can be detected by an additional property that was attached to 
        # them. In those cases the signature of the callback is different which is why we need to check this.
        # I know that this is essentially bad design, but the necessity crept up only later and at that 
        # point I needed to maintain back-comp and couldn't generally change the signature.
        if hasattr(callback, 'requires_molecule') and getattr(callback, 'requires_molecule'):
            value = callback(mol, element)
        else:
            value = callback(element)
        
        return value

    def process(self,
                value: Union[str, Chem.Mol],
                double_edges_undirected: bool = False,
                use_node_coordinates: bool = False,
                graph_labels: list = [],
                ) -> dict:
        """
        Converts SMILES string into graph representation.

        This command processes the SMILES string and creates the full graph representation of the
        corresponding molecular graph. The node and edge feature vectors will be created in the exact
        same manner as the rest of the dataset, making the resulting representation compatible with
        any model trained on the original VGD.

        This command outputs the JSON representation of the graph dictionary representation to the console.

        :param value: The SMILES string of the molecule to be converted
        :param double_edges_undirected: A boolean flag of whether to represent an edge as two undirected
            edges
        :param use_node_coordinates: A boolean flag of whether to include 3D node_coordinates into the
            graph representation. These would be created by using an RDKit conformer. However, this
            process can fail.
        :param graph_labels: A list containing the various ground truth labels to be additionally associated
            with the element
        """
        # 01.06.23 - When working with the counterfactuals I have noticed that there is a problem where
        # it is not easily possible to maintain the original molecules atom indices through a SMILES
        # conversion, which seriously messes with the localization of which parts were modified.
        # So as a solution we introduce here the option to generate the graph based on a Mol object
        # directly instead of having to use the SMILES every time. This could potentially be slightly
        # more efficient as well.
        if isinstance(value, Chem.Mol):
            mol = value
            smiles = Chem.MolToSmiles(mol)
        else:
            smiles = value
            mol = Chem.MolFromSmiles(smiles)

        atoms = mol.GetAtoms()
        # First of all we iterate over all the atoms in the molecule and apply all the callback
        # functions on the atom objects which then calculate the actual attribute values for the final
        # node attribute vector.
        node_indices: list[int] = []
        node_attributes: list[list[float]] = []
        # Here we want to maintain a list of the the string atom symbols for each of the atom nodes as 
        # we process the molecule graph structure.
        # The goal is to have some kind of information such that a human could reconstruct 
        # the molecule from the graph structure later on as well.
        node_atoms: list[str] = []
        for atom in atoms:
            node_indices.append(atom.GetIdx())

            attributes = []
            # "node_attribute_callbacks" is a dictionary which specifies all the transformation functions
            # that are to be applied on each atom to calculate part of the node feature vector
            for name, data in self.node_attribute_map.items():
                callback: Callable[[Chem.Mol, Chem.Atom], list] = data['callback']
                value: list = self.apply_callback(callback, mol, atom)

                attributes += value

            node_attributes.append(attributes)

            # "symbol_encoder" is an Encoder specifically designed to encode the atom symbols into a 
            # one-hot encoded vector. It has the additional method "encode_string" which encodes the 
            # symbol into a human readable string.
            if self.symbol_encoder:
                atom_symbol = self.symbol_encoder.encode_string(atom.GetSymbol())
                node_atoms.append(atom_symbol)

        bonds = mol.GetBonds()
        # Next up is the same with the bonds
        edge_indices: list[tuple[int, int]] = []
        edge_attributes: list[list[float]] = []
        # Here we want to maintain a list of the bond types for each of the edge nodes as we process the
        # molecule graph structure. More specifically, we want to safe a human readable string representation 
        # of the bond type. 
        # The goal is to have some kind of information such that a human could reconstruct 
        # the molecule from the graph structure later on as well.
        edge_bonds: list[str] = []
        for bond in bonds:
            i = int(bond.GetBeginAtomIdx())
            j = int(bond.GetEndAtomIdx())

            edge_indices.append([i, j])
            if double_edges_undirected:
                edge_indices.append([j, i])

            attributes = []
            for name, data in self.edge_attribute_map.items():
                callback: Callable[[Chem.Bond], list] = data['callback']
                value: list = self.apply_callback(callback, mol, bond)
                attributes += value

            # We have to be careful here to really insert the attributes as often as there are index
            # tuples.
            edge_attributes.append(attributes)
            if double_edges_undirected:
                edge_attributes.append(attributes)
                
            if self.bond_encoder:
                bond_type: str = self.bond_encoder.encode_string(bond.GetBondType())
                edge_bonds.append(bond_type)
                if double_edges_undirected:
                    edge_bonds.append(bond_type)

        # Then there is also the option to add global graph attributes. The callbacks for this kind of
        # attribute take the entire molecule object as an argument rather than just atom or bond
        graph_attributes = []
        for name, data in self.graph_attribute_map.items():
            callback: Callable[[Chem.Mol], list] = data['callback']
            value: list = callback(mol)
            graph_attributes += value

        # Now we can construct the preliminary graph dict representation. All of these properties form the
        # core of the graph dict - they always have to be present. All the following code after that is
        # optional additions which can be added to support certain additional features
        graph: tc.GraphDict = {
            'node_indices':         np.array(node_indices, dtype=int),
            'node_attributes':      np.array(node_attributes, dtype=float),
            'edge_indices':         np.array(edge_indices, dtype=int),
            'edge_attributes':      np.array(edge_attributes, dtype=float),
            'graph_attributes':     np.array(graph_attributes, dtype=float),
            'graph_labels':         graph_labels,
            # 14.02.24 - Initially I wanted to avoid addding the string graph representation to the graph 
            # dictionary itself, because I kind of wanted all of the values to be numpy array. However, I 
            # have now hit a problem where this is pretty much necessary.
            'graph_repr':           smiles,
        }
        
        # 28.10.24 - Here we save some domain-specific additional information on the node and edge level.
        # More specifically these are lists of human readable strings which contain the atom symbols and
        # bond types respectively. The goal of this additional information is to provide a way for a human 
        # to easily reconstruct the molecule from the graph representation as well.
        graph['node_atoms'] = np.array(node_atoms, dtype=str)
        graph['edge_bonds'] = np.array(edge_bonds, dtype=str)

        # Optionally, if the flag is set, this will apply a conformer on the molecule which will
        # then calculate the 3D coordinates of each atom in space.
        if use_node_coordinates and mol.GetNumConformers() == 0:
            try:
                # # https://sourceforge.net/p/rdkit/mailman/message/33386856/
                # try:
                #     rdkit.Chem.AllChem.EmbedMolecule(mol)
                #     rdkit.Chem.AllChem.MMFFOptimizeMolecule(mol)
                # except:
                #     rdkit.Chem.AllChem.EmbedMolecule(mol, useRandomCoords=True)
                #     rdkit.Chem.AllChem.UFFOptimizeMolecule(mol)
                    
                # 05.02.2024
                # Previously we tried to do the MMFF optimization here, but now we simply commit to the UFF 
                # optimization which is a bit faster and should be sufficient for our purposes.
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol, AllChem.ETKDG())
                AllChem.UFFOptimizeMolecule(mol)
                mol = Chem.RemoveHs(mol)

            except Exception:
                raise ValueError(f'Cannot calculate node_coordinates for the given '
                                 f'molecule with smiles code: {smiles}')

        # 21.10.24 - with the processing of Mol objects that were directly loaded from xyz files, there is 
        # now actually the possibility that they already have a pre-defined set of node coordinates when 
        # they enter the processing method. So in that case we detect this here and properly add those 
        # as the "node_coordinates" property of the graph dict.
        if mol.GetNumConformers() > 0:
            
            conformer = mol.GetConformers()[0]
            node_coordinates = np.array(conformer.GetPositions())
            graph['node_coordinates'] = node_coordinates

            # Now that we have the 3D coordinates for every atom, we can also calculate the length of all
            # the bonds from that!
            edge_lengths = []
            for i, j in graph['edge_indices']:
                c_i = graph['node_coordinates'][i]
                c_j = graph['node_coordinates'][j]
                length = la.norm(c_i - c_j)
                edge_lengths.append([length])

            graph['edge_lengths'] = np.array(edge_lengths)

        return graph

    def visualize_as_figure(self,
                            value: str,
                            width: int,
                            height: int,
                            additional_returns: Optional[dict] = None,
                            **kwargs,
                            ) -> Tuple[plt.Figure, np.ndarray]:
        """
        The normal "visualize" method has to return a numpy array representation of the image. While that
        is a decent choice for printing it to the console, it is not a good choice for direct API usage.
        This method will instead return the visualization as a matplotlib Figure object.

        This method should preferably used when directly interacting with the processing functionality
        with code.
        
        :param value: The domain specific string representation of the graph
        :param width: The width of the resulting visualization image in pixels.
        :param height: The height of the resulting visualization image in pixels.
        :param additional_returns: Optional. A dict object can be passed to this method to collect additional 
            returns of this method. In this specific case, this dict will contain the following keys:
            "svg_string" - the raw string representation of the original svg string that created the molecule 
            visualization
            
        :returns: a tuple where the first value is a plt.Figure object containing the visualization of the 
            molecular graph and the second element is a numpy array of shape (V, 2) where V is the number of 
            nodes in the graph and which assigns the 2D pixel coordinates of each of the nodes.
        """
        smiles = value
        mol = mol_from_smiles(smiles)
        fig, ax = create_frameless_figure(width=width, height=height)
        node_positions, svg_string = visualize_molecular_graph_from_mol(
            ax=ax,
            mol=mol,
            image_width=width,
            image_height=height
        )
        
        # The "node_positions" which are returned by the above function are values within the axes object
        # coordinate system. Using the following piece of code we transform these into the actual pixel
        # coordinates of the figure image.
        node_positions = [[int(v) for v in ax.transData.transform((x, y))]
                          for x, y in node_positions]
        node_positions = np.array(node_positions)

        # 24.10.23 - made the additional returns optional and checking for the None state here because that 
        # is better practice than having a mutable dictionary instance as a default value of a method argument.
        if additional_returns is not None:
            additional_returns['svg_string'] = svg_string

        # 24.10.23 - Desperate attempt to somehow fix the performance degradation of this method.
        del ax, mol

        return fig, node_positions

    # -- utils --
    
    def get_attribute_map_descriptions(self, attribute_map: Dict) -> Dict[int, str]:
        
        descriptions: dict[int, str] = {}
        index: int = 0
        for name, info in attribute_map.items():
            
            # The actual callback can either be just a function (such as a simple lambda) or it can 
            # be a more complex subclass of the EncoderBase class. In the latter case, there might 
            # be more sophisticated descriptions embedded in the object itself which we want to 
            # access.
            
            # We can access the callback object itself here as a property of the decorated function here.
            # If that object implements the EncodingDescriptionMixin interface, then we can extract the 
            # descriptions from that object.
            callback = info['callback']
            if hasattr(callback, 'callback'):
                
                obj = callback.callback
                if isinstance(obj, EncodingDescriptionMixin):
                    for description in obj.descriptions:
                        descriptions[index] = description
                        index += 1
                        
                    continue
                
            # Otherwise we can assume that the callback is a simple function which only maps a singular 
            # chemical property. which we'll be able to capture with just the overall description of 
            # the attribute.
            description = info['description']
            descriptions[index] = description
            index += 1
            
        return descriptions
            
    def summary(self, echo: bool = True) -> str:
        
        # "get_attribute_map_descriptions" is a method which will return a dictionary which maps the
        # indices of the node and edge attributes values to their string and human-readable descriptions.
        # We do this for both nodes and edges.
        node_descriptions: dict[int, str] = self.get_attribute_map_descriptions(self.node_attribute_map)
        edge_descriptions: dict[int, str] = self.get_attribute_map_descriptions(self.edge_attribute_map)
        
        # Using that information we can then assemble the rich display object which we can then use to 
        # display that information in the console.
        
        summary = RichProcessingSummary({
            'Node Attributes': node_descriptions,
            'Edge Attributes': edge_descriptions,
        })
        if echo:
            print(summary)
        
        return summary

    def save_svg(self, content: str, path: str):
        with open(path, mode='w') as file:
            file.write(content)
