# Custom Pre-Processing of Graphs

The ``chem_mat_data`` package provides dataset in an already pre-processed graph format. This pre-processed graph 
format makes some opinionated choices about *which* kinds of node and edge features are inlcuded to represent 
information about each individual atom or bond in the molecular graph.

If this pre-defined format for some reason isn't sufficient for a given dataset, there exists the possibility to 
define a custom ``Processing`` class to construct a pre-processing pipeline that converts the SMILES representations 
into graph structures.

## Creating a ``MoleculeProcessing`` Subclass

To define a custom pre-processing structure, one can define a custom subclass of the ``MoleculeProcessing`` subclass 
and define the desired node and edge features by modifying the ``node_attributes`` and ``edge_attributes`` class 
properties. Both properties have to be dictionary objects that define the node and edge features by providing 
a callback function or class that derives the desired property from the corresponding ``rdkit.Atom`` or ``rdkit.Bond`` 
objects.

In the following example we can define a customized processing class which only encodes a subset of atom types and 
only includes the mass of the atom as an additional feature. For the edge attributes we only encode the difference 
between single and double bonds.

```python
from chem_mat_data.processing import MoleculeProcessing
from chem_mat_data.processing import OneHotEncoder, chem_prop, list_identity

# Has to inherit from MoleculeProcessing!
class CustomProcessing(MoleculeProcessing):

    node_attribute_map = {

        'mass': {
            # "chem_prop" is a wrapper function which will call the given 
            # property method on the rdkit.Atom object - in this case the 
            # GetMass() method - and pass the output to the transformation 
            # function given as the second argument. "list_identity" means 
            # that the value is simply converted to a list as it is.
            # Therefore, this configuration will result in outputs such as 
            # [12.08], [9.88] etc. as parts of the overall feature vector.
            'callback': chem_prop('GetMass', list_identity),
            # Provide a human-readable description of what this section of 
            # the node feature vector represents.
            'description': 'The mass of the atom'
        },
        
        'symbol': {
            # "OneHotEncoder" is a special callable class that can be used 
            # to automatically define one-hot encodings. The object will 
            # accept the output of the given chem prop - in this case the 
            # GetSymbol action on the rdkit.Atom - and create an integer 
            # one-hot vector according to the provided list. In this case, 
            # the encoding will encode a carbon as [1, 0, 0, 0], 
            # a oxygen as [0, 1, 0, 0] etc.
            'callback': chem_prop('GetSymbol', OneHotEncoder(
                ['C', 'O', 'N', 'S'],
                add_unknown=False,
                dtype=str,
            )),
            'description': 'One hot encoding of the atom type',
            'is_type': True,
            'encodes_symbol': True,
        },
    }

    edge_attributes = {
        'type': {
            'callback': chem_prop()
        }
    }

```

## Applying the Custom Processing

After the custom processing class has been defined it can be used in the same manner as the orginal ``MoleculeProcessing`` 
class to convert the SMILES string representations of the dataset into the graph representation by using the ``process`` method.

```python
from rich.pretty import pprint

processing = CustomProcessing()

graph: dict = processing.process('CCCC')
pprint(graph)
```

## Defining Custom Transformation Callbacks

As introduced in the previous example, the ``chem_prop`` wrapper can be used to cast the output of an ``rdkit.Atom`` or 
``rdkit.Bond`` atom to some transformation callback which is then supposed to return a list that will become part 
of the final node/edge feature vector. 

The most simple usage is the ``list_identity`` transformation which will simply wrap the output value in a list as it is.
An alternative is to use the existing OneHotEncoder class to convert the output of a property getter method into an 
integer one-hot encoded vector.

Alternatively, it is also possible to define a completely custom callback to derive properties from the atom / bond 
objects directly. The callback functions simply have to accept a single positional argument ``entity: Atom | Bond``. 
callback function has to return a list of numeric values which will be appended to the overall feature vector.

```python
import rdkit.Chem as Chem
from rich.pretty import pprint
from typing import List
from chem_mat_data.processing import MoleculeProcessing

def custom_callback(atom: Chem.Atom) -> List[float]:
    
    # Mass multiplied with the charge
    return [atom.GetMass() * atom.GetCharge()]


class CustomCallbackProcessing(MoleculeProcessing):

    node_attributes = {
        'mass_times_charge': {
            'callback': custom_callback,
            'description': 'atom mass multiplied with the charge',
        }
    }


processing = CustomCallbackProcessing()
graph = processing.process('CCCC')
pprint(graph)
```



