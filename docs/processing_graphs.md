# Processing New Graphs

The ``chem_mat_data`` package provides a pre-processed **graph** format in which the individual 
molecules of a dataset can directly be loaded in a full graph representation with numeric node and 
edge feature vectors to specifically simplify the training of *graph neural network (GNN)* models.

Assuming one trains a GNN model on such a pre-processed dataset, one might also want to use such 
a model for *inference* to predict the target properties of *new* elements which weren't in the 
initial dataset. In such a case, the new elements will have to be processed into the same graph 
format to be compatible to be used as input to the trained model.

To process new graphs into the same graph format, one can use the package's ``MoleculeProcessing`` class. 
The ``process`` method of such a processing instance takes a molecule SMILES string representation as 
an input and returns the corresponding graph dict representation:

```python
from rich.pretty import pprint
from chem_mat_data.processing import MoleculeProcessing

processing = MoleculeProcessing()

smiles: str = 'C1=CC=CC=C1CCN'
graph: dict = processing.process(smiles)
pprint(graph)
```