# Loading Datasets

In ``chem_mat_data``, each dataset is provided in a **raw** and a **processed/graph** format.

The **raw** format resembles the format in which the dataset was originally published in and is 
usually *more compressed* and therefore more storage- and bandwidth-efficient to work with. For molecular 
datasets, this raw format usually simply consists of a list of SMILES strings that represents different 
molecules and their corresponding target value annotations. Since it is more data efficient, this 
format is recommended when working with machine learning methods that do not require the full graph 
representation, such as methods based on [molecular fingerprints](https://greglandrum.github.io/rdkit-blog/posts/2023-01-18-fingerprint-generator-tutorial.html).

The **processed/graph** format contains the already pre-processed full graph information for each molecule 
in the dataset. This format represents each molecule as a graph structure where all atoms are represented as 
graphs and all bonds as the corresponding edges. This format is recommended for machine learning 
methods based on *graph neural networks (GNNs)* since it removes the time-consuming graph pre-processing step.

## Loading Raw Datasets

### SMILES Dataset

For molecular property prediction tasks, the raw dataset format consists of a list of SMILES strings that 
represent the various molecules and their corresponding target value annotations.

A raw SMILES dataset can be loaded using the ``load_smiles_dataset`` function. This function will return 
a ``pandas.DataFrame`` object that contains a "smiles" column as well as various other columns that contain 
the target property annotations (whose names differe between the various datasets.)

```python
import pandas as pd
from chem_mat_data import load_smiles_dataset

df: pd.DataFrame = load_smiles_dataset('clintox')
print(df.head())
```

### XYZ Datasets

Alternative to SMILES representations, some datasets are stored in the form of [.XYZ files](https://en.wikipedia.org/wiki/XYZ_file_format). 
The main difference of an XYZ dataset w.r.t. to a SMILES dataset is that the SMILES dataset contains the *graph structure* in the form of the bond 
information and the XYZ dataset contains additional geometry information in the form of atom coordinates.

A raw XYZ dataset can be loaded using the ``load_xyz_dataset`` function. This function will return a 
``pandas.DataFrame`` object that most importantly contains the "mol" column as well as various other columns that 
contain the target property annotations or other additional values.

```python
import pandas as pd
from chem_mat_data import load_xyz_dataset

df: pd.DataFrame = load_xyz_dataset('qm9', parser_cls='qm9')
print(df.head())
```

!!! info "Different XYZ File Formats"

    There exist slightly different format specifications for .xyz files which vary in how the include which information. The 
    ``parser_cls`` parameter of the ``load_xyz_dataset`` function can be used to specify a distinct parser to be used to load 
    the dataset in question. To find out which parser to use, one can use the ``cmdata info`` command for the dataset in 
    question.

## Loading Processed Datasets

The processed/graph format of a dataset can be loaded with the ``load_graph_dataset`` function. This function 
will return a list of ``dict`` objects which contain various key value pairs that describe the full graph 
structure of the molecule. For more information on the structure of these graph representations visis the 
[Graph Representation](graph_representation.md) documentation.

```python
from rich.pretty import pprint
from chem_mat_data import load_graph_dataset

graphs: list[dict] = load_graph_dataset('clintox')
print(f'loaded {len(graphs)} graphs')
example_graph = graphs[0]
pprint(example_graph)
```
