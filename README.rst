|made-with-python| |python-version| |ruff| 


.. |made-with-python| image:: https://img.shields.io/badge/Made%20with-Python-1f425f.svg
   :target: https://www.python.org/

.. |python-version| image:: https://img.shields.io/badge/python-3.8%20|%203.9%20|%203.10%20|%203.11%20|%203.12-blue
   :target: https://www.python.org/

.. |ruff| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
   :target: https://github.com/astral-sh/ruff

.. |pypi| image:: https://img.shields.io/pypi/v/chem_mat_data.svg
   :target: https://pypi.org/project/ruff/

=================
⚗️ ChemMatData
=================

.. image:: chem_mat_data/ChemMatData_logo_final.png
   :alt: ChemMatData Logo
   :align: center

The ``chem_mat_data`` package provides easy access to a large range of property prediction datasets from Chemistry and Material Science. 
The aim of this package is to provide the datasets in a unified format suitable to *machine learning* applications and specifically to train 
*graph neural networks (GNNs)*.

Specifically, ``chem_mat_data`` addresses these aims by providing simple, single-line command line (CLI) and programming (API) interfaces to download 
datasets either in *raw* or in *processed* (graph) format.

Features:

- 🐍 Easily installable via ``pip``
- 📦 Instant access to a collection of datasets across the domains of *chemistry* and *material science* 
- 🤖 Direct support of popular graph deep learning libraries like [Torch/PyG](https://pytorch-geometric.readthedocs.io/en/latest/) and [Jax/Jraph](https://jraph.readthedocs.io/en/latest/)
- 🤝 Large python version compatibility
- ⌨️ Comprehensive command line interface (CLI)
- 📖 Documentation: https://the16thpythonist.github.io/chem_mat_data 

Getting ready to train a PyTorch Geometric model can be as easy as this:

.. code-block:: python

    from chem_mat_data import load_graph_dataset, pyg_data_list_from_graphs
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    
    # Load the dataset of graphs
    graphs: list[dict] = load_graph_dataset('clintox')
    
    # Convert the graph dicts into PyG Data objects
    data_list: list[Data] = pyg_data_list_from_graphs(graphs)
    data_loader: DataLoader = DataLoader(data_list, batch_size=32, shuffle=True)
    
    # Network training...


📦 Pip Installation
===================

Install the latest stable release using ``pip`` from the Python Package Index (PyPI):

.. code-block:: console

    pip install chem_mat_database

Or install the latest development versin directly from the GitHub repository:

.. code-block::

    pip install git+https://github.com/the16thpythonist/chem_mat_data.git


⌨️ Command Line Interface (CLI)
===============================

The package provides the ``cmdata`` command line interface (CLI) to interact with the remote database.

To see the list of all available commands, simply use the ``--help`` flag:

.. code-block:: bash

    cmdata --help

Listing Available Datasets
--------------------------

To which datasets are available to be downloaded from the remote file share server, use the ``list`` command:

.. code-block:: bash

    cmdata list

This will print a table containing all the dataset which are currently available to download from the database. Each row of the 
table represents one dataset and contains the name of the dataset, the number of molecules in the dataset and the number of
target properties as additional columns.


Downloading Datasets
--------------------

Finally, to download this dataset, use the ``download`` command:

.. code-block:: bash

    cmdata donwload "clintox"

This will download the dataset ``clintox.csv`` dataset file to your current working directory.

One can also specify the path to wich the dataset should be downloaded as following:

.. code-block:: bash

    cmdata download --path="/tmp" "clintox"


🚀 Quickstart
=============

Alternatively, the ``chem_mat_data`` functionality can be used programmatically as part of python code. The 
package provides each dataset either in **raw** or **processed/graph** format (For further information on the 
distincation visit the [Documentation](https://the16thpythonist.github.io/chem_mat_data/api_datasets/)).

Raw Datasets
------------

You can use the ``load_smiles_dataset`` function to download the raw dataset format. This function will 
return the dataset as a ``pandas.DataFrame`` object which contains a "smiles" column along with the specific 
target value annotations as separate data frame columns.

.. code-block:: python

    import pandas as pd
    from chem_mat_data import load_smiles_dataset

    df: pd.DataFrame = load_smiles_dataset('clintox')
    print(df.head())


Graph Datasets
--------------

You can also use the ``load_graph_dataset`` function to download the same dataset in the *pre-processed* graph 
representation. This function will return a list of ``dict`` objects which contain the full graph representation 
of the corresponding molecules.

.. code-block:: python

    from rich.pretty import pprint
    from chem_mat_data import load_graph_dataset

    graphs: list[dict] = load_graph_dataset('clintox')
    example_graph = graphs[0]
    pprint(example_graph)


For further information on the graph representation, visit the [Documentation](https://the16thpythonist.github.io/chem_mat_data/graph_representation/).


Training Graph Neural Networks
------------------------------

Finally, the following code snippet demonstrates how to train a graph neural network (GNN) model using the
PyTorch Geometric library with the dataset loaded from the ``chem_mat_data`` package.

.. code-block:: python

    from torch import Tensor
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn.models import GIN
    from rich.pretty import pprint
    
    from chem_mat_data import load_graph_dataset, pyg_data_list_from_graphs
    
    # Load the dataset of graphs
    graphs: list[dict] = load_graph_dataset('clintox')
    example_graph = graphs[0]
    pprint(example_graph)
    
    # Convert the graph dicts into PyG Data objects
    data_list = pyg_data_list_from_graphs(graphs)
    data_loader = DataLoader(data_list, batch_size=32, shuffle=True)
    
    # Construct a GNN model
    model = GIN(
        in_channels=example_graph['node_attributes'].shape[1],
        out_channels=example_graph['graph_labels'].shape[0],
        hidden_channels=32,
        num_layers=3,  
    )
    
    # Perform model forward pass with a batch of graphs
    data: Data = next(iter(data_loader))
    out_pred: Tensor = model.forward(
        x=data.x, 
        edge_index=data.edge_index, 
        batch=data.batch
    )
    pprint(out_pred)


🤝 Credits
===========

We thank the following packages, institutions and individuals for their significant impact on this package.

* PyComex_ is a micro framework which simplifies the setup, processing and management of computational
  experiments. It is also used to auto-generate the command line interface that can be used to interact
  with these experiments.

.. _PyComex: https://github.com/the16thpythonist/pycomex.git
.. _Cookiecutter: https://github.com/cookiecutter/cookiecutter
