|made-with-python| |python-version|

.. |made-with-python| image:: https://img.shields.io/badge/Made%20with-Python-1f425f.svg
   :target: https://www.python.org/

.. |python-version| image:: https://img.shields.io/badge/Python-3.8.0-green.svg
   :target: https://www.python.org/

=================
⚗️ ChemMatData
=================

.. image:: chem_mat_data/ChemMatData_logo_final.png
   :alt: ChemMatData Logo
   :align: center

ChemMatData is a database consisting of a collection of datasets from the fields of chemistry and material science. 
Each dataset contains various molecules and/or crystal structures which have been annotated with specific target properties. 
The main purpose of these datasets is to be used for the training of machine learning models with a special focus on - but not exclusive to -
the training of graph neural networks (GNNs).

The primary goal of this package is to provide a simple and convenient *command line* as well as *programming* interface 
to access these datasets. Each dataset is available in two formats: The raw format consists of a CSV file containing the 
SMILES string representation of the molecules and the target annotations. In the processed format, each molecule is already 
represented as a full graph structure and ready for GNN training.

Getting ready to train for PyTorch Geometric (PyG) is as easy as this:

.. code-block:: python

   from chem_mat_data import load_graph_dataset, pyg_data_loader_from_graphs

   graphs = load_graph_dataset('clintox')
   data_list = pyg_data_list_from_graphs(graphs)

   # train network...

=========================
📦 Installation by Source
=========================

First, clone the repository:

.. code-block:: console

    git clone {your github url}/chem_mat_data.git

Install using ``pip``:

.. code-block:: console

    cd chem_mat_data
    python3 -m pip install .

**(Optional)** Afterwards, you can check the installation by invoking the CLI:

.. code-block:: console

    python3 -m chem_mat_data.cli --version
    python3 -m chem_mat_data.cli --help

=========================
📦 Installation by Source
=========================

Install the latest stable release using ``pip``

.. code-block::

    pip3 install chem_mat_data

=============
🚀 Quickstart
=============

The package provides a simple and convenient interface to access the datasets. 

.. code-block:: python

    from pandas import DataFrame
    from rich import print
    from chem_mat_data import load_smiles_dataset, load_graph_dataset
    from chem_mat_data import pyg_data_list_from_graphs

    # ~ LOADING RAW DATASETS
    # Datasets are generally available in the "raw" format and the "processed" format.
    # The most common raw format is simply as a CSV file containing the SMILES string 
    # representations of hte various molecules in one column and the corresponding 
    # target value annotations in another column.
    # The "load_smiles_dataset" function can be used to load such a dataset as a 
    # pandas data frame.

    df: DataFrame = load_smiles_dataset('_test')
    print('dataset:\n', df.head())

    # ~ LOADING PROCESSED DATASETS
    # Alternatively, datasets are available in the "processed" format as well. In this 
    # format, every molecule is already represented as a graph structure where nodes 
    # represent the atoms and edges represent the bonds.
    # A structure like this is especially suited for deep learning applications such as 
    # graph neural networks (GNNs). The "load_graph_dataset" function can be used to 
    # directly load such graph representations. 

    graphs: list[dict] = load_graph_dataset('_test')
    # In practice, these graphs are represented as dictionaries with various keys 
    # whose values are numpy arrays that represent different aspects of the graph.
    print('graph keys:', list(graphs[0].keys()))

    # ~ DEEP LEARNING INTEGRATION
    # The package also provides convenient functions to easily convert these graphs 
    # dictionaries into a PyTorch Geometric (PyG) DataLoader instance which can then 
    # be directly employed to train a GNN model!

    import torch_geometric.loader
    data_list = pyg_data_list_from_graphs(graphs)
    data_loader = torch_geometric.loader.DataLoader(
        data_list, 
        batch_size=32, 
        shuffle=False
    )

==============================
⌨️ Command Line Interface (CLI)
==============================

In addition to the programming interface, the package also provides a command line interface (CLI) ``chemdata`` to interact with the database.
To see the available commands, simply use the ``--help`` flag:

.. code-block:: console

   chemdata --help

Listing Available Datasets
--------------------------

To see the available datasets execute the ``list`` in the terminal

.. code-block:: console 

   chemdata list

This will print a table containing all the dataset which are currently available to download from the database. Each row of the 
table represents one dataset and contains the name of the dataset, the number of molecules in the dataset and the number of
target properties as additional columns.


Listing Dataset Information
---------------------------

Additional information for a specific dataset is obtained by the ``info`` command. 
For example for the "clintox" dataset, execute this

.. code-block:: console 

   chemdata info "clintox"

This command will print all available information about a given dataset to the console - including, for example, a short 
textual description of the dataset as well as information about where it was originated from.


Downloading Datasets
--------------------

Finally, to download this dataset, use the ``download`` command:

.. code-block:: console

   chemdata donwload "clintox"

This will download the dataset "clintox" to your current working directory. 
One can also specify the path to wich the dataset should be downloaded as following:

.. code-block:: console

   chemdata download --path="/absolute/path/to/desired/directory"


===========
🤝 Credits
===========

We thank the following packages, institutions and individuals for their significant impact on this package.

* PyComex_ is a micro framework which simplifies the setup, processing and management of computational
  experiments. It is also used to auto-generate the command line interface that can be used to interact
  with these experiments.

.. _PyComex: https://github.com/the16thpythonist/pycomex.git
.. _Cookiecutter: https://github.com/cookiecutter/cookiecutter
