|made-with-python| |python-version|

.. |made-with-python| image:: https://img.shields.io/badge/Made%20with-Python-1f425f.svg
   :target: https://www.python.org/

.. |python-version| image:: https://img.shields.io/badge/Python-3.8.0-green.svg
   :target: https://www.python.org/

=================
⭐ ChemMatData
=================

.. image:: chem_mat_data/ChemMatData_logo_final.png
   :alt: ChemMatData Logo
   :align: center
ChemMatData is a database consisting of a collection of datasets from physics, chemistry, phyisology and material science. Each dataset contains various molecules and/or crystal structures associated with a specific property. The main purpose of these datasets is to be used for the training of machine learning models and thus the prediction of these properties. The datasets are available in their original format as well in a graph dictionary structure.

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

    **NOTE.** delete this section if your code is not to be published as a python package

Install the latest stable release using ``pip``

.. code-block::

    pip3 install chem_mat_data

============
🚀 Quckstart
============
Not sure how the python import example would look like.(Will be adjusted soon)

.. code-block:: python

    # The following code is just an example and not executable
    from chem_mat_data.dataset import Dataset


To see the available datasets execute the following in the terminal

.. code-block:: console 

   chemdata list

Additional information for a specific dataset is obtained by the "info" command. For example for the clintox dataset, execute this

.. code-block:: console 

   chemdata info clintox

To download this dataset, one uses the "download" command:

.. code-block:: console

   chemdata donwload clintox

This will download the dataset "clintox" to your current working directory. One can also specify the path to wich the dataset should be downloaded as following:

.. code-block:: console

   chemdata download --path="/absolute/path/to/desired/directory"

The dataset will be in a graph dictionary structure.
If one is interested in the original format of the dataset and the graph dictionary format, use the "full" flag:

.. code-block:: console

   chemdata download --full clintox

One can thus download both formats of the dataset into a desired directory like this:

.. code-block:: console

   chemdata download --full --path="/absolute/path/to/desired/directory" clintox
==========
🤝 Credits
==========

We thank the following packages, institutions and individuals for their significant impact on this package.

* PyComex_ is a micro framework which simplifies the setup, processing and management of computational
  experiments. It is also used to auto-generate the command line interface that can be used to interact
  with these experiments.

.. _PyComex: https://github.com/the16thpythonist/pycomex.git
.. _Cookiecutter: https://github.com/cookiecutter/cookiecutter
