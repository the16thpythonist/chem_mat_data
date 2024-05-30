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

ChemMatData is a database wich contains molecular datasets for machine learning purposes from pyhsics,chemistry and physiology. The datasets are available in ther original format as well as in a graph dictionary structure.

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

    **NOTE.** Use this section to create a minimal example of how to use the code in this repository. If your repository is mainly based on a number 
    of scripts, you could show how the most important scripts can be executed and what the most important parameters are. If your code is rather 
    used as a library you can write a simple code block that shows how to use the features of that library.

.. code-block:: python

    # The following code is just an example and not executable
    from chem_mat_data.dataset import Dataset
    from chem_mat_data.compute import Computation

    dataset = Dataest('name')
    computation = Computation(dataset)
    result = computation.compute()
    print(result)

==============
📖 Referencing
==============

    **NOTE** Delete this section if you are not working / are not planning on a publication of your project

If you use, extend or otherwise reference our work, please cite the corresponding paper as follows:

.. code-block:: bibtex

    @article{
        title={Your Publication title},
        author={Mustermann, Max and Doe, John},
        journal={arxiv},
        year={2023},
    }

==========
🤝 Credits
==========

We thank the following packages, institutions and individuals for their significant impact on this package.

* PyComex_ is a micro framework which simplifies the setup, processing and management of computational
  experiments. It is also used to auto-generate the command line interface that can be used to interact
  with these experiments.

.. _PyComex: https://github.com/the16thpythonist/pycomex.git
.. _Cookiecutter: https://github.com/cookiecutter/cookiecutter
