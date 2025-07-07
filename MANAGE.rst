==========================
Information for Management
==========================

This file contains some information about the management and the maintance of the *database* itself (further 
information about the *package* can be found in the `DEVELOP.rst` file).

In the ``chem_mat_data`` package, the ``manage.py`` script can be used to manage the datasets:

.. code-block:: bash

    cd chem_mat_data
    python manage.py --help


Triggering Experiment Modules for Dataset Creation 
==================================================

The processed *graph* datasets are created by running the experiment modules in the ``chem_mat_data/scripts`` 
folder. The ``manage.py dataset create`` command can be used to trigger the execution of these modules to 
(re-)create the datasets.

.. code-block:: bash

    python manage.py dataset create "aqsoldb"

Alternatively one can use the ``--all`` flag to trigger the execution of all available experiment modules. 
However, note that this may take a very long time.

.. code-block:: bash

    python manage.py dataset create --all

Following the execution of the dataset creation modules, the datasets are stored in the ``chem_mat_data/scripts/results`` 
folder in the format of experiment archive folders.

Uploading Datasets to the Database
==================================

After creating the datasets with the "create" command, the datasets can be uploaded to the database using the 
``manage.py dataset upload`` command:

.. code-block:: bash

    python manage.py dataset upload "aqsoldb"

Alternatively, the ``--all`` flag can be used to upload all datasets to the database.

.. code-block:: bash

    python manage.py dataset upload --all

In the default setting, if there are multiple experiment archives related to one dataset, the most recent one 
will be used to upload the dataset to the database.


Uploading Metadata.yml file to Remote File Share
================================================

The ``metadata.yml`` file stored on the remote file share is the primary source of information for 
accessing the remote datasets. To effectively include the datasets in the database, the ``metadata.yml``
file has to be updated with the new datasets.

The ``manage.py metadata upload`` command can be used to upload the local version of ``metadata.yml`` 
to the remote file share. The command will automatically *replace* the version of the file on the remote
file share with the local version.

.. code-block:: bash

    python manage.py metadata upload


Collecting Metadata Information locally
=======================================

To include the newest metadata information that was produced as a side product of the creation/processing 
of the datasets, the ``manage.py metadata collect`` command can be used. This command will collect the 
metadata information from the experiment archives in the ``chem_mat_data/scripts/results`` folder and
store it in the local ``metadata.yml`` file.

.. code-block:: bash

    python manage.py metadata collect