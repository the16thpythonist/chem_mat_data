

.. code-block:: python

    graph: dict = {
        'graph_index': 0,
        'graph_smiles': 'C1CC1',
        'node_attributes': [
            [0, 0, 1, 1],
            # ...
        ],
        'edge_attributes': [
            [0, 0, 1, 1],
            # ...
        ],
        'graph_labels': [
            0.2 # solubity
        ],
    }

.. code-block:: python

    dataset: list[dict] = [...]
    string = json.dumps(dataset)
    # save to file

    # on nextcloud
    solubility_3.csv
    solubility_3.json

    # the api
    from chem_mat_data.pipeline import download_dataset

    download_dataset('solubility_3', full=True)
