# Graph Representation

The ``chem_mat_data`` package provides various chemical property prediction datasets in a pre-processed format, 
in which molecules are mapped to graph structures that represent the atoms as nodes and the bonds as corresponding 
edges.

## Graph Dict Structure

Practically, this graph structure is represented as a native python ``dict`` object with a set of specific attributes.

Generally, the graph dict structure follows the following naming convention

- Prefix ``node_`` for node-level attributes with shape $(V, ?)$
- Prefix ``edge_`` for edge-level attributes with shape $(E, ?)$
- Prefix ``graph_`` for graph-level attributes with shape $(?, )$

More specifically, each graph dict object has the following *minimal set* of attributes:

<style>
table th:first-child, table td:first-child {
    min-width: 150px;
}
</style>

| Attribute               | Description |
|-------------------------|-------------|
| `node_indices`          | Integer numpy array of shape $(V, )$ where $V$ is the number of nodes in the graph. Contains the unique integer indices of the graph nodes. |
| `node_attributes`       | Float numpy array of shape $(V, N)$ where $V$ is the number of nodes and $N$ is the number of numeric node features. Contains the numeric feature vectors that represent each node. |
| `edge_indices`          | Integer numpy array of shape $(E, 2)$ where $E$ is the number of edges in the graph. Contains tuples $(i, j)$ of node indices that indicate the existence of an edge between nodes $i$ and $j$. |
| `edge_attributes`       | Float numpy array of shape $(E, M)$ where $E$ is the number of edges in the graph and $M$ is the number of values in the graph. |
| `graph_labels`          | Float numpy array of shape $(T, )$ where $T$ is the number of target values associated with each element of the dataset. The target values can either be continuous regression targets such as ``[1.43, -9.4]`` or could be one-hot classification labels such as ``[0, 1, 0]``. |
| `graph_repr`            | The string numpy array of shape $(1, )$ containing the string SMILES representation of the original molecule. |

Depending on the dataset, the graph dict representation may contain additional attributes that represent custom properties such as the 3D coordinates of nodes/atoms, for exmaple.