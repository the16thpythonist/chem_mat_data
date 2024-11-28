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

## Default Features

The graph representations that can be downloaded already come with a pre-defined encoding of numeric node and edge feature vectors that encode specific information about the atoms and bonds of the molecular graphs. The selection of *which* attributes are included here is pre-determined to be a basic selection of attributes which will be explained in detail in the following sections.

All features were calculated based on the ``rdkit.Mol`` representation of the corresponding elements.

If the default feature selection is for some reason insufficient for a given task, there is the option to define custom pre-processing classes with a 
custom selection of features.
[Custom Pre-Processing](custom_pre_processing.md)

### Node Features

The standard node features are the following:

| Feature Index | Name    | Description |
|---------------|---------|-------------|
| $0$ | is Carbon (C)? | Integer one-hot label if the node represents a carbon atom. | 
| $1$ | is Oxygen (O)? | Integer one-hot label if the node represents an oxygen atom. |
| $2$ | is Nitrogen (N)? | Integer one-hot label if the node represents a nitrogen atom. |
| $3$ | is Sulfur (S)? | Integer one-hot label if the node represents a sulfur atom. |
| $4$ | is Phosphorus (P)? | Integer one-hot label if the node represents a phosphorus atom. |
| $5$ | is Fluorine (F)? | Integer one-hot label if the node represents a fluorine atom. |
| $6$ | is Chlorine (Cl)? | Integer one-hot label if the node represents a chlorine atom. |
| $7$ | is Bromine (Br)? | Integer one-hot label if the node represents a bromine atom. |
| $8$ | is Iodine (I)? | Integer one-hot label if the node represents an iodine atom. |
| $9$ | is Silicon (Si)? | Integer one-hot label if the node represents a silicon atom. |
| $10$ | is Boron (B)? | Integer one-hot label if the node represents a boron atom. |
| $11$ | is Sodium (Na)? | Integer one-hot label if the node represents a sodium atom. |
| $12$ | is Magnesium (Mg)? | Integer one-hot label if the node represents a magnesium atom. |
| $13$ | is Calcium (Ca)? | Integer one-hot label if the node represents a calcium atom. |
| $14$ | is Iron (Fe)? | Integer one-hot label if the node represents an iron atom. |
| $15$ | is Aluminum (Al)? | Integer one-hot label if the node represents an aluminum atom. |
| $16$ | is Copper (Cu)? | Integer one-hot label if the node represents a copper atom. |
| $17$ | is Zinc (Zn)? | Integer one-hot label if the node represents a zinc atom. |
| $18$ | is Potassium (K)? | Integer one-hot label if the node represents a potassium atom. |
| $19$ | is Uknown atom? | Integer one-hot label for an unknown atom type not covered by the above list of atoms. |
| $20$ | S hybrid | Integer one-hot label if the atom has S hybridization. |
| $21$ | SP hybrid | Integer one-hot label if the atom has SP hybridization. |
| $22$ | SP2 hybrid | Integer one-hot label if the atom has SP2 hybridization. |
| $23$ | SP3 hybrid | Integer one-hot label if the atom has SP3 hybridization. |
| $24$ | SP2D hybrid | Integer one-hot label if the atom has SP2D hybridization. |
| $25$ | SP3D hybrid | Integer one-hot label if the atom has SP3D hybridization. |
| $26$ | Unknown hybridization | Integer one-hot label for an unknown hybridization type not covered by the above list. |
| $27$ | 0 Neighbors | Integer one-hot label if the atom has 0 neighbors. |
| $28$ | 1 Neighbor | Integer one-hot label if the atom has 1 neighbor. |
| $29$ | 2 Neighbors | Integer one-hot label if the atom has 2 neighbors. |
| $30$ | 3 Neighbors | Integer one-hot label if the atom has 3 neighbors. |
| $31$ | 4 Neighbors | Integer one-hot label if the atom has 4 neighbors. |
| $32$ | 5 Neighbors | Integer one-hot label if the atom has 5 neighbors. |
| $33$ | 0 Hydrogens | Integer one-hot label if the atom has 0 attached hydrogen atoms. |
| $34$ | 1 Hydrogen | Integer one-hot label if the atom has 1 attached hydrogen atom. |
| $35$ | 2 Hydrogens | Integer one-hot label if the atom has 2 attached hydrogen atoms. |
| $36$ | 3 Hydrogens | Integer one-hot label if the atom has 3 attached hydrogen atoms. |
| $37$ | 4 Hydrogens | Integer one-hot label if the atom has 4 attached hydrogen atoms. |
| $38$ | Mass | Float value representing the mass of the atom. |
| $39$ | Charge | Integer value representing the electrical charge of the atom. |
| $40$ | Is Aromatic? | Integer one-hot label if the atom is aromatic. |
| $41$ | Is in Ring? | Integer one-hot label if the atom is in a ring. |
| $42$ | Crippen Contributions | Float value representing the Crippen logP contributions of the atom as computed by RDKit. |

### Edge Features

The standard edge features are the following:

| Feature Index | Name             | Description                               |
|---------------|------------------|-------------------------------------------|
| $0$           | Single Bond      | One hot encoding if the bond is a single bond. |
| $1$           | Double Bond      | One hot encoding if the bond is a double bond. |
| $2$           | Triple Bond      | One hot encoding if the bond is a triple bond. |
| $3$           | Aromatic Bond    | One hot encoding if the bond is an aromatic bond. |
| $4$           | Ionic Bond       | One hot encoding if the bond is an ionic bond. |
| $5$           | Hydrogen Bond    | One hot encoding if the bond is a hydrogen bond. |
| $6$           | Unknown Bond     | One hot encoding if the bond type is unknown. |
| $7$           | Stereo None      | One hot encoding if the bond has no stereo property. |
| $8$           | Stereo Any       | One hot encoding if the bond has any stereo property. |
| $9$           | Stereo Z         | One hot encoding if the bond has Z stereo property. |
| $10$          | Stereo E         | One hot encoding if the bond has E stereo property. |
| $11$          | Is Aromatic      | Integer flag if the bond is aromatic. |
| $12$          | Is in Ring       | Integer flag if the bond is part of a ring. |
| $13$          | Is Conjugated    | Integer flag if the bond is conjugated. |