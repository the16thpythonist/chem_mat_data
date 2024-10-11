# Custom Graph Representation

## Status 

implemented

## Context 

The package also provides the various datasets to be downloaded in a processed format where the 
various molecules are in a generic graph representation. The question that arises in this situation 
is how to structure this graph representation, especially since different datasets may have different 
requirements.

The main requirements for this graph representation are:
- support for rich metadata at each level of the graph: node-, edge- and graph-level
- support for graph-level feature and label annotations
- support for node-level feature vectors
- support for edge-level feature vectors

## Decision

The decision is to use a custom graph representation specifically created by and for this package called 
a GraphDict representation. This represents a graph as a simple python native dictionary structure where 
special keys identify the various parts of the graph representation. The most important keys are the following:

- ``node_indices``: The integer indices of the nodes
- ``node_attributes``: A node feature vector for each node in the same order as the node indices
- ``edge_indices``: a list of node index tuples which define which nodes are connected by edges
- ``edge_attributes``: A edge feature vector for each edge in the same order as the edge indices
- ``graph_labels``: A list of label annotations for the graph as a whole

Besides these main keys, the primary feature of the graph dict representation is that it is supposed to be 
flexible and that additional properties can be added dynamically by using one of the special prefixes 
``node_``, ``edge_`` and ``graph_`` to indicate at which level of the graph to attach the information 
to.

## Consequences

### Advantages

**Native.** The major advantage of this graph representation is that it only relies on python native 
datastructures such as dictionaries and lists and by extension is therefore also easily JSON encodable. 
Being able to JSON encode is very important for the transfer of the datasets over the internet and 
also the potential compatibility with other programming languages when compared to a pickle dump for 
example. 

**Generic and Flexible.** The flexible format allows to add additional information for datasets that 
need it without having to over-complicate the core data structure.

### Disadvantages

**Custom Format.** It is a custom format that most people will not be initially familiar with. This is 
in contrast to using more common formats such as networkx Graph instances for example, which some 
people might already have familarity with. However, we argue that the format is simple enough to 
understand and also simple enough to convert to a more common format such as networkx - especially if 
the necessary adapaters are provided as part of the package as well.