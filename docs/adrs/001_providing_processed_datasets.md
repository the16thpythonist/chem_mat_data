# Providing Processed Datasets

## Status

implemented

## Context

The ``chem_mat_data`` package mainly aims to provide a Python and command line API which can be used to easily 
download and access chemistry and material science datasets for machine learning with a specific focus on 
graph neural networks. One core question in the design of this package is in which *format* to provide these 
datasets to the end user. This is a tricky matter because the datasets may have slightly different formats 
themselves and might even have slighlty different objectives. For example, there can be a distinction between 
node-level and graph-level tasks which changes the way in which the ground truth labels have to be provided. 
Another difference could be that some datasets may consists of different molecules while other datasets consist 
of the same molecule over and over again, only with different geometries.

## Decision

The decision made for this package is to provide datasets in two formats simultaneously:
- **raw.** This format should be as close as possible to the original domain-specific representation that the 
  dataset is provided in. For most datasets this will likely be a CSV file containing the SMILES codes of 
  different molecules associated with ground truth classification and regression labels. Additionally, this 
  raw representation could be extended with additional information about the geometric configurations.
- **processed.** In addition to the raw format, each dataset should also be provided in the already processed 
  format. In this format, the individual elements of the dataset are already present as abstract graph 
  structures consisting of nodes (atoms) connected by edges (bonds). Additionally each node and edge should 
  be associated with a numeric feature vector which has been extracted using the Cheminformatics library 
  ``RDKit``.

## Consequences

### Advantages

**Ease of use.** The main advantage of providing the processed version of the dataset directly is that this makes 
it immensly easy to use the dataset to train a graph neural network. The graph format used in the package is a 
generic one and the package specifically aims to provide adapters that transform these generic graph structures 
into the necessary object instances required for the most common graph learning libraries such as pytorch geometric 
and jraph. This ultimately means that a dataset is training-ready in a couple lines of code.

**Standardization.** The processed graph structures already contain numeric feature vectors for the nodes and edges 
which have been obtained from RDKit. If everyone were to use the same processing / featurization pipeline, this would 
lead to more comparable results in the end.

### Disadvantages

**Dataset size.** A main disadvantage of the processed datasets is the significantly increased size required 
to encode the full graph structure which will affect the download speed and the storage requirements especially 
for larger datasets.

**Rigidity.** The processed graph structures already contain numeric feature vectors for the nodes and edges 
which have been obtained from RDKit. Besides a standardization effect, this also means that.
However, the package also provides the means to implement custom processing on top of the raw dataset 
representations if desired.