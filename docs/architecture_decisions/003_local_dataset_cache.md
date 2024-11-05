# Local Dataset Cache

## Status 

implemented

## Context 

The ChemMatData package provides the possibility to download graph datasets from a remote file share 
server. These datasets can either be in the "raw" format - in the case of purely molecular datasets 
this usually means a CSV file with the molecule SMILES representations annotated with the target values.
But more importantly, these datasets can be downloaded in the already pre-processed format and easily 
loaded into the popular graph deep learning libraries.

There exists a command line interface and a programming interface to very easily load a dataset with a 
single function call. The problem now is that repeated calls to this function would download the dataset 
anew each time. This would require a constant internet connection and could take a significant amount of 
time for larger datasets.

## Decision

The mitigate the recurring runtime of re-downloading the datasets each time, a local caching mechanism was 
added which was inspired by package managers such as ``pip`` which also maintain a similar caching mechanism.
Whenever a dataset is downloaded, the downloaded files will be placed into a user-specific caching folder. 
For all subsequent retrievals of the dataset, the cached version will be used instead.

Inside the cache, each dataset is enumerated by its unique string name and the dataset type (raw or processed).

## Consequences

### Advantages

**Runtime.** The clear advantage of a local cache is the runtime. Especially for large datasert and/or repeated 
execution of the dataset loading functionality, a local cahce will result in a clear reduction in runtime and 
bandwith.

### Disadvantages

**User Storage.** For the user, the cache will eat up some storage capacity, which may or may not be significant 
depending on the number and size of the stored datasets. Although, this likely won't be a problem as the archived 
versions of even the largest datasets currently do now exceed a GB in size.

**Management Overhead.** On the development side, the cache introduces another layer of complexity. Before fetching 
the datasets, we need to check if the dataset exists in the cache and after downloading the dataset we need to 
add the dataset to the cache.