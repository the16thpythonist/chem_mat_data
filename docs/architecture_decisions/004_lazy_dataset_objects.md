# Lazy Dataset Objects

## Status 

22.07.2025: in progress

## Context 

The ChemMatData package provides the possibility to download graph datasets from a remote file share 
server. One feature of the package is that it provides also the already pre-processed versions of these 
datasets in the form of graph dictionary objects which can then easily be converted into Pytorch Geometric
objects etc. The pre-processing for those datasets is already opinionated and standardized for comparability.

However, there is instances where the user might want to apply a different pre-processing. Another case could 
be that there are datasets which are simply too large to be loaded into memory at once. I've recently come into 
more contact of how this is solved in Pytorch Dataset objects which allow to lazy load the data from the disk. 
To mitigate the processing bottleneck it is possible to use multiple workers to load and process the data in 
parallel. This was surprisingly fast and efficient compared to loading it all into the memory at once.

It would be nice to have the same option for ChemMatData datasets. Where one downloads the raw format (which 
is much smaller and not opionionated) and then use such a lazy loading mechanism to load the data in-time.




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