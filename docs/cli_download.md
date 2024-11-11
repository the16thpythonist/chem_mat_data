# ``download`` - Download Dataset

You can use the ``download`` command to download the datasets to the local file system by 
supplying the unique string identifier of the corresponding dataset (see [List Available Datasets](cli_list.md)):

```bash
cmdata download "clintox"
```

This command will download the *raw* dataset files into the *current working directory (CWD)*. 

## Changing the Destination Path

You can change the destination directory for the download using the ``--path`` option:

```bash
cmdata download --path="/tmp" "clintox"
```

## Downloading the Graph Dataset

By default the command downloads the *raw* version of each dataset. You can supply ``--full`` flag 
to also download the *processed/graph* version of the dataset as well:

```bash
cmdata download --full "clintox"
```

This command will download the raw ``clintox.csv`` file *and* the ``clintox.mpack`` file. The processed 
dataset is stored in the [MessagePack](https://msgpack.org/index.html) file format, which is essentially 
a binary version of the JSON format. When decoding this file with a programming language of choice, it 
will yield a list/sequence of graph dictionaries/objects as described in the [Graph Representation](graph_representation.md)
documentation.