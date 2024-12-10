"""
This module implements the conversion of a dataset of molecules (SMILES strings) to 
fully pre-processed dataset of GraphDict object in message pack format that can be 
directly loaded for the training of graph neural networks.

This module heavily makes use of parallel processing to speed up the conversion 
process. In the main process, the raw data of the molecule elements is fed into a 
multiprocessing queue which is then consumed by several worker processes that 
use the ``MoleculeProcessing`` instance to turn the SMILES string of the molecule 
into a full GraphDict representation. These graph dicts are sent back to the 
main process where they are collected and then later saved into a message pack 
file in the experiments results folder. As an additional optimization those 
message pack files are then compressed into a gzip file to reduce the file size
even further.

**Base Experiment**

This module is actually just the base experiment implementation that should NOT be 
modified directly. The implementation of processing *specific* datasets should only 
be done by creating new sub experiments which inherit the main functionality from 
this base experiment. This can be done by using the ``Experiment.extend`` method.

These sub experiment can inject custom code / functionality to the processing process 
by using the various hooks that are defined in the base experiment.
"""
import os
import time
import csv
import gzip
import shutil
import datetime
import multiprocessing
from typing import Union

import msgpack
import numpy as np
import rdkit.Chem as Chem
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

import chem_mat_data._typing as typc
from chem_mat_data.processing import MoleculeProcessing
from chem_mat_data.data import default, ext_hook
from chem_mat_data.data import save_graphs
from chem_mat_data.utils import SCRIPTS_PATH
import yaml

# == SOURCE PARAMETERS ==
# These global parameters can be used to configure the source files from which the 
# dataset will be created. In the default implementation, this source file is assumed 
# to be a CSV file which contains at least the molecule smiles representation and 
# the target values.

# :param SOURCE_PATH:
#       This is the absolute string path of the dataset CSV file from which the graph
#       dataset should be created. 
SOURCE_PATH: str = os.path.join(SCRIPTS_PATH, 'assets', '_test.csv')
# :param SMILES_COLUMN:
#       This is the string name of the CSV column which contains the SMILES strings of
#       the molecules.
SMILES_COLUMN: str = 'smiles'
# :param TARGET_COLUMNS:
#       This is a list of string names of the CSV columns which contain the target values
#       of the dataset. This can be a single column for regression tasks or multiple columns
#       for multi-target regression or classification tasks. For the final graph dataset
#       the target values will be merged into a single numeric vector that contains the 
#       corresponding values in the same order as the column names are defined here.
TARGET_COLUMNS: list[str] = ['target']
# :param DATASET_TYPE:
#       Either 'regression' or 'classification' to define the type of the dataset. This
#       will also determine how the target values are processed.
DATASET_TYPE: str = 'regression'
# :param DESCRIPTION:
#       This is a string description of the dataset that will be stored in the experiment
#       metadata.
DESCRIPTION: str = 'the description of the dataset'
# :param METADATA:
#       A dictionary which will be used as the basis for the metadata that will be added 
#       as additional information to the file share server.
METADATA: dict = {
    'tags': ['Molecules']
}

# == PROCESSING PARAMETERS ==
# These parameters can be used to configure the processing functionality of the script 
# itself. This includes for example whether or not the molecule coordinates should be 
# created by RDKIT as well or not. Also the compression of the final dataset file can
# be configured here.

# :param DATASET_NAME:
#       This string determines the name of the message pack dataset file that is then 
#       stored into the "results" folder of the experiment as the result of the 
#       processing process. The corresponding file extensions will be added 
#       automatically.
DATASET_NAME: str = 'dataset'
# :param USE_COORDINATES:
#       If this is True, then the graph representations of the molecules will also contain 
#       the "node_coordinates" field which will be populated with the 3D coordinates of
#       of the nodes created by RDKIT. Setting this to True will increase the processing 
#       runtime and may cause the processing of some molecules to fail entirely.
USE_COORDINATES: bool = False
# :param COMPRESS:
#       If this is True, then the final message pack dataset file will be compressed into
#       a gzip file to reduce the file size. This is recommended because the raw message
#       pack files are still relatively large. However, the compression will likely take 
#       a while to complete. 
COMPRESS: bool = True

# == EXPERIMENT PARAMETERS ==

# :param __DEBUG__:
#       In debug mode, the experiment will not create a unique results folder but instead will
#       overwrite the results of the last experiment. This is useful for debugging the experiment
#       code itself.
__DEBUG__ = True
# :param __TESTING__:
#       If this is True, then the experiment will be started in "testing" mode 
__TESTING__ = False

experiment = Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)


class ProcessingWorker(multiprocessing.Process):
    """
    This class implements the actual processing of the SMILES representation to the 
    graph representation. It inheritted from the multiprocessing.Process class which means 
    that it will be run in a different process which enables multiple molecules to be 
    processed in parallel.
    
    Communication bween the main process and this worker process is done via two
    multiprocessing queues. The input queue is used to send the raw data to be processed 
    to the worker process and the output queue is used to send the processed graph 
    representations back to the main process.
    """
    
    def __init__(self,
                 input_queue: multiprocessing.Queue,
                 output_queue: multiprocessing.Queue,
                 *args,
                 ):
        super(ProcessingWorker, self).__init__()

        self.input_queue = input_queue
        self.output_queue = output_queue

        # This is the object that handles the actual processing of the SMILES.
        self.processing = MoleculeProcessing()
        
    def run(self):
        """
        This code will be executed in a separate process. It essentially consists of an 
        infinite loop that waits for new data from the input queue, then attempts to process 
        the smiles contained in that input data into a graph and then sends the graph back 
        to the main process via the output queue.
        
        It is possible that the processing of molecule fails due to some reason. In this case, 
        the process will put a "None" value into the output queue instead of the graph object.
        The important thing is that for every input element there will be exactly one output.
        """
        for data in iter(self.input_queue.get, None):
            
            try:
                # 10.12.24 - If we want to support the processing of xyz file based datasets as well we need 
                # to not only support the conversion based on a smiles string representation but also the 
                # direct conversion based from a Mol object directly. The processing.process method supports 
                # this inherently to use either - we only have to check here what we are actually getting.
                value: Union[Chem.Mol, str]
                if 'mol' in data:
                    value = data['mol']
                else:
                    value = data['smiles']
                
                graph: typc.GraphDict = self.processing.process(
                    value=value, 
                    use_node_coordinates=experiment.USE_COORDINATES,
                )
                
                graph['graph_labels'] = np.array(data['targets'])
                experiment.apply_hook(
                    'add_graph_metadata',
                    data=data,
                    graph=graph,
                )
                
            except Exception as exc:
                print(f' ! error processing {value} - {exc.__class__.__name__}: {exc}')
                graph = None
            
            graph_encoded = msgpack.packb(graph, default=default)            
            self.output_queue.put(graph_encoded)


@experiment.hook('add_graph_metadata', default=True, replace=False)
def add_graph_metadata(e: Experiment,
                       data: dict,
                       graph: typc.GraphDict
                       ) -> dict:
    """
    This hook is invoked in the processing worker after the SMILES code has been converted 
    to the graph dict already. The hook receives the original data dict and the graph dict 
    as arguments and provides the opportunity to add additional metadata to the graph dict.
    
    ---
    
    This hook is not used by default.
    """
    pass


@experiment.hook('load_dataset', default=True, replace=False)
def load_dataset(e: Experiment) -> dict[int, dict]:
    """
    In the experiment, this hook is invoked at the very beginning to obtain the actual 
    raw data of the dataset that should be processed. The output of this function should 
    be a dictionary whose keys are the integer indices of the data elements and the values 
    are in turn dictionary objects that should contain AT LEAST the following keys:
    - 'smiles': The SMILES representation of the molecule
    - 'targets': A list of float target values for the molecule

    ---

    This default implementation assums that the dataset is stored in a CSV file and that
    the SMILES representation of the molecules is stored in a column name determined by 
    SMILES_COLUMN and that the target values are stored in columns determined by TARGET_COLUMNS.    

    This default implementation may be overwritten in sub experiments to achieve a more 
    customized behavior.
    """
    if os.path.exists(e.SOURCE_PATH) and os.path.isfile(e.SOURCE_PATH):
        
        # ~ reading csv file
        dataset = {}
        with open(e.SOURCE_PATH) as file:
            dict_reader = csv.DictReader(file)
            for index, row in enumerate(dict_reader):
                dataset[index] = row
                
        # ~ constructing target values
        for index, data in dataset.items():
            targets = []
            
            if e.DATASET_TYPE == 'regression':
                for key in e.TARGET_COLUMNS:
                    targets.append(data[key])
            
            data['targets'] = targets
            data['smiles'] = data[e.SMILES_COLUMN]
            
        return dataset
    
    e.log(f'dataset @ "{e.SOURCE_PATH}" not found!')


@experiment
def experiment(e: Experiment):
    """
    This is the main experiment code that is executed when the script is run.
    """
    
    e.log('starting experiment...')
    
    # We need to setup the multiprocessing at the beginning here because of the following 
    # technical problem:
    # When python spawns a subprocess it doesnt actually spawn a completely blank python runtime
    # but it *copies* the current process. This also means that it copies the complete memory 
    # of the current process. If we started these child processes after loading the dataset we'd 
    # be copying the entire dataset X times which could very easily lead to a memory overflow...
    e.log('creating processing workers...')
    input_queue = multiprocessing.Queue(maxsize=100)
    output_queue = multiprocessing.Queue(maxsize=100)
    workers = []
    for _ in range(os.cpu_count()):
        worker = ProcessingWorker(
            input_queue=input_queue,
            output_queue=output_queue,
        )
        worker.start()
        workers.append(worker)
        e.log(' * started worker')
    
    e.log('loading dataset...')
    dataset: dict[int, dict] = e.apply_hook(
        'load_dataset'
    )
    num_elements = len(dataset)
    e.log(f'loaded dataset with {num_elements} elements')
    
    # when starting the experiment in testing mode we want to limit the number of elements in 
    # the dataset to only a few examples so that the experiment overall finishes really quickly 
    # and so that all the code in the experiment (from start to finish) can be tested quickly 
    # for obvious errors (e.g. syntax errors, etc.) 
    if e.__TESTING__:
        
        e.log('running experiment in testing mode...')
        
        e.log(' * limiting to only 50 elements')
        num_elements = min(num_elements, 50)
        dataset = dict(list(dataset.items())[:num_elements])

    e.log('processing dataset...')
    indices = list(dataset.keys())
    num_indices = len(indices)
    graphs = []
    
    start_time = time.time()
    count = 0
    prev_count = 0
    
    # The actual processing happens in the additional worker processes and in this loop we simply 
    # feed the data to be processed into the input queue for the workers processes and then collect 
    # the processed graphs from the output queue and store that in the local "graphs" list.
    # In the end all the graph dicts in that graphs list will be saved into a msgpack file to the 
    # disk.
    while count < num_indices:
        
        # In each iteration we will completely fill up the input queue so that it is always full
        # and the workers are always busy
        while not input_queue.full() and len(indices) != 0:
            index = indices.pop()
            data = dataset[index]
            input_queue.put(data)
            
        # Then we also want to clear all the contents of output queue and transfer them to the 
        # local graph list
        while not output_queue.empty():
            graph_encoded = output_queue.get()
            graph = msgpack.unpackb(graph_encoded, ext_hook=ext_hook)
            if graph:
                graphs.append(graph)
            
            count += 1
            
        if count % 1000 == 0 and count != prev_count:
            prev_count = count
            time_passed = time.time() - start_time
            num_remaining = num_elements - (count + 1)
            time_per_element = time_passed / (count + 1)
            time_remaining =time_per_element * num_remaining
            eta = datetime.datetime.now() + datetime.timedelta(seconds=time_remaining)
            e.log(f' * {count:05d}/{num_elements} done'
                  f' - time passed: {time_passed / 60:.2f}m'
                  f' - time remaining: {time_remaining / 60:.2f}m'
                  f' - eta: {eta:%a %d.%m %H:%M}')
            
    end_time = time.time()
    duration = end_time - start_time
    e.log(f'finished processing dataset after {duration / 3600:.2f}h')

    # We need to stop the subprocesses here because if we don't do that then the main process won't be 
    # able to properly exit either...
    e.log('stopping the workers...')
    for worker in workers:
        input_queue.put(None)
        worker.terminate()
        worker.join()
        
    del input_queue
    del output_queue

    e.log('Description:')
    e.log(e.DESCRIPTION)

    # ~ saving the dataset into the msgpack format
    e.log(f'saving the dataset with {len(graphs)} graphs...')
    dataset_path = os.path.join(e.path, e.DATASET_NAME + '.mpack')
    save_graphs(graphs, dataset_path)
    
    file_size = os.path.getsize(dataset_path)
    file_size_mb = file_size / (1024 * 1024)
    e.log(f'wrote file with {file_size_mb:.1f} MB')
    
    # ~ compressing the file
    # The raw message pack files are actually still relatively large (surprisingly larger than JSON)
    # so we additionally compress those files into a ZIP here so that we can then put those zipped 
    # files onto the server to be actually downloaded. This should greatly reduce the download size 
    # and therefore speed by another 10x or so.
    if e.COMPRESS:
        
        e.log('compressing the dataset file, this may take a while...')
        compressed_path = os.path.join(e.path, e.DATASET_NAME + '.mpack.gz')
        with open(dataset_path, mode='rb') as file:
            with gzip.open(compressed_path, mode='wb') as compressed_file:
                shutil.copyfileobj(file, compressed_file)
                
        compressed_size = os.path.getsize(compressed_path)
        compressed_size_mb = compressed_size / (1024 * 1024)
        e.log(f'compressed file with {compressed_size_mb:.1f} MB')
        
    # ~ saving metadata
    # Alongside the actual dataset information, we also save metadata about the dataset which will later 
    # on be available to be fetched from the remote file share server and includes information such as 
    # a short description about the dataset, some relevant tags but also automatically determined information 
    # such as the number of elements in the dataset and the number of 
    e.log('saving metadata...')
        
    # first of all we need to construct the actual metadata dict.
    example_graph = graphs[0]
    metadata: dict = {
        'compounds': len(graphs),
        'targets': len(example_graph['graph_labels']),
        'target_type': [e.DATASET_TYPE],
        'raw': ['csv'],
        'sources': [],
    }
    
    # We also want to apply the user-defined overwrites to the metadata dict.
    metadata.update(e.METADATA)
    
    # Finally we save the metadata in yml format to the experiment archive folder.
    metadata_path = os.path.join(e.path, 'metadata.yml')
    with open(metadata_path, 'w') as metadata_file:
        yaml.dump(metadata, metadata_file)
        
    e.log(f'saved metadata @ {metadata_path}')
        
    # :hook after_save:
    #       Action hook that is called after the graph dataset itself has been saved to the disk inside 
    #       the experiment archive folder. Receives the dataset in the form of the index_data_map and the 
    #       list of graphs as parameters.
    e.apply_hook(
        'after_save',
        index_data_map=dataset,
        graphs=graphs,
    )

experiment.run_if_main()