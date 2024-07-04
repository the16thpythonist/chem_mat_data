import os
import time
import csv
import datetime
import multiprocessing

import msgpack
import numpy as np
import matplotlib.pyplot as plt
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

import chem_mat_data._typing as typc
from chem_mat_data.processing import MoleculeProcessing
from chem_mat_data.data import default, ext_hook
from chem_mat_data.data import save_graphs
from chem_mat_data.utils import SCRIPTS_PATH

# == SOURCE PARAMETERS ==

SOURCE_PATH: str = os.path.join(SCRIPTS_PATH, 'assets', '_test.csv')
TARGET_COLUMNS: list[str] = ['target']
DATASET_TYPE: str = 'regression'

# == PROCESSING PARAMETERS ==

DATASET_NAME: str = 'dataset'

# == EXPERIMENT PARAMETERS ==

__DEBUG__ = True
__TESTING__ = False

experiment = Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)


class ProcessingWorker(multiprocessing.Process):
    
    def __init__(self,
                 input_queue: multiprocessing.Queue,
                 output_queue: multiprocessing.Queue,
                 *args,
                 ):
        super(ProcessingWorker, self).__init__()

        self.input_queue = input_queue
        self.output_queue = output_queue

        self.processing = MoleculeProcessing()
        
    def run(self):
        
        for data in iter(self.input_queue.get, None):
            
            smiles = data['smiles']
            graph: typc.GraphDict = self.processing.process(smiles)
            
            graph['graph_labels'] = np.array(data['targets'])
            experiment.apply_hook(
                'add_graph_metadata',
                data=data,
                graph=graph,
            )
            
            graph_encoded = msgpack.packb(graph, default=default)            
            self.output_queue.put(graph_encoded)


@experiment.hook('add_graph_metadata')
def add_graph_metadata(e: Experiment,
                       data: dict,
                       graph: typc.GraphDict
                       ) -> dict:
    print("this works?")


@experiment.hook('load_dataset')
def load_dataset(e: Experiment) -> dict[int, dict]:
    
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
            
        return dataset
    
    e.log(f'dataset @ "{e.SOURCE_PATH}" not found!')


@experiment
def experiment(e: Experiment):
    
    e.log('starting experiment...')
    
    e.log('creating processing workers...')
    input_queue = multiprocessing.Queue(maxsize=1000)
    output_queue = multiprocessing.Queue(maxsize=1000)
    workers = []
    for _ in range(os.cpu_count()):
        worker = ProcessingWorker(
            input_queue=input_queue,
            output_queue=output_queue,
        )
        worker.start()
        workers.append(worker)
        e.log(f' * started worker')
    
    e.log('loading dataset...')
    dataset: dict[int, dict] = e.apply_hook(
        'load_dataset'
    )
    num_elements = len(dataset)
    e.log(f'loaded dataset with {num_elements} elements')

    e.log('processing dataset...')
    indices = list(dataset.keys())
    graphs = []
    
    start_time = time.time()
    count = 0
    prev_count = 0
    
    while len(indices) != 0 or not input_queue.empty() or not output_queue.empty():
        
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

    # ~ stopping the workers
    for worker in workers:
        worker.terminate()
        worker.join()

    # ~ saving the dataset into the msgpack format
    e.log('saving the dataset...')
    dataset_path = os.path.join(e.path, e.DATASET_NAME + '.mpack')
    save_graphs(graphs, dataset_path)
    
    file_size = os.path.getsize(dataset_path)
    file_size_mb = file_size / (1024 * 1024)
    e.log(f'wrote file with {file_size_mb:.1f} MB')

experiment.run_if_main()