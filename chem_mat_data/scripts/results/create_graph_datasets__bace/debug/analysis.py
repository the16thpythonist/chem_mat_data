#! /usr/bin/env python3
"""
This python module was automatically generated.

This module can be used to perform analyses on the results of an experiment which are saved in this archive
folder, without actually executing the experiment again. All the code that was decorated with the
"analysis" decorator was copied into this file and can subsequently be changed as well.
"""
import os
import pathlib

# Useful imports for conducting analysis
from pycomex.functional.experiment import Experiment

# Importing the experiment

from create_graph_datasets import *

from experiment_code import *

PATH = pathlib.Path(__file__).parent.absolute()
CODE_PATH = os.path.join(PATH, 'experiment_code.py')
experiment = Experiment.load(CODE_PATH)
experiment.analyses = []


experiment.execute_analyses()