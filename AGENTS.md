# Guide for AI Agents

## Overview

This project implements the *Chemistry and Materials Database* (ChemMatData) which is supposed to be a database 
containing various datasets related to chemistry and materials science in a format which is accessible for 
the training of neural network models and specifically graph neural networks (GNNs). The project is designed to 
have an intuitive python interface for easy downloading and processing of the datasets as well as a 
command line interface (CLI) to interact with the remote database.

## Project Structure

- `/chem_mat_data`: Python source files
    - `/chem_mat_data/examples`: Example scripts which demonstrate how to use the project
    - `/chem_mat_data/scripts`: Pycomex experiment files which implement the conversion of the various datasets in their 
    native data formats into the commong graph format used by the project.
    - `/chem_mat_data/templates`: Jinja2 templates
- `/tests`: Pytest unit tests which are names "tests_" plus the name of the source python file
    - `/tests/assets`: Additional files etc. which are needed by some of the unittests
    - `/tests/artifacts`: Temp folder in which the tests save their results

## Documentation

Type hints should be used wherever possible.
Every function/method should be properly documented by a Docstring using the **ReStructuredText** documentation style.
The doc strings should start with a brief summary of the function, followed by the parameters as illustrated by this example:

```python
def multiply(a: float, b: float) -> float:
    """
    Returns the product of ``a`` and ``b``.

    :param a: first float input value
    :param b: second float input value
    
    :returns: The float result
    """
    return a * b
```

## Computational Experiments

This project uses the [pycomex](https://github.com/the16thpythonist/pycomex) micro-framework for implementing and executing computational experiments.
A computational experiment can be defined according to the following example:

```python
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

# :param PARAM1:
#       Description for the parameter...
PARAM1: int = 100

__DEBUG__ = True # enables/disables debug mode

experiment = Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)

@experiment.hook('util_function')
def util_function(e: Experiment, param: int):
    return param ** 2

@experiment
def experiment(e: Experiment):
    
    # automatically pushes to log file as well as to the stdout
    # parameters are accessed as attributes of the Experiment object
    e.log(f'this is a parameter value: {e.PARAM1}')

    # Store values into the experiment automatically
    e['value'] = value

    # The e.path contains the path to the artifacts folder of the 
    # current experiment run.
    e.log('artifacts path: {}')

    # Easily store figures to the artifacts folder
    fig: plt.Figure = ...
    e.commit_fig(fig, 'figure1.png')

    # using hooks instead of plain functions
    result = e.apply_hook(
        'util_function',
        param=20,
    )


experiment.run_if_main()
```

## Code Convention

1. functions with many parameters should be split like this:

```python
def function(arg1: List[int],
             arg2: List[float],
             **kwargs
             ) -> float:
    # ...

```

## Testing

Unittests use `pytest` in the `/tests` folder with this command

```bash
pytest -q -m "not localonly"
```

## Pull Requests / Contributing

Pull Requests should always start with a small summary of the changes and a list of the changed files.
Additionally a PR should contain a small summary of the tests results.