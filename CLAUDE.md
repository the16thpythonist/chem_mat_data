# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ChemMatData is a Python package that provides unified access to chemistry and material science datasets for machine learning applications, specifically designed for training graph neural networks (GNNs). The package offers both raw (CSV/pandas) and processed (graph) formats of datasets.

## Core Architecture

- **CLI Interface**: Two main commands - `cmdata` (data operations) and `cmmanage` (management)
- **Data Loading**: Unified API through `load_smiles_dataset()` and `load_graph_dataset()` functions
- **Graph Processing**: Custom graph representation format with conversion utilities for PyTorch Geometric
- **Remote Storage**: Nextcloud-based file sharing system for dataset distribution
- **Caching**: Local caching system for downloaded datasets

## Virtual Environment

This project uses a virtual environment which should be activated before running any command line tools or scripts.
To activate the virtual environment, run:

```bash
source .venv/bin/activate
```

## Development Guidelines

### Docstrings

Docstrings should use the ReStructuredText (reST) format. This is important for generating documentation and for consistency across the codebase. Docstrings should always start with a one-line summary followed by a more detailed paragraph - also including usage examples, for instance. If appropriate, docstrings should not only describe a method or function but also shed some light on the design rationale.

Documentation should also be *appropriate* in length. For simple functions, a brief docstring is sufficient. For more complex functions or classes, more detailed explanations and examples should be provided.

An example docstring may look like this:

```python

def multiply(a: int, b: int) -> int:
    """
    Multiply two integers `a` and `b`.

    This function takes two integers as input and returns their product.

    Example:
    
    ... code-block:: python

        result = multiply(3, 4)
        print(result)  # Output: 12

    :param a: The first integer to multiply.
    :param b: The second integer to multiply.

    :return: The product of the two integers.
    """
    return a * b

```

### Key Modules

- `main.py`: Core dataset loading functions and API
- `cli.py`: Command line interface using rich_click
- `data.py`: Data serialization/deserialization with msgpack
- `web.py`: Nextcloud file share client
- `graph.py`: Graph conversion utilities (to PyG, Jraph)
- `processing.py`: Dataset processing and validation
- `config.py`: Configuration management

## Development Commands

### Testing
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_cli.py
```

### Building
```bash
# Build package
uv build

# Install in development mode
pip install -e .
```

## Environment Setup

The project requires a `.env` file with:
```
url=<nextcloud_server_url>
```

## Package Structure

- Main package: `chem_mat_data/`
- Tests: `tests/`
- Scripts: `chem_mat_data/scripts/`: The pycomex experiment modules which implement the processing pipelines for each of the datasets.
- Configuration: Uses `pyproject.toml` with hatchling build system

## Data Format

The package works with two main data formats:
1. **Raw**: Pandas DataFrames with SMILES strings and target properties
2. **Graph**: Dictionary format with node/edge features, labels, and metadata

Graph format includes:
- `node_indices`, `node_attributes`
- `edge_indices`, `edge_attributes` 
- `graph_labels`, `node_labels`, `edge_labels`
- Metadata fields

## Dependencies

Key dependencies include:
- RDKit for chemical processing
- ASE for atomic structure handling
- NumPy/Pandas for data manipulation
- Rich/Rich-Click for CLI interface
- PyComex for experiment management
- msgpack for efficient serialization