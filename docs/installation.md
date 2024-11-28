---
title: Installation
---

# Installation

You can install the latest stable version from the Python Package Index (PyPi) like this:

```bash
uv pip install chem_mat_data
```

!!! Warning

    The stable version of the package is *not* yet officially released. For the time being, please use the source installation
    from the github repository.

Alternatively, you can install the latest development version of the package directly from the Github
repository:

```bash
uv pip install git+https://github.com/the16thpythonist/chem_mat_data
```

**Check your installation.** Installing the package should provide access to the ``cmdata`` command line interface.
You can check this with the following command, which should print the current version of the ``chem_mat_data`` package:

```bash
cmdata --version
```