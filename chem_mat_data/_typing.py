import typing as t

import numpy as np


"""
Type alias for specific dictionary format to represent graph structures.

The GraphDict format is generally designed to be flexible - being able to dynamically attach 
additional properties - and to be JSON serializable.
"""
GraphDict = t.Dict[str, t.Union[list, np.ndarray]]



