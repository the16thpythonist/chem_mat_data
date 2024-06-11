"""
This module implements the saving and loading of the datasets from and to the persistent 
file storage representations.
"""
import os
import pathlib

import msgpack
import numpy as np

from chem_mat_data._typing import GraphDict


def default(obj):
    
    if isinstance(obj, np.ndarray):
        return msgpack.ExtType(1, obj.tobytes())
    
    raise TypeError("Unknown type: %r" % (obj,))


def ext_hook(code, data):
    
    if code == 1:
        return np.frombuffer(data, dtype=np.float32)
    
    return data



def save_graphs(graphs: list[GraphDict],
                path: str,
                ) -> None:

    packed = msgpack.packb(graphs, default=default)
    with open(path, mode='wb') as file:
        file.write(packed)


def load_graphs(path: str) -> list[GraphDict]:
    
    with open(path, mode='rb') as file:
        content: bytes = file.read()
        return msgpack.unpackb(content, ext_hook=ext_hook)