"""
This module is used to collect common utility functions that thematically don't fit elsewhere.
"""
import os
import pathlib

import requests
# GLOBAL VARIABLES
# ================

# This is the absolute string path to the folder that contains all the code modules. Use this whenever 
# you need to access files from within the project folder.
PATH: str = pathlib.Path(__file__).parent.absolute()

# Based on the package path we can now define the more specific sub paths
VERSION_PATH: str = os.path.join(PATH, 'VERSION')


# MISC FUNCTIONS
# ==============

def get_version(path: str = os.path.join(PATH, 'VERSION')) -> str:
    """
    This function returns the string representation of the package version.
    """
    with open(path, mode='r') as file:
        content = file.read()
        version = content.replace(' ', '').replace('\n', '')
        
    return version

def download_dataset(url, destination):
    # Makes a request to the above specified URL
    response = requests.get(url, stream = True)

    # Open the file..
    with open(destination, 'wb') as file:
        # .. and iterate over the contents of the file in little chunks.
        for chunk in response.iter_content(chunk_size=1024):
            # We check if it actually contains data and then write it to a file in a destination folder
            if chunk:
                file.write(chunk)



