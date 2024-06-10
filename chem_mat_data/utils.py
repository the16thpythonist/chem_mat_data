"""
This module is used to collect common utility functions that thematically don't fit elsewhere.
"""
import os
import math
import pathlib

import requests
import jinja2 as j2
import rdkit.Chem as Chem
from rich.console import Console, ConsoleOptions, RenderResult
from rich.text import Text
from rich.style import Style
from rich.progress import Progress

# GLOBAL VARIABLES
# ================

# This is the absolute string path to the folder that contains all the code modules. Use this whenever 
# you need to access files from within the project folder.
PATH: str = pathlib.Path(__file__).parent.absolute()

# Based on the package path we can now define the more specific sub paths
VERSION_PATH: str = os.path.join(PATH, 'VERSION')

TEMPLATE_PATH = os.path.join(PATH, 'templates')
TEMPLATE_ENV = j2.Environment(loader=j2.FileSystemLoader(TEMPLATE_PATH))

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
    total_size = int(response.headers.get('content-length',0))

    # Open the file..
    with open(destination, 'wb') as file, Progress() as progress:
        task = progress.add_task('[cyan]Downloading...', total = total_size)
        bytes_written = 0
        # .. and iterate over the contents of the file in little chunks.
        for chunk in response.iter_content(chunk_size=1024):
            # We check if it actually contains data and then write it to a file in a destination folder
            if chunk:
                file.write(chunk)
                bytes_written += len(chunk)
                progress.update(task, advance=len(chunk))
        
        


def mol_from_smiles(smiles: str
                    ) -> Chem.Mol:
    """
    Given the `smiles` string of a molecule, this function will convert that SMILES string into a valid
    RDKit molecule object.
    
    :raises ValueError: If the SMILES string cannot be converted into a valid RDKit molecule object aka 
        if the SMILES string is invalid and the conversion returns None.
    
    :param smiles: The SMILES string of the molecule to be converted.
    
    :returns: The RDKit molecule object corresponding to the given SMILES string
    """
    mol = Chem.MolFromSmiles(smiles)
    # 06.02.24 
    # This is important to check here because the conversion function sometimes fails silently by just returning 
    # None instead of raising an exception. This is a problem because we need to know if the conversion was
    # successful or not for downstream applications of this function.
    if mol is None:
        raise ValueError(f'Could not convert SMILES string "{smiles}" into a valid RDKit molecule object!')

    return mol


class RichMixin:
    """
    Implements a mixin/interface for objects whose string representations should be rendered using 
    the "rich" library.
    
    Specifically, this mixin requires a subclass to implement the magic method "__rich_console__". 
    This method should implement a generator of rich renderable objects which together form the 
    final string rendering of the custom subclass.
    
    Based on this custom implementation of the __rich_console__ method, this mixin provides a default 
    implementation of the __str__ method which converts the rich renderable objects into a string
    representation and returns it.
    
    References:
    
    https://rich.readthedocs.io/en/stable/protocol.html
    """
    
    def __str__(self):
        console = Console()
        with console.capture() as capture:
            console.print(self)
        
        return capture.get()
    
    def __rich__console__(self, 
                          console: Console, 
                          console_options: ConsoleOptions
                          ) -> RenderResult:
        """
        This function is called by the Rich library when the object is being printed to the console. It 
        allows to customize the output of the object in the console.
        
        This method will have to be implemented by the subclass and should return a generator of rich
        renderable objects which together form the final string representation of the object.
        
        :param console: The console object which is used to print the object to the console.
        :param console_options: The options that are used to format the output of the object.
        
        :returns: A generator of rich renderable objects which together form the final string representation
            of the object.
        """
        raise NotImplementedError('This method has to be implemented by the subclass!')
    
    
class RichHeader(RichMixin):
    
    def __init__(self, title: str, rule='=', color='white'):
        self.title = title
        self.rule = rule
        self.color = color
        
    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        rule_width: int = math.floor((options.max_width - len(self.title) - 2) / 2)
        
        style = Style(bold=True, color=self.color)
        yield Text(f'{self.rule * rule_width} {self.title} {self.rule * rule_width}', style=style)

