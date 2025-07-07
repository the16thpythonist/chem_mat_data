from typing import Dict
from rdkit import Chem
import pandas as pd
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

import numpy as np

DATASET_NAME: str = 'synth_binary_local'

__TESTING__ = False
__DEBUG__ = True

experiment = Experiment.extend(
    'create_graph_datasets.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)

# This dictionary defines all the relevant SMARTS patterns that we want to use later on 
# to check for local patterns. The keys of this dict are human-readable names for the patterns 
# and the values are the actual SMARTS strings.
SMARTS_PATTERNS: Dict[str, str] = {
    'benzene'           : 'c1ccccc1',
    'nitro'             : '[NX3](=O)=O',
    'halogen'           : '[F,Cl,Br,I]',
    'carboxylic_acid'   : 'C(=O)[OH]',
    'tertiary_amine'    : '[$([NX3]([#6])([#6])[#6])]',  # N with three C neighbors
    'sulfoxide'         : '[#16](=O)',
    'ether'             : 'C-O-C',
    'pyridine'          : 'n1ccccc1',
    'furan'             : 'o1cccc1',
    'ketone'            : 'C(=O)[#6]',
    'alcohol'           : '[OX2H]',
    'thiol'             : '[SX2H]'
}

# compile the SMARTS patterns
SMARTS_PATTERNS_COMPILED = {
    name: Chem.MolFromSmarts(smarts)
    for name, smarts in SMARTS_PATTERNS.items()
}


def extract_motifs(mol: Chem.Mol, smarts_patterns: dict = SMARTS_PATTERNS_COMPILED) -> dict:
    """Return a dict mapping motif name → bool for whether
       that SMARTS appears in the molecule."""
    return {
        name: bool(mol.HasSubstructMatch(pat))
        for name, pat in smarts_patterns.items()
    }


def motif_label(mol: Chem.Mol, smarts_patterns: dict) -> int:
    """
    A rather complicated rule:
    
      Label = 1 if any of these three big clauses holds:
      
        Clause 1:
          benzene AND (nitro OR (halogen AND carboxylic_acid))
        
        Clause 2:
          tertiary_amine AND NOT sulfoxide
          AND (ether OR halogen)
        
        Clause 3:
          (pyridine OR furan) AND ketone AND NOT alcohol
        
      Otherwise 0.
    """
    m = extract_motifs(mol, smarts_patterns=smarts_patterns)
    
    # These are the individual clauses that will suffice to determine the label of the molecule
    # if any of these claususes are true, then the label is 1 otherwise 0.
    c1 = m['benzene'] and (m['nitro'] or (m['halogen'] and m['carboxylic_acid']))
    c2 = m['tertiary_amine'] and not m['sulfoxide'] and (m['ether'] or m['halogen'])
    c3 = (m['pyridine'] or m['furan']) and m['ketone'] and not m['alcohol']
    
    return int(c1 or c2 or c3)


@experiment.hook('add_graph_metadata', default=False, replace=True)
def add_graph_metadata(e: Experiment, data: dict, graph: dict) -> dict:
    """
    We add the compound id for identification and the molecular weight
    """
    pass #graph['graph_id'] = data['ID']


@experiment.hook('load_dataset', default=False, replace=True)
def load_dataset(e: Experiment) -> dict[int, dict]:
    df = pd.read_csv('/media/data/Downloads/zinc_250k.csv')
    e.log(f'dataset of size {len(df)} loaded')
    e.log(df.head())

    dataset: dict[int, dict] = {}
    index: int = 0
    for c, data in enumerate(df.to_dict('records')):
        data['smiles'] = data['smiles']
        
        # == MOLECULE FILTERS ==
        # We don't want to use compounds with '.' in the smiles (separate molecules)
        if '.' in data['smiles']:
            continue
        
        # We don't want to use compounds that only consist of a single atom
        mol = Chem.MolFromSmiles(data['smiles'])
        if not mol:
            continue
        
        # We also don't want to accept "molecules" that are essentially just individual atoms
        if len(mol.GetAtoms()) < 2:
            continue
        
        # == TARGETS ==
        
        # Use the previously defined function to calculate the binary target label based on the 
        # local smarts patterns.
        label: int = motif_label(mol, SMARTS_PATTERNS_COMPILED)
        
        # In the end we dont need the integer but we need a one-hot encoded vector to represent the 
        # target classification label, which we construct here.
        graph_labels = np.array([1 - label, label], dtype=np.float32)
        
        data['targets'] = graph_labels
        dataset[index] = data
        
        index += 1
        
        if c % 1000 == 0:
            print(f' * {c} elements loaded')

    e.log('returning dataset...')
    return dataset

experiment.run_if_main()