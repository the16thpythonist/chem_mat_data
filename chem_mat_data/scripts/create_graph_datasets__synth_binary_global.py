from typing import Dict, Any
from rdkit import Chem
import pandas as pd
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, QED
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

# This dict defines the 
DRUGLIKENESS_CRITERIA: Dict[str, Any] = {
    'MW':        ('<=', 500.0),
    'LogP':      ('<=', 5.0),
    'HBD':       ('<=', 5),
    'HBA':       ('<=', 10),
    'TPSA':      ('<=', 140.0),
    'RotB':      ('<=', 10),
    'QED':       ('>=', 0.50),
}

def compute_descriptors(mol: Chem.Mol) -> dict:
    """
    Compute the set of global descriptors used for druglikeness.
    """
    
    return {
        'MW':   Descriptors.MolWt(mol),
        'LogP': Crippen.MolLogP(mol),
        'HBD':  rdMolDescriptors.CalcNumHBD(mol),
        'HBA':  rdMolDescriptors.CalcNumHBA(mol),
        'TPSA': rdMolDescriptors.CalcTPSA(mol),
        'RotB': rdMolDescriptors.CalcNumRotatableBonds(mol),
        'QED':  QED.qed(mol),
    }

def druglikeness_label(mol: Chem.Mol, criteria: dict) -> int:
    """
    Return 1 if molecule meets *all* drug-likeness criteria, else 0.
    
      - MW ≤ 500
      - LogP ≤ 5
      - HBD ≤ 5
      - HBA ≤ 10
      - TPSA ≤ 140
      - RotB ≤ 10
      - QED ≥ 0.50
    """
    desc = compute_descriptors(mol)
    for key, (op, thresh) in criteria.items():
        val = desc[key]
        if   op == '<=' and not (val <=  thresh): 
            return 0
        elif op == '>=' and not (val >=  thresh): 
            return 0
    return 1


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
        
        # Here we invoke the custom function that we've defined to obtain the binary target label.
        label: int = druglikeness_label(mol, DRUGLIKENESS_CRITERIA)
        
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