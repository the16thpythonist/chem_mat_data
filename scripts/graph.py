import numpy as np
import json
import pandas as pd 
from rdkit import Chem
from rdkit.Chem import Draw
#Transforms smiles into graphs
def read_smiles(csv_file):
    file = pd.read_csv(csv_file,skiprows=0)
    smiles_column_index = 0
    smiles_list = file.iloc[:, smiles_column_index].tolist()
    return smiles_list

def smiles_to_graph(smiles_list):
    graph_dicts = []
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            # Initialize graph dictionary
            graph_dict = {
                'graph_index': i,
                'graph_smiles': smiles,
                'node_attributes': [],
                'edge_attributes': []
               # 'graph_labels': [graph_labels[i]] if graph_labels else []
            }
            
            # Extract node and edge attributes
            for atom in mol.GetAtoms():
                graph_dict['node_attributes'].append([atom.GetAtomicNum()])  # Atom type
            
            for bond in mol.GetBonds():
                start_idx = bond.GetBeginAtomIdx()
                end_idx = bond.GetEndAtomIdx()
                bond_type = bond.GetBondTypeAsDouble()
                graph_dict['edge_attributes'].append([bond_type, 0])  # Bond type (assuming no direction)
            
            graph_dicts.append(graph_dict)
    return graph_dicts

file = 'clintox.csv'
file_path = file

smiles_list = read_smiles(file_path)
graph_dict = smiles_to_graph(smiles_list)

file = file.split('.')[0]
output_file = '/home/mohit/smiles_graph/' + file + '.json'
with open(output_file, 'w',) as f:
    json.dump(graph_dict, f, indent=4)
    print("Graphs saved to", output_file)
