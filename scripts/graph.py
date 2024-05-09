import numpy as np
import json
import pandas as pd 
from rdkit import Chem
from rdkit.Chem import Draw
# A script that transforms smiles into a graph in dictionary style
def read_smiles(csv_file):
    file = pd.read_csv(csv_file,skiprows=0)
    smiles_column_index = 0
    smiles_list = file.iloc[:, smiles_column_index].tolist()
    return smiles_list

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        #mol = Chem.RemoveHs(mol)

        #Creat dictionaries for nodes and edges
        nodes_dict = {}
        edges_dict = {}
        
        #This commented method gives errors..
        """
        for atom in mol.GetAtoms():
            nodes_dict[atom.GetIdx()] = {'element': atom.GetSymbol(), 'position': list(atom.GetIdx())}

        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            #Gives us the bond type as a numerical value
            bond_type = bond.GetBondTypeAsDouble()
            #We probably work if undirected graphs
            edges_dict[(start, end)] = {'bond_type': bond_type, 'length': np.linalg.norm(np.array(nodes_dict[start]['position']) - np.array(nodes_dict[end]['position']))}
            edges_dict[(end, start)] = {'bond_type': bond_type, 'length': np.linalg.norm(np.array(nodes_dict[start]['position']) - np.array(nodes_dict[end]['position']))}

        """
        # use this simpler method for now

        # Add nodes (atoms) to nodes_dict
        for atom in mol.GetAtoms():
            atom_index = atom.GetIdx()
            atom_symbol = atom.GetSymbol()
            nodes_dict[atom_index] = {'atom_symbol': atom_symbol}

        # Add edges (bonds) to edges_dict
        for bond in mol.GetBonds():
            begin_atom_index = bond.GetBeginAtomIdx()
            end_atom_index = bond.GetEndAtomIdx()
            bond_type = str(bond.GetBondType())
            if begin_atom_index not in edges_dict:
                edges_dict[begin_atom_index] = {}
            edges_dict[begin_atom_index][end_atom_index] = bond_type
            if end_atom_index not in edges_dict:
                edges_dict[end_atom_index] = {}
            edges_dict[end_atom_index][begin_atom_index] = bond_type

        mol_dict = {'nodes': nodes_dict, 'edges': edges_dict}

        return mol_dict
    else:
        return None

#Augment the file path to your liking
file = 'toxcast_data.csv'
file_path = '/home/mohit/datasets/' + file

smiles_list = read_smiles(file_path)


#Let's turn the graph into a json file
all_graphs = []
for smiles in smiles_list:
   mol_dict = smiles_to_graph(smiles)
   if mol_dict:
       all_graphs.append(mol_dict)

file = file.split('.')[0]
output_file = '/home/mohit/smiles_graph/' + file + '.json'
with open(output_file, 'w',) as f:
    json.dump(all_graphs, f, indent=4)
    print("Graphs saved to", output_file)
    
