import os
import glob
import csv

def parse_xyz_file(filepath):
    """
    Parse a QM9 XYZ file and extract properties and SMILES.
    """
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    
    n_atoms = int(lines[0])
    
    properties_line = lines[1].split('\t')
    properties_line = [p.strip() for p in properties_line if p.strip()]
    
    tag_and_index = properties_line[0].split()
    index = int(tag_and_index[1]) if len(tag_and_index) > 1 else None
    
    A = float(properties_line[1]) if len(properties_line) > 1 else None
    B = float(properties_line[2]) if len(properties_line) > 2 else None
    C = float(properties_line[3]) if len(properties_line) > 3 else None
    mu = float(properties_line[4]) if len(properties_line) > 4 else None
    alpha = float(properties_line[5]) if len(properties_line) > 5 else None
    homo = float(properties_line[6]) if len(properties_line) > 6 else None
    lumo = float(properties_line[7]) if len(properties_line) > 7 else None
    gap = float(properties_line[8]) if len(properties_line) > 8 else None
    r2 = float(properties_line[9]) if len(properties_line) > 9 else None
    zpve = float(properties_line[10]) if len(properties_line) > 10 else None
    U0 = float(properties_line[11]) if len(properties_line) > 11 else None
    U = float(properties_line[12]) if len(properties_line) > 12 else None
    H = float(properties_line[13]) if len(properties_line) > 13 else None
    G = float(properties_line[14]) if len(properties_line) > 14 else None
    Cv = float(properties_line[15]) if len(properties_line) > 15 else None
    
    smiles_line = lines[n_atoms + 3] if len(lines) > n_atoms + 3 else ''
    smiles_parts = smiles_line.split('\t')
    smiles_parts = [s.strip() for s in smiles_parts if s.strip()]
    smiles_gdb9 = smiles_parts[0] if len(smiles_parts) > 0 else ''
    smiles_relaxed = smiles_parts[1] if len(smiles_parts) > 1 else smiles_gdb9
    
    return {
        'index': index,
        'smiles': smiles_relaxed,
        'A_GHz': A,
        'B_GHz': B,
        'C_GHz': C,
        'mu_Debye': mu,
        'alpha_Bohr3': alpha,
        'homo_Hartree': homo,
        'lumo_Hartree': lumo,
        'gap_Hartree': gap,
        'r2_Bohr2': r2,
        'zpve_Hartree': zpve,
        'U0_Hartree': U0,
        'U_Hartree': U,
        'H_Hartree': H,
        'G_Hartree': G,
        'Cv_cal_mol_K': Cv
    }

def main():
    base_dir = '/media/ssd/Programming/chem_mat_data/_test_'
    output_csv = os.path.join(base_dir, 'qm9_dataset.csv')
    
    print("Starting to parse QM9 dataset...")
    print(f"Looking for XYZ files in: {base_dir}")
    
    xyz_files = sorted(glob.glob(os.path.join(base_dir, 'dsgdb9nsd_*.xyz')))
    print(f"Found {len(xyz_files)} XYZ files")
    
    if len(xyz_files) == 0:
        print("ERROR: No XYZ files found!")
        return
    
    fieldnames = [
        'index', 'smiles',
        'A_GHz', 'B_GHz', 'C_GHz',
        'mu_Debye', 'alpha_Bohr3',
        'homo_Hartree', 'lumo_Hartree', 'gap_Hartree',
        'r2_Bohr2', 'zpve_Hartree',
        'U0_Hartree', 'U_Hartree', 'H_Hartree', 'G_Hartree',
        'Cv_cal_mol_K'
    ]
    
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for i, xyz_file in enumerate(xyz_files, 1):
            try:
                data = parse_xyz_file(xyz_file)
                writer.writerow(data)
                
                if i % 10000 == 0:
                    print(f"Processed {i}/{len(xyz_files)} files...")
            except Exception as e:
                print(f"Error processing {xyz_file}: {e}")
    
    print(f"\nCompleted! CSV file saved to: {output_csv}")
    print(f"Total molecules processed: {len(xyz_files)}")

if __name__ == '__main__':
    main()
