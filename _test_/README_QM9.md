# QM9 Dataset

## Overview
This dataset contains quantum chemistry properties for 133,885 small organic molecules from the QM9 database, as published in:

**Ramakrishnan, R., Dral, P., Rupp, M., & von Lilienfeld, O. A. (2014). Quantum chemistry structures and properties of 134 kilo molecules. Scientific Data, 1, 140022.**
https://doi.org/10.1038/sdata.2014.22

## Source
Downloaded from Figshare: https://figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904

## Dataset Description
The QM9 dataset contains neutral organic molecules with up to 9 heavy atoms (C, N, O, F) plus hydrogens. All molecular geometries were relaxed and properties calculated at the DFT/B3LYP/6-31G(2df,p) level of theory.

## CSV Format
The dataset has been processed into CSV format with the following columns:

- **index**: Molecule identifier (1-133885)
- **smiles**: SMILES string representation of the molecule
- **A_GHz**: Rotational constant A (GHz)
- **B_GHz**: Rotational constant B (GHz)
- **C_GHz**: Rotational constant C (GHz)
- **mu_Debye**: Dipole moment (Debye)
- **alpha_Bohr3**: Isotropic polarizability (Bohr³)
- **homo_Hartree**: Energy of highest occupied molecular orbital (Hartree)
- **lumo_Hartree**: Energy of lowest unoccupied molecular orbital (Hartree)
- **gap_Hartree**: HOMO-LUMO gap (Hartree)
- **r2_Bohr2**: Electronic spatial extent (Bohr²)
- **zpve_Hartree**: Zero point vibrational energy (Hartree)
- **U0_Hartree**: Internal energy at 0 K (Hartree)
- **U_Hartree**: Internal energy at 298.15 K (Hartree)
- **H_Hartree**: Enthalpy at 298.15 K (Hartree)
- **G_Hartree**: Free energy at 298.15 K (Hartree)
- **Cv_cal_mol_K**: Heat capacity at 298.15 K (cal/(mol·K))

## Statistics
- Total molecules: 133,885
- File size: 21 MB
- Format: CSV with header

## Citation
If you use this dataset, please cite the original publication:
```
Ramakrishnan, R., Dral, P. O., Rupp, M., & von Lilienfeld, O. A. (2014). 
Quantum chemistry structures and properties of 134 kilo molecules. 
Scientific Data, 1, 140022. 
https://doi.org/10.1038/sdata.2014.22
```

## License
The dataset is released under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
