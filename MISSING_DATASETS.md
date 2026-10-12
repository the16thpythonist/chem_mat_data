# Missing Datasets — Confirmed Positives Not in ChemMatData

Datasets confirmed as genuine molecular/materials dataset releases by the `agent identify` validation but not yet represented in the ChemMatData collection. Sourced from the upstream `discovery_positives.jsonl` training data.

**Date:** 2026-04-12
**Total confirmed positives:** 104
**Already in ChemMatData:** ~7 (QM9, Ames, AqSolDB, DUD-E, ZINC, OQMD-related, AFLOW-related)
**Missing:** ~97

## High-Priority — Molecular Property Datasets (GNN-ready)

These datasets contain molecules with SMILES/structures and computed or measured properties, directly suitable for ChemMatData's graph format:

| # | Dataset / Paper | Data Type | Size | URL |
|---|-----------------|-----------|------|-----|
| 2 | PFAS database (structure-function relationships) | SMILES + physicochemical properties | 1000s of PFAS compounds | https://doi.org/10.1038/s41597-021-00798-x |
| 3 | Polyuniverse | Polymer SMILES + thermal/mechanical properties | >1M hypothetical polymers | https://doi.org/10.1039/D4DD00196F |
| 7 | Open-source QSAR pKa models | SMILES + experimental pKa values | 7,912 chemicals | https://doi.org/10.1186/s13321-019-0384-1 |
| 11 | Frag20 / Plati20 / CSD20 | 3D molecular geometries + DFT properties | >500K molecules | https://doi.org/10.1021/acs.jcim.1c00007 |
| 29 | pKa prediction via Atomic Sensitivity Analysis | SMILES + pKa datasets | Multiple datasets via Zenodo | https://doi.org/10.1021/acs.jcim.4c01472 |
| 30 | Nucleophilicity/Electrophilicity ML | SMILES + QM-calculated MCA/MAA | ~50K molecules | https://doi.org/10.1039/D4SC07297A |
| 32 | QO2Mol database | Molecular conformers + QM properties | 120K molecules, 20M conformers | https://doi.org/10.48550/arXiv.2410.19316 |
| 37 | ReactAIvate CRM dataset | Reaction SMILES + mechanistic steps | 100K elementary steps | https://doi.org/10.48550/arXiv.2407.10090 |
| 72 | Amino acid conformers | Conformer geometries + DFT energies | 45,892 conformers | https://doi.org/10.1038/sdata.2016.9 |
| 87 | Ames mutagenicity benchmark | SMILES + Ames test results | ~6,500 compounds | https://doi.org/10.1021/ci900161g |
| 88 | PubChemQC | Electronic structure data | Millions of molecules | https://doi.org/10.1021/acs.jcim.7b00083 |
| 90 | PEPCONF | Peptide conformational energies | Diverse peptide set | https://doi.org/10.1038/sdata.2018.310 |
| 96 | P-Glycoprotein Transport QSAR | SMILES + transport classification | 332 compounds | https://doi.org/10.1021/ci3002809 |
| 99 | QM40 | QM properties for realistic molecules | 40 representative molecules | https://doi.org/10.1038/s41597-024-04206-y |
| 101 | Ligand-Based Drug Discovery benchmark | SMILES + bioactivity | Multiple target datasets | https://doi.org/10.1021/acs.jcim.5c00374 |
| 116 | BORON1000 (C-H Borylation selectivity) | Reaction SMILES + selectivity | 1000 reactions | https://doi.org/10.1021/acs.jcim.4c00137 |
| 120 | PPARgamma binding affinity QSAR | SMILES + binding data | Large dataset | https://doi.org/10.1021/acs.est.0c07040 |
| 123 | Contaminant Reactivity QSAR | SMILES + oxidant reactivity | Multiple oxidant datasets | https://doi.org/10.1021/acs.est.1c04883 |
| 125 | P-glycoprotein substrate models | SMILES + molecular descriptors | 332 structures | https://doi.org/10.1021/ci2001583 |
| 129 | Lipophilicity of peptide anions | Peptide structures + Gibbs energies | Systematic measurements | https://doi.org/10.1021/jp034387e |
| 132 | Brain exposure relationships | Drug structures + Kp,uu values | Novel unbound drug data | https://doi.org/10.1021/jm901036q |
| 138 | Diverse drug dataset (correlation/benchmarking) | SMILES + lipophilicity/pKa/solubility/permeability | 24 diverse drugs | https://doi.org/10.1021/jm0506219 |
| 141 | Molecular Conformations for Virtual Screening | 3D conformer benchmark | Large-scale dataset | https://doi.org/10.3390/molecules28165982 |
| 142 | ACtriplet (activity cliffs) | SMILES + bioactivity pairs | Activity cliff dataset | https://doi.org/10.1016/j.jpha.2025.101317 |

## High-Priority — Materials Property Datasets

| # | Dataset / Paper | Data Type | Size | URL |
|---|-----------------|-----------|------|-----|
| 1 | BEAST DB (electrocatalyst properties) | Surface calculations + properties | >20K calculations | https://doi.org/10.1021/acs.jpcc.4c06826 |
| 5 | GMTKN55 database | Thermochemistry/kinetics/NCIs | 1505 relative energies | https://doi.org/10.1039/C7CP04913G |
| 6 | Barrier heights/enthalpies/rate coefficients | Reaction barriers + rate constants | ~12K reactions | https://doi.org/10.1038/s41597-022-01529-6 |
| 14 | Benchmark Crystalline Organic Semiconductors (BMCOS1) | Crystal structures + DFT properties | 67 semiconductors | https://doi.org/10.1021/acs.jctc.3c00861 |
| 17 | 50 benchmark datasets for materials informatics | Diverse materials properties | 50 datasets | https://doi.org/10.1016/j.dib.2021.107262 |
| 18 | Materials property prediction with UQ | Crystal materials + formation energy/band gap | 4 datasets | https://doi.org/10.1063/5.0133528 |
| 24 | Lattice thermal conductivity prediction | Experimental thermal conductivity | ~100 compounds | https://doi.org/10.1016/j.commatsci.2019.109155 |
| 25 | Hard-sphere packing optimization | Packing fractions + simulation data | 494K simulations | https://doi.org/10.1016/j.dib.2023.109487 |
| 34 | Isomer stability in small clusters | Cluster geometries + DFT properties | 44K isomers | https://doi.org/10.1039/D2MA01088G |
| 40 | Amorphous silicon hydride structures | a-Si:H configurations + properties | 2500+ structures | https://doi.org/10.1016/j.jnoncrysol.2023.122641 |
| 56 | Stillinger-Weber parameters for 2D crystals | Phonon spectra + force field params | 156 crystals | https://doi.org/10.5772/intechopen.71929 |
| 67 | TMC151 (transition metal reactions) | Reaction energies + barrier heights | 151 data points | https://doi.org/10.1021/acs.jctc.9b00239 |
| 84 | MOF water stability (WS24) | MOF structures + stability labels | 911 MOFs | https://doi.org/10.1021/jacs.4c05879 |
| 91 | Open Catalyst 2022 (OC22) | Oxide electrocatalyst structures + energies | Large-scale DFT | https://doi.org/10.1021/acscatal.2c05426 |
| 104 | MOFSimplify (MOF stability) | MOF structures + stability data | 3000 MOFs | https://doi.org/10.1038/s41597-022-01181-0 |
| 110 | Strain effects on adsorption energies | Catalyst surfaces + adsorption properties | GNN-ready | https://doi.org/10.1126/sciadv.abq5944 |
| 112 | MOFormer (MOF properties) | MOF structures + multiple properties | Large-scale | https://doi.org/10.1021/jacs.2c11420 |
| 113 | Polymers for gas separation membranes | Polymer structures + permeability | Interpretable features | https://doi.org/10.1126/sciadv.abn9545 |
| 114 | Universal neural network potential | Multi-element structures + energies | Broad chemical space | https://doi.org/10.1038/s41467-022-30687-9 |

## Medium-Priority — Benchmark/Interaction Datasets

| # | Dataset / Paper | Data Type | URL |
|---|-----------------|-----------|-----|
| 12 | OpenQDC | Consolidated 37+ QM datasets (400M geometries) | https://doi.org/10.48550/arXiv.2411.19629 |
| 13 | Reference Vertical Excitation Energies | Transition metal excitation data | https://doi.org/10.1021/acs.jctc.3c01080 |
| 22 | Materials Transformers datasets | Materials compositions (>400K entries) | https://doi.org/10.1088/2632-2153/acadcd |
| 27 | MatDeepLearn benchmark | Materials chemistry GNN benchmarks | https://doi.org/10.1038/s41524-021-00554-0 |
| 28 | Polymer Glass Transition Temperature | Polymer Tg data (566 polymers) | https://doi.org/10.1021/acs.jcim.1c01031 |
| 36 | 3D Molecular Geometry benchmark | ~4M molecular geometries | https://doi.org/10.1137/1.9781611978032.39 |
| 63 | MGCDB84 database | ~5000 molecular property data points | https://doi.org/10.1080/00268976.2017.1333644 |
| 70 | A24 dataset extensions | Noncovalent interaction energies | https://doi.org/10.1039/C5CP03151F |
| 71 | NCI Atlas HB300SPX | Hydrogen bonding benchmark | https://doi.org/10.1021/acs.jctc.0c00715 |
| 75 | S66a8 extensions | Intermolecular interaction energies | https://doi.org/10.1021/ct200523a |
| 80 | NCI Atlas HB375x10 | Hydrogen bonding benchmark | https://doi.org/10.1021/acs.jctc.9b01265 |
| 82 | NCI Atlas R739x5 | Repulsive contacts benchmark | https://doi.org/10.1021/acs.jctc.0c01341 |
| 127 | MPCONF196 | Peptide/macrocycle conformational energies | https://doi.org/10.1021/acs.jctc.7b01074 |

## Lower Priority — Protein/Structural Biology Datasets

These are confirmed datasets but focus on protein structures rather than small molecules/materials:

| # | Dataset / Paper | URL |
|---|-----------------|-----|
| 31 | AI2BMD (protein dipeptide conformations) | https://doi.org/10.1038/s41586-024-08127-z |
| 54 | Atomic cluster expansion (Cu/Si datasets) | https://doi.org/10.1038/s41524-021-00559-9 |
| 55 | CascadesDB (radiation damage simulations) | https://cascadesdb.iaea.org/ |
| 62 | Worldwide Protein Data Bank | https://doi.org/10.1038/nsb1203-980 |
| 69 | sc-PDB (druggable binding sites) | https://doi.org/10.1021/ci050372x |
| 73 | ProPairs (protein-protein docking) | https://doi.org/10.1021/acs.jcim.5b00082 |
| 77 | LEADS-PEP (peptide docking) | https://doi.org/10.1021/acs.jcim.5b00234 |
| 78 | PDBbind (binding affinity data) | https://doi.org/10.1093/bioinformatics/btu626 |
| 83 | OMEGA validation sets | https://doi.org/10.1021/ci100031x |
| 86 | Allosteric sites benchmarks | https://doi.org/10.1016/j.patter.2021.100408 |
| 89 | Protein-protein interface dataset | https://doi.org/10.1002/prot.20580 |
| 92 | sc-PDB-Frag | https://doi.org/10.1021/ci500282c |
| 93 | RNAsolo | https://doi.org/10.1093/bioinformatics/btac386 |
| 94 | Cross-Docked dataset | https://doi.org/10.1021/acs.jcim.0c00411 |
| 98 | BepiPred-3.0 (B-cell epitopes) | https://doi.org/10.1002/pro.4497 |
| 106 | Protein solubility prediction | https://doi.org/10.1002/jcc.27249 |
| 118 | Mg-binding sites in RNA | https://doi.org/10.1093/nar/gkv225 |
| 119 | SimBoost (drug-target binding) | https://doi.org/10.1186/s13321-017-0209-z |
| 121 | PDBbind (scoring functions) | https://doi.org/10.1021/acs.accounts.6b00491 |
| 130 | CSAR Benchmark (protein-ligand complexes) | https://doi.org/10.1021/ci200082t |
| 134 | AbDb (antibody structures) | https://doi.org/10.1093/database/bay040 |

## DCP Probe Datasets

These are ChEMBL-sourced bioactivity data for donated chemical probes. They could be bundled into a single dataset:

| # | Probe |
|---|-------|
| 41 | BI 639667 |
| 45 | MRL-SYKi |
| 46 | BI-1935 |
| 47 | A-1155463 |
| 48 | BAY-549 |
| 49 | TP-030-1 |
| 50 | TP-030-2 |
| 51 | MLi-2 |
| 52 | Borussertib |

All from the "Deep Annotation of Donated Chemical Probes (DCP)" paper (https://doi.org/10.1021/acschembio.2c00877) — 97 probes with bioactivity data across hundreds of protein targets.
