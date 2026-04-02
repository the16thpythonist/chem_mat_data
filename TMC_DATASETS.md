# Transition Metal Complex (TMC) Datasets for HDF Benchmarking

## Context and Motivation

Our paper introduces **Hyperdimensional Fingerprints (HDF)** — a molecular representation that combines hyperdimensional computing with message-passing on molecular graphs. So far, HDF has been evaluated exclusively on **small organic molecules** (QM9, FreeSolv, BACE, Lipophilicity, etc.) using Morgan and RDKit fingerprints as baselines.

Reviewer feedback (W7) asked us to evaluate HDF on broader compound classes: **organometallics, MOFs, perovskites, polymers, and biomolecules**. The key insight is that Morgan fingerprints — our primary baseline — are inapplicable or severely limited for most of these domains (e.g., RDKit cannot sanitize many metal complexes, periodic structures have no molecular graph). Rather than comparing HDF against a broken baseline, the strategy is to benchmark against **domain-specific fingerprints** that were designed for transition metal chemistry:

- **ELECTRUM** (Orsi & Frei, Digital Discovery 2025) — 598-bit fingerprint combining ECFP-like ligand encoding with metal electron configuration
- **RACs** (Janet & Kulik, J. Phys. Chem. A 2017) — ~153 autocorrelation features over the molecular graph using atomic properties

Both are deterministic, training-free, fixed-length, and don't require 3D coordinates — making them fair comparison targets for HDF.

This document catalogs the datasets suitable for this benchmarking effort.

---

## Tier 1: Primary Datasets

These are large, freely available, have SMILES representations, and have been used to benchmark ELECTRUM and/or RACs. They are the most directly useful for an HDF comparison.

### 1.1 tmQM (2024 release)

The foundational large-scale dataset for ML on transition metal complexes.

| Property | Value |
|---|---|
| **Size** | ~108,000 mononuclear TMCs (original 2020 release: 86,665) |
| **Source** | Cambridge Structural Database (CSD) — Werner, bioinorganic, and organometallic complexes |
| **Metals** | All 30 transition metals (3d, 4d, 5d series, groups 3–12), >30,000 different ligands |
| **DFT Level** | TPSSh-D3BJ/def2-SVP (single-point); GFN2-xTB (geometry optimization, polarizability) |
| **Properties** | Electronic energy, dispersion energy, dipole moment, metal natural charge, HOMO energy, LUMO energy, HOMO-LUMO gap, polarizability |
| **Formats** | XYZ (geometries), CSV (properties + SMILES), charges, Wiberg bond orders |
| **License** | MIT |
| **Download** | [github.com/uiocompcat/tmQM](https://github.com/uiocompcat/tmQM) |
| **Reference** | Balcells & Skjelstad, *J. Chem. Inf. Model.* 2020, 60, 6135. DOI: [10.1021/acs.jcim.0c01041](https://doi.org/10.1021/acs.jcim.0c01041) |

### 1.2 tmQMg (Graph Dataset)

Graph-enriched version of tmQM with NBO-derived bond information. This is the dataset used in the ELECTRUM paper for regression benchmarking.

| Property | Value |
|---|---|
| **Size** | 74,547 TMCs (74,539 after Nov 2025 erratum) |
| **Source** | Subset of tmQM with NBO-derived graph representations |
| **Properties** | 20 QM properties: HOMO/LUMO energies, HOMO-LUMO gap, electronic energy, dispersion energy, enthalpy, Gibbs energy, ZPE correction, heat capacity, entropy, dipole moment, polarizability, lowest/highest vibrational frequency, plus delta corrections |
| **Formats** | GML graph files (baseline, u-NatQG, d-NatQG), CSV, XYZ, SMILES |
| **License** | CC-BY-NC-4.0 |
| **Download** | [github.com/uiocompcat/tmQMg](https://github.com/uiocompcat/tmQMg); bulk data at [archive.sigma2.no](https://archive.sigma2.no/dataset/569AEAF9-5F8E-4BE3-B30E-0D439BC417C2) |
| **Reference** | Kneiding et al., *Digital Discovery* 2023, 2, 618. DOI: [10.1039/D2DD00129B](https://doi.org/10.1039/D2DD00129B) |

**Note:** The ELECTRUM repo includes a pre-processed version of this dataset at `electrum_val/datasets/tmQMg.csv` (63,466 complexes after matching with CSD coordination data to add LigandSMILES). This is the easiest starting point.

### 1.3 ELECTRUM CSD Coordination Number Dataset

Large classification benchmark from the ELECTRUM paper.

| Property | Value |
|---|---|
| **Size** | 217,517 mononuclear TMCs |
| **Source** | Curated from CSD; counterions removed; ligands extracted via RDKit + NetworkX |
| **Metals** | 35 transition metals |
| **Task** | Multi-class classification: 11 coordination number classes (CN 2–12), each with ≥1,000 examples |
| **Largest classes** | CN=6 (70.8k, 32.5%), CN=4 (64.9k, 29.9%), CN=5 (28.8k, 13.2%) |
| **Formats** | CSV with columns: `Name` (CSD refcode), `LigandSmiles`, `Metal`, `classification` |
| **Download** | [electrum_val/datasets/coordnumber.csv](https://github.com/TheFreiLab/electrum_val/tree/main/datasets) |
| **Reference** | Orsi & Frei, *Digital Discovery* 2025, 4, 3567. DOI: [10.1039/D5DD00145E](https://doi.org/10.1039/D5DD00145E) |

### 1.4 ELECTRUM CSD Oxidation State Dataset

Classification benchmark from the ELECTRUM paper.

| Property | Value |
|---|---|
| **Size** | 39,166 TMCs |
| **Source** | Subset of the coordination number dataset where oxidation states could be assigned from CSD metadata |
| **Metals** | 29 transition metals |
| **Task** | Multi-class classification: 7 oxidation state classes (0, +1, +2, +3, +4, +5, +6) |
| **Largest class** | OS=+2 (20.5k, 52.4%) |
| **Formats** | CSV with columns: `smiles`, `Name`, `LigandSmiles`, `Metal`, `bondorder`, `oxidation_states`, `classification` |
| **Download** | [electrum_val/datasets/oxidationstate_46k.csv](https://github.com/TheFreiLab/electrum_val/tree/main/datasets) |
| **Reference** | Same as 1.3 |

---

## Tier 2: Smaller or More Specialized Datasets

These are well-curated but smaller, or cover a more specific chemical subspace. Useful for supplementary evaluation or specialized tasks.

### 2.1 tmQMg* (Excited States)

Extension of tmQMg with TD-DFT excited-state properties.

| Property | Value |
|---|---|
| **Size** | 74,273 TMCs |
| **DFT Level** | TD-DFT wB97xd/def2SVP |
| **Properties** | UV-Vis-NIR absorption wavelengths and oscillator strengths (first 30 excited states), natural transition orbitals, charge transfer character, solvatochromic shifts |
| **Download** | [github.com/uiocompcat/tmQMg_star](https://github.com/uiocompcat/tmQMg_star) |
| **License** | CC-BY-4.0 |
| **Reference** | Kneiding & Balcells, *J. Chem. Inf. Model.* 2025. DOI: [10.1021/acs.jcim.5c01958](https://doi.org/10.1021/acs.jcim.5c01958) |

### 2.2 tmQM_wB97MV (Improved DFT Energies)

Re-computed energies for tmQM at a higher-quality DFT level.

| Property | Value |
|---|---|
| **Size** | Subset of tmQM (structures with missing hydrogens removed) |
| **DFT Level** | wB97M-V/def2-SVPD |
| **Properties** | Electronic energies at improved accuracy |
| **Download** | [github.com/ulissigroup/tmQM_wB97MV](https://github.com/ulissigroup/tmQM_wB97MV); also on [Figshare](https://figshare.com/collections/6964854) |
| **Reference** | Garrison et al., *J. Chem. Inf. Model.* 2023, 64, 1. DOI: [10.1021/acs.jcim.3c01226](https://doi.org/10.1021/acs.jcim.3c01226) |

### 2.3 tmCAT / tmPHOTO / tmBIO / tmSCO (Application-Labeled Subsets)

NLP-curated subsets of tmQM linked to functional applications.

| Dataset | Size | Application |
|---|---|---|
| **tmCAT** | 21,631 | Catalysis |
| **tmPHOTO** | 4,599 | Photophysics |
| **tmBIO** | 2,782 | Biological activity |
| **tmSCO** | 983 | Spin crossover / magnetism |

| Property | Value |
|---|---|
| **Source** | Subsets of tmQM labeled via NLP analysis of associated literature |
| **Download** | Zenodo (referenced in paper) |
| **Reference** | Kevlishvili et al., *Faraday Discuss.* 2025. DOI: [10.1039/D4FD00087K](https://doi.org/10.1039/D4FD00087K) |

### 2.4 Kulik Group Redox / Spin-Splitting Dataset

Small but carefully curated dataset with RAC benchmarks already published.

| Property | Value |
|---|---|
| **Size** | 874 octahedral open-shell TMCs |
| **Metals** | Co, Cr, Fe, Mn in M(II)/M(III), both high-spin and low-spin |
| **DFT Level** | B3LYP / LANL2DZ (metal) + 6-31G* (ligands) |
| **Properties** | HOMO energy (0.15 eV MAE), HOMO-LUMO gap, ionization/redox potential, spin-state splitting, metal-ligand bond lengths |
| **Download** | [Figshare](https://acs.figshare.com/articles/dataset/7182242/1) (SI of Nandy et al. 2018) |
| **Reference** | Nandy et al., *Ind. Eng. Chem. Res.* 2018, 57, 13973. DOI: [10.1021/acs.iecr.8b04015](https://doi.org/10.1021/acs.iecr.8b04015) |

### 2.5 CSD Ligand Coordination Dataset

Large-scale ligand-level dataset for predicting how ligands coordinate to metals.

| Property | Value |
|---|---|
| **Size** | 70,069 unique ligands from CSD |
| **Task** | Predict denticity and coordinating atoms from SMILES |
| **Formats** | SMILES with denticity/coordination labels |
| **Download** | [github.com/hjkgrp/pydentate](https://github.com/hjkgrp/pydentate); data on [Zenodo](https://doi.org/10.5281/zenodo.13840776) |
| **Reference** | Toney et al., *PNAS* 2025, 122(41), e2415658122. DOI: [10.1073/pnas.2415658122](https://doi.org/10.1073/pnas.2415658122) |

### 2.6 tmQMg-L (Ligand Library)

Library of unique ligands extracted from tmQMg.

| Property | Value |
|---|---|
| **Size** | 35,466 unique ligands |
| **Properties** | NBO-derived formal charge, denticity, hapticity, coordinating atom indices, electronic/steric/cheminformatic descriptors |
| **Download** | [github.com/uiocompcat/tmQMg-L](https://github.com/uiocompcat/tmQMg-L); also on [Zenodo](https://doi.org/10.5281/zenodo.10374523) |
| **Reference** | Kneiding et al., *Nat. Comput. Sci.* 2024, 4, 263. DOI: [10.1038/s43588-024-00616-5](https://doi.org/10.1038/s43588-024-00616-5) |

### 2.7 CSD TMC SMILES Collection ("SMILES All Around")

The largest validated collection of SMILES for transition metal complexes. Useful as a structure source to pair with property labels from other datasets.

| Property | Value |
|---|---|
| **Size** | 227,124 RDKit-parsable SMILES for mononuclear TMCs |
| **Source** | CSD, using xyz2mol_tm and extended Hückel theory for SMILES generation |
| **Formats** | SMILES (validated for RDKit parsing) |
| **Download** | [Figshare](https://springernature.figshare.com/collections/7792138) |
| **Reference** | Kneiding et al., *J. Cheminformatics* 2025, 17, 63. DOI: [10.1186/s13321-025-01008-1](https://doi.org/10.1186/s13321-025-01008-1) |

### 2.8 SCO-95 (Spin Crossover)

Very small but experimentally validated dataset for spin-crossover prediction.

| Property | Value |
|---|---|
| **Size** | 95 Fe(II) complexes |
| **Properties** | Experimental spin transition temperatures, DFT energetics (30 functionals benchmarked) |
| **Reference** | Vennelakanti & Kulik, *J. Chem. Phys.* 2023, 159, 024120 |

---

## Practical Notes for HDF Implementation

### Recommended Starting Point

Use the **pre-processed tmQMg CSV from the ELECTRUM repo** (`electrum_val/datasets/tmQMg.csv`, 63,466 complexes). This gives you:
- LigandSMILES for fingerprint computation
- Metal identity as a separate column
- 20 regression targets already cleaned and aligned
- Direct comparability with published ELECTRUM results

For classification, add the **coordination number dataset** (217k, same repo) — it's the largest and most diverse.

### Recommended Regression Targets

Pick 2–3 from tmQMg for a focused comparison:
1. **HOMO-LUMO gap** (`tzvp_homo_lumo_gap`) — the standard benchmark target
2. **Polarizability** (`polarisability`) — well-predicted by fingerprint methods
3. **Dipole moment** (`tzvp_dipole_moment`) — more challenging, tests spatial sensitivity

### HDF Adaptations Needed

1. **Element dictionary expansion:** Current HDF encodes atoms via a dictionary of element types. Transition metals (Sc–Zn, Y–Cd, Hf–Hg) need to be added. This is straightforward — just expand the random HV dictionary to cover all 30 TM elements.

2. **Dative/coordinate bonds:** Many TMC SMILES use `->` or `<-` for dative bonds. RDKit parses these but they may not be in HDF's bond type dictionary. Need to add dative bond types or map them to single bonds.

3. **RDKit sanitization failures:** Some TMC SMILES will fail RDKit sanitization (unusual valences, haptic bonding). Record the failure rate — the ELECTRUM paper handles this by filtering to parseable structures. Expect 5–15% failure rate depending on the dataset.

4. **Metal as special node:** Consider whether the metal center needs special treatment in message passing (e.g., higher weight, separate encoding) or whether the standard HDF pipeline handles it naturally. Start with the standard pipeline and see.

### Baseline Fingerprint Setup

| Fingerprint | Install | Input | Output | Similarity |
|---|---|---|---|---|
| **ELECTRUM** | `pip install electrum-fp` or clone [TheFreiLab/electrum_val](https://github.com/TheFreiLab/electrum_val) | LigandSMILES + Metal | 598-bit binary | Tanimoto |
| **RACs** | `pip install molSimplify` or clone [hjkgrp/molSimplify](https://github.com/hjkgrp/molSimplify) | XYZ or SMILES | ~153 continuous features | Euclidean / cosine |
| **Morgan (if applicable)** | `rdkit` (standard) | SMILES | 2048-bit binary | Tanimoto |

**Note on RACs:** The full scoped RACs (axial/equatorial) assume octahedral geometry and need 3D coordinates. The `f/all` (full-molecule) scope works from topology alone and is the fair comparison for 2D fingerprints like HDF. Use depth=3 to match the standard setup.

### Experiment Design

For a minimal but convincing comparison in the SI:
1. Take tmQMg (63k, ELECTRUM repo version) with 3 regression targets
2. Compute HDF, ELECTRUM, and RACs for all parseable molecules
3. Train the same ML models (RF, NN) with the same train/test splits
4. Report MAE and R² in a table
5. Optionally add coordination number classification (217k) for a second task type

This mirrors the ELECTRUM paper's experimental setup, making results directly comparable.
