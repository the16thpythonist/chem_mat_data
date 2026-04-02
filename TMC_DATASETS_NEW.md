# Additional TMC Datasets for ChemMatData

Literature research conducted 2026-04-02. These datasets are **not yet implemented** in ChemMatData and complement the existing collection (tmQM, tmQMg, tmQMg*, ELECTRUM coord/oxstate, tmCAT/PHOTO/BIO/SCO, Kulik spin-splitting).

---

## Tier A: Strong Candidates

Large enough for ML, freely available, property labels present, mononuclear TMCs.

### A1. TM-GSspin (Ground State Spin Classification)

| Property | Value |
|---|---|
| **Size** | 2,063 mononuclear first-row TMCs |
| **Metals** | Cr, Mn, Fe, Co, Ni (d3-d8) |
| **Task** | Classification: ground-state spin multiplicity |
| **DFT Level** | B3LYP* |
| **Formats** | XYZ coordinates, Chemiscope JSON |
| **SMILES** | Not directly provided; would need xyz2mol conversion or tmQM SMILES overlap |
| **Download** | [Materials Cloud](https://doi.org/10.24435/materialscloud:jx-a5) (5.4 MB) |
| **License** | CC-BY 4.0 |
| **Reference** | Cho et al., *Digital Discovery* 2024. DOI: [10.1039/D4DD00093E](https://doi.org/10.1039/D4DD00093E) |

**Why interesting:** Spin state classification is a fundamental TMC property not yet covered. Complements the Kulik spin-splitting regression dataset.

### A2. IrCytoToxDB (Iridium Cytotoxicity)

| Property | Value |
|---|---|
| **Size** | 803 unique Ir(III) complexes, 2,694 IC50 values across 127 cell lines |
| **Metals** | Ir(III) (bis-cyclometalated and half-sandwich) |
| **Task** | Regression: IC50 cytotoxicity |
| **Source** | Curated from 222 papers (2008-2022) |
| **Formats** | CSV with per-ligand SMILES via RDKit |
| **SMILES** | Yes |
| **Download** | [Zenodo](https://doi.org/10.5281/zenodo.13120939) |
| **License** | CC-BY 4.0 |
| **Reference** | *Scientific Data* 11, 870 (2024). DOI: [10.1038/s41597-024-03735-w](https://doi.org/10.1038/s41597-024-03735-w) |

**Why interesting:** First experimental bioactivity dataset for metal complexes. Completely different property domain from QM targets.

### A3. RuCytoToxDB (Ruthenium Cytotoxicity)

| Property | Value |
|---|---|
| **Size** | 3,255 unique Ru complexes, 12,292 IC50 values across 600+ cell lines |
| **Metals** | Ru |
| **Task** | Regression: IC50 cytotoxicity, antibacterial activity |
| **Source** | Curated from 921 papers (2001-2025) |
| **Formats** | Ligand-composition-based representation |
| **SMILES** | Partial (ligand-composition SMILES) |
| **Download** | [Streamlit app](https://rucytotoxdb.streamlit.app/) |
| **License** | Unknown (ChemRxiv preprint) |
| **Reference** | ChemRxiv 2025. [Link](https://chemrxiv.org/engage/chemrxiv/article-details/6882592efc5f0acb52d49343) |

**Why interesting:** Larger than IrCytoToxDB, same property domain. Download mechanism (Streamlit app) needs investigation.

### A4. MetalCytoToxDB (Multi-Metal Cytotoxicity)

| Property | Value |
|---|---|
| **Size** | 7,050 unique complexes, 26,500 IC50 values across 754 cell lines |
| **Metals** | Ru, Ir, Rh, Re, Os |
| **Task** | Regression: IC50 cytotoxicity |
| **Source** | Curated from 1,921 papers |
| **Formats** | Ligand-composition representation |
| **SMILES** | Partial |
| **Download** | [Streamlit app](https://biometaldb.streamlit.app/) |
| **License** | Unknown (ChemRxiv preprint) |
| **Reference** | ChemRxiv 2025. [Link](https://chemrxiv.org/engage/chemrxiv/article-details/68c67a6523be8e43d6e68b0c) |

**Why interesting:** Largest cytotoxicity dataset, multi-metal. Same download concern as RuCytoToxDB.

### A5. IUPAC Stability Constants

| Property | Value |
|---|---|
| **Size** | 32,459 log K values, 3,585 ligands, 102 metal ions |
| **Metals** | 102 metal ions (73 elements, including all TMs) |
| **Task** | Regression: metal-ligand binding constants (log K1) |
| **Source** | IUPAC critically evaluated data |
| **Formats** | Tabulated with ligand SMILES |
| **SMILES** | Yes (ligand SMILES, Chemprop D-MPNN compatible) |
| **Download** | *JCIM* 2025 SI |
| **License** | Unknown |
| **Reference** | *J. Chem. Inf. Model.* 2025. DOI: [10.1021/acs.jcim.5c01546](https://doi.org/10.1021/acs.jcim.5c01546) |

**Why interesting:** Largest dataset here, experimental property, broadest metal coverage (102 ions). Completely unique property type.

### A6. Cross-Coupling Catalyst Dataset (Corminboeuf)

| Property | Value |
|---|---|
| **Size** | 7,054 DFT-computed catalysts (18,062 total with ML-predicted) |
| **Metals** | Ni, Pd, Pt, Cu, Ag, Au (with 91 ligands) |
| **Task** | Regression: oxidative addition energy (Suzuki-Miyaura descriptor) |
| **DFT Level** | B3LYP-D3/def2-TZVP |
| **Formats** | XYZ coordinates in ESI |
| **SMILES** | Not directly provided |
| **Download** | [Materials Cloud](https://archive.materialscloud.org/records/aw44a-z2597) (32.3 MB) |
| **License** | CC-BY 4.0 |
| **Reference** | Meyer, Sawatlon et al., *Chem. Sci.* 2018. DOI: [10.1039/C8SC01949E](https://doi.org/10.1039/C8SC01949E) |

**Why interesting:** First catalysis-specific reaction energy dataset. Six metals across groups 10-11.

### A7. Metal Phototherapy Classifier

| Property | Value |
|---|---|
| **Size** | 4,640 complexes (balanced from 9,775 initial) |
| **Metals** | Pt, Ir, Ru, Rh |
| **Task** | Binary classification: absorbs in 500-850 nm therapeutic window |
| **Source** | UV-Vis data from literature |
| **Formats** | CSV with SMILES |
| **SMILES** | Yes |
| **Download** | [GitHub](https://github.com/vorsamaqoy/MetalComplexClassifier) |
| **License** | CC-BY-NC-ND 4.0 |
| **Reference** | Vigna et al., *J. Cheminform.* 2024. DOI: [10.1186/s13321-024-00939-5](https://doi.org/10.1186/s13321-024-00939-5) |

**Why interesting:** Binary classification on optical absorption. SMILES directly available. Multi-metal.

### A8. Redox2k (Iron Redox Potentials)

| Property | Value |
|---|---|
| **Size** | 2,267 iron complexes (1,450 after filtering) |
| **Metals** | Fe(II)/Fe(III) |
| **Task** | Regression: redox potential |
| **DFT Level** | TPSSh/def2-TZVP with SMD solvation |
| **Formats** | XYZ, RDKit Mol objects, graph representations |
| **SMILES** | tmQM-derived structures |
| **Download** | *Digital Discovery* 2026 SI |
| **License** | CC-BY-NC 3.0 |
| **Reference** | Bhuiyan et al., *Digital Discovery* 2026. DOI: [10.1039/D5DD00431D](https://doi.org/10.1039/D5DD00431D) |

**Why interesting:** Electrochemical property, derived from tmQM structures (easy SMILES matching).

### A9. Ir(III) Phosphor HTE Dataset

| Property | Value |
|---|---|
| **Size** | 1,380 Ir(III) complexes [Ir(CN)2(NN)]+ |
| **Metals** | Ir(III) |
| **Task** | Regression: emission energy, excited-state lifetime, spectral integral |
| **Source** | High-throughput experimental measurements (60 CN x 23 NN ligands) |
| **Formats** | Ligand XYZ in ESI; ML models on Zenodo |
| **SMILES** | Partial (ligand structures available) |
| **Download** | [Zenodo](https://zenodo.org/records/7090416) |
| **License** | CC-BY 3.0 |
| **Reference** | Terrones et al., *Chem. Sci.* 2023. DOI: [10.1039/D2SC06150C](https://doi.org/10.1039/D2SC06150C) |

**Why interesting:** Experimental photophysical properties from HTE. Well-structured combinatorial design.

### A10. Zinc NNP Dataset

| Property | Value |
|---|---|
| **Size** | 771 Zn(II) complexes, 39,599 conformations |
| **Metals** | Zn(II) (coordination numbers 2-8) |
| **Task** | Regression: single-point energies |
| **DFT Level** | r2SCAN-3c (ORCA) |
| **Formats** | XYZ, SMILES for ligands |
| **SMILES** | Yes (ligand SMILES) |
| **Download** | [GitHub](https://github.com/Neon8988/Zinc_NNPs) |
| **License** | CC-BY 4.0 |
| **Reference** | Jin & Merz, *JCIM* 2024. DOI: [10.1021/acs.jcim.4c00095](https://doi.org/10.1021/acs.jcim.4c00095) |

**Why interesting:** Conformational diversity (39k conformers), single-metal focus with diverse coordination environments.

### A11. tmQM+ (QTAIM-Enriched tmQM)

| Property | Value |
|---|---|
| **Size** | ~60,000 TMCs |
| **Metals** | All TMs (same as tmQM) |
| **Task** | Regression: >20 QTAIM descriptors + original tmQM properties |
| **DFT Level** | Multiple levels of theory |
| **Formats** | CSV (shares tmQM structure identifiers) |
| **SMILES** | Yes (tmQM SMILES) |
| **Download** | *Digital Discovery* 2025 SI |
| **License** | Unknown |
| **Reference** | *Digital Discovery* 2025. DOI: [10.1039/D5DD00220F](https://doi.org/10.1039/D5DD00220F) |

**Why interesting:** Extends tmQM with QTAIM features at multiple theory levels. Easy integration since structures overlap with existing tmQM support.

### A12. PDT Photosensitizer Dataset

| Property | Value |
|---|---|
| **Size** | Hexacoordinate TMCs (exact count TBD) |
| **Metals** | Ru, Ir, Re |
| **Task** | Regression: singlet oxygen quantum yield |
| **Source** | DFT-computed descriptors + experimental quantum yields |
| **Formats** | Unknown |
| **SMILES** | Unknown |
| **Download** | [Figshare](https://figshare.com/articles/dataset/30500939) |
| **License** | Unknown |
| **Reference** | *ACS Omega* 2025. DOI: [10.1021/acsomega.5c08727](https://doi.org/10.1021/acsomega.5c08727) |

**Why interesting:** Photodynamic therapy application. Singlet oxygen quantum yield is experimentally and therapeutically important.

### A13. HCat-GNet (Rh Asymmetric Catalysis)

| Property | Value |
|---|---|
| **Size** | Rh-catalyzed asymmetric 1,4-addition reactions |
| **Metals** | Rh |
| **Task** | Regression: enantioselectivity |
| **Formats** | SMILES-based |
| **SMILES** | Yes |
| **Download** | [GitHub](https://github.com/EdAguilarB/hcatgnet) + [Zenodo](https://zenodo.org/records/13954130) |
| **License** | CC-BY 4.0 |
| **Reference** | *iScience* 2025 |

**Why interesting:** Enantioselectivity prediction from SMILES. Asymmetric catalysis is a high-value application domain.

### A14. Rh C-H Activation Dataset

| Property | Value |
|---|---|
| **Size** | 1,743 reactant-intermediate pairs |
| **Metals** | Rh (Rh(PLP)(Cl)(CO)) |
| **Task** | Regression: reaction energy (DeltaE) for C-H activation |
| **Formats** | XYZ, Gaussian log files |
| **SMILES** | Not provided |
| **Download** | [Zenodo](https://zenodo.org/records/11109592) |
| **License** | CC-BY 4.0 |
| **Reference** | Huang et al., 2024 |

**Why interesting:** C-H activation is a key catalytic transformation. DFT-computed reaction energies.

---

## Tier B: Catalysis / Reaction Yield Datasets

These focus on reaction outcomes (yield, selectivity) rather than isolated complex properties. The "input" is a full reaction (catalyst + substrate + conditions), which is architecturally different from the current single-complex data loading pattern.

### B1. Dreher-Doyle Buchwald-Hartwig

| Property | Value |
|---|---|
| **Size** | 3,955 reactions (15 aryl halides, 4 Pd catalysts, 3 bases, 23 additives) |
| **Metals** | Pd |
| **Task** | Regression: reaction yield (%) for C-N cross-coupling |
| **SMILES** | Convertible from provided structures |
| **Download** | [GitHub](https://github.com/doylelab/rxnpredict) |
| **License** | Freely available (Science SI) |
| **Reference** | Ahneman et al., *Science* 360, 186 (2018). DOI: [10.1126/science.aar5169](https://doi.org/10.1126/science.aar5169) |

**Note:** The most widely used ML yield prediction benchmark in catalysis.

### B2. Perera Suzuki-Miyaura

| Property | Value |
|---|---|
| **Size** | 5,760 reactions (5 electrophiles, 6 nucleophiles, 11 ligands, 7 bases, 4 solvents) |
| **Metals** | Pd |
| **Task** | Regression: yield (%) for C-C cross-coupling |
| **SMILES** | Available in converted form |
| **Download** | [GitHub](https://github.com/leojklarner/gauche) |
| **License** | Freely available (Science SI) |
| **Reference** | Perera et al., *Science* 359, 429 (2018). DOI: [10.1126/science.aap9112](https://doi.org/10.1126/science.aap9112) |

### B3. Denmark Closed-Loop Suzuki

| Property | Value |
|---|---|
| **Size** | ~400 automated reactions across 5 optimization rounds |
| **Metals** | Pd |
| **Task** | Regression: yield for heteroaryl Suzuki-Miyaura coupling |
| **Download** | [Zenodo](https://doi.org/10.5281/zenodo.7099435) |
| **License** | Unknown |
| **Reference** | Angello et al., *Science* 2022. DOI: [10.1126/science.adc8743](https://doi.org/10.1126/science.adc8743) |

### B4. AbbVie Suzuki Coupling Library

| Property | Value |
|---|---|
| **Size** | 24,203 reactions, 23,236 unique products (15 years of data) |
| **Metals** | Pd |
| **Task** | Regression: yield for Suzuki cross-coupling |
| **SMILES** | Yes |
| **Download** | [GitHub](https://github.com/priyanka-rag/suzuki_yield_predict_external) + JACS SI |
| **License** | Check SI (pharma-company data) |
| **Reference** | Raghavan et al., *JACS* 146, 15070 (2024). DOI: [10.1021/jacs.4c00098](https://doi.org/10.1021/jacs.4c00098) |

**Note:** Largest real-world Suzuki dataset from actual medicinal chemistry campaigns.

### B5. NiCOlit (Ni C-O Couplings)

| Property | Value |
|---|---|
| **Size** | Small (literature-mined) |
| **Metals** | Ni |
| **Task** | Regression: yield for C-O cross-coupling |
| **SMILES** | Yes |
| **Download** | [GitHub](https://github.com/truejulosdu13/NiCOlit) |
| **License** | Open source |
| **Reference** | Saebi et al., *JACS* 144, 14722 (2022). DOI: [10.1021/jacs.2c05302](https://doi.org/10.1021/jacs.2c05302) |

### B6. Ni Borylation (BMS)

| Property | Value |
|---|---|
| **Size** | 1,632 reactions (33 substrates, 36 monophosphine ligands, 2 solvents) |
| **Metals** | Ni (NiCl2) |
| **Task** | Regression: yield for aryl (pseudo)halide borylation |
| **Download** | SI at [Organometallics](https://pubs.acs.org/doi/10.1021/acs.organomet.2c00089) |
| **License** | Unknown |
| **Reference** | *Organometallics* 41, 1847 (2022). DOI: [10.1021/acs.organomet.2c00089](https://doi.org/10.1021/acs.organomet.2c00089) |

### B7. AHO Database (Asymmetric Hydrogenation of Olefins)

| Property | Value |
|---|---|
| **Size** | >12,000 literature-mined entries |
| **Metals** | Rh, Ir, Ru |
| **Task** | Regression: enantioselectivity (ee) |
| **SMILES** | Likely included |
| **Download** | [GitHub](https://github.com/licheng-xu-echo/AHO) + [asymcatml.net](http://asymcatml.net/) |
| **License** | Unknown |
| **Reference** | Xu et al., *Angew. Chem. Int. Ed.* 60, 22804 (2021). DOI: [10.1002/anie.202106880](https://doi.org/10.1002/anie.202106880) |

### B8. ChemAHNet (Asymmetric Hydrogenation)

| Property | Value |
|---|---|
| **Size** | Multiple reaction types |
| **Metals** | TM catalysts |
| **Task** | Classification/regression: stereoselectivity, absolute configuration |
| **SMILES** | Yes |
| **Download** | [Zenodo](https://doi.org/10.5281/zenodo.17346605) + [GitHub](https://github.com/CHENGLi-96/ChemAHNet) |
| **License** | Unknown |
| **Reference** | Cheng et al., *Nat. Comput. Sci.* 2025. DOI: [10.1038/s43588-025-00920-8](https://doi.org/10.1038/s43588-025-00920-8) |

### B9. Rh Asymmetric Hydrogenation HTE (Pidko Group)

| Property | Value |
|---|---|
| **Size** | 3,552 reactions (960 curated for ML), 192 Rh-chiral ligand catalysts, 5 substrates |
| **Metals** | Rh |
| **Task** | Regression: conversion (%), enantiomeric excess (ee) |
| **Download** | [4TU.ResearchData](https://doi.org/10.4121/ecbd4b91-c434-4bdf-a0ed-4e9e0fb05e94) + [GitHub](https://github.com/EPiCs-group/obelix-ml-pipeline) |
| **License** | Open access |
| **Reference** | Kalikadien et al., *Chem. Sci.* 2024. DOI: [10.1039/d4sc03647f](https://doi.org/10.1039/d4sc03647f) |

### B10. Zirconocene Barriers

| Property | Value |
|---|---|
| **Size** | >700 zirconocene systems |
| **Metals** | Zr |
| **Task** | Regression: ethylene migratory insertion barriers (DFT) |
| **Download** | [Figshare](https://figshare.com/articles/dataset/25047818) |
| **License** | Unknown |
| **Reference** | *JCIM* 2024. DOI: [10.1021/acs.jcim.3c01575](https://doi.org/10.1021/acs.jcim.3c01575) |

---

## Tier C: Small Benchmarks

Valuable as test sets or gold-standard references, not as training sets.

### C1. MOBH35 (Metal-Organic Barrier Heights)

| Property | Value |
|---|---|
| **Size** | 35 reactions, 70 barrier heights |
| **Metals** | Various 3d, 4d, 5d TMs |
| **Task** | Benchmark: forward/reverse barrier heights at DLPNO-CCSD(T)/CBS |
| **Download** | SI at [JPC A](https://pubs.acs.org/doi/10.1021/acs.jpca.9b01546) |
| **Reference** | Dohm et al., *J. Phys. Chem. A* 123, 5266 (2019). DOI: [10.1021/acs.jpca.9b01546](https://doi.org/10.1021/acs.jpca.9b01546) |

### C2. MOR41 (Metal-Organic Reaction Energies)

| Property | Value |
|---|---|
| **Size** | 41 closed-shell organometallic reactions (up to 120 atoms) |
| **Metals** | 3d, 4d, 5d TMs |
| **Task** | Benchmark: reaction energies at DLPNO-CCSD(T)/CBS |
| **Download** | [Figshare](https://figshare.com/articles/6083288) |
| **Reference** | Dohm et al., *JCTC* 14, 2596 (2018). DOI: [10.1021/acs.jctc.7b01183](https://doi.org/10.1021/acs.jctc.7b01183) |

### C3. SSE17 (Experimental Spin-State Energetics)

| Property | Value |
|---|---|
| **Size** | 17 first-row TMCs (9 SCO + 8 non-SCO) |
| **Metals** | Fe(II), Fe(III), Co(II), Co(III), Mn(II), Ni(II) |
| **Task** | Benchmark: experimental adiabatic/vertical spin-state splittings |
| **Download** | [ioChem-BD](https://www.iochem-bd.org/handle/10/364100) + RSC SI |
| **License** | CC-BY (RSC open access) |
| **Reference** | Vela et al., *Chem. Sci.* 2024. DOI: [10.1039/D4SC05471G](https://doi.org/10.1039/D4SC05471G) |

### C4. CASPT2/CC Spin Gap Benchmark

| Property | Value |
|---|---|
| **Size** | 50 octahedral complexes (d4-d6), expanded to 500 via delta-ML |
| **Metals** | First-row (Fe, Mn, Co, etc.) |
| **Task** | Benchmark: spin energy gaps at CASPT2/CC multireference level |
| **Download** | RSC SI |
| **Reference** | Dey et al., *PCCP* 2026. DOI: [10.1039/D5CP03964A](https://doi.org/10.1039/D5CP03964A) |

### C5. WCCR10 (Ligand Dissociation Energies)

| Property | Value |
|---|---|
| **Size** | 10 large cationic TMCs |
| **Metals** | Cu, Pd, Ru, Pt, Ag |
| **Task** | Benchmark: experimental gas-phase ligand dissociation energies |
| **Download** | SI at [JCTC](https://pubs.acs.org/doi/10.1021/ct500248h) |
| **Reference** | Weymuth et al., *JCTC* 2014. DOI: [10.1021/ct500248h](https://doi.org/10.1021/ct500248h) |

### C6. TMPHOTCAT-137 (Photocatalyst UV-Vis Benchmark)

| Property | Value |
|---|---|
| **Size** | 137 TMC photocatalysts |
| **Metals** | Cu, Ru, Ir, Fe, Au, Mo, W |
| **Task** | Benchmark: digitized UV-Vis spectra + TDDFT with 14 functionals |
| **Download** | [GitHub](https://github.com/PeterF1234/photocatalyst-TDDFT-benchmark) |
| **License** | Unknown |
| **Reference** | *Chemistry-Methods* 2024. DOI: [10.1002/cmtd.202400071](https://doi.org/10.1002/cmtd.202400071) |

### C7. MME55 (Metalloenzyme Model Reactions)

| Property | Value |
|---|---|
| **Size** | 55 data points across 10 enzyme models (up to 116 atoms) |
| **Metals** | 8 different TMs |
| **Task** | Benchmark: barrier heights + reaction energies at DLPNO-CCSD(T) |
| **Download** | SI at [JCTC](https://pubs.acs.org/doi/10.1021/acs.jctc.3c00558) |
| **Reference** | Wappett & Goerigk, *JCTC* 2023. DOI: [10.1021/acs.jctc.3c00558](https://doi.org/10.1021/acs.jctc.3c00558) |

**Note:** Not strictly mononuclear (metalloenzyme active site models).

### C8. 16OSTM10 / 16TMCONF543 (Conformational Energies)

| Property | Value |
|---|---|
| **Size** | 16OSTM10: 16 complexes, 160 structures; 16TMCONF543: 16 complexes, 543 conformers |
| **Metals** | 3d TMs |
| **Task** | Benchmark: conformational energies across DFT/semiempirical/FF methods |
| **Download** | ESI at [PCCP 2022](https://doi.org/10.1039/D2CP01659A) and [Organometallics 2024](https://doi.org/10.1021/acs.organomet.4c00246) |
| **Reference** | Otlyotov et al. |

---

## Tier D: Large-Scale / Requires Extraction

### D1. OMol25 Metal Complex Subset (Meta FAIR)

| Property | Value |
|---|---|
| **Size** | ~83M unique molecular systems total; monometallic TMC subset generated with 723 ligands via Architector |
| **Metals** | Nearly all TMs + lanthanides |
| **Task** | Regression: energies, forces, charges, orbital energies |
| **DFT Level** | wB97M-V/def2-TZVPD |
| **Download** | [HuggingFace](https://huggingface.co/datasets/facebook/OMol25) |
| **License** | CC-BY-NC 4.0 |
| **Reference** | Meta FAIR, arXiv:2505.08762 (2025) |

**Note:** Would need to extract the monometallic TMC portion. Massive scale but no SMILES.

---

## Tier E: Ligand-Level Datasets

Not full complexes, but ligand libraries with descriptors. Different data loading pattern.

### E1. KRAKEN (Phosphine Ligands)

| Property | Value |
|---|---|
| **Size** | 1,558 DFT-calculated monodentate phosphines; 300k+ ML-predicted; 190 descriptors each |
| **Task** | Ligand electronic/steric descriptors |
| **SMILES** | Yes |
| **Download** | [MolSSI](https://descriptor-libraries.molssi.org/kraken/) + [GitHub](https://github.com/aspuru-guzik-group/kraken) |
| **License** | Open access |
| **Reference** | Gensch et al., *JACS* 144, 1205 (2022). DOI: [10.1021/jacs.1c09718](https://doi.org/10.1021/jacs.1c09718) |

### E2. LKB-PP (Bidentate P,P-Donor Ligands)

| Property | Value |
|---|---|
| **Size** | 334 bidentate P,P/P,N-donor ligands |
| **Task** | Electronic/steric descriptors with PCA maps |
| **Download** | SI at [Organometallics](https://pubs.acs.org/doi/10.1021/om300312t) |
| **Reference** | Jover & Fey, *Organometallics* 32, 5801 (2013). DOI: [10.1021/om300312t](https://doi.org/10.1021/om300312t) |

---

## Tier F: Specialized / Niche

### F1. SC1MC-2022 (DMRG Entropies)

| Property | Value |
|---|---|
| **Size** | 7,259 artificial mononuclear TMCs, 971k two-orbital entropy samples |
| **Metals** | First-row TMs |
| **Task** | Regression: one-site/two-site entropies, mutual information (DMRG) |
| **Download** | Described in [arXiv:2101.06090](https://arxiv.org/abs/2101.06090) (no clear repository) |

### F2. Kulik Methane-to-Methanol Catalyst Space

| Property | Value |
|---|---|
| **Size** | Design space of 16M Mn/Fe catalysts; active learning DFT subset |
| **Metals** | Mn, Fe (macrocyclic ligands) |
| **Task** | Regression: HAT barriers, methanol release energies |
| **Download** | ESI of [JACS Au](https://doi.org/10.1021/jacsau.2c00176) |
| **License** | CC-BY-NC-ND 4.0 |
| **Reference** | Nandy et al., *JACS Au* 2022. DOI: [10.1021/jacsau.2c00176](https://doi.org/10.1021/jacsau.2c00176) |

### F3. Kulik Multireference Diagnostic Dataset

| Property | Value |
|---|---|
| **Size** | >4,800 open-shell TMCs |
| **Task** | Regression: FON-based multireference diagnostics |
| **Download** | ESI / Kulik group data |
| **Reference** | Liu et al., *J. Phys. Chem. Lett.* 2020. DOI: [10.1021/acs.jpclett.0c02288](https://doi.org/10.1021/acs.jpclett.0c02288) |

### F4. Kulik Chromophore Active Learning

| Property | Value |
|---|---|
| **Size** | Design space of 32.5M Fe(II)/Co(III) TMCs; active learning DFT subset |
| **Task** | Regression: absorption energies (visible), 23-DFA consensus |
| **Download** | ESI of [JACS Au](https://doi.org/10.1021/jacsau.2c00547) |
| **License** | ACS AuthorChoice |
| **Reference** | Duan et al., *JACS Au* 2023. DOI: [10.1021/jacsau.2c00547](https://doi.org/10.1021/jacsau.2c00547) |

### F5. Ir(III) Photocatalyst Redox Potentials

| Property | Value |
|---|---|
| **Size** | Ir(III) and Os photocatalysts |
| **Task** | Regression: ground-state and excited-state redox potentials |
| **Download** | ESI at [Angew. Chem.](https://doi.org/10.1002/anie.202517393) |
| **Reference** | Li et al., *Angew. Chem. Int. Ed.* 2025. DOI: [10.1002/anie.202517393](https://doi.org/10.1002/anie.202517393) |

### F6. SIMDAVIS (Single-Ion Magnets)

| Property | Value |
|---|---|
| **Size** | 1,411 lanthanide SIM entries, >10,000 data points |
| **Metals** | Lanthanides (Pr, Nd, Sm, Gd, Tb, Dy, Ho, Er, Tm, Yb) |
| **Task** | Regression: Ueff, tau_0, blocking temperature |
| **Formats** | CIF structures |
| **Download** | [SIMDAVIS](https://go.uv.es/rosaleny/SIMDAVIS) |
| **License** | CC-BY 4.0 |
| **Reference** | *Nat. Commun.* 13, 7626 (2022). DOI: [10.1038/s41467-022-35336-9](https://doi.org/10.1038/s41467-022-35336-9) |

**Note:** Lanthanides, not transition metals. Included for completeness.

### F7. PBDD (Metalloporphyrin Database)

| Property | Value |
|---|---|
| **Size** | 12,096 porphyrins (10,080 metalloporphyrins) |
| **Task** | Regression: HOMO/LUMO energies and gaps |
| **Download** | [CMR at DTU](https://cmr.fysik.dtu.dk/dssc/dssc.html) |
| **Reference** | Kulichenko et al., *Catalysts* 2022. DOI: [10.3390/catal12111485](https://doi.org/10.3390/catal12111485) |

### F8. Metal-CF3 BDE Dataset

| Property | Value |
|---|---|
| **Size** | 2,219 M-CF3 bond dissociation energies |
| **Task** | Regression: metal-trifluoromethyl BDEs |
| **Download** | ESI at [Chinese J. Chem.](https://doi.org/10.1002/cjoc.202500083) |
| **Reference** | Shao et al., *Chinese J. Chem.* 2025. DOI: [10.1002/cjoc.202500083](https://doi.org/10.1002/cjoc.202500083) |

### F9. Non-Heme Iron BDE Dataset

| Property | Value |
|---|---|
| **Size** | >600 non-heme iron complexes, ~900 diabatic BDEs |
| **Task** | Regression: Fe-X and Fe-OH bond dissociation energies |
| **Download** | ESI at [OBC](https://doi.org/10.1039/D5OB00007F) |
| **Reference** | *Org. Biomol. Chem.* 2025. DOI: [10.1039/D5OB00007F](https://doi.org/10.1039/D5OB00007F) |

### F10. Stability Constants (Kanahashi, Sci. Rep.)

| Property | Value |
|---|---|
| **Size** | 19,810 data points, 57 metals, 2,706 ligands |
| **Task** | Regression: overall stability constants (beta) |
| **SMILES** | Yes (ligand SMILES) |
| **Download** | Data available from authors upon request |
| **License** | CC-BY 4.0 (paper); raw data requires contacting authors |
| **Reference** | Kanahashi et al., *Sci. Rep.* 12, 12689 (2022). DOI: [10.1038/s41598-022-15300-9](https://doi.org/10.1038/s41598-022-15300-9) |

### F11. Zenodo Spin State Dataset (Leeds/York)

| Property | Value |
|---|---|
| **Size** | Unknown (3 ZIP files, 3.7 MB total) |
| **Task** | Spin state energies: DFT vs ligand field theory |
| **Metals** | First-row TMs |
| **Download** | [Zenodo](https://zenodo.org/records/18182735) |
| **License** | CC-BY 4.0 |
| **Reference** | Nguyen & Mace (University of Leeds), 2026 |

### F12. Fe(II) Conformer Spin-Splitting Dataset

| Property | Value |
|---|---|
| **Size** | 23,000+ conformers in LS and HS states |
| **Task** | Regression: total energy, spin-splitting energy as a function of conformation |
| **Download** | JCTC SI |
| **Reference** | *JCTC* 2024. DOI: [10.1021/acs.jctc.4c00063](https://doi.org/10.1021/acs.jctc.4c00063) |

### F13. Antibacterial Ru Arene Complexes

| Property | Value |
|---|---|
| **Size** | 288 Ru arene Schiff-base complexes (training), 54 validation, 77M virtual library |
| **Task** | Classification/regression: antibacterial activity against MRSA |
| **Download** | SI at [Angew. Chem.](https://doi.org/10.1002/anie.202317901) |
| **Reference** | *Angew. Chem. Int. Ed.* 2024. DOI: [10.1002/anie.202317901](https://doi.org/10.1002/anie.202317901) |

---

## Relevant Review Papers

1. **"Computational Discovery of TMCs: From HT Screening to ML"** — Kulik et al., *Chem. Rev.* 2022. DOI: [10.1021/acs.chemrev.1c00347](https://doi.org/10.1021/acs.chemrev.1c00347)
2. **"AI Approaches to Homogeneous Catalysis with TMCs"** — *ACS Catal.* 2025. DOI: [10.1021/acscatal.5c01202](https://doi.org/10.1021/acscatal.5c01202)
3. **"Exploring Beyond Experiment: HQ Datasets of TMCs with QC and ML"** — Kulik group, *Coord. Chem. Rev.* 2025.
4. **"ML in Homogeneous Catalysis: Basic Concepts and Best Practices"** — *ACS Catal.* 2025. DOI: [10.1021/acscatal.5c06439](https://doi.org/10.1021/acscatal.5c06439)

---

## Reaction Databases with Organometallic Content

### Open Reaction Database (ORD)

| Property | Value |
|---|---|
| **Size** | Millions of reactions (growing) |
| **Source** | Academic papers + patents |
| **Download** | [open-reaction-database.org](https://open-reaction-database.org/) + [GitHub](https://github.com/Open-Reaction-Database) |
| **License** | CC-BY-SA |
| **Reference** | Kearnes et al., *JACS* 143, 18820 (2021). DOI: [10.1021/jacs.1c09820](https://doi.org/10.1021/jacs.1c09820) |

**Note:** Mostly organic, but contains metal-catalyzed reactions. Includes Doyle and Perera datasets.

### Gold-DIGR / RDBCO (Reaction Database for Catalysis and Organometallics)

| Property | Value |
|---|---|
| **Size** | Large (from 50+ journals, 7 publishers) |
| **Properties** | Reactant/product/TS geometries, IRC traces, reaction classes, ligand/metal descriptors |
| **Download** | ChemRxiv preprint (Nov 2025); data availability TBD |
| **Reference** | ChemRxiv 2025. DOI: [10.26434/chemrxiv-2025-ccgfs](https://doi.org/10.26434/chemrxiv-2025-ccgfs) |

**Note:** Specifically designed for organometallic reaction ML. Full pathways including transition states.

### NCCR Catalysis Datasets

| Property | Value |
|---|---|
| **Size** | Multiple small datasets |
| **Source** | Swiss NCCR Catalysis program (Corminboeuf, Cramer groups) |
| **Download** | [nccr-catalysis.ch/research/datasets](https://www.nccr-catalysis.ch/research/datasets/) (links to Zenodo) |

**Note:** Small but high-quality. CO2 reduction barriers, carboamination/cyclopropanation barriers, metalloenzyme catalysis.
