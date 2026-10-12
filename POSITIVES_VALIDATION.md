# Upstream Positives Validation Report

Validation of all 142 entries from `discovery_positives.jsonl` (upstream Dataset-Discovery-Pipeline training data) using the `cmmanage agent identify` command. The goal was to check whether the upstream `label=1` assignments are correct.

**Date:** 2026-04-12

## Summary

| Metric | Count | % |
|--------|------:|----:|
| Agent confirmed (yes) | 104 | 73% |
| Agent rejected (no) | 37 | 26% |
| Unclear errors | 1 | 1% |

Of the 142 entries, 139 had `label=1` and 3 had `label=0` in the upstream training data. Some "error" results contained a clear yes/no decision in the response body (JSON parse failures due to preamble text) and were counted accordingly.

A **26% disagreement rate** on the positive set indicates significant label noise in the upstream training data, which directly impacts classifier quality.

## Confirmed Positives (104/142)

The agent confirmed these papers genuinely release molecular/materials datasets. Examples include well-known databases (OQMD, GMTKN55, AFLOW, OpenQDC, QM9, PDBbind, DUD-E), curated benchmark sets, and Zenodo/Figshare data deposits with SMILES, DFT calculations, or crystal structures.

## Rejected Positives (37/142) — Likely Mislabeled

### Wrong Domain (10 papers)

These papers release datasets, but not in chemistry or materials science:

| # | Title | Agent Reasoning |
|---|-------|-----------------|
| 15 | DiLiGenT102: A Photometric Stereo Benchmark Dataset | Computer vision dataset for 3D shape reconstruction. "Material variation" refers to surface reflectance, not chemical properties. |
| 16 | Multi-View Photometric Stereo: A Robust Solution and Benchmark Dataset | Computer vision paper. Contains images and surface normal maps, not molecular data. |
| 20 | Mechanical MNIST: A benchmark dataset for mechanical metamodels | Computational mechanics (FE simulations). Continuum-level, not molecular-scale. |
| 23 | Quartet protein reference materials and datasets for label-free proteomics | Proteomics data (protein quantification). Not chemistry/materials ML data. |
| 26 | Material structure-property linkages using 3D CNNs | Synthetic 3D microstructures of composites. No molecular structures or chemical identifiers. |
| 59 | Identifying the missing protein in human proteome | Protein structure prediction with bioinformatics tools. Not molecular property data. |
| 68 | Improved protein structure prediction using predicted interresidue orientations | Protein structure methodology (trRosetta). Training data is not molecular property data. |
| 115 | A knowledge graph to interpret clinical proteomics data | Clinical proteomics knowledge graph. Biomedical, not chemistry/materials. |
| 131 | The molecular basis of ubiquitin-like protein NEDD8 deamidation | Structural biology of specific protein-protein interactions. Not a dataset. |
| 140 | Characterization of Molecular Recognition Features, MoRFs | Structural protein analysis from PDB. Not molecular property data for ML. |

### Methodology Papers — No New Dataset Released (13 papers)

These use existing data but don't release new datasets:

| # | Title | Agent Reasoning |
|---|-------|-----------------|
| 21 | Can ML identify the next high-temperature superconductor? | Methodological study evaluating ML extrapolation. No new public dataset. |
| 33 | Data-driven parametrization of force fields for expansive chemical space | Generates 2.4M geometries but no data availability section. Data not publicly shared. |
| 35 | Using ML in QSPR to estimate boiling/critical temperatures | QSPR methodology paper. No data repository links found. |
| 57 | Ch. 11: Junction Tree VAE for Molecular Graph Generation | Introduces JT-VAE architecture. Evaluates on existing benchmarks. |
| 61 | Modeling of Proteins with the Integrative Modeling Platform | Software methodology paper (IMP). Provides scripts, not datasets. |
| 64 | PDB_REDO: automated re-refinement of X-ray structures | Methodology for re-refining existing PDB structures. Not a new dataset. |
| 65 | FieldScreen: Virtual Screening Using Molecular Fields | Validates a screening method using existing DUD dataset. |
| 74 | GTA: Graph Truncated Attention for Retrosynthesis | Methodological contribution using existing USPTO-50k benchmark. |
| 85 | BDB: Databank of PDB files with consistent B-factors | Standardizes existing PDB files. Not new molecular property data. |
| 109 | ML enables polymer cloud-point engineering via inverse design | Uses existing polymer data for inverse design methodology. |
| 122 | ff19SB: Amino-Acid-Specific Protein Backbone Parameters | Force field parameter development. Methodology, not a dataset release. |
| 124 | Chalcogen Bonding in Protein-Ligand Complexes: PDB Survey | PDB survey/analysis. Does not publish new downloadable datasets. |
| 126 | Tautomer Preference in PDB Complexes | Analyzes existing data. No new dataset provided. |

### Too Small / Not a Dataset (8 papers)

These report data for only 1-3 compounds or lack a downloadable dataset:

| # | Title | Agent Reasoning |
|---|-------|-----------------|
| 38 | Synthetic Control of Excited-State Properties in Ir(III) Complexes | Small series of Ir complexes. Data in paper tables, no downloadable dataset. |
| 39 | Phosphorescence vs TADF: Controlling Singlet-Triplet Splitting in Cu(I) Compounds | Exactly 2 Cu(I) complexes studied. Not a dataset. |
| 42 | Data for DCP probe BI 665915 | ChEMBL entry with only 2 compounds (probe + control). |
| 44 | Data for DCP probe BAY-1797 | Single-compound characterization. Not a comprehensive dataset. |
| 53 | Data for DCP probe BAY1125976 | ChEMBL entry for a single probe. Not a dataset publication. |
| 58 | Osmium Containing Double Perovskite Ba2XOsO6 | DFT study of 3 materials. Too small for ML. |
| 135 | Determination of crystal structures from limited powder data sets | Single crystal structure determination. Predates modern data sharing. |
| 139 | Syntheses and Photoluminescence of d10 Metal Complexes | Only 2 novel coordination polymers. Not a dataset. |

### Not Findable / Inaccessible (4 papers)

The agent could not locate the paper or verify data availability:

| # | Title | Agent Reasoning |
|---|-------|-----------------|
| 43 | Data for DCP probe BAY-707 | Could not locate a publication with this title. |
| 111 | Robust recognition of zinc binding sites in proteins | Could not verify dataset availability. |
| 128 | Boiling Point and Critical Temperature: QSAR with Atom Type Electrotopological State Indices | 1996 paper. Predates online data repositories. |
| 136 | Hydrogen Bonds with Fluorine in Ligand-Protein Complexes | PDB analysis without downloadable dataset publication. |

### Borderline / Debatable (2 papers)

| # | Title | Agent Reasoning |
|---|-------|-----------------|
| 107 | The TensorMol-0.1 model chemistry | Provides trained models but unclear if underlying training data is downloadable. |
| 117 | QupKake: Integrating ML and Quantum Chemistry for Micro-pKa Prediction | Methodological paper. Data in supplementary info but no dedicated dataset release. |

## Implications for Retraining

1. **~26% of upstream positive labels are incorrect or highly debatable.** Of 142 entries, 37 were rejected by the agent. This represents significant noise in the training data.

2. **The mislabeled positives cluster into clear categories:**
   - Papers from wrong domains (computer vision, proteomics, mechanical engineering)
   - Methodology papers that use but don't release datasets
   - Single-compound or very small studies mislabeled as dataset releases
   - Older papers predating modern data sharing practices

3. **Cleaning these labels before retraining should meaningfully improve classifier precision.** Removing the 37 disputed entries from the positive set (or relabeling them as negatives) would give the model a cleaner signal of what a genuine dataset release looks like.

4. **Full results are in `_positives_validation.csv`** with columns: `index`, `title`, `upstream_label`, `agent_decision`, `agent_reasoning`.
