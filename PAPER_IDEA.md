# ChemMatData — Paper Planning Document

## 1. Target Journal

**Primary: Journal of Cheminformatics** (Springer Nature / BioMed Central)

- **Article type:** Software
- **Impact Factor:** ~5.7 (2024 JCR); 5-year IF: 8.9
- **Open Access:** Fully open access (Gold OA)
- **APC:** ~$2,390 USD (institutional Read & Publish agreements may cover this)
- **Review:** Single-anonymous; median ~10 days to first decision; 5–9 months submission-to-publication
- **URL:** https://jcheminf.biomedcentral.com

### Why This Journal

- Has a dedicated **Software** article type purpose-built for packages like ChemMatData.
- Scope explicitly covers chemical databases, ML on chemical/biological data, and software tools.
- Strong reproducibility and open-source requirements align with ChemMatData's MIT license and public PyPI/GitHub presence.
- High visibility in the cheminformatics community; regularly publishes ML-for-chemistry tools (e.g., QSPRpred, MolScore, fastprop, Deepmol).
- No formal word limit — allows space for architecture description, comparison, and case study.

### Submission Requirements Checklist

- [ ] Source code publicly available on GitHub under OSI-approved license (MIT — done)
- [ ] Archived release on Zenodo with DOI (done: 10.5281/zenodo.19234789)
- [ ] Software installable and testable by anonymous reviewers (PyPI — done)
- [ ] All tests pass without privileged access (network-free test subset needed)
- [ ] Documentation site up to date
- [ ] Source code archive included as supplementary file
- [ ] Abstract contains "Scientific Contribution" subsection (max 3 sentences)
- [ ] "Availability and Requirements" structured section in manuscript
- [ ] Double-line spacing, line/page numbering in manuscript
- [ ] All URLs cited as numbered references with access dates

---

## 2. Fallback Journals

### Patterns (Cell Press) — IF ~7.4

- **Article type:** Descriptor (covers both software and data resources)
- Higher impact factor and Cell Press prestige. 
- Good fit because the Descriptor format explicitly supports software + data infrastructure.
- **Downsides:** High APC (~$3,000+), competitive standards, broad data-science audience (less chemistry-specific).

### JCIM — Journal of Chemical Information and Modeling (ACS) — IF ~5.3

- **Article type:** Application Note (max ~6 pages / 5,000 words)
- Highest visibility in cheminformatics; strong track record for ML tools.
- **Downsides:** Concise format limits depth. Hybrid OA (APC for open access). May expect methodological novelty.

### Scientific Data (Nature) — IF ~6.9

- **Article type:** Data Descriptor
- Best option if the paper is reframed around the *dataset collection* rather than the software.
- Nature-branded prestige, FAIR emphasis.
- **Downsides:** Software/API aspects would be secondary; purely descriptive (no hypothesis testing).

---

## 3. Selling Points — How to Position ChemMatData

The key review criterion is demonstrating a **"significant advance over previously published software."** 
The narrative should be built around four pillars:

### 3.1 Unified, Framework-Agnostic Data Access

No existing tool provides a single API that serves chemistry/materials datasets as pre-processed 
molecular graphs to **both** PyTorch Geometric and Jax/Jraph. ChemMatData makes this a 2–3 line operation:

```python
from chem_mat_data import load_graph_dataset, pyg_data_list_from_graphs

graphs = load_graph_dataset('clintox')
data_list = pyg_data_list_from_graphs(graphs)
```

This is the strongest differentiator and should be the lead argument.

### 3.2 Chemistry + Materials Science Scope

Most existing tools are siloed: TDC covers drug discovery, Materials Project covers inorganic materials, 
OGB is domain-agnostic. ChemMatData bridges chemistry and materials science in a single package 
with 30+ datasets spanning toxicology, solubility, DFT properties, bioactivity, and more.

### 3.3 Chemically-Informed, Extensible Graph Processing

The `MoleculeProcessing` pipeline provides configurable, semantically rich featurization — not just 
adjacency matrices but atom-level and bond-level features derived from RDKit (element type, hybridization, 
aromaticity, bond order, etc.). Users can extend or customize the processing pipeline for their needs.

### 3.4 CLI for Accessibility

`cmdata list` and `cmdata download "clintox"` make datasets accessible to researchers who prefer 
not to write Python code. No other chemistry dataset tool provides a first-class CLI experience.

---

## 4. Competitor Comparison

This table should be adapted for the manuscript's Results/Discussion section.

| Feature | ChemMatData | MoleculeNet | TDC | OGB | DeepChem | PyG Built-in |
|---|---|---|---|---|---|---|
| **Maintained software** | Yes (active) | No (paper only; datasets scattered across DeepChem) | Yes | Yes | Yes | Yes |
| **Domain** | Chemistry + Materials Science | Chemistry (drug discovery focus) | Drug discovery / therapeutics | General (social, bio, chem) | Chemistry | General |
| **Pre-processed graphs** | Yes (unified GraphDict format) | Partial (via DeepChem) | No (raw data) | Yes (framework-specific) | Yes (internal format) | Yes (PyG-only) |
| **PyTorch Geometric support** | Yes | Via DeepChem | No native | Yes | No (own format) | Yes (native) |
| **Jax/Jraph support** | Yes | No | No | No | No | No |
| **Framework-agnostic** | Yes (intermediate dict format) | No | N/A (raw data) | Partial | No | No |
| **CLI interface** | Yes (`cmdata`) | No | No | No | No | No |
| **Extensible featurization** | Yes (`MoleculeProcessing` pipeline) | Fixed | N/A | Fixed | Configurable | Fixed |
| **Materials science datasets** | Yes (QM9, COMPAS, etc.) | QM7/QM8/QM9 only | No | No | QM7/QM8/QM9 only | QM7/QM9 only |
| **Install** | `pip install chem_mat_database` | Part of DeepChem | `pip install PyTDC` | `pip install ogb` | `pip install deepchem` (heavy) | Part of PyG |
| **Lightweight** | Yes (focused on data access) | N/A (not standalone) | Yes | Yes | No (full ML framework) | N/A (part of PyG) |
| **Local caching** | Yes | Partial | Yes | Yes | Yes | Yes |
| **Dataset count** | 30+ | ~17 | 60+ (but drug-only) | ~5 molecular | ~17 | ~20 molecular |

### Key Differentiation Arguments

1. **MoleculeNet** is a benchmark *paper* (2018), not maintained software. Its datasets live inside 
   DeepChem with no standalone access. ChemMatData provides a standalone, pip-installable, actively 
   maintained package.

2. **TDC** has more datasets but is restricted to drug discovery / therapeutics. It provides raw data 
   (SMILES + labels), not pre-processed graphs. ChemMatData covers a broader scientific scope and 
   delivers graph-ready data.

3. **OGB** is domain-agnostic (social networks, citation graphs, etc.) with only ~5 molecular datasets. 
   ChemMatData is purpose-built for chemistry with chemically meaningful featurization.

4. **DeepChem** is a full ML framework — a heavyweight dependency when all you need is data loading. 
   ChemMatData is lightweight and focused.

5. **PyG built-in datasets** lock users into one framework. ChemMatData's intermediate `GraphDict` 
   format decouples data from framework, supporting both PyG and Jraph.

---

## 5. Manuscript Outline

### Abstract (~250 words)

- Problem: Fragmented landscape of chemistry/materials datasets; no unified, framework-agnostic tool.
- Solution: ChemMatData — a Python package providing CLI and API access to 30+ datasets as 
  pre-processed molecular graphs, with native support for PyTorch Geometric and Jax/Jraph.
- Key result: Demonstrate simplicity and reproducibility through comparison with existing tools.

**Scientific Contribution** (max 3 sentences, required subsection):
> ChemMatData provides a unified, framework-agnostic Python package for accessing chemistry and 
> materials science datasets as pre-processed molecular graphs, directly supporting both PyTorch 
> Geometric and Jax/Jraph. Unlike existing tools that are either framework-locked, domain-restricted, 
> or unmaintained, ChemMatData offers a single API and CLI spanning 30+ datasets with configurable, 
> chemically-informed featurization. This lowers the barrier for reproducible GNN benchmarking 
> across chemical domains.

### Keywords (3–10)
cheminformatics, graph neural networks, molecular graphs, benchmark datasets, machine learning, 
materials science, property prediction, open-source software

### 1. Introduction (~1,000 words)

- Growing importance of GNNs for molecular property prediction.
- The dataset fragmentation problem: datasets scattered across papers, frameworks, and formats.
- Existing tools and their limitations (MoleculeNet/TDC/OGB/DeepChem) — set up the gap.
- ChemMatData's vision: one package, any framework, any chemical domain.

### 2. Implementation (~2,000 words)

- **2.1 Architecture Overview** — Package structure, data flow diagram (remote server → download → 
  cache → raw/graph format → framework conversion). Describe the Nextcloud-based distribution system.
- **2.2 Graph Representation** — The `GraphDict` format: node/edge indices, attributes, labels, 
  metadata. Why a dictionary-based intermediate format enables framework independence.
- **2.3 Molecular Processing Pipeline** — `MoleculeProcessing` class: SMILES → RDKit Mol → graph. 
  Configurable atom and bond featurization (one-hot encoding, descriptors). Extensibility via 
  custom callbacks.
- **2.4 Framework Conversion** — `pyg_data_list_from_graphs()` and Jraph conversion utilities. 
  How the intermediate format maps to each framework's data structures.
- **2.5 CLI Design** — `cmdata` and `cmmanage` commands. Rich-formatted output. How the CLI 
  serves users who prefer not to write Python.
- **2.6 Caching and Configuration** — Local caching system, `.env` configuration, TOML config files.

### 3. Results and Discussion (~2,000 words)

- **3.1 Dataset Collection** — Overview table of all available datasets with statistics (count, 
  task type, domain). Briefly describe sourcing and curation methodology.
- **3.2 Comparison with Existing Tools** — Adapted version of the competitor comparison table (Section 4).
  Discuss qualitative and quantitative differences (lines of code for a typical workflow, install 
  size, supported frameworks, dataset coverage).
- **3.3 Usage Case Study** — End-to-end example: load a dataset, process into PyG format, train 
  a simple GIN model, evaluate. Show the same workflow in Jraph to demonstrate framework independence. 
  Compare the code complexity with achieving the same via DeepChem or raw PyG datasets.
- **3.4 Limitations and Future Work** — Acknowledge limitations honestly (e.g., dataset count vs. TDC, 
  dependency on Nextcloud server, current Python version constraints). Discuss planned features 
  (new datasets, additional framework support, community contributions).

### 4. Conclusions (~300 words)

- Restate the core contribution: unified, framework-agnostic access to chemistry/materials datasets.
- Emphasize reproducibility and lowered barriers to entry.
- Call to action for community adoption and dataset contributions.

### Availability and Requirements (structured section)

- **Project name:** ChemMatData
- **Project home page:** https://github.com/the16thpythonist/chem_mat_data
- **Operating system(s):** Platform independent
- **Programming language:** Python (3.9–3.12)
- **Other requirements:** RDKit, NumPy, Pandas, ASE, msgpack (see pyproject.toml)
- **License:** MIT
- **Any restrictions to use by non-academics:** None

### Figures (planned)

1. **Architecture diagram** — Data flow from remote server through caching to framework-specific output.
2. **GraphDict schema** — Visual representation of the intermediate graph format.
3. **Code comparison** — Side-by-side: loading a dataset with ChemMatData vs. DeepChem vs. raw PyG.
4. **Dataset overview** — Table or chart summarizing available datasets by domain and task type.
5. **CLI screenshot** — Terminal output of `cmdata list` and `cmdata download`.

---

## 6. Pre-Submission Action Items

- [ ] **Gap analysis**: Download and test MoleculeNet (via DeepChem), TDC, and OGB. Verify all 
  claims in the comparison table. Identify any missing features or inaccuracies.
- [ ] **Repo cleanup**: Ensure README is current, all tests pass, documentation is complete, 
  `pip install chem_mat_database` works cleanly on a fresh environment.
- [ ] **Network-free tests**: Ensure a subset of tests can run without network access (for reviewers).
- [ ] **Case study code**: Write and test the end-to-end example (PyG + Jraph) that will go 
  in the Results section.
- [ ] **Architecture diagram**: Create a clear data flow diagram for the Implementation section.
- [ ] **Graphical abstract** (optional): 920×300px, white background, summarizing the package visually.
- [ ] **Draft manuscript**: Write following the outline above; target 5,000–7,000 words.
- [ ] **Co-author coordination**: Align with Mohit Singh on contributions and review.
- [ ] **Institutional OA agreement**: Check if KIT has a Springer Nature Read & Publish agreement 
  to cover the APC.
- [ ] **Submit**: Via https://jcheminf.biomedcentral.com — consider the special collection 
  "Evaluating AI and ML models in cheminformatics" if still open.
