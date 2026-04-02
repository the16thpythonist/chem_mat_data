# Transition Metal Complex Support

## Status

implemented

## Context

The ``chem_mat_data`` package was originally designed exclusively for small organic molecules, using 
``MoleculeProcessing`` to convert SMILES strings into graph representations via RDKit. Reviewer feedback 
on the associated HDF (Hyperdimensional Fingerprints) paper requested evaluation on broader compound 
classes, particularly organometallics and transition metal complexes (TMCs).

TMCs present fundamental challenges that the existing organic pipeline cannot handle:

- **RDKit sanitization failures.** RDKit enforces valence rules designed for organic chemistry. Transition 
  metals routinely violate these rules (e.g., Fe with 6 coordinate bonds), causing ``MolFromSmiles`` to 
  return ``None`` for 5-70% of TMC SMILES depending on the dataset.
- **Dative bonds.** The dominant bond type in coordination chemistry (ligand donates an electron pair to 
  the metal) is not in the organic bond type vocabulary (single, double, triple, aromatic).
- **Haptic bonding.** Metallocenes and arene complexes involve multi-center bonding where a metal 
  interacts with an entire ring's pi system, which cannot be faithfully represented as atom-pair bonds.
- **Element coverage.** The organic atom one-hot encoding covers only 19 elements. TMC datasets involve 
  all 30 transition metals (Sc-Zn, Y-Cd, Hf-Hg).
- **Metal-specific properties.** d-electron count, oxidation state, coordination geometry, and spin state 
  are critical for predicting TMC properties but have no analogue in organic chemistry.

## Decision

### Decomposed Raw Format

Instead of representing a TMC as a single whole-complex SMILES (which is fragile and lossy), the raw 
format uses a **decomposed representation**: a metal element symbol, a list of individual ligand SMILES 
strings, and connecting atom indices specifying which ligand atoms coordinate to the metal. This mirrors 
how the most successful TMC ML methods work (ELECTRUM, RACs, molSimplify) and keeps ligand SMILES fully 
organic and RDKit-compatible.

A dedicated ``load_tmc_dataset()`` function handles loading this richer tabular format, separate from 
the organic ``load_smiles_dataset()``.

### Sibling Processing Class

A new ``MetalOrganicProcessing`` class was created as a **sibling** to ``MoleculeProcessing`` (both 
conceptually inherit from ``AbstractProcessing``) rather than a subclass. The organic assumptions in 
``MoleculeProcessing`` (single SMILES input, organic element vocabulary, no dative bonds, Crippen logP) 
are too deeply embedded to override cleanly. The sibling approach avoids fighting inherited behavior 
while reusing the same encoder infrastructure (``OneHotEncoder``, ``chem_prop``, ``EncoderBase``).

The ``process()`` method takes the decomposed input and assembles a unified molecular graph by:

1. Creating the metal center as node 0 with TMC-specific features.
2. Parsing each ligand SMILES separately (with lenient sanitization that skips valence checks).
3. Adding intra-ligand bonds as standard edges.
4. Adding dative bond edges from each ligand's donor atoms to the metal node.

### Feature Encoding (91 / 18 / 5 dimensions)

**Node features (91 dims):** 51-element one-hot (20 common organic + 30 TMs + unknown), hybridization 
(10 dims, including SP3D2), total degree (13 dims, covering CN 0-12), hydrogen count, atomic mass, 
formal charge, aromaticity, ring membership, plus TMC-specific continuous features: is_metal_center flag, 
Pauling electronegativity, covalent radius, van der Waals radius, periodic table period and group, 
d-electron count, and outer electron count. Electronegativity and group number use bundled Python 
dictionary lookup tables (~50 entries each).

**Edge features (18 dims):** Bond type one-hot (7 dims, adding DATIVE and ZERO to the organic types), 
stereo, aromaticity, ring membership, conjugation, bond order as continuous value, plus three manually 
computed TMC features: is_dative flag, dative_direction (+1 for donor-to-metal, -1 for reverse, 0 for 
non-dative), and is_metal_metal flag for future polynuclear support.

**Graph features (5 dims):** Molecular weight, total charge, number of metal centers (always 1 for now), 
coordination number, and metal atomic number.

### Haptic Bonding Strategy

The **star pattern** (Strategy A) is used as the default: each atom in a haptic ligand (e.g., each of 
the 5 carbons in a cyclopentadienyl ring) gets an individual edge to the metal. This over-counts the 
coordination number but is compatible with all standard GNN architectures (GCN, GAT, GIN, MPNN). The 
``is_haptic`` edge feature is available in the full encoding scheme (when metadata is provided) to let 
models distinguish haptic from sigma/dative bonds.

### Scope: Mononuclear Only

The initial implementation targets **mononuclear** complexes (single metal center), which covers the 
primary benchmark datasets (tmQMg, ELECTRUM coordination number and oxidation state datasets). Graph-level 
features like coordination number and metal atomic number assume a single metal. Polynuclear support 
(metal-metal bonds, bridging ligands) is designed for but not yet implemented.

## Consequences

### Advantages

**High compatibility.** By decomposing TMCs into organic ligand fragments, the entire RDKit ecosystem 
works for ligand processing without workarounds. Lenient sanitization (skipping valence checks) handles 
the remaining edge cases where ligand fragments have metal-influenced valences.

**Clean separation.** Organic and TMC processing pipelines do not interfere with each other. Users working 
only with organic molecules are completely unaffected. TMC users get purpose-built features validated 
against the ELECTRUM, RACs, and tmQMg benchmarks.

**Extensibility.** The decomposed format and sibling class pattern make it straightforward to add support 
for new compound classes (MOFs, polymers) as additional processing classes in the future.

### Disadvantages

**Different feature dimensions.** TMC graph datasets (91-dim nodes) and organic graph datasets (44-dim 
nodes) cannot be mixed in the same model without a compatibility layer. This is an inherent consequence 
of the different chemical information being encoded.

**Heuristic donor atom detection.** When connecting atom indices are not provided in the source data 
(as with the tmQMg dataset), donor atoms must be inferred heuristically. The current heuristic (prefer 
negatively charged atoms, then heteroatoms with lone pairs, then fall back to index 0) works well for 
classical coordination compounds but may be wrong for carbon-donor ligands like NHC carbenes.

**No whole-complex graph.** The decomposed representation loses inter-ligand interactions (steric clashes, 
hydrogen bonding between ligands, trans influence) that require knowing the full 3D arrangement. For 
tasks where these matter, 3D coordinates would need to be incorporated as an additional input.
