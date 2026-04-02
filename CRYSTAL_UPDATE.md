# Crystal Structure Support — Implementation Plan

## Overview

Add support for periodic crystal structures as a new category alongside `organic` and `tmc`.
This introduces a `CrystalProcessing` class that converts crystal structures (lattice + atom
positions + species) into the existing GraphDict format, with additional crystal-specific fields
for periodicity. The first dataset will be the **Matbench `matbench_mp_e_form`** task (~132k
structures, formation energy regression).

---

## 1. New File: `chem_mat_data/crystal_processing.py`

### 1.1 Design Principles

- **Sibling to `MoleculeProcessing` and `MetalOrganicProcessing`**, not a subclass.
  Follows the precedent set by ADR 006: crystal assumptions (no bonds, periodic, geometry-defined
  edges) are fundamentally different from molecular assumptions.
- **ASE-based**, since it's already a dependency. Use `ase.neighborlist.neighbor_list` for
  periodic neighbor finding. Add `spglib` as a new dependency for space group detection.
- **Same encoder infrastructure**: reuse `EncoderBase`, `OneHotEncoder`, `LookupEncoder`,
  `PeriodicTableEncoder` from the existing codebase.
- **Attribute map pattern**: `node_attribute_map`, `edge_attribute_map`, `graph_attribute_map`
  dicts, same as `MoleculeProcessing` and `MetalOrganicProcessing`.

### 1.2 `CrystalProcessing` Class

```python
class CrystalProcessing(AbstractProcessing):
    """
    Processing class for periodic crystal structures.

    Takes an ASE Atoms object (with cell and pbc) and constructs a graph using
    a radius-based distance cutoff with periodic boundary conditions. Nodes are
    atoms in the unit cell; edges connect all atom pairs within the cutoff radius,
    including across periodic boundaries.

    Example:

        from ase.io import read

        processing = CrystalProcessing()
        atoms = read('structure.cif')
        graph = processing.process(
            atoms=atoms,
            graph_labels=[formation_energy],
        )
    """
```

### 1.3 `process()` Method Signature

```python
def process(self,
            atoms: 'ase.Atoms',
            graph_labels: list = [],
            cutoff: float = 8.0,
            ) -> GraphDict:
```

**Steps:**

1. **Validate input**: Ensure `atoms` has a cell and `pbc` set.
2. **Compute periodic neighbor list** using `ase.neighborlist.neighbor_list('ijdS', atoms, cutoff)`:
   - `i`, `j`: source/destination atom indices (unit cell)
   - `d`: distances
   - `S`: image offset vectors (N×3 integers)
3. **Encode node features** via `node_attribute_map` callbacks. Since there are no RDKit atoms,
   encoders operate on atomic numbers (integers) and ASE atom properties directly.
4. **Encode edge features**: raw interatomic distance as a float (shape E×1).
5. **Encode graph features** via `graph_attribute_map`.
6. **Detect symmetry** via spglib: space group number, symbol, crystal system, Wyckoff positions.
7. **Assemble GraphDict** with standard + crystal-specific fields.

### 1.4 Node Attribute Map

Crystal-specific encoders that work on atomic numbers (not RDKit Atom objects). New encoder
classes needed: `AtomicNumberEncoder`, `ElementPropertyEncoder`.

| Feature | Encoding | Dims | Description |
|---|---|---|---|
| `atomic_number` | One-hot | 119 | All elements (1-118 + unknown) |
| `electronegativity` | Normalized float | 1 | Pauling electronegativity / 3.98 |
| `covalent_radius` | Normalized float | 1 | Covalent radius / 2.6 Å |
| `vdw_radius` | Normalized float | 1 | Van der Waals radius / 3.0 Å |
| `atomic_mass` | Normalized float | 1 | Atomic mass / 238.0 |
| `period` | Normalized float | 1 | Period / 7 |
| `group` | Normalized float | 1 | Group / 18 |
| `block` | One-hot | 4 | s, p, d, f |
| `ionization_energy` | Normalized float | 1 | First ionization energy (normalized) |
| `electron_affinity` | Normalized float | 1 | Electron affinity (normalized) |
| `n_outer_electrons` | Normalized float | 1 | Outer electrons / 18 |

**Estimated total: ~132 dims** (dominated by the 119-dim one-hot).

Note: These encoders will NOT use RDKit Atom objects. They will be a new family of encoders
that take an integer atomic number as input. This is a key difference from the molecular
and TMC processing classes.

### 1.5 Edge Attribute Map

| Feature | Encoding | Dims | Description |
|---|---|---|---|
| `distance` | Raw float | 1 | Euclidean distance in Å |

Kept minimal intentionally. Downstream models (CGCNN, ALIGNN, etc.) typically apply their
own distance expansions (Gaussian basis, Bessel, etc.). A utility function
`gaussian_basis_expansion(distances, centers, width)` can be provided separately for
convenience.

### 1.6 Graph Attribute Map

| Feature | Encoding | Dims | Description |
|---|---|---|---|
| `density` | Float | 1 | atoms / volume (Å⁻³) |
| `volume` | Float | 1 | Unit cell volume (ų) |
| `num_elements` | Float | 1 | Number of distinct elements |

---

## 2. Crystal-Specific GraphDict Fields

In addition to the standard fields (`node_indices`, `node_attributes`, `edge_indices`,
`edge_attributes`, `graph_labels`), crystal graphs include:

| Field | Type | Shape | Description |
|---|---|---|---|
| `node_coordinates` | np.float32 | (N, 3) | Cartesian coordinates in the unit cell |
| `node_frac_coordinates` | np.float32 | (N, 3) | Fractional coordinates [0, 1) |
| `node_atomic_numbers` | np.int32 | (N,) | Atomic numbers (for PyG `z` field) |
| `edge_image` | np.int32 | (E, 3) | Periodic image offsets per edge |
| `edge_distances` | np.float32 | (E,) | Interatomic distances per edge |
| `graph_lattice` | np.float32 | (3, 3) | Lattice vectors as rows (Å) |
| `graph_pbc` | np.bool_ | (3,) | Periodic boundary conditions |
| `graph_structure_type` | str | — | Always `"crystal"` |
| `graph_cutoff` | float | — | Cutoff radius used for edge construction |
| `graph_repr` | str | — | Reduced formula (e.g. `"Fe2O3"`) |

### Symmetry Metadata (stored as graph-level fields)

| Field | Type | Description |
|---|---|---|
| `graph_spacegroup_number` | int | International space group number (1-230) |
| `graph_spacegroup_symbol` | str | Hermann-Mauguin symbol (e.g. `"Fm-3m"`) |
| `graph_crystal_system` | str | Crystal system (e.g. `"cubic"`) |
| `graph_point_group` | str | Point group symbol |
| `node_wyckoff` | np.ndarray (str) | Wyckoff position label per atom |

---

## 3. New Encoder Classes

Since crystal atoms are not RDKit `Atom` objects, we need a parallel set of encoders that
work on atomic numbers (integers) and ASE Atoms properties.

```python
class AtomicNumberOneHotEncoder(EncoderBase):
    """One-hot encoding over all 118 elements + unknown. Input: int (atomic number)."""

class ElementPropertyEncoder(EncoderBase):
    """
    Looks up a property from a static dict keyed by atomic number, normalizes.
    Similar to LookupEncoder but takes int instead of RDKit Atom.
    """

class BlockEncoder(EncoderBase):
    """One-hot encoding of the periodic table block (s, p, d, f). Input: int."""
```

These should be defined in `crystal_processing.py` alongside the processing class, following
how `MetalFlagEncoder`, `DElectronCountEncoder`, etc. are defined in `tmc_processing.py`.

Lookup tables for electronegativity, ionization energy, electron affinity, covalent radius,
etc. should be defined as module-level constants. Many of these already exist in
`tmc_processing.py` (`PAULING_ELECTRONEGATIVITY`, `ELEMENT_GROUP`) and can be imported.

---

## 4. PyG Conversion Extension

Extend `pyg_from_graph()` in `main.py` to detect crystal-specific fields and map them:

```python
# Inside pyg_from_graph(), after existing code:

if 'graph_lattice' in graph:
    data.cell = torch.tensor(graph['graph_lattice'], dtype=torch.float).unsqueeze(0)  # (1, 3, 3)

if 'edge_image' in graph:
    data.cell_offsets = torch.tensor(graph['edge_image'], dtype=torch.long)  # (E, 3)

if 'node_atomic_numbers' in graph:
    data.z = torch.tensor(graph['node_atomic_numbers'], dtype=torch.long)  # (N,)

if 'graph_pbc' in graph:
    data.pbc = torch.tensor(graph['graph_pbc'], dtype=torch.bool)
```

No separate function needed — the existing `pyg_from_graph` handles it transparently.

---

## 5. Graph Validation Extension

Extend `assert_graph_dict()` in `graph.py` or add `assert_crystal_graph_dict()`:

- If `graph_structure_type == "crystal"`:
  - `graph_lattice` must exist with shape (3, 3)
  - `graph_pbc` must exist with shape (3,)
  - `node_coordinates` must exist
  - `edge_image` must exist with shape (E, 3)
  - Allow duplicate (i, j) pairs in `edge_indices` (different periodic images)

---

## 6. New Dependency: `spglib`

Add `spglib` as an optional dependency (like torch/pyg) or a required one.
It's lightweight (~2 MB) and is the standard library for space group detection.

```toml
# pyproject.toml
dependencies = [
    ...
    "spglib>=2.0",
]
```

---

## 7. First Dataset: `matbench_mp_e_form`

### Why This Dataset

- **132,752 structures** with DFT formation energies from Materials Project
- Part of the **Matbench** benchmark suite — the community standard for crystal ML
- Well-defined, standardized train/test splits
- CC-BY-4.0 license (Materials Project)
- Installable via `pip install matbench`
- Regression task (formation energy in eV/atom) — straightforward target

### Dataset Script

New file: `chem_mat_data/scripts/create_graph_datasets__matbench_mp_e_form.py`

Following the same pattern as the TMC dataset scripts:

1. Download/load via the `matbench` Python package
2. Iterate over structures (pymatgen `Structure` objects → convert to ASE `Atoms`)
3. Process each with `CrystalProcessing`
4. Save as msgpack via `save_graphs()`
5. Generate metadata.yml

```python
DATASET_NAME: str = 'matbench_mp_e_form'
DESCRIPTION: str = (
    'Formation energy dataset of ~132,000 inorganic crystal structures from the '
    'Materials Project, part of the Matbench benchmark suite. Target is the DFT-computed '
    'formation energy in eV/atom, a key indicator of thermodynamic stability. '
    'Covers a wide range of compositions and crystal systems.'
)
METADATA: dict = {
    'category': 'crystal',
    'min_version': '1.9.0',
    'tags': ['Crystal', 'Periodic', 'DFT', 'FormationEnergy', 'MaterialsProject', 'Matbench'],
    'sources': [
        'https://doi.org/10.1038/s41524-020-00406-3',
        'https://materialsproject.org',
    ],
    'verbose': 'Matbench Formation Energy (Materials Project)',
}
```

### New Dependencies for the Script

The dataset creation script (not the main package) needs:
- `matbench` — for loading the dataset with standardized splits
- `pymatgen` — matbench returns pymatgen Structure objects (convert to ASE for processing)

---

## 8. Metadata & CLI Integration

### 8.1 New Category: `crystal`

Add `crystal` as a recognized category in the metadata system and the web frontend.

- `metadata.yml`: datasets get `category: crystal`
- Landing page: new filter pill color for "Crystal" (e.g., blue/teal)
- CLI: `cmdata list --category crystal`

### 8.2 Loading API

Add a new loader function in `main.py`:

```python
def load_crystal_dataset(name: str) -> List[GraphDict]:
    """Load a crystal structure dataset as a list of graph dicts."""
```

This mirrors `load_smiles_dataset()` and `load_graph_dataset()`. Under the hood it uses
the same download + cache + msgpack pipeline, but the returned GraphDicts contain the
crystal-specific fields.

Alternatively, `load_graph_dataset()` could work as-is since the GraphDict format is the
same — just with extra fields. A separate function is still useful for discoverability and
documentation.

---

## 9. Implementation Order

1. **Encoder classes** in `crystal_processing.py` — `AtomicNumberOneHotEncoder`,
   `ElementPropertyEncoder`, `BlockEncoder`, lookup tables
2. **`CrystalProcessing` class** with `process()` method
3. **Unit tests** for the processing class (mock ASE Atoms, verify GraphDict output)
4. **PyG conversion extension** in `main.py`
5. **Graph validation extension** in `graph.py`
6. **Dataset script** for `matbench_mp_e_form`
7. **Metadata & CLI integration** (new category, loader function)
8. **Web frontend** (crystal category pill, updated filters)
9. **ADR 008** documenting the crystal support decision

---

## 10. Out of Scope (Future Work)

- Asymmetric unit representation (store symmetry in metadata for now, implement later as
  an optional processing mode following the coGN paper)
- Gaussian basis expansion / Bessel basis expansion (provide as utility, not baked into
  edge features)
- Additional Matbench tasks (add after the first one works)
- k-NN graph construction (radius-based only for now)
- Force/stress targets for ML interatomic potentials
- Integration with Materials Project / JARVIS APIs for on-demand download
