# Metadata

The `METADATA` dict you declare in the script is merged *over* the fields the base
template derives automatically. Getting it right matters more than it looks: these
fields are what `cmdata list`, `cmdata info`, and the generated documentation show,
and they are the only description a user of the dataset ever sees.

> Confirm field names against the metadata-writing block near the end of
> `chem_mat_data/scripts/create_graph_datasets.py`, and browse
> `chem_mat_data/metadata.yml` for ~46 real examples.

## Derived automatically — do not set these

| Field | Source |
|---|---|
| `compounds` | number of graphs that survived processing |
| `targets` | length of `graph_labels` on the first graph |
| `target_type` | `[DATASET_TYPE]` |
| `description` | the `DESCRIPTION` parameter |
| `raw` | `['csv']` |
| `sources` | `[]` — override this, it is empty by default |

You *can* override `target_type` and `description` through `METADATA`, but prefer
setting `DATASET_TYPE` and `DESCRIPTION` as module parameters — one source of truth.

## Fields you set

### `verbose` — required in practice (43/46 datasets have it)

The human-readable display name. Title case, expand acronyms on first use.

```python
'verbose': 'HOPV15 — Harvard Organic Photovoltaic Dataset'
'verbose': 'Ames Mutagenicity Assays'
```

### `sources` — required (36/46)

Provenance URLs, most authoritative first: the paper DOI, then the data repository,
then any mirror. Use `https://doi.org/...` form for the paper when you have a DOI.

```python
'sources': [
    'https://doi.org/10.1021/acs.jcim.0c01041',
    'https://github.com/uiocompcat/tmQM',
]
```

### `tags` — required (45/46)

See the vocabulary below.

### `target_descriptions` — required for every target, including a single one

Maps target index → description. **Use string keys** — `'0'`, `'1'`, … Both string
and int keys occur in the existing data (24 string vs 2 int), and string is the
convention to follow.

Format each value as `<column_name> - <what it is, with units>`:

```python
'target_descriptions': {
    '0': 'HOMO - Highest occupied molecular orbital energy (eV)',
    '1': 'LUMO - Lowest unoccupied molecular orbital energy (eV)',
    '2': 'PCE - Power conversion efficiency (%)',
}
```

The keys must cover every index from `0` to `target_count - 1`, in the same order as
the list your `load_dataset` builds. The metadata gate checks exactly this and does not
exempt single-target datasets: a regression dataset with one target still needs `'0'`,
and that is the case where the units matter most, because there is no sibling column to
infer them from.

Do not do what `tmqm` does — `{'0': 'Electronic_E', '1': 'Dispersion_E', ...}` is a
column name with no explanation and no units, which tells a downstream user nothing.
If the README gives units, they belong here.

For classification targets, say what the classes mean:

```python
'0': 'mutagenic - 1 if the compound tested positive in the Ames assay, 0 otherwise'
```

Note the `-1` sentinel wherever you used it:

```python
'4': 'PCE - Power conversion efficiency (%); -1 where not measured'
```

Only write that when `-1` really is impossible as a measurement for this target. A
power conversion efficiency cannot be negative, so the sentinel is unambiguous there.
A boiling point in °C, a formation energy or anything centred near zero can genuinely
be `-1`, and using the sentinel on such a target quietly corrupts the data — skip those
rows instead and say so here.

### `license` — set it whenever stage 1 established one

```python
'license': {
    'identifier': 'CC-BY-SA-3.0',
    'url': 'https://creativecommons.org/licenses/by-sa/3.0/',
    'note': (
        'The molecular structures derive from ChEMBL, which is licensed CC BY-SA 3.0. '
        'The ShareAlike condition propagates, so this dataset is distributed under the '
        'same licence.'
    ),
}
```

A mapping, not a bare string, and the `note` is the load-bearing part. `CC-BY-SA-3.0`
alone does not tell a reader *why* it applies or what they inherit; the note does.

Two things it exists to express that an identifier cannot:

- **Inherited obligations.** A dataset whose own source states no terms, but whose
  structures come from a ShareAlike database, carries the obligation without ever having
  received a grant. Write that down rather than implying the authors licensed it.
- **Honest absence.** Omit the field entirely when no licence was established. An absent
  `license` means "not researched"; it is **not** a claim that the data is freely usable,
  and nothing should read it that way.

The full policy and what it commits the project to is in
`docs/architecture_decisions/009_dataset_license_field.md`.

### Optional fields

| Field | Used by | Meaning |
|---|---|---|
| `category` | 4 | Groups the dataset — e.g. `'tmc'` for transition-metal complexes. |
| `min_version` | 4 | Minimum `chem_mat_data` version that can load it. Set when the dataset needs processing features added in a specific release (e.g. `'1.7.0'` for TMC support). |
| `full` | 15 | Marks the dataset as complete/curated rather than a subset. |
| `notes` | 6 | Caveats a user should know before training on it. |

## Tag vocabulary

Tags are free-form but consistency is what makes them useful for filtering. Reuse an
existing tag over inventing a synonym.

**Always include `Molecules`** — 45 of 46 datasets have it.

**Representation** — use `SMILES`. (`Smiles` appears on 15 older datasets; `SMILES`
on 23. Standardize on the uppercase form.) Also `3D coordinates`.

**Domain**: `Drug Discovery`, `Bioactivity`, `Toxicity`, `Physiology`, `Biology`,
`Biophysics`, `Quantum Chemistry`, `Physical Chemistry`, `Physical Property`,
`Molecular Properties`, `Electronic Properties`, `Organic Electronics`, `Solar Cells`,
`Environment`

**Task shape**: `Classification`, `Regression`, `Multi-Target`, `Multi-Task`,
`Multi-Label Classification`, `QSAR`, `Virtual Screening`, `Similarity`, `Benchmark`

**Method / provenance**: `DFT`, `QM`, `GFN2-xTB`, `Experimental`, `Molecular Docking`,
`Molecular Dynamics`, `ChEMBL`, `PubChem`, `CSD`, `Curated`, `Filtered`

**Chemistry**: `TransitionMetals`, `TMC`, `Polycyclic Aromatic Hydrocarbons`, `PAH`,
`Protein Target`, `Protein Binding`, `Inhibitors`, `OpenShell`, `Octahedral`

**Specific property**: `Solubility`, `Melting Point`, `Boiling Point`, `Vapor Pressure`,
`Half-Life`, `Partition Coefficient`, `Hydration Free Energy`, `OxidationState`,
`SpinSplitting`, `Skin Irritation`, `Skin Sensitization`

Four to seven tags is the useful range: representation, domain, method, and the
specific property.

## Writing `DESCRIPTION`

Two to five sentences, and it should answer:

1. **What** — how many compounds, of what kind.
2. **What was measured or computed** — the targets, with units.
3. **How** — experimental assay, or level of theory for computed data.
4. **Why it matters** — what the dataset is typically used for.

Good, from `open_melting_point`:

> The Jean-Claude Bradley Open Melting Point Dataset contains 28k melting point
> measurements for organic compounds. A highly curated subset of 3,041 validated
> measurements is also available, containing only data with multiple measurements and
> temperature ranges between 0.01°C and 5°C. Released as open data under a CC0
> license, this dataset is widely used in machine learning and computational chemistry
> research for developing models to predict melting points from molecular structure.

Name the level of theory when the data is computed — `TPSSh-D3BJ/def2-SVP`,
`GFN2-xTB` — since it determines whether the values are comparable to another source.
Do not pad with marketing; do not claim curation the source does not claim.
