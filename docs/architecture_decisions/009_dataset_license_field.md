# 009 — Recording a licence on each dataset

**Status:** accepted
**Date:** 2026-08-19
**Decided by:** the maintainer

## Context

ChemMatData redistributes other people's data, and until now carried no record of the
terms it does so under. `metadata.yml` had thirteen fields — `category`, `compounds`,
`description`, `full`, `min_version`, `notes`, `raw`, `sources`, `tags`,
`target_descriptions`, `target_type`, `targets`, `verbose` — and none of them was a
licence.

That was tolerable while licences were established informally. It stopped being tolerable
when the extraction pipeline (ADR 005) began *systematically* establishing a licence for
every dataset it produces: stage 1 researches the terms, records them in the run manifest,
and blocks on anything non-permissive until a human decides. The conclusion then went no
further. A maintainer could accept a ShareAlike obligation on the project's behalf and the
person who downloaded the dataset would never learn of it.

The immediate trigger was ChEMBL. Ten datasets in the collection derive their structures
from it, and ChEMBL is **CC BY-SA 3.0** — attribution *and* ShareAlike. The attribution
half was added on 2026-08-18. The ShareAlike half is an obligation that propagates to
downstream users, and there was nowhere to put it. The full analysis is in
[`../chembl_licensing_analysis.md`](../chembl_licensing_analysis.md).

## Decision

**Add an optional `license` field to dataset metadata**, shaped as a mapping:

```yaml
license:
    identifier: CC-BY-SA-3.0
    url: https://creativecommons.org/licenses/by-sa/3.0/
    note: >
        The molecular structures derive from ChEMBL, which is licensed CC BY-SA 3.0.
        The ShareAlike condition propagates, so this dataset is distributed under the
        same licence.
```

A mapping rather than a bare SPDX string, for two reasons. First, provenance is usually
the interesting part: "CC-BY-SA-3.0" alone does not say *why*, and the why is what a user
needs in order to reason about their own obligations. Second, some datasets cannot be
described by an identifier alone — `esnuel_mca` and `esnuel_maa` carry a ShareAlike
obligation inherited from ChEMBL while their own upstream archive states no terms at all,
and a bare identifier would assert a grant that was never made.

**The field is optional and its absence means "not established".** Most of the collection
predates any licence research. An absent `license` is not a claim that the data is freely
usable, and nothing should read it that way.

### Applied now

The twelve ChEMBL-derived datasets are labelled `CC-BY-SA-3.0`: `dud_e`, `riniker_1`,
`riniker_1_filtered`, `riniker_2`, `bl_chembl_reg`, `bl_chembl_cls`, `lipophilicity`,
`dpp4`, `moleculeace`, `pparg_binding`, `esnuel_mca`, `esnuel_maa`.

The rest of the collection is untouched. Licences will be filled in as datasets are
touched, and automatically for anything the extraction pipeline produces from now on.

## Consequences

**Users of the twelve inherit a reciprocal obligation.** Someone building on `lipophilicity`
or `esnuel_mca` must offer their derivative under CC-BY-SA-3.0 or a compatible licence.
This is a real restriction the project is now placing on its users, deliberately, because
it is the obligation ChEMBL attached to the data in the first place. Note it is **not** a
restriction on commercial use — ChEMBL is explicit that ShareAlike does not preclude it.

**Scope is bounded by the licence text itself.** CC BY-SA 3.0 §4(a) and §4(b) both end by
stating that the obligation does not extend to "the Collection apart from the
Work/Adaptation itself". The obligation attaches to the individual dataset. It does not
reach the catalogue, the other datasets, or the `chem_mat_data` package.

**Three code paths had to learn about the field**, all of which enumerate metadata keys
explicitly and would otherwise have dropped it:

- `chem_mat_data/cli.py` — `cmdata info` renders a **License** section, placed before
  Targets because whether a reader may use the dataset is more urgent than what is in it.
  Without this the label would exist only in a YAML file nobody reads.
- `chem_mat_data/manage.py` — the field list in `docs_compile_command`.
- `chem_mat_data/agent/gates.py` — the metadata gate warns when a draft carries no
  licence; the verification gate **errors** when a draft declares one and the shipped
  `metadata.yml` does not. The asymmetry is deliberate: forgetting to research a licence
  is a warning, but establishing one and failing to ship it defeats the entire point.

**What this does not settle.** Whether ShareAlike *legally* attaches to a filtered,
re-encoded dataset of computed properties is genuinely uncertain — facts are not
copyrightable, the protected element would be ChEMBL's selection, and CC BY-SA 3.0
Unported does not mention database rights at all (verified: zero occurrences of
"database" or "sui generis" in the legal code). This decision does not resolve that
question; it takes the conservative side of it. Asking EMBL-EBI directly would settle it,
and remains worth doing.

## Alternatives considered

**A bare `license: CC-BY-SA-3.0` string.** Simpler and easier to filter on, but cannot
express inheritance or an unstated upstream without asserting something false.

**Attribute without propagating** — credit ChEMBL, decline to pass the obligation on.
This is what DeepChem/MoleculeNet, TDC and OGB effectively do. Rejected as a legal
judgement the project would be making without counsel; the conservative option costs
users a licence label, the permissive one costs them a defence.

**Licence every dataset now.** Rejected as disproportionate: most have never had their
licence researched, so it would mean either weeks of work or a great many honest
"unknown" entries that add nothing.
