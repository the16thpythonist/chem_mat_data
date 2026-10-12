# ChEMBL and ShareAlike — how ChemMatData answered it

**Status: DECIDED, 2026-08-19, by the maintainer — option A, comply.**
The twelve ChEMBL-derived datasets are labelled `CC-BY-SA-3.0` via a new
`license` metadata field. See [ADR 009](architecture_decisions/009_dataset_license_field.md).
Attribution (§4 step 1) was completed 2026-08-18; labelling (§4 steps 2 and 4)
on 2026-08-19. This file is kept as the evidence behind that decision.
Written 2026-08-18. Not legal advice — this is research assembled so that a person can
decide, with the evidence in one place instead of being re-derived every time.

---

## Why this file exists

The dataset-extraction run `esnuel_mca` (Ree et al., *Chem. Sci.* 2025,
[10.1039/D4SC07297A](https://doi.org/10.1039/D4SC07297A)) blocked at its licence gate on
two grounds. The first is specific to that paper and is dealt with in
[`../.dataset_work/esnuel_mca/reviews/license.md`](../.dataset_work/esnuel_mca/reviews/license.md).

The second is not specific to it at all: **the molecules come from ChEMBL, and ChEMBL is
CC BY-SA 3.0 — ShareAlike.** The extraction skill's policy escalates ShareAlike to a
human, correctly. But answering it for this one dataset would leave the same question to
be re-answered on the next ChEMBL-derived paper, and there will be many: ChEMBL underlies
a large share of med-chem ML data.

**And the question is not hypothetical or forward-looking.** ChemMatData already
redistributes eight ChEMBL-derived datasets, none of which attributes ChEMBL. That was
discovered while investigating this run, not by design.

So the decision to take is a project policy, and it has a retrospective half as well as a
prospective one.

---

## 1. What ChEMBL's licence actually says

Verified against three independent current sources — this is not inferred from context.

**ChEMBL interface documentation, "Data Licensing"**
(<https://chembl.gitbook.io/chembl-interface-documentation/about>):

> "Access to the web interface of ChEMBL is made under the EBI's Terms of Use. The ChEMBL
> data is made available on a Creative Commons Attribution-Share Alike 3.0 Unported
> License."

**ChEMBL FAQ** — this is the one that spells out propagation
(<https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/general-questions>):

> "The ChEMBL database is licensed under a Creative Commons Attribution-Share Alike 3.0
> Unported License. This allows use, redistribution and adaption of ChEMBL as long as
> appropriate attribution is given (e.g., cite the current ChEMBL paper) and **ensure that
> any adaptations are redistributed under the same license**."

**The current release's own files** —
<https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/README>:

> "The data content in ChEMBL is licensed under a highly permissive Creative Commons
> license - specifically the "CC Attribution-ShareAlike 3.0 Unported license", see LICENSE
> file. The required attribution should contain the url of the ChEMBL resource, **and also
> the release version**, e.g.:
>
>     ChEMBL data is from http://www.ebi.ac.uk/chembl - the version of ChEMBL is chembl_37.
>
> This should be visible on the entry portal for a web resource in which ChEMBL is
> integrated, or **contained with the documentation for any further distribution**."

**No change of terms was found.** Consistent from the 2009 licensing FAQ through
`chembl_37` today. No migration to CC-BY or CC0.

**One near-miss, recorded so nobody trips on it.** An August 2025 ChEMBL blog post
releases a text-mined IUPAC-name file "under a CC0 license". That is **SureChEMBL**, a
different resource and a single auxiliary file. It is not ChEMBL core and does not change
anything here.

**EMBL-EBI's umbrella terms neither add nor remove anything** — they place "no additional
restrictions … other than those provided by the original data owners", and resource-specific
terms "prevail" (<https://www.ebi.ac.uk/about/terms-of-use/>, rev. 2024-02-05).

### Three things ShareAlike here is *not*

Worth stating plainly, because ShareAlike gets pattern-matched to NonCommercial and they
are not the same animal:

- **It is not NonCommercial.** ChEMBL has been explicit since 2012 that the SA condition
  does not preclude commercial use. Nothing here stops a company training on the data.
- **It is not a prohibition on redistribution.** ChEMBL's own FAQ says the licence "allows
  use, redistribution and adaption". Redistribution is invited, on conditions.
- **It is not the CSD situation.** The tmQMg-L / CSD entry in
  [`rejected_datasets.md`](rejected_datasets.md) rests on a rightsholder whose published
  position is that redistribution requires a signed agreement. ChEMBL's position is the
  opposite. These two cases should not be reasoned about together.

---

## 2. What ShareAlike actually obliges — read from the licence text

The load-bearing distinction is **Adaptation vs Collection**, and CC BY-SA 3.0 Unported
defines both (<https://creativecommons.org/licenses/by-sa/3.0/legalcode.en>):

> **Adaptation** means a work based upon the Work … including in any form recognizably
> derived from the original, **except that a work that constitutes a Collection will not be
> considered an Adaptation** for the purpose of this License.

> **Collection** means a collection of literary or artistic works … in which the Work is
> included **in its entirety in unmodified form** along with one or more other
> contributions, each constituting separate and independent works in themselves, which
> together are assembled into a collective whole.

§4(b), the ShareAlike clause, ends with the sentence that bounds how far the obligation
travels:

> "This Section 4(b) applies to the Adaptation as incorporated in a Collection, **but this
> does not require the Collection apart from the Adaptation itself to be made subject to
> the terms of the Applicable License**."

§4(a) says the same thing about the unmodified Work.

**What that means concretely for this project.** ShareAlike would attach to *the individual
ChEMBL-derived dataset*. It would **not** reach:

- the ChemMatData catalogue as a whole,
- the other 38 datasets in it,
- the `chem_mat_data` package source code.

So the worst case is per-dataset licensing, not a copyleft event that swallows the project.
That is a materially smaller problem than "ShareAlike" sounds like at first reading, and
it is the single most useful fact in this document.

### The genuine open questions in the licence text

Three, and they are why this is a judgement call rather than a lookup:

1. **Is a processed ChemMatData dataset an "Adaptation" or a "Collection"?** A Collection
   requires the Work "in its entirety in unmodified form". A ChemMatData dataset is a
   filtered subset re-encoded as graphs — closer to an Adaptation. But the boundary is not
   crisp for data, and CC's definitions are written for literary and artistic works.

2. **Are the individual data points even protected?** A SMILES string and a measured Ki
   are facts. Facts are not copyrightable. What copyright can subsist in is the *selection
   and arrangement* of the database. Whether extracting 50,000 structures takes a
   protected "substantial part" of that selection is exactly the kind of question that has
   no clean answer without a lawyer.

3. **CC BY-SA 3.0 Unported does not mention database rights at all.** Checked directly
   against the legal code: **zero occurrences of "database" and zero of "sui generis"**.
   The §3 grant is a copyright grant, and the "Work" definition reaches "a compilation of
   data **to the extent it is protected as a copyrightable work**". The EU/UK *sui generis*
   database right is a separate right that 3.0 Unported does not expressly license — CC
   only addressed this properly in the ported EU 3.0 licences and in 4.0. EMBL-EBI sits in
   that jurisdiction. This cuts both ways: the licence may not clearly *grant* database
   rights, and equally may not clearly *impose* ShareAlike on extractions that engage them.

None of these is resolvable by reading harder. They are why this needs a person.

---

## 3. What the project's exposure actually is today

Ten scripts in `chem_mat_data/scripts/` are ChEMBL-derived. Eight are published in
`chem_mat_data/metadata.yml`; two exist as scripts only.

| Dataset | Published | Compounds | ChEMBL release recorded | ChEMBL in `sources` |
|---|---|---|---|---|
| `dud_e` | yes | 400,040 | **chembl_09** | **added 2026-08-18** |
| `riniker_1` | yes | 168,326 | **chembl_14** † | **added 2026-08-18** |
| `riniker_1_filtered` | yes | 105,294 | **chembl_14** † | **added 2026-08-18** |
| `bl_chembl_reg` | yes | 52,484 | **chembl_25** | **added 2026-08-18** |
| `riniker_2` | yes | 20,922 | not established | **added 2026-08-18** |
| `bl_chembl_cls` | yes | 8,780 | **chembl_25** | **added 2026-08-18** |
| `lipophilicity` | yes | 4,199 | not established | **added 2026-08-18** |
| `dpp4` | yes | 3,933 | not established | **added 2026-08-18** |
| `moleculeace` | script only | — | no | — |
| `pparg_binding` | script only | — | no | — |
| **`esnuel_mca`** (this run) | **blocked** | ~47,921 | **no** | — |

**≈764,000 compounds, about 12% of the collection's 6.2 million.**

Provenance is not in doubt for any of them — each script says so in its own docstring or
description (`dud_e`: "extracted from ChEMBL09 database"; `lipophilicity`: "from the ChEMBL
database"; `dpp4`: "was extract from ChEMBL with DPP-4 target"; `riniker_2`: "37 biological
targets from ChEMBL"). `lipophilicity` even carries `graph_chemblid` per molecule.

**As first found, none of the eight listed ChEMBL in `sources` and none carried a licence
note.** That gap was closed on 2026-08-18 — see §4 below.

† `chembl_14` for the two `riniker_1*` datasets was **not** recorded anywhere in this
project; it was recovered from the source publication (Riniker & Landrum, *J. Cheminform.*
**5**, 26 (2013), [PMC3686626](https://europepmc.org/article/PMC/PMC3686626)), which states:
*"This selection process was repeated with ChEMBL version 14 (Heikamp and Bajorath
extracted the actives from ChEMBL version 9)."* `riniker_2` (Subset II) comes from a later
paper and the benchmarking-platform README does not state a version, so it remains
unestablished rather than assumed to be 14.

Note also that `dud_e` and `bl_chembl_*` *do* record their release version — so for those,
ChEMBL's prescribed attribution string is writable today. For `esnuel_mca` it is not: the
paper never says which ChEMBL release Bayer's 50k subset came from, so
`"the version of ChEMBL is chembl_NN"` cannot be completed without asking the authors.

---

## 4. What compliance would concretely require

If the decision is to comply rather than to conclude SA does not attach, the work is
small and mostly mechanical:

1. **Attribution — DONE 2026-08-18.** Added to each affected dataset's `sources` and a
   `notes` entry:
   `ChEMBL data is from http://www.ebi.ac.uk/chembl - the version of ChEMBL is chembl_NN`,
   plus the ChEMBL paper (Mendez et al., *Nucleic Acids Res.* 2019, 47, D930–D940,
   [10.1093/nar/gky1075](https://doi.org/10.1093/nar/gky1075)). Where the release is
   unknown, say so explicitly rather than omitting the line.
2. **A per-dataset licence field.** `metadata.yml` has no licence key today. Adding one is
   independently worth doing — it is the natural place for this, and for the CC-BY
   attributions the extraction pipeline already collects and then drops.
3. **Mark that changes were made.** §3(b) requires an Adaptation to "take reasonable steps
   to clearly label, demarcate or otherwise identify that changes were made". The
   descriptions largely do this already by describing the processing.
4. **State CC-BY-SA-3.0 on the ChEMBL-derived datasets** if they are treated as Adaptations.
5. **Decide what this means for `cmdata`'s users** — arguably nothing more than that the
   dataset they downloaded carries a licence they can read.

Steps 1–3 are worth doing under *any* of the options below, including "SA does not attach":
attribution is a scientific-credit obligation before it is a legal one, and ChEMBL asks for
it in a specific form.

### What was actually done on 2026-08-18

Step 1 only. **Steps 2, 4 and 5 are untouched, because they encode a licence decision that
has not been taken** — nothing here labels any dataset CC-BY-SA-3.0.

- `https://www.ebi.ac.uk/chembl/` added to `sources` for all eight datasets.
- A `notes` entry added to all eight, naming ChEMBL as the source of the structures,
  stating that ChEMBL is licensed CC BY-SA 3.0, and citing Mendez et al.

  **The shipped credit is deliberately generic and names no release version** — a
  maintainer decision on 2026-08-18, taken for uniformity across the eight. Note that
  this is *technically incomplete* against ChEMBL's own formula, which asks that the
  attribution "contain the url of the ChEMBL resource, **and also the release version**".
  The versions that were established are not lost: they are recorded in the exposure
  table above, and can be folded into the shipped notes at any time if the project
  later decides to comply to the letter.
- The same text was written into **both** `chem_mat_data/metadata.yml` (so it is live now)
  **and** each dataset's `METADATA` dict in `chem_mat_data/scripts/` (so that re-running
  `cmmanage metadata collect` regenerates it rather than dropping it). The two were then
  checked to be byte-identical.
- `create_graph_datasets__dpp4.py` had **no `METADATA` dict at all** — its catalogue entry
  is hand-curated and a regeneration would have silently lost `verbose`, `tags` and
  `sources`. One was added mirroring the existing catalogue entry exactly.

**Not done, and worth someone's attention separately:** `dpp4` also declares no
`DATASET_TYPE`, so it would regenerate as `regression` while the catalogue records
`classification`. That is a pre-existing bug unrelated to licensing and was left alone.

---

## 5. The decision — taken 2026-08-19: **option A**

The maintainer chose to comply: label the ChEMBL-derived datasets CC-BY-SA-3.0 and
accept that the obligation propagates to downstream users. Recorded in
[ADR 009](architecture_decisions/009_dataset_license_field.md), which also documents
the field shape and the three code paths that had to learn about it.

Option C (ask EMBL-EBI) remains worth doing — it would settle whether ShareAlike
legally attaches at all, and option A is the conservative answer taken without that
confirmation. Doing so could only relax the position, never tighten it.

The four positions as they were put, for the record:

**A. Comply — treat ChEMBL-derived datasets as Adaptations under CC-BY-SA-3.0.**
Label them CC-BY-SA-3.0, add ChEMBL attribution with release versions where known.
*Cost:* downstream users of those eight datasets inherit a reciprocal obligation, and the
catalogue becomes licence-heterogeneous. *Benefit:* unambiguously correct, and matches what
ChEMBL asks for. Under this option `esnuel_mca` can proceed, with the caveat that its
release version is unknown.

**B. Attribute but do not propagate** — take the position that a ChemMatData dataset of
*computed properties keyed to structures* is not an Adaptation of ChEMBL's selection, or
that the underlying facts are unprotected. Add full attribution regardless.
*Cost:* rests on a legal judgement the project would be making without counsel; should be
written down as a deliberate position, not left implicit. *Benefit:* no obligation
propagates to users; matches what the wider field does — DeepChem/MoleculeNet, TDC and OGB
all redistribute ChEMBL-derived data under permissive licences.

**C. Ask EMBL-EBI.** They have a licensing contact and answer this kind of question.
*Cost:* time. *Benefit:* the answer is authoritative, settles all eight existing datasets
and every future one, and is quotable in this file forever.

**D. Stop adding ChEMBL-derived datasets** and decide separately what to do about the eight
already published. *Cost:* excludes a large fraction of med-chem ML data. Probably only
attractive if C comes back badly.

**What was recommended at the time:** **C first, then A or B
depending on the answer** — and do steps 1–3 of §4 immediately regardless, because they are
required under every option and cost an afternoon. C is cheap, and the difference between A
and B is precisely the thing EMBL-EBI is placed to answer and this project is not.

Whatever is decided, record it in this file with the date and who decided, and add a line to
[`rejected_datasets.md`](rejected_datasets.md) under "Checked and approved — do not
re-litigate" so the extraction agent stops re-researching it.

---

## 6. Not in question

- **ChEMBL permits redistribution.** Nobody needs to establish that; the FAQ says so.
- **Commercial use is fine.** SA is not NC.
- **The `chem_mat_data` package licence is unaffected.** §4(b) does not reach the
  Collection, let alone the code.
- **The MCA/MAA *values* in `esnuel_mca` are not ChEMBL's.** They are the authors' own
  r2SCAN-3c//GFN1-xTB computations. Only the `smiles` column is ChEMBL-derived. The same
  is true of most datasets in the table above — the structures are ChEMBL's, the
  measurements and computations are the papers'.

## 7. How to tell whether a new dataset is ChEMBL-derived

For the extraction agent, since this will keep coming up. ChEMBL provenance is usually one
step removed and stated only in passing:

- a `CHEMBL\d+` identifier column, or target names like `CHEMBL1862_Ki`;
- "drug-like subset", "bioactivity data", "curated from the literature" in a methods
  section, with ChEMBL named once in an acknowledgement rather than in the data statement
  — which is exactly how `esnuel_mca` states it;
- benchmark lineage: MoleculeNet `lipophilicity`, MoleculeACE, the RDKit benchmarking
  platform, DUD-E actives and Briem–Lessel recreations are all ChEMBL underneath.

`bace_cls` / `bace_reg` were checked and are **not** flagged as ChEMBL-derived by their own
scripts; they are literature-compiled. That was verified rather than assumed, and is
recorded here so it is not re-checked.
