# Rejected and restricted datasets

Datasets that have been investigated and **must not be added to ChemMatData**, with the
reason and the evidence behind it.

This is the complement to `MISSING_DATASETS.md`, which lists candidates *to* add. The
point of both is the same: not to spend an afternoon re-deriving a conclusion someone
already reached.

**If you are the extraction agent:** read this before stage 1. If the publication URL,
the data record, or the upstream database you are about to investigate appears below,
stop and tell the human, quoting the entry. Do not re-research it and do not proceed on
your own judgement — the entries here were decided by a person, and only a person can
change one.

A rejection is not always permanent. Each entry records what would have to change.

---

## Rejected

### tmQMg-L — ligand library of transition metal complexes

| | |
|---|---|
| **Publication** | Kneiding, H., Nova, A. & Balcells, D. *Directional multiobjective optimization of metal complexes at the billion-system scale.* Nat. Comput. Sci. **4**, 263–273 (2024) — [10.1038/s43588-024-00616-5](https://doi.org/10.1038/s43588-024-00616-5) |
| **Data record** | https://github.com/uiocompcat/tmQMg-L · Zenodo [10.5281/zenodo.10374523](https://doi.org/10.5281/zenodo.10374523) |
| **What it is** | 35,466 ligands cut out of the tmQMg complexes, with SMILES, metal-coordinating atom indices, NBO-derived formal charges and computed descriptors |
| **Decided** | 2026-08-13, by the maintainer |
| **Evidence** | [`rejected/tmqmg_l_charge-license-review.md`](rejected/tmqmg_l_charge-license-review.md) |

**Two independent reasons, either sufficient on its own.**

**1. The data is CC BY-NC 4.0.** The NonCommercial condition would bind every downstream
ChemMatData user, not just this project. Verified rather than assumed: the `LICENSE` file
is the unmodified CC BY-NC 4.0 legal code (MD5 `fff4b145b044f5755732fda1d3aeda8c`, first
line `Attribution-NonCommercial 4.0 International`), zero occurrences of *ShareAlike* or
*NoDerivatives*, and Zenodo's API independently reports `cc-by-nc-4.0`.

> **Do not read GitHub's sidebar as evidence here.** The GitHub API reports this repo as
> `"spdx_id": "NOASSERTION", "name": "Other"`, which looks like an unclear licence and is
> not one — CC BY-NC 4.0 simply is not in GitHub's licence picklist. This misled the
> initial briefing for the run; the LICENSE file itself is unambiguous.

**2. The upstream is the Cambridge Structural Database**, reached via tmQM and tmQMg. The
CCDC's published position is that CSD data may not be redistributed, and its licence
requires a separate signed agreement to distribute derived datasets externally. CSD
refcodes appear verbatim in the redistributed files. No evidence of CCDC permission was
found.

**Revisit if:** the CCDC grants written permission for redistribution of CSD-derived
data, or the authors relicense the record under permissive terms. Neither is likely
without asking.

**Note:** this was extracted once, on 2026-08-13, as a deliberate end-to-end test of the
extraction pipeline, with the licence override approved *for the test only*. The dataset,
its script and its 400 MB of artifacts were removed the same day. The licence review is
kept because re-deriving the CCDC position took real work.

---

### polymer_tg — polymer glass transition temperatures

| | |
|---|---|
| **Publication** | *J. Chem. Inf. Model.* **61** (2021) — [10.1021/acs.jcim.1c01031](https://doi.org/10.1021/acs.jcim.1c01031) |
| **Data record** | The article's Supporting Information, files `ci1c01031_si_002.xlsx` and `ci1c01031_si_003.xlsx` |
| **What it is** | 566 polymers as repeat-unit SMILES with MD-simulated glass transition temperatures in °C |
| **Decided** | 2026-08-19, by the maintainer |
| **Status** | Script-only, never published |

**The article reserves rights and the one permissive branch of its SI terms does not
apply.** `pubs.acs.org` is Cloudflare-walled, which is why an earlier pass could only
argue from ACS's site-wide policy. A Wayback snapshot of the article record page
(2025-06-22) renders the Supporting Information block in full, and it shows: "Copyright
(c) 2021 American Chemical Society", a "Request reuse permissions" link, a hard paywall
("Purchase this article for 48 hours $48.00"), and **zero occurrences of "Creative
Commons"** anywhere on the page. The SI terms read:

> "Such files may be downloaded **by article for research use** (*if there is a public use
> license linked to the relevant article*, that license may permit other uses). Permission
> may be obtained from ACS for other uses through requests via the RightsLink permission
> system."

That permission is conditional and **the condition fails** — six independent indexes agree
no public-use licence is linked to this article. So what exists is permission to download
for research use, which is not permission to redistribute.

**A second, independent reason:** the upstream is **PoLyInfo (NIMS MatNavi)**, whose terms
are registration-gated and prohibit bulk acquisition ("Web scraping of data is
prohibited!").

**Revisit if:** ACS grants a RightsLink permission for redistribution of
`ci1c01031_si_002.xlsx` and `ci1c01031_si_003.xlsx` as a downloadable dataset — note ACS's
permissions page pre-declares that it does not grant permission for some redistribution
classes — or the authors republish the data under an open licence elsewhere. **Do not
re-research the licence itself**: this was settled from the authoritative record, not
inferred, and no further searching will change it.

---

### polymer_gas_permeability — polymer gas separation membranes

| | |
|---|---|
| **Publication** | *Sci. Adv.* (2022) — [10.1126/sciadv.abn9545](https://doi.org/10.1126/sciadv.abn9545) |
| **Data record** | https://github.com/jsunn-y/PolymerGasMembraneML |
| **What it is** | 778 polymers with experimentally measured gas permeabilities (log10 Barrer) for He, H2, O2, N2, CO2 and CH4 |
| **Decided** | 2026-08-19, by the maintainer |
| **Status** | Script-only, never published |

**A permissive wrapper licence contradicted by its own repository, over a database that
forbids bulk acquisition.** The repo carries a root `LICENSE` file containing the MIT
licence, but:

1. The repo's **own README points to a CC-BY-NC (NonCommercial) paper**, contradicting
   that MIT file.
2. The upstream is **PoLyInfo (NIMS MatNavi)** — registration-gated, bulk acquisition
   prohibited — together with the Membrane Society of Australasia Polymer Gas Separation
   Membrane Database.

This is deliberately treated differently from `zinc250k`, where an Apache-2.0 repository
licence *was* accepted: there the data file sat inside the licensed repository with
nothing contradicting it, and the upstream (ZINC) affirmatively permits sharing the
results of a screen. Here the repository contradicts itself and the upstream prohibits
the acquisition.

**Revisit if:** the authors clarify which licence governs the data files, or NIMS grants
redistribution permission for the PoLyInfo-derived values.

---

### price_small — Mcule purchasable-compound prices

| | |
|---|---|
| **Publication** | none — this is a commercial vendor export, not a published dataset |
| **Data record** | Mcule "Purchasable (In Stock)" export (`mcule_stock_210622_prices.csv`), mcule.com Kft. (Budapest) / Mcule, Inc. (Palo Alto) |
| **What it is** | Mcule IDs, SMILES and per-tier USD prices for 8.49M compounds; the script keeps a ~1% (~85k-row) sample as the target `price 4 (USD)` |
| **Decided** | 2026-08-19, by the maintainer |
| **Status** | Script-only, never published; script now refuses to run |

**No licence exists, and the vendor's terms affirmatively forbid the redistribution.**
This is the most thoroughly established negative in the licence review. Checked, over a
decade of snapshots: the current and June-2021 Terms of Use (byte-identical), the database
download page across **8 Wayback snapshots spanning 2015–2025**, `/legal` (404),
`/privacy`, the entire `doc.mcule.com` wiki, FAIRsharing and re3data. **Zero licence,
copyright or reuse statements anywhere.** Mcule's Terms of Use then go further and
prohibit what redistribution here would require.

This is a commercial vendor's price list — the compound prices are the product Mcule
sells. It is categorically unlike the research datasets elsewhere in this collection.

**Revisit if:** Mcule grants written permission to redistribute a sample of the
purchasable collection with prices. Nothing short of that changes the answer — **do not
re-research the licence**, it was settled from the vendor's own terms across a decade of
captures.

---

### tadf — TADF/OLED virtual screening molecules

| | |
|---|---|
| **Publication** | Gómez-Bombarelli, R. *et al.* *Design of efficient molecular organic light-emitting diodes by a high-throughput virtual screening and experimental approach.* Nat. Mater. **15**, 1120–1127 (2016) — [10.1038/nmat4717](https://doi.org/10.1038/nmat4717) |
| **Data record** | **None exists.** The file used was `source/tadf.csv` on a KIT group file share |
| **What it is** | 460,199 organic molecules from a high-throughput virtual screen for thermally activated delayed fluorescence OLED emitters |
| **Decided** | 2026-08-19, by the maintainer |
| **Status** | Was published (460,199 compounds); **removed from `metadata.yml`** and the script now refuses to run. The files were left on the remote share. |

**This is a provenance and consent failure, not a publisher-copyright one.** An earlier
pass blocked it on the Springer "all rights reserved" notice; **that reasoning was wrong
and has been corrected**, though the conclusion stands.

**1. The Springer SI notice does not govern this data.** The article's SI PDF contains
roughly 300 structures (46 lead candidates, ~168 fragments, ~88 calibration compounds) —
not 460,199. Decisively: all 46 SI lead-candidate SMILES were canonicalised and matched
**46/46** against `tadf.csv`, and our file holds the **full-precision** values the SI only
prints rounded — SI `0.121 / 0.061 / 2.94 / 0.07` against our
`0.121228915 / 0.06096116 / 2.936687531 / 0.065375355`. The SI is therefore a hand-rounded
0.01% extract **of this file**, not its source. Springer Nature also affirmatively
disclaims intellectual property in "datasets connected to published papers", so the
publisher is not a plausible rights-holder here at all.

**2. But nobody who might hold rights has granted anything.** An exhaustive first-hand
sweep — Zenodo, Figshare, Dataverse, OSF, GitHub, Hugging Face, Kaggle, DataCite under both
corresponding-author names, PubChem, and the Matter Lab's own open-data page — found **no
open release of this data anywhere**, ten years on. The nearest neighbours are all
different data: the Clean Energy Project (Figshare 9640427, CC BY 4.0) is a photovoltaic
screen with a different schema; Tartarus' `gdb13.csv` carries `license: null`; and
`the-matter-lab/Artificial-Design-of-Organic-Emitters` claims Apache-2.0 in its README with
no LICENSE file and is a different molecule set regardless.

**3. The authors' own most recent word on this exact dataset is access-gated.** Friederich,
Krenn, Tamblyn & Aspuru-Guzik, *Mach. Learn.: Sci. Technol.* **2**, 025027 (2021) — itself
published CC-BY-4.0, so these authors plainly know how to signal openness when they intend
it — states the data is **"available upon reasonable request from the authors"**. That is
the opposite of an open release.

**4. Samsung co-authorship clouds who could even grant it.** The original screen was a
joint academic–industrial project with Samsung SAIT / Samsung Research America co-authors,
so a willing academic author may lack unilateral authority to release it.

**5. The copy reached this project informally.** Researcher-to-researcher, with no
documented grant of redistribution rights.

**An open legal question is deliberately left open.** Springer's site-terms restriction is
contractual rather than copyright-based, which raises but does not settle whether a
computed SMILES + TD-DFT property table is "facts" below the originality threshold
(*Feist*) in the first place. Given the Ilm-NMR-P31 precedent in this same file — where EU
*sui generis* database-right terms were taken seriously rather than assumed away — and
ChemMatData's German context, that is a maintainer-level legal call, not one to settle in
passing.

**Revisit if:** the provenance can be established internally. The obvious lead is to ask
**Pascal Friederich (AiMat, KIT)** how `source/tadf.csv` came to be on the group's
bwSyncAndShare share, where it has sat since `visual_graph_datasets`' first release — he
was a postdoc in the Aspuru-Guzik group, which is the natural explanation for a 460k-row
file that exists nowhere public. If the answer is that the authors provided it, that is
provenance no amount of web searching can recover, and the dataset may be revivable on
that basis together with a permission from the authors. Failing that, a Springer Nature
permission request would be needed.

---

## Partially restricted

### Ilm-NMR-P31 — ³¹P NMR chemical shifts

| | |
|---|---|
| **Publication** | [10.1186/s13321-023-00792-y](https://doi.org/10.1186/s13321-023-00792-y) |
| **Data record** | https://zenodo.org/records/20762603 — record licence CC BY 4.0 |
| **Decided** | 2026-08-12, by the maintainer |
| **Evidence** | [`dataset_redistribution_analysis.md`](dataset_redistribution_analysis.md) |

The record's own licence is clean; the problem is upstream, and it splits the dataset in
two:

| Source block | Structures | Position |
|---|---|---|
| **HBP** — Tebby, *Handbook of ³¹P NMR Data*, CRC Press, **1991** | 6,608 | Sui generis term **expired 31 Dec 2012** (§137g(2) UrhG); US maker, likely outside Art. 11. **Usable.** |
| **Lan1/2/3** — Landolt-Börnstein III/40H, Springer, **2013/14** | 5,114 | Term runs to **~2029**; German maker. **Not usable.** |

**Decision: the middle path** — only the HBP-derived block may be used. The
Landolt-Börnstein-derived structures are held back until the database right expires
(~2029), unless Springer grants permission first.

**Second, independent reason not to publish this dataset at all as it stands:** 196 of
its geometries are a chemically different species from their own published formula —
gas-phase optimisation let ion pairs react and waters of hydration dissociate, with one
"bond" measuring 7.6 Å. That is a data-quality decision that has not been made, separate
from the licence one.

---

## Already published, needs a decision

**`tmqm` and `tmqmg` are live in the collection and derive from the CSD** through the same
chain documented in the tmQMg-L entry above. The CCDC position recorded there applies to
them by the same reasoning.

This is deliberately *not* filed as a rejection: they are already distributed, and
withdrawing a published dataset is a different decision from declining to add one. But it
is the same question, and it is now written down rather than implicit. Their own record
licences have not been examined in this project — that is the first thing to check if
this is ever taken up.

---

## Checked and approved — do not re-litigate

Not rejections. Recorded here so the same upstream question is not researched a third
time.

### DataWarrior `pKaInWater.dwar` — the source behind `pka_water` and `pka_acidic`

The article layer is clean (BMC applies the CC0 waiver to data in additional files). The
values themselves come from the DataWarrior reference file, which carries **no licence at
all** — and its author states on the project forum that the data was compiled from the
literature and *"the original paper reference got lost or was never collected"*, so no
upstream licence can be established even in principle.

**Approved by the maintainer on 2026-08-13** with the reason recorded verbatim in
`.dataset_work/pka_acidic/manifest.yml`. The decision covers the same Additional file 1
for both datasets.

---

## Settled — do not re-litigate

### ChEMBL and ShareAlike

**Decided 2026-08-19 by the maintainer.** ChEMBL is **CC BY-SA 3.0** — attribution and
ShareAlike. The project **complies**: ChEMBL-derived datasets are attributed to ChEMBL and
labelled `CC-BY-SA-3.0` in their metadata, so the reciprocal obligation is passed on to
whoever downloads them.

Applies to twelve datasets: `dud_e`, `riniker_1`, `riniker_1_filtered`, `riniker_2`,
`bl_chembl_reg`, `bl_chembl_cls`, `lipophilicity`, `dpp4`, `moleculeace`,
`pparg_binding`, `esnuel_mca`, `esnuel_maa`.

- Evidence and the options that were weighed: [`chembl_licensing_analysis.md`](chembl_licensing_analysis.md)
- The field, its shape, and what it commits the project to: [ADR 009](architecture_decisions/009_dataset_license_field.md)

**If you are the extraction agent and your dataset is ChEMBL-derived:** this no longer
needs escalating on the ChEMBL ground alone. Record `CC-BY-SA-3.0` in the licence stage
with ChEMBL named as `upstream_database`, put the same identifier in the metadata draft's
`license` field, and note that the gate will still escalate ShareAlike — that is expected,
and the human override should cite this entry. Any *other* licence problem with your
dataset is still yours to escalate on its own merits.

**Still worth doing, and would only relax this:** asking EMBL-EBI whether ShareAlike
legally attaches to a filtered, re-encoded dataset of computed properties. Option A was
taken as the conservative answer without that confirmation.

---

## Adding an entry

Record enough that nobody has to redo the research:

- **Every identifier the dataset can be reached by** — publication DOI, record URL,
  mirrors. An agent arriving from a different link must still find the entry.
- **What was actually verified**, not what was concluded. File hashes, quoted licence
  text, the URL of the terms. A conclusion with no evidence gets re-derived anyway.
- **Who decided and when.** These are human decisions; an agent may not add or remove one.
- **What would have to change** for the answer to be different.

If the reasoning is long, put it in `docs/rejected/` and link it, as with the tmQMg-L
licence review.
