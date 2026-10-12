reviewer: subagent

# Licence determination — tmQMg-L

**Verdict: ESCALATE TO HUMAN.** Two independent reasons, either of which is sufficient
under `references/license.md`:

1. The data licence is **CC-BY-NC-4.0** — a NonCommercial variant, explicitly in the
   ESCALATE column of the policy table.
2. The data is **derived from the Cambridge Structural Database (CSD)**, a commercial
   licensed database whose operator (CCDC) states in writing that CSD data may not be
   redistributed and that external distribution of *derived datasets* requires a
   separate signed agreement.

There is no charitable reading available here. Both findings point the same way.

---

## 1. Licence on the data record

### GitHub — `uiocompcat/tmQMg-L` (this is where the extraction script would download from)

- Repo: https://github.com/uiocompcat/tmQMg-L
- LICENSE file: https://github.com/uiocompcat/tmQMg-L/blob/main/LICENSE
  - Raw: https://raw.githubusercontent.com/uiocompcat/tmQMg-L/main/LICENSE
  - 19,346 bytes, MD5 `fff4b145b044f5755732fda1d3aeda8c`
  - First line, verbatim: `Attribution-NonCommercial 4.0 International`
  - It is the full, unmodified CC BY-NC 4.0 legal code. I checked the body: it contains
    the `NonCommercial means not primarily intended for or directed towards commercial
    advantage or monetary compensation` definition (line 116), and Section 2 grants only
    `reproduce and Share the Licensed Material, in whole or in part, for NonCommercial
    purposes only` and `produce, reproduce, and Share Adapted Material for NonCommercial
    purposes only` (lines 152–155). Greps for `ShareAlike` and `NoDerivatives` return
    **zero** hits, so this is plain BY-NC, not BY-NC-SA and not BY-NC-ND.
- README licence notice (bottom of https://github.com/uiocompcat/tmQMg-L#readme),
  verbatim:
  > This work is licensed under a
  > [Creative Commons Attribution-NonCommercial 4.0 International License](http://creativecommons.org/licenses/by-nc/4.0/).
  It carries the CC BY-NC 88x31 badge. The notice sits at the end of a README that
  describes only data files, so it covers the data, not merely code.
- **GitHub sidebar detection:** the GitHub API reports
  `"license": {"key": "other", "spdx_id": "NOASSERTION", "name": "Other"}`
  (https://api.github.com/repos/uiocompcat/tmQMg-L). This is *not* evidence of an unclear
  licence — CC BY-NC 4.0 simply is not in GitHub's licensee picklist, so any repo using it
  detects as "Other". The LICENSE file text itself is unambiguous.

### Zenodo — record 10374524 (concept DOI 10.5281/zenodo.10374523)

- Record page: https://zenodo.org/records/10374524
- Machine-readable metadata (human-openable): https://zenodo.org/api/records/10374524
  - `"license": {"id": "cc-by-nc-4.0"}`, `"access_right": "open"`
- DataCite view of the same record confirms:
  ```
  "rightsList": [{
    "rights": "Creative Commons Attribution Non Commercial 4.0 International",
    "rightsIdentifier": "cc-by-nc-4.0",
    "rightsIdentifierScheme": "spdx",
    "rightsUri": "https://creativecommons.org/licenses/by-nc/4.0/legalcode"
  }, {"rightsUri": "info:eu-repo/semantics/openAccess"}]
  ```
- The Zenodo record also ships a `LICENSE` file. I downloaded it and it is
  **byte-identical** to the GitHub one (same MD5 `fff4b145b044f5755732fda1d3aeda8c`).

### Do GitHub and Zenodo disagree?

**No — they agree on the licence.** Both say CC-BY-NC-4.0, and the LICENSE blobs are
byte-identical. I looked for a discrepancy because the task asked me to; there isn't one.

There *is* a **content** discrepancy worth recording, though it is not a licence issue:

| | GitHub `main` | Zenodo 10374524 |
|---|---|---|
| Ligand count | 35,466 (README) | 29k (record description) |
| Files | `ligands_misc_info.csv`, `ligands_fingerprints.csv`, `ligands_descriptors.csv`, `benchmarks/`, `xyz/`, `stages/` | `ligands_misc_info.csv`, `ligands_fingerprints.csv`, `ligands_descriptors.csv`, `stable.csv`, `weisfeiler_lehman_graph_hashes.csv`, `descriptors.zip`, `xyz.zip`, `EXCLUDED.txt` |
| Cites | the Nature Comput. Sci. article | the ChemRxiv preprint |
| Last data push | 2026-02-25 | 2023-12 |

So the Zenodo record is the older, smaller snapshot; GitHub `main` is the current one and
is what `raw.githubusercontent.com` would serve. Since both carry the same licence, the
choice of source does not change the licence answer — but the parent should record which
one it actually downloaded, because the row counts differ.

## 2. LICENSE / COPYING file in the repo tree

Repo root listing (via `GET /repos/uiocompcat/tmQMg-L/contents/`):

```
.gitignore (88)   LICENSE (19346)   README.md (7342)   benchmarks/
ligands_descriptors.csv (18824298)  ligands_fingerprints.csv (4213807)
ligands_misc_info.csv (19287178)    stages/   tmQMg-L.png   xyz/
```

Exactly one `LICENSE`, no `COPYING`, no `NOTICE`, no per-directory licence overrides.
Contents as described above: verbatim CC BY-NC 4.0.

## 3. Article licence and data policy (context only)

- **Published version — Nature Computational Science 4, 263–273 (2024)**,
  https://doi.org/10.1038/s43588-024-00616-5. The article is **not open access**. The
  rights line reads: *"The Author(s), under exclusive licence to Springer Nature America,
  Inc."* No Creative Commons licence on the article.
- **Data availability statement**, verbatim:
  > The tmQMg-L dataset can be accessed at https://github.com/hkneiding/tmQMg-L,
  > including the data for the charge assignment benchmark and the 1.37M space. The
  > dataset is also available via Zenodo at https://doi.org/10.5281/zenodo.10374523 …
  > All data are openly available.

  ("All data are openly available" is a statement about *access*, not about
  redistribution terms. The actual terms are the CC BY-NC 4.0 on the record.)
  `github.com/hkneiding/tmQMg-L` 301-redirects to `github.com/uiocompcat/tmQMg-L`, so
  the paper's link and the licence I read are the same repo.
- **Preprint — ChemRxiv**, https://doi.org/10.26434/chemrxiv-2023-k3tf2-v2 /
  https://chemrxiv.org/engage/chemrxiv/article-details/651051d4ed7d0eccc32252ea. The
  preprint manuscript is **CC-BY-NC-ND-4.0**. *(Trap 2 note: this is the licence on the
  preprint text, not on the data. It does not govern the CSV files, which come from
  GitHub/Zenodo under CC-BY-NC-4.0. But if anyone were to lift tables or figures out of
  the preprint PDF, NoDerivatives would apply and would forbid redistributing a processed
  derivative outright.)*
- No CCDC acknowledgement appears anywhere in the Nature article's Acknowledgements
  section (funding only: Marie Skłodowska-Curie 945371, RCN 314321 / 325003 / 262695,
  Sigma2 NN4654K).

## 4. Redistribution and attribution under CC-BY-NC-4.0

**Permitted:** copy, redistribute in any medium or format; remix, transform and build
upon the material — including as a processed ChemMatData dataset — **for NonCommercial
purposes only**.

**Forbidden / conditional:**
- Any use "primarily intended for or directed towards commercial advantage or monetary
  compensation" (Section 1(i)). ChemMatData is redistributed to unrestricted downstream
  users, some of whom will be commercial; the NC condition travels with the data and
  binds them, which is a commitment the project would be making on their behalf.
- Removing attribution, or applying "effective technological measures" that restrict
  further use (Section 2(a)(5)(B) / 3(a)(4)).
- Note Section 4 (Sui Generis Database Rights) applies: where database rights subsist,
  extracting and reusing a substantial portion of the contents is itself a licensed act
  under the same NC condition.

**Attribution required** (Section 3(a)): retain identification of the creators, the
copyright notice, a notice referring to the licence and the disclaimer, a URI/link to the
licence, and indicate if the material was modified. Concretely:

- Creators: **Hannes Kneiding, David Balcells, Ainara Nova** (Zenodo creator list) —
  Hylleraas Centre for Quantum Molecular Sciences, University of Oslo.
- Cite the paper: Kneiding, H., Nova, A. & Balcells, D. *Directional multiobjective
  optimization of metal complexes at the billion-system scale.* **Nat. Comput. Sci. 4**,
  263–273 (2024). https://doi.org/10.1038/s43588-024-00616-5
- Cite the data record: Kneiding, H., Balcells, D. & Nova, A. *tmQMg-L* (Zenodo, 2023).
  https://doi.org/10.5281/zenodo.10374523
- State the licence: CC BY-NC 4.0, https://creativecommons.org/licenses/by-nc/4.0/
- State that ChemMatData modified the data (reprocessed into graph/SMILES format).
- The predecessor chain should also be credited: tmQM (Balcells & Skjelstad, *J. Chem.
  Inf. Model.* **60**, 6135–6146, 2020, https://doi.org/10.1021/acs.jcim.0c01041) and
  tmQMg (Kneiding et al., *Digital Discovery*, 2023,
  https://doi.org/10.1039/D2DD00129B), plus the CSD itself (Groom, Bruno, Lightfoot &
  Ward, *Acta Cryst.* **B72**, 171–179, 2016).

## 5. Provenance — the upstream database

**This publication is NOT the original source of the structural values.** The chain is:

```
Cambridge Structural Database (CSD, CCDC)      ← experimental crystal structures, licensed/commercial
  └─ tmQM      (Balcells & Skjelstad, JCIM 2020; 2024 release = 108k TMCs)
        CSD structures filtered, then GFN2-xTB geometry optimisation + DFT properties
        repo: github.com/uiocompcat/tmQM  — GitHub licence: MIT
  └─ tmQMg     (Kneiding et al., Digital Discovery 2023; 74,547 TMC graphs)
        NBO-analysis-derived graph representations of tmQM
        repo: github.com/uiocompcat/tmQMg — GitHub licence: MIT
  └─ tmQMg-L   (this dataset; 35,466 ligands)
        ligands cut out of the tmQMg complexes + computed descriptors
        repo: github.com/uiocompcat/tmQMg-L — licence: CC-BY-NC-4.0
```

The tmQMg-L README states this in its first sentence, verbatim:
> This repository contains the data files of the ligands dataset tmQMg-L containing
> 35,466 ligands extracted from the tmQMg dataset containing transition metal complexes
> **reported in the Cambridge Structural Database**.

And the tmQM README:
> Structures were extracted from the 2024 release of the Cambridge Structural Database
> (CSD) with a series of filters…

**The CSD content is not merely an ancestor — it is present in the redistributed files.**
I range-read the first bytes of `ligands_misc_info.csv` from GitHub without downloading
it. The header is:

```
Unnamed: 0;name;stoichiometry;occurrence;parent_metal_occurrences;metal_bond_node_idx_groups;smiles;smiles_metal_bond_node_idx_groups;weisfeiler_lehman_graph_hash;stable_occurrence
```

and the first data row contains:

```
{'Pd': ['ELELEM-subgraph-0', 'CEBNON-subgraph-1', 'WEXDIO-subgraph-0', 'FUVMOZ-subgraph-2', ...]}
```

`ELELEM`, `CEBNON`, `WEXDIO`, `FUVMOZ` are **CSD refcodes** — CCDC-assigned identifiers,
confirmed by the README ("the TMC with CSD code XXYYZZ"). So the files carry CCDC
identifiers verbatim alongside geometries that descend from CSD coordinates (xTB-relaxed,
so transformed but not independent measurements).

### CCDC's stated terms

**Terms URL (most directly on point):**
https://support.ccdc.cam.ac.uk/support/solutions/articles/103000339607-can-i-redistribute-data-from-the-csd-

Title of that page is literally *"Can I redistribute data from the CSD?"*. Verbatim:

> - **No**; in order to ensure the database remains available for future generations, and
>   to keep The CCDC sustainable as a charity so we can continue to maintain and improve
>   the data, access is licenced.
> - If you have a full paid licence, this allows you to search, view, and use the data in
>   calculations and in the training of algorithms and models. **The licence does not
>   allow external sharing of original data from the CSD**, such as making bulk CIF files
>   available to others outside your organization.
> - **If you would like to share derived data or models, please contact us to check if
>   this is permitted within your license terms or if a separate agreement is needed.**

**Supporting URL — the standard CSD licence agreement:**
https://www.ccdc.cam.ac.uk/licence-agreement/

Clause 5, verbatim (emphasis mine):

> Subject to clauses (1) and (2), the Licensee may (i) integrate the licensed products
> with other software or systems used by the Licensee … and (ii) **derive datasets from
> the licensed products**. … **External distribution of such integrated software or
> systems or derived datasets requires a separate signed agreement with the Licensor,
> which may involve payment of extra fees.**

Clause 3, verbatim:

> The Licensee shall not sub-lease or assign the use of the products to any third party,
> or in any way (physically or online) redistribute or promulgate the products or any of
> their contents (including data and documentation) without the written and signed
> agreement of the Licensor.

Clause 11 adds that for derived products intended for external distribution, ownership of
new IP "will be subject to negotiation between the Licensee and Licensor".

So CCDC's position is explicit and recent (that FAQ was last modified 23 January): CSD
data is not redistributable, and *derived* datasets are the case they specifically ask you
to come and ask about. This is exactly Trap 1 from the policy — a permissive-looking
licence on the publication over a restrictive upstream — except here the publication's
licence is not even permissive.

CCDC does carve out two openly-licensed subsets, neither of which is tmQMg-L's source:
the **CSD MOF Collection** (CC-BY-NC-SA) and the **CSD Teaching Subset** (unrestricted).
tmQMg-L derives from the full CSD (2020 release for the original tmQM, 2024 release for
the current one), not from either carve-out.

### How the predecessors in the same group handled it

- `github.com/uiocompcat/tmQM` — GitHub licence **MIT**.
- `github.com/uiocompcat/tmQMg` — GitHub licence **MIT**.
- `github.com/hkneiding/PL-MOGA` (the algorithm, not data) — **MIT**.
- `github.com/uiocompcat/tmQMg-L` — **CC-BY-NC-4.0**.

Two observations:

1. The group **tightened** the licence for tmQMg-L relative to its own predecessors —
   MIT on tmQM/tmQMg, but BY-**NC** on tmQMg-L. That looks deliberate, not accidental,
   and reads as a decision not to permit commercial reuse of the ligand library.
2. Applying MIT to tmQM/tmQMg does not resolve the upstream question, it just doesn't
   raise it. MIT is a *software* licence applied to a repository of CSD-derived data; the
   authors cannot MIT-licence rights they do not hold in the underlying CSD content. The
   tmQM paper (open access, CC-BY, https://pmc.ncbi.nlm.nih.gov/articles/PMC7768608/)
   calls tmQM "an open data set freely available at GitHub" and contains **no CCDC
   acknowledgement or permission statement** in its Acknowledgements.

### Evidence of CCDC permission

**None found.** I looked in: the Nature article (Acknowledgements, Data availability,
Methods), the tmQMg-L README and repo tree, the Zenodo record metadata and description,
the tmQM README, and the tmQM paper on PMC. No statement anywhere that CCDC granted
permission to publish or redistribute the derived data, and no CCDC acknowledgement of the
kind CCDC's own licence clause 4 requires of publications. Reporting "none found" rather
than inferring one either way.

I want to be fair about what this does and does not mean: the authors publicly published
this data themselves, and the most likely explanation is that they either hold a licence
they read as permitting it, or obtained permission and did not record it publicly. What is
*not* available is a public basis on which ChemMatData — a third party with no CSD licence
— can rely to re-host it. CC BY-NC 4.0 Section 2(b)/5 grants only the rights the licensor
had to give and disclaims warranties, so a downstream re-host inherits the upstream
question rather than being shielded from it.

---

## Verdict under the project policy

**ESCALATE TO HUMAN.** Do not download-and-redistribute on my authority.

Mapping to the policy table in `references/license.md`:

| Finding | Policy row |
|---|---|
| CC-BY-NC-4.0 on the data record | ESCALATE — "CC-BY-NC and any NonCommercial variant" |
| Upstream CSD/CCDC, proprietary, derived-dataset redistribution needs a signed agreement | ESCALATE — non-permissive upstream (Trap 1) |

What redistributing this would commit the project to, stated plainly for the human:

1. **The NC condition propagates.** ChemMatData would be handing every downstream user a
   dataset they may not use for anything "primarily intended for or directed towards
   commercial advantage". That is a restriction ChemMatData's other datasets do not carry
   and that its users will not expect from a catalogue entry. If ChemMatData is happy to
   host NC data, it needs to be labelled as such at the point of download.
2. **The CSD question is not answered by the authors' licence.** Even with the NC
   condition accepted, CCDC's published position is that derived datasets require a
   separate signed agreement for external distribution. ChemMatData re-hosting 35k
   CSD-derived ligand geometries plus CSD refcodes is squarely the activity that FAQ
   describes. The clean resolutions are (a) ask CCDC directly — they explicitly invite
   this — or (b) ask the tmQMg-L authors whether their CSD licence covers third-party
   re-hosting of tmQMg-L, or (c) link rather than host.
3. **Option (c) is worth flagging** given the repo already has a "link-don't-host"
   proposal in its history: tmQMg-L is stably available at both GitHub and Zenodo with a
   concept DOI, so a catalogue entry that points at the upstream record and ships only the
   processing script would give users the dataset without ChemMatData redistributing
   anything. That sidesteps both findings at once.

A human decision is required before the prepare stage runs. If they approve, record their
words verbatim in `human_override.reason` and merge into — not over — the fields below.

---

## Machine-readable finding

```json
{
  "identifier": "CC-BY-NC-4.0",
  "evidence_url": "https://github.com/uiocompcat/tmQMg-L/blob/main/LICENSE",
  "redistribution": "restricted",
  "attribution": "CC BY-NC 4.0 requires retaining creator identification, the copyright and licence notices, a link to https://creativecommons.org/licenses/by-nc/4.0/, and an indication that the material was modified. Credit Hannes Kneiding, David Balcells and Ainara Nova (Hylleraas Centre, University of Oslo). Cite the paper: Kneiding, H., Nova, A. & Balcells, D. 'Directional multiobjective optimization of metal complexes at the billion-system scale.' Nat. Comput. Sci. 4, 263-273 (2024), https://doi.org/10.1038/s43588-024-00616-5. Cite the data record: Kneiding, H., Balcells, D. & Nova, A. 'tmQMg-L', Zenodo, https://doi.org/10.5281/zenodo.10374523. Credit the provenance chain: tmQM (Balcells & Skjelstad, J. Chem. Inf. Model. 60, 6135-6146, 2020) and tmQMg (Kneiding et al., Digital Discovery, 2023, https://doi.org/10.1039/D2DD00129B), and the CSD (Groom et al., Acta Cryst. B72, 171-179, 2016). NonCommercial condition: reuse must not be primarily intended for or directed towards commercial advantage or monetary compensation, and that restriction binds every downstream user. Zenodo (https://zenodo.org/api/records/10374524) states the identical licence and ships a byte-identical LICENSE file (MD5 fff4b145b044f5755732fda1d3aeda8c), so the two records agree.",
  "upstream_database": "Cambridge Structural Database (CSD), operated by the CCDC; reached via tmQM and tmQMg",
  "upstream_terms_url": "https://support.ccdc.cam.ac.uk/support/solutions/articles/103000339607-can-i-redistribute-data-from-the-csd-",
  "upstream_license": "proprietary"
}
```
