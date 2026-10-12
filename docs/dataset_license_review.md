# Dataset licence review — the 41 that need a decision

**Status: open. Awaiting maintainer decisions.**
Generated 2026-08-19 from a 92-agent research pass over 46 unlabelled datasets.

Each dataset was researched by its own agent (licence on the data, not the article;
upstream provenance named explicitly) and then assessed against the policy in
`.claude/skills/dataset-extraction/references/license.md`. Three were clear enough to
apply automatically. Two more were already decided and are recorded below. The remaining
**41 need a person**, but they are not 41 separate questions - they collapse into ten
families, and four of those families account for 17 datasets.

**Nothing in this file has been applied.** No script was edited for any dataset listed
here.

## Already settled, for completeness

| Dataset | Licence | How |
|---|---|---|
| `hopv15_exp` | CC-BY-4.0 | Applied automatically - Figshare record states it |
| `pfas_classify` | CC0-1.0 | Applied automatically - Figshare record states it |
| `reactaivate` | MIT | Applied automatically - repo LICENSE, the only record the paper names |
| `pka_water` | CC0-1.0 | Applied from your existing 2026-08-13 approval |
| `pka_acidic` | CC0-1.0 | Applied from your existing 2026-08-13 approval |

The two `pka_*` datasets were escalated by the agents even though you had already decided
that exact DataWarrior question on 2026-08-13. That is a briefing error of mine - the
extraction skill's stage 1 begins by checking `rejected_datasets.md` and I did not pass
that instruction to the research agents. Their conclusions matched yours; the work was
simply redundant.

---

## ZINC — 5 datasets

Five datasets whose structures trace to ZINC. One decision about ZINC's terms settles all five.

| Dataset | Own licence | Conf. | Redistribution | Upstream |
|---|---|---|---|---|
| `qm40_smiles` | CC-BY-4.0 | high | allowed | ZINC |
| `qm40_xyz` | CC-BY-4.0 | high | allowed | ZINC |
| `synth_binary_global` | none-found | low | restricted | ZINC (ZINC15/ZINC12, Shoichet Lab, UCSF — docking.org), via the ZINC-250k subset… |
| `synth_binary_local` | LicenseRef-ZINC-Terms-of-Use | medium | restricted | ZINC (docking.org, Irwin & Shoichet labs / UC Regents) — the 250k random drug-li… |
| `zinc250k` | Apache-2.0 | medium | restricted | ZINC |

**`qm40_smiles`** — Researcher found CC-BY-4.0 stated directly on the Figshare data record (article 25993060, high confidence) for the QM40 SMILES/property compilation itself — that part is clean. But QM40's underlying molecules are a screened subset of ZINC (162,954 of ZINC20's ~1B, ~0.016%) and the script keeps ZINC ids as graph_id, so ZINC's own terms of use are also in play. Those terms are custom, not SPDX-permissive: sharing 'results of a screen' is allowed, but redistributing 'major portions' of ZINC requires John Irwin's written permission (chemistry4biology@gmail.com). The researcher's own reasoning is that 0.016% clearly isn't a 'major portion' so redistribution is fine, but that's an interpretive judgment call on an undefined term, not a stated grant — exactly the kind of upstream-database question the policy reserves for a human. Decide: (a) accept the researcher's 'not a major portion' reading 
  · evidence: https://api.figshare.com/v2/articles/25993060

**`qm40_xyz`** — Do not apply. The data itself (QM40 Figshare record 10.6084/m9.figshare.25993060.v1) is CC-BY-4.0 with high confidence — verified via the Figshare API against the exact file the script downloads (MD5-matched), corroborated by the paper's Code availability section. That part alone would pass. But the molecular identities (ZINC ids/SMILES retained as graph_id) come from ZINC, whose docking.org terms of use are custom/bespoke — not CC0/CC-BY/MIT/BSD/Apache/ODC-BY — and explicitly state: "you may not redistribute major portions of ZINC without the express written permission of John Irwin." The researcher argues this doesn't apply because QM40's ~163k molecules are a small fraction (~0.02-0.07%) of ZINC's ~700M and fall under ZINC's own carve-out for sharing "the results of a ZINC search or a screen," but that is an interpretive judgment call about a non-permissive upstream licence, exactly w
  · evidence: https://figshare.com/articles/dataset/QM40_A_More_Realistic_QM_Dataset_for_Machine_Learning_in_Molecular_Science/25993060

**`synth_binary_global`** — Do not apply a license to synth_binary_global; it needs a human decision. The 249,455 SMILES in this dataset are the ZINC-250k subset (Gomez-Bombarelli et al. 2018, itself drawn from ZINC/docking.org, Shoichet Lab UCSF) — confirmed by matching compound count against the sibling zinc250k dataset already in this repo. ZINC's own terms of use (https://wiki.docking.org/index.php/ZINC) say it is "freely available to everyone to use" but that "significant portions... may not be re-distributed without express written permission of John Irwin" — free-to-use, not free-to-redistribute, with no SPDX equivalent. The three plausible download mirrors give conflicting/non-authoritative signals: Kaggle's copy self-declares ODbL-1.0 (ShareAlike copyleft — would itself escalate, and an uploader who isn't the rights holder can't grant ZINC's rights anyway); the chemical_vae GitHub repo wraps the CSV in a r
  · evidence: https://wiki.docking.org/index.php/ZINC

**`synth_binary_local`** — Do not apply a licence to synth_binary_local — escalate to a human. The dataset's 249,455 molecules are byte-identical to ZINC's public "250k_rndm_zinc_drugs_clean_3.csv" (the ChemicalVAE 250k subset); the labels are ChemMatData's own SMARTS-derived synthetic classification, so the only third-party content redistributed is the ZINC SMILES themselves. ZINC's live terms of use (wiki.docking.org/index.php/Terms_And_Conditions, current revision dated 2025-03-21) permit sharing "the results of a ZINC search or a screen of molecules from ZINC" but forbid redistributing "major portions of ZINC without the express written permission of John Irwin" — a threshold ZINC never defines. The chemical_vae GitHub repo that hosts the file carries a blanket Apache-2.0 LICENSE.txt, but that grant comes from the ChemicalVAE authors (A. Dorsk / Aspuru-Guzik group), who hold no rights in the ZINC compound coll
  · evidence: https://wiki.docking.org/index.php/Terms_And_Conditions

**`zinc250k`** — Do not apply a license to zinc250k (script: chem_mat_data/scripts/create_graph_datasets__zinc250k.py) — the file was not edited. The repository hosting the CSV (github.com/the-matter-lab/chemical_vae, formerly aspuru-guzik-group/chemical_vae) carries a repo-level Apache-2.0 LICENSE, but the researcher rated this only "medium" confidence because the license is software-oriented and the CSV itself carries no data-specific notice. More importantly, the underlying data is ~250,000 SMILES drawn from ZINC12 (Irwin et al. 2012), and ZINC's own terms of use (wiki.docking.org/index.php/Terms_And_Conditions, "ZINC - All versions") state you "may not redistribute major portions of ZINC without the express written permission of John Irwin" (chemistry4biology@gmail.com) — a restriction that travels with the data regardless of what license the Aspuru-Guzik group put on their own repo/code. Whether 250
  · evidence: https://github.com/the-matter-lab/chemical_vae/blob/master/LICENSE.txt

---

## GDB (Reymond group) — 4 datasets

Four datasets built on the Reymond group's enumerated chemical universes (GDB-13/GDB-17). `qm9_smiles` is among them, which makes this a decision about one of the most widely redistributed datasets in the field.

| Dataset | Own licence | Conf. | Redistribution | Upstream |
|---|---|---|---|---|
| `polyuniverse_epoxy` | CC-BY-4.0 | medium | restricted | GDB-13, GDB-17, PubChem |
| `polyuniverse_polyolefin` | CC-BY-4.0 | high | restricted | GDB-13 and GDB-17 (Reymond group, University of Bern) and PubChem — the olefin/v… |
| `polyuniverse_polyurethane` | CC-BY-4.0 | high | restricted | GDB-13 and GDB-17 (Reymond group, University of Bern) and PubChem (NIH/NCBI); Po… |
| `qm9_smiles` | CC0-1.0 | high | allowed | GDB-17 / GDB-9 (Reymond group chemical universe database, University of Bern) — … |

**`polyuniverse_epoxy`** — Do not apply a licence tag to polyuniverse_epoxy — this needs a human call. The Zenodo data record (12585902) itself is tagged CC-BY-4.0 (high confidence, machine-readable rights block), but the researcher flagged the overall redistribution verdict as "restricted", not "allowed": the record bulk-hosts GDB-13/GDB-17/PubChem-derived files alongside the polymer products, and the Reymond group's custom upstream terms for GDB explicitly forbid redistributing "GDB and large portions thereof" without written permission — there's no evidence such permission was obtained before the CC-BY-4.0 deposit. Epoxy_1M_p.csv (the file this script downloads) itself carries two monomer-SMILES columns (Smiles_Compound_1/2) that are verbatim GDB/PubChem entries, not generated structures, and the script propagates both into every graph as graph_monomer_1/graph_monomer_2 — so republishing as-is republishes rough
  · evidence: https://zenodo.org/records/12585902

**`polyuniverse_polyolefin`** — Escalating, not applying. The Zenodo data record (10.5281/zenodo.12585902) carries a machine-readable CC-BY-4.0 licence, confirmed via the Zenodo API and DataCite metadata — but the researcher marked redistribution "restricted", not "allowed", because that record also blankets multi-gigabyte verbatim extracts of GDB-13/GDB-17 (Reymond group) whose own terms forbid redistributing "large portions" without written permission from Jean-Louis Reymond. The polyolefin file we'd ingest (Polyolefin_1M_p.csv) is roughly 0.1% of GDB-13 and carries monomer SMILES columns (graph_monomer_1/graph_monomer_2) that trivially re-encode the source olefin structures — the researcher's read is that this is probably below "large portions" but flags it as an unresolved judgement call on a third party's undefined term, not something to wave through. A secondary, weaker concern: the 15 target columns are ML predi
  · evidence: https://zenodo.org/records/12585902

**`polyuniverse_polyurethane`** — The polyurethane CSV on Zenodo (record 12585902) is CC-BY-4.0, high confidence — but the file's Smiles_Compound_1/Smiles_Compound_2 columns carry ~1-2M verbatim GDB-13/GDB-17/PubChem monomer SMILES, which the script's add_graph_metadata hook republishes as graph_monomer_1/graph_monomer_2. The Reymond group's GDB terms (gdb.unibe.ch/downloads) forbid redistributing "large portions thereof" without written permission from Jean-Louis Reymond, conflicting with their own CC-BY-4.0 Zenodo deposit of GDB. Maintainer needs to choose: (a) drop the two monomer columns from the metadata hook and publish the clean CC-BY-4.0 polymer+target data, or (b) seek written permission from Reymond to keep them, or (c) accept the redistribution risk as-is. No file was edited.
  · evidence: https://zenodo.org/records/12585902

**`qm9_smiles`** — Do not apply yet — the data record itself is clean (Figshare collection 978904, article 1057646 holding dsgdb9nsd.xyz.tar.bz2, carries an explicit CC0-1.0 license via the Figshare API, high confidence, redistribution allowed), but QM9's molecules are a subset of GDB-17, and GDB's own download-page terms forbid redistributing "GDB and large portions thereof" without Jean-Louis Reymond's written permission — a custom, non-permissive clause, not CC0/CC-BY/MIT/etc. The researcher argues this likely doesn't bite (QM9 is ~8e-7 of GDB-17, the QM9 authors credit "access to GDB-17 data" from the Reymond group, and MoleculeNet/DeepChem/PyG have redistributed it openly for ~12 years unchallenged) but explicitly flagged it as the one point a human might want to overrule rather than resolving it charitably. Maintainer decision needed: (1) accept the "not a large portion / acknowledged access" reading
  · evidence: https://api.figshare.com/v2/articles/1057646

---

## US EPA EPI Suite / PHYSPROP — 4 datasets

Four datasets whose values trace to EPI Suite / the SRC PHYSPROP database. Note this is the same upstream already recorded as **proprietary** in `references/license.md` as a worked example of the upstream trap.

| Dataset | Own licence | Conf. | Redistribution | Upstream |
|---|---|---|---|---|
| `aqsoldb` | CC0-1.0 | high | allowed | OECD eChemPortal (6,110 records); US EPA EPI Suite / SRC datasets — WaterFragmen… |
| `elanos_bp` | CC-BY-4.0 | high | restricted | US EPA EPI Suite (experimental physicochemical property values; the bundled PHYS… |
| `elanos_vp` | CC-BY-4.0 | medium | restricted | US EPA EPI Suite (PHYSPROP database) |
| `open_melting_point` | CC0-1.0 | high | allowed | Alfa Aesar product catalogue; PhysProp / EPI Suite (SRC + US EPA); DrugBank; MDP… |

**`aqsoldb`** — Do not apply CC0-1.0 automatically. The AqSolDB Dataverse record (doi:10.7910/DVN/OVHAW8) is CC0-1.0 with high-confidence evidence (DataCite rightsList entry), but AqSolDB compiles 9 upstream sources and the AqSolDB authors state no licence for any of them. Roughly 1,210 records trace to the proprietary Beilstein/Reaxys database, ~1,291 to AQUASOL/SRC PHYSPROP (commercial), and the largest block, 6,110 records, to OECD eChemPortal under its own site terms — none established as permissive. A depositor's CC0 waiver can only release rights the depositor holds; it does not on its face clear rights (if any exist, e.g. EU sui generis database right) held over the upstream compilations. The maintainer needs to decide: accept AqSolDB's CC0 self-declaration as sufficient for the whole merged dataset (in line with widespread unchallenged redistribution since 2019 via DeepChem, Hugging Face, etc.),
  · evidence: https://api.datacite.org/dois/10.7910/dvn/ovhaw8

**`elanos_bp`** — Do NOT apply — escalate to a human. The Zenodo data record itself (zenodo.org/records/14392754) is unambiguously CC-BY-4.0 (high confidence, machine-readable SPDX id 'cc-by-4.0'), but the researcher deliberately recorded redistribution as 'restricted' rather than 'allowed' because the underlying boiling-point values trace upstream to US EPA EPI Suite / the PHYSPROP© database (Syracuse Research Corp.), reached via Zang et al. 2017's curated filtering. EPA's own copyright notice for EPI Suite is non-permissive ('Users may not alter, modify, merge, adapt, distribute, or prepare derivative works from the software'), with no data-specific carve-out or explicit grant that would let NovaMechanics relicense those values under CC-BY. Nothing in the Zenodo record, the companion Molecular Diversity article, or EPA's pages resolves whether NovaMechanics had standing to do that. Decision the maintain
  · evidence: https://zenodo.org/records/14392754

**`elanos_vp`** — Not applied — escalating. File untouched: chem_mat_data/scripts/create_graph_datasets__elanos_vp.py.

The Zenodo data record (https://zenodo.org/records/14364265, matching companion article Koutroumpa et al., Mol. Divers. 2025) is itself explicitly CC-BY-4.0 ("Creative Commons Attribution 4.0 International" / API license.id "cc-by-4.0"), and confidence in that reading is medium — the researcher would otherwise call it high. The blocker is upstream: the record's own description states the values "originate from the US EPA EPI Suite, filtered according to Zang et al. (2017)," and EPI Suite's copyright notice ("Users may not alter, modify, merge, adapt, distribute, or prepare derivative works from the software... PHYSPROP©") is restrictive, not permissive. So the entity that granted CC-BY-4.0 (NovaMechanics) does not obviously hold the right to relicense the underlying measurements it is re
  · evidence: https://zenodo.org/records/14364265

**`open_melting_point`** — The Figshare record for the Bradley Open Melting Point Dataset (article 1031637, the exact file the script downloads) is CC0-1.0, stated explicitly and corroborated independently via the Figshare API and DataCite/DOI metadata (high confidence). However the dataset is a compilation, and the named source pool (Alfa Aesar, MDPI, Bergstrom, PhysProp, DrugBank, Bell, Oxford MSDS, Hughes, Griffiths, Chemical Information Validation Spreadsheet) includes components the depositors do not appear to have had clear standing to release as CC0: DrugBank (CC BY-NC 4.0, redistribution requires a licence per DrugBank's own terms), Oxford University MSDS pages (no licence statement), and several ACS-copyrighted journal SI sets. Alfa Aesar and PhysProp/EPI, the two largest and most commercially sensitive components, were verifiably cleared by written agreement (documented on the project's 2011 blog) — but 
  · evidence: https://api.figshare.com/v2/articles/1031637

---

## PubChem — 4 datasets

Four datasets whose assay data comes from PubChem BioAssay.

| Dataset | Own licence | Conf. | Redistribution | Upstream |
|---|---|---|---|---|
| `muv` | none-found | medium | unclear | PubChem BioAssay |
| `muv_` | none-found | low | unclear | PubChem (PubChem BioAssay confirmatory screens, AIDs 466, 548, 600, 644, 652, 68… |
| `skin_irritation` | none-found | medium | restricted | ECHA REACH study results database (registered-substances dissemination portal, E… |
| `tox21` | none-found | low | unclear | U.S. Tox21 Program (NCATS/NIH Chemical Genomics Center, U.S. EPA, NIEHS/National… |

**`muv`** — No file was edited. The script currently pulls MUV compound lists from github.com/rdkit/benchmarking_platform, which carries no licence at all (GitHub API reports license: null, no LICENSE/COPYING/NOTICE file exists in the repo). The MUV originators (TU Braunschweig/Baumann group) gate their own copy behind an email request rather than an open licence, and the underlying 2009 ACS paper (Rohrer & Baumann, J. Chem. Inf. Model.) is all-rights-reserved with no Crossref licence entry. A genuinely open path does exist but was not adopted by the current script: Riniker & Landrum republished the identical MUV compound lists as Additional File 5 of their J. Cheminform. 5:26 (2013) article, which carries an unqualified CC BY 2.0 licence (https://creativecommons.org/licenses/by/2.0), and that article explicitly argues MUV has no redistribution restrictions (unlike MDDR). The upstream bioactivity da
  · evidence: https://www.tu-braunschweig.de/en/pharmchem/forschung/baumann/translate-to-english-muv

**`muv_`** — No licence exists for the MUV data anywhere in the chain the research agent could find. The authoritative source (Baumann group, TU Braunschweig) gates access behind an email request and states no terms. The GitHub mirror (rdkit/benchmarking_platform) has no LICENSE file and GitHub's API reports license:null. The actual host the muv_ script pulls from (DeepChem's S3 bucket, muv.csv.gz — confirmed by column-name schema 'MUV-466'...'MUV-859', 'mol_id', 'smiles') carries no terms of its own (DeepChem's MIT licence covers only the repo code, not the S3 data). The original journal article (Rohrer & Baumann 2009, JCIM) is closed-access with no Crossref licence. The one open-licence artifact in the chain — Riniker & Landrum 2013 (CC BY 2.0), whose Additional file 5 contains an equivalent compound list — is a third party's redistribution of Baumann's data under a journal blanket licence, and it 
  · evidence: https://www.tu-braunschweig.de/en/pharmchem/forschung/baumann/translate-to-english-muv

**`skin_irritation`** — No licence exists on the skin_irritation data anywhere in its provenance chain, so this needs a human decision, not an SPDX identifier. The dataset traces to ECHA's REACH study-results database (via either the ChemPharos ds14 record, marked &quot;All rights reserved&quot;, or the EHP paper's supplementary Excel, whose licence page explicitly defers to &quot;the terms of use of the cited database&quot;). ECHA's own Legal Notice permits reuse of individual data points under conditions (mandatory source credit in every copy) but explicitly forbids reproducing &quot;whole or substantial parts&quot; of its databases without prior written permission — and this 1,012-compound extract may or may not count as &quot;substantial.&quot; CAS numbers, if present in the shipped columns, are separately excluded (ACS property). The maintainer needs to decide: (1) whether a 1,012-record endpoint extract f
  · evidence: https://ehp.niehs.nih.gov/about-ehp/license

**`tox21`** — tox21 (chem_mat_data/scripts/create_graph_datasets__tox21.py) must go to a human — no licence is established for the data. Chain of custody: U.S. Tox21 Program (NCATS/NIH, EPA, NIEHS/NTP, FDA) qHTS assays -> 2014 Tox21 Data Challenge / PubChem BioAssay -> MoleculeNet benchmark -> DeepChem's tox21.csv.gz (identified by exact match of all 12 target columns + mol_id to deepchem's loader) -> ChemMatData's own hosted CSV. None of the four places checked state a data licence: the Tox21 challenge page's only licence text ("Public Domain Code") scopes explicitly to three code repos, not the datasets; DeepChem's repo-level MIT LICENSE covers "the Software", and PandeLab (the repo owner) never held rights in NCATS-generated data to license anyway; PubChem/NCBI's policy is permissive in tone but explicitly refuses to grant permission on the depositor's behalf ("NCBI cannot provide comment or unrest
  · evidence: https://tripod.nih.gov/tox21/challenge/data.jsp

---

## US EPA (other, public domain) — 2 datasets

Two datasets on US EPA data. US federal government works are generally public domain, so these are the likeliest of all the families to clear quickly - note `pfas_classify`, on the same EPA upstream, was applied automatically.

| Dataset | Own licence | Conf. | Redistribution | Upstream |
|---|---|---|---|---|
| `pfas_ttr_binding` | CC0-1.0 | high | allowed | US EPA CompTox Chemicals Dashboard - PFAS Master List (https://comptox.epa.gov/d… |
| `toxcast` | CC0-1.0 | medium | allowed | US EPA ToxCast / invitroDB (which itself aggregates 20+ assay vendors plus the N… |

**`pfas_ttr_binding`** — The Figshare deposit that hosts this file (article 12584645, "PFAS-Map", https://api.figshare.com/v2/articles/12584645) is CC0-1.0 at high confidence — but the 22 TTR-binding-percentage values in this specific dataset are not original data from that deposit's authors. They are transcribed from Weiss et al. 2009, Toxicol. Sci. 109:206-216 (doi:10.1093/toxsci/kfp055, Oxford University Press), a journal article with no open license recorded by Crossref (i.e. conventional all-rights-reserved copyright). The compound identities/SMILES additionally trace to the US EPA CompTox PFAS Master List, whose own terms were not independently verified by the researcher. The maintainer needs to decide whether a CC0 dedication applied by a downstream compiler (Su & Rajan) to a compiled deposit can be relied on to cover 22 numeric measurements originally published under conventional copyright by a different
  · evidence: https://api.figshare.com/v2/articles/12584645

**`toxcast`** — The ultimate data origin (US EPA ToxCast/invitroDB) is clearly CC0-1.0, but the actual file ChemMatData redistributes is not that EPA record — it's DeepChem's S3-hosted toxcast_data.csv, MoleculeNet's own processed/binarized compilation of 617 endpoints, published in an article licensed CC-BY-NC 3.0 with no separate license on the hosted CSV itself. That NonCommercial layer sits directly in the provenance chain of the redistributed bytes, which is exactly the upstream/version trap the license policy warns about, so I did not edit the script. The maintainer needs to decide whether to accept the researcher's argument that the NC clause can't re-encumber public-domain EPA facts (and approve with a recorded override), or require re-sourcing directly from the EPA Figshare record (DOI 10.23645/epacomptox.6062623) to get a clean CC0 grant on the file actually shipped; metadata.yml also currentl
  · evidence: https://www.epa.gov/comptox-tools/downloadable-computational-toxicology-data

---

## PoLyInfo (NIMS) — 2 datasets

Two polymer datasets whose properties come from NIMS PoLyInfo, which has its own registration-gated terms.

| Dataset | Own licence | Conf. | Redistribution | Upstream |
|---|---|---|---|---|
| `polymer_gas_permeability` | MIT | medium | forbidden | PoLyInfo (NIMS MatNavi) and the Membrane Society of Australasia (MSA) Polymer Ga… |
| `polymer_tg` | none-found | medium | forbidden | PoLyInfo (NIMS MatNavi polymer database) |

**`polymer_gas_permeability`** — Do not apply any licence to chem_mat_data/scripts/create_graph_datasets__polymer_gas_permeability.py; no edit was made. The GitHub repo's root LICENSE is plain MIT, but the paper states Dataset A was "manually collected from the PoLyInfo database" (pre-2005 rows) and "merged with data from the MSA database" (post-2005 rows). PoLyInfo/NIMS MatNavi's terms explicitly forbid third-party redistribution ("No reproduction, republication or distribution to third parties of any content is permitted without written permission of NIMS") and separately ban bulk/manual scraping. The MSA/CSIRO database states no licence at all, only a citation request. A downstream MIT grant (or the CaltechDATA CC0 deposit, which may not even contain this specific file) issued by a party that never held redistribution rights over the upstream values does not cure that restriction. Separately, the repo's README "Licen
  · evidence: https://github.com/jsunn-y/PolymerGasMembraneML/blob/main/LICENSE

**`polymer_tg`** — polymer_tg (script: chem_mat_data/scripts/create_graph_datasets__polymer_tg.py) must NOT be redistributed as-is. The 566-row Tg dataset comes from ACS Supporting Information (Tao, Varshney & Li, J. Chem. Inf. Model. 2021, 61, 5395-5413, DOI 10.1021/acs.jcim.1c01031), which is a closed-access article — no CC licence applies, and ACS's permissions policy routes all reuse through RightsLink and explicitly forbids posting ACS material online. Independently, the underlying polymer structures trace to NIMS's PoLyInfo database, whose terms state "No reproduction, republication or distribution to third parties of any content is permitted without written permission of NIMS" and prohibit bulk download/scraping. Evidence: https://pubs.acs.org/doi/10.1021/acs.jcim.1c01031 (article, closed access) and https://polymer.nims.go.jp/en/ (PoLyInfo terms). To publish this dataset the maintainer would need e
  · evidence: https://pubs.acs.org/doi/10.1021/acs.jcim.1c01031

---

## ECHA REACH — 1 dataset

One dataset drawing on ECHA REACH registration dossiers, whose dissemination terms restrict reuse.

| Dataset | Own licence | Conf. | Redistribution | Upstream |
|---|---|---|---|---|
| `skin_sensitizers` | none-found | low | restricted | ECHA REACH study results database (dissemination portal / eChemPortal, ECHA and … |

**`skin_sensitizers`** — No licence exists on this data record -- do not edit chem_mat_data/scripts/create_graph_datasets__skin_sensitizers.py. The researcher found no licence grant anywhere in the chain: the STopTox GitHub repo (github.com/joyvb/stoptox) reports license:null with no LICENSE file, the ChemPharos ds15 record page carries no licence/terms statement, and the EHP journal's own licence page states research-article SI data are copyright-free but the *presentation* (e.g. the Excel file) is not, and explicitly defers to any cited database's own terms. About half the ~1,000 compounds trace to the ECHA REACH study-results database, whose Legal Notice permits reuse only under conditions (mandatory "Source: European Chemicals Agency" credit in every copy, no alteration, no replication of a "substantial part" of the database, no automated/systematic extraction, and CAS numbers remain ACS property and cannot 
  · evidence: https://ehp.niehs.nih.gov/about-ehp/license

---

## NonCommercial / ShareAlike on the dataset itself — 6 datasets

Six datasets where the restriction is on the dataset's own licence, not an upstream. `half_life` is the enviPath case already documented in `references/license.md`; `compas_*` are CC-BY-NC-SA. These are the clearest-cut refusals unless you accept NC/SA terms.

| Dataset | Own licence | Conf. | Redistribution | Upstream |
|---|---|---|---|---|
| `ames` | CC-BY-NC-4.0 | high | restricted | CCRIS (NLM), GENE-TOX, US EPA, and Vitic (Lhasa Limited) — plus the published Ka… |
| `compas_1x` | CC-BY-NC-SA | low | restricted | null |
| `compas_2x` | CC-BY-NC-SA-4.0 | low | restricted | null |
| `compas_3x` | CC-BY-NC-SA-4.0 | low | restricted | null |
| `half_life` | CC-BY-NC-SA-4.0 | low | restricted | enviPath (EAWAG-SOIL package) |
| `sider` | CC-BY-SA-4.0 | medium | restricted | SIDER 4.1 (Side Effect Resource, EMBL / Kuhn et al.) — reached via MoleculeNet/D… |

**`ames`** — Did not edit chem_mat_data/scripts/create_graph_datasets__ames.py — this needs a human decision, not an APPLY.

What the licence appears to be: CC BY-NC 4.0, stated directly on the ACS/figshare Supporting Information record for the source paper (Hansen et al. 2009, J. Chem. Inf. Model. 49(9), 2077-2081, DOI 10.1021/ci900161g; figshare article 2825590, confirmed via both the rendered page and the figshare API's license.name field). The zip already present in the repo (chem_mat_data/agent/artifacts/downloads/ames_mutagenicity_benchmark_dataset.zip) is byte-identical (md5 match) to that figshare file, so this is confidently the same dataset the script consumes (column names Canonical_Smiles/CAS_NO/Activity and the 6512-compound count both match). Researcher confidence: high.

What the evidence was: CC BY-NC 4.0 permits redistribution/modification but only non-commercially and with attributi
  · evidence: https://acs.figshare.com/articles/dataset/Benchmark_Data_Set_for_in_Silico_Prediction_of_Ames_Mutagenicity/2825590

**`compas_1x`** — compas_1x's licence cannot be resolved from the record: the GitLab repo porannegroup/compas (source of compas-1x.csv) contains a direct, unresolved conflict — the root LICENSE file is verbatim CC BY 4.0, but both README.md files explicitly state "licensed under a CC-BY-NC-SA license," and that sentence has survived edits made well after the LICENSE file's last change (2024-01-03). The NC restriction is independently corroborated by the paper's own ACS Figshare deposit (collection 6118490, both records tagged CC BY-NC 4.0). A CC0 LICENSE file also sat in the repo for 9 days in Dec 2023 before being overwritten, ruling out a stable public-domain reading. No upstream database is implicated — data is de novo GFN2-xTB enumeration, not a compilation. Researcher reports identifier CC-BY-NC-SA, redistribution "restricted", confidence "low". The maintainer needs to decide: (a) email porannegroup@
  · evidence: https://gitlab.com/porannegroup/compas/-/blob/main/COMPAS-1/README.md

**`compas_2x`** — Do not apply — this needs a human decision. The GitLab repo the script actually downloads compas-2x.csv from (porannegroup/compas) states in two places — the root README.md and COMPAS-2/readme.md, the latter sitting right next to the CSV — "The COMPAS database ... is licensed under a CC-BY-NC-SA license" (no version given; researcher assumed 4.0). That text has survived four README edits after 2024-01-03 and is still live today. Conflicting evidence points the other way: the Figshare data deposit (10.6084/m9.figshare.24347152.v3, the canonical record for the data files) is tagged CC BY 4.0 in its metadata; the repo's LICENSE file was deliberately replaced with verbatim CC BY 4.0 text on 2024-01-03 (same day as the Figshare deposit); and the Sci Data 2024 paper's availability statement calls the GitLab repo "licensed under a CC-BY license" — though that statement is about "All code", and 
  · evidence: https://gitlab.com/porannegroup/compas/-/blob/main/COMPAS-2/readme.md

**`compas_3x`** — The COMPAS-3 GitLab repo (gitlab.com/porannegroup/compas), which the script downloads compas-3x.csv from, carries two conflicting current licence statements: the root LICENSE file is verbatim CC BY 4.0, but both the root README and the COMPAS-3-specific README state "It is licensed under a CC-BY-NC-SA license" (added after the CC-BY LICENSE swap, reaffirmed through July 2025 edits, and consistent with the group previously removing an MIT LICENSE for "incompatibility with the current license. CC-BY-NC-SA"). Under the project's binding-conservative policy this reads as CC-BY-NC-SA, which would forbid ChemMatData's current redistribution (no NC restriction, no ShareAlike re-licensing on the mirrored copy). The maintainer should email porannegroup@technion.ac.il (contact given in both READMEs) for a written, binding answer on which licence governs the data, then apply that same answer to com
  · evidence: https://gitlab.com/porannegroup/compas/-/blob/main/COMPAS-3/README.md

**`half_life`** — Do not edit the script. This is the skill's own documented Trap-1 example: the paper (Hafner, Fenner & Scheidegger, Environ. Sci. Technol. Lett. 2023) is genuinely CC-BY-4.0 — confirmed via Crossref (license URL creativecommons.org/licenses/by/4.0) and the Europe PMC permissions block — but the half-life data was extracted from the EAWAG-SOIL package on enviPath, whose own licence documentation (https://community.envipath.org/t/envipath-license/109) states the Soil package is CC-BY-NC-SA-4.0 and explicitly claims a viral effect: "even if only a very small part of the data set is based on enviPath, your full data set needs to be published under CC BY-NC-SA 4.0 and is not allowed to be used in a commercial context." The script's current METADATA['license'] entry (identifier CC-BY-4.0) understates this and should be treated as wrong, not confirmed. A person needs to decide: (1) whether Chem
  · evidence: https://community.envipath.org/t/envipath-license/109

**`sider`** — Do not apply. The sider script (chem_mat_data/scripts/create_graph_datasets__sider.py) was not edited. The upstream SIDER 4.1 database (reached via MoleculeNet/DeepChem, the actual source of ChemMatData's sider.csv per column-name/task-count match) states its download-page terms as CC-BY-SA-4.0 as of today, quoted from https://sideeffects.embl.de/download/: "The results of this mapping are available under a Creative Commons Attribution-Share Alike 4.0 License." ShareAlike means any redistribution of ChemMatData's raw CSV or processed graph dataset would itself have to carry CC BY-SA 4.0 (or a CC-compatible licence) — it cannot go out under ChemMatData's presumably more permissive collection-wide terms, which is a policy question for the maintainer, not something an agent can decide. Additionally the researcher flags real uncertainty worth a human's attention before any decision: (a) ever
  · evidence: https://sideeffects.embl.de/download/

---

## Commercial / proprietary supplier — 5 datasets

Five datasets whose structures or values come from a commercial catalogue or subscription database.

| Dataset | Own licence | Conf. | Redistribution | Upstream |
|---|---|---|---|---|
| `beet` | none-found | low | forbidden | PPDB — Pesticide Properties DataBase (Agriculture & Environment Research Unit, U… |
| `boron1000` | MIT | medium | allowed | ETHmodlab/lsfml literature borylation dataset (Nippa et al., Nat. Chem. 2024, DO… |
| `contaminant_reactivity` | none-found | medium | forbidden | NIST Solution Kinetics Database (NDRL/NIST, NIST Standard Reference Database 40,… |
| `photo_oliogomers` | CC-BY-4.0 | high | allowed | eMolecules |
| `price_small` | none-found | high | forbidden | Mcule (mcule.com Kft. / Mcule, Inc.) purchasable compound database |

**`beet`** — Do not host beet's data. The research agent traced a three-hop provenance chain (Wei-lab data library -> JPCLett 2021 paper -> Venko et al. 2018 SAR/QSAR paper -> PPDB/Como et al. compilation) and found no licence grant at any hop: the Wei-lab download page carries no licence or copyright statement (checked via two Wayback captures since the live site is down for maintenance), the associated GitHub repo has license:null and doesn't even contain the beet data, the origin article (Venko et al. 2018) is paywalled with Crossref license:null, and the likely upstream compilation (PPDB, University of Hertfordshire) is explicitly "All Rights Reserved" with republication requiring prior written permission or a paid licence. No file was edited. To proceed, the maintainer would need to either obtain explicit written permission from the Wei lab and/or the Venko/Novič authors (katja.venko@ki.si, marj
  · evidence: https://web.archive.org/web/20260607200058/https://weilab.math.msu.edu/DataLibrary/2D/

**`boron1000`** — Do not apply a license to this script as-is. The research agent found the download is not actually the BORON1000 dataset the script's name/DESCRIPTION/sources claim: real BORON1000 is Reaxys-derived and proprietary (Elsevier license) and was deliberately never published as structures -- only Reaxys IDs. What this script actually pulls (github.com/ruslankotl/rxn-data-proc, data/roche/classification/*) is instead a byte-verified copy of the Nippa et al. literature borylation set from ETHmodlab/lsfml, which does carry a root MIT license (https://github.com/ETHmodlab/lsfml/blob/master/LICENSE) -- but the immediate GitHub host has no license file of its own, so the MIT permission is read as traveling through an unlicensed intermediate copy, and MIT's text nominally covers 'the Software' rather than data files explicitly (hence the researcher's 'medium', not 'high', confidence). A maintainer n
  · evidence: https://github.com/ETHmodlab/lsfml/blob/master/LICENSE

**`contaminant_reactivity`** — Do not host this dataset. The only source is the paywalled Supporting Information of an ACS journal article (Zhong, Zhang & Zhang, Environ. Sci. Technol. 2022, DOI 10.1021/acs.est.1c04883); no open licence exists (Unpaywall: closed/no OA locations; Crossref license field lists only STM sharing policies, not an open licence). ACS's Terms of Use explicitly forbid redistribution, incorporation into databases, and posting to third-party repositories for this "free-to-read" SI content. Independently, the SI's own "Original sources" sheet names the NIST Solution Kinetics Database (NIST SRD 40) as an original source for the ClO2 and O3 subsets, and NIST SRD is copyrighted by the U.S. Secretary of Commerce with reproduction requiring prior permission (Public Law 90-396) — the usual "US government work = public domain" rule does not apply. A maintainer would need to decide: (a) whether to pursue 
  · evidence: http://web.archive.org/web/20250112130223/https://pubs.acs.org/page/policy/termsofuse.html

**`photo_oliogomers`** — Do not apply yet — needs a human call on the eMolecules exposure. The data itself is CC-BY-4.0 (Zenodo record 11580890, high confidence; an MIT LICENSE file is also bundled in the archive but that's a second permissive option, not a conflict) and would otherwise pass cleanly. The blocker is that the 2,200-molecule space's acceptor building blocks were purchased from eMolecules, a commercial catalogue whose terms of use the researcher could not find (all candidate ToS URLs 404). ChemMatData would not carry any eMolecules-proprietary data directly — only SMILES + measured photostability targets — but MonomerFeatures.csv (an intermediate file in the pipeline) does carry eMolecules catalogue fields (A_VersionID, A_ParentID, A_CAS, plate location), which the researcher recommends dropping before publishing. Decision needed: (1) confirm eMolecules' terms don't restrict even this limited deriva
  · evidence: https://zenodo.org/records/11580890

**`price_small`** — Do not publish/keep price_small. The script (chem_mat_data/scripts/create_graph_datasets__price_small.py) has no data-source citation and no METADATA license block; it reads a local file, /media/data/Downloads/mcule_stock_210622_prices.csv, whose name and 'price 4 (USD)' column identify it as an export from Mcule (mcule.com), a commercial compound-supplier aggregator, not a research database. Mcule's Terms of Use (https://mcule.com/terms/) grant no redistribution licence: clause 2.9 forbids disclosing/sub-licensing 'any material ... from the Website to third parties' and limits downloads to the client's own business use; clause 2.10 forbids editing/altering site content; clause 2.8 bars automated scraping. Free public Mcule downloads cover structures/IDs only — prices are a subscriber-gated feature, so the target column here is the commercially sensitive part of Mcule's product, not the 
  · evidence: https://mcule.com/terms/

---

## No licence found, no single shared upstream — 8 datasets

Eight datasets where no licence statement could be found at all, and no common upstream ties them together. Each needs its own judgement. Several are MoleculeNet staples that the field redistributes universally without a stated licence.

| Dataset | Own licence | Conf. | Redistribution | Upstream |
|---|---|---|---|---|
| `bace_cls` | none-found | low | unclear | null |
| `bace_reg` | none-found | low | unclear | ChEMBL |
| `bbbp` | none-found | low | restricted | MoleculeNet / DeepChem (which itself redistributes the Martins et al. 2012 liter… |
| `clintox` | none-found | low | unclear | SWEETLEAD (FDA-approved drug structures) and AACT — Aggregate Analysis of Clinic… |
| `esol` | none-found | medium | forbidden | Abraham group (University College London) literature compilation of aqueous solu… |
| `freesolv` | CC-BY-4.0 | medium | restricted | null |
| `hiv` | none-found | medium | allowed | NCI Developmental Therapeutics Program (DTP) AIDS Antiviral Screen |
| `tadf` | none-found | low | forbidden | null |

**`bace_cls`** — bace_cls's license cannot be established: no license is stated anywhere in the chain (DeepChem/MoleculeNet S3 data file, its README, and the repo's MIT LICENSE which the researcher confirms does not cover hosted datasets), redistribution is unclear rather than allowed, and confidence is low. The underlying compound set traces to a closed-access ACS article (Subramanian et al. 2016, J. Chem. Inf. Model., DOI 10.1021/acs.jcim.6b00290) whose Supporting Information carries ACS copyright with no stated reuse terms — per policy, unstated-terms SI escalates by default. The only actual license found anywhere (CC BY-NC 3.0) sits on the MoleculeNet paper's own text, not the data, and would forbid commercial redistribution regardless. A maintainer with institutional journal access would need to check the ACS SI directly for any data-sharing statement, and separately confirm whether the compound set
  · evidence: https://pmc.ncbi.nlm.nih.gov/articles/PMC5868307/

**`bace_reg`** — No file was edited — chem_mat_data/scripts/create_graph_datasets__bace_reg.py is unchanged. bace_reg re-reads ChemMatData's own raw 'bace' CSV (no ingestion script exists for it in-repo), which the researcher identified by exact column-schema match as DeepChem/MoleculeNet's bace.csv, sourced from Subramanian, Ramsundar, Pande & Denny, J. Chem. Inf. Model. 2016 (DOI 10.1021/acs.jcim.6b00290, evidence: https://pubs.acs.org/doi/10.1021/acs.jcim.6b00290). No license statement exists anywhere in the chain: the ACS Supporting Information carries an explicit "Copyright 2016 American Chemical Society" with no CC/AuthorChoice marker (Unpaywall confirms the article is closed access); DeepChem's repo-level MIT LICENSE is scoped to "the Software" and arguably does not cover the CSV data files hosted on a separate S3 bucket outside the repo (no LICENSE/README there, bucket listing AccessDenied); the 
  · evidence: https://pubs.acs.org/doi/10.1021/acs.jcim.6b00290

**`bbbp`** — bbbp has no valid license anywhere in its provenance chain and must not be marked APPLY. The script downloads from https://huggingface.co/datasets/chao1224/MoleculeSTM (resolve/main/MoleculeNet_data/bbbp/raw/BBBP.csv) — a mirror that declares zero data license (HF API cardData:null, no license tag, no LICENSE file among 149 files). Its companion GitHub repo's NVIDIA Source Code License restricts use to non-commercial research/evaluation. Tracing further up: MoleculeNet/DeepChem grants nothing over this specific file (BBBP.csv lives outside DeepChem's MIT-licensed repo, on an unlicensed S3 bucket; DeepChem's own docs say "MoleculeNet already supports datasets released under different licenses"). The ultimate origin, Martins et al. 2012 (J. Chem. Inf. Model., DOI 10.1021/ci300124c), is a closed-access ACS paper with no license registered anywhere (confirmed via Crossref, OpenAlex, Unpaywal
  · evidence: https://huggingface.co/datasets/chao1224/MoleculeSTM

**`clintox`** — ClinTox (chem_mat_data/scripts/create_graph_datasets__clintox.py) has no licence anywhere in its chain, and the researcher's search was thorough, not lazy: no source URL was ever recorded for this dataset in the repo, so provenance was reconstructed from a byte-level column/row match (smiles,FDA_APPROVED,CT_TOX; 1484 rows) to DeepChem/MoleculeNet's ClinTox, served from a bare S3 bucket (deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz) that carries no licence metadata and is not even inside the deepchem GitHub repo (whose MIT LICENSE, scoped to "the Software," would not plausibly cover it anyway). The MoleculeNet docs page and the underlying Chem. Sci. 2018 paper (itself CC-BY-NC-3.0, and about the article, not the data) state no data licence. Upstream compilation sources — SWEETLEAD (FDA-approved drug list) and AACT (ClinicalTrials.gov toxicity-failure list) — each publis
  · evidence: https://deepchem.readthedocs.io/en/latest/api_reference/moleculenet.html#clintox-datasets

**`esol`** — Do not apply a licence to esol; escalate to a human. The dataset is MoleculeNet's delaney-processed.csv, which is a re-columned copy of the Supporting Information of Delaney 2004 (J. Chem. Inf. Comput. Sci., DOI 10.1021/ci034243x). The researcher fetched the actual ACS SI file and found no copyright or licence text on the record itself, but ACS's own copyright policy (https://www.acs.org/copyright.html) explicitly forbids reproducing/redistributing/web-posting SI content without written permission, and Crossref lists no open licence for the article (not OA). No redistribution mirror supplies a real licence either: moleculenet.org is dead/unverifiable, the deepchem/moleculenet GitHub repo's MIT licence covers no actual data files, and while deepchem/deepchem does check in delaney-processed.csv under its root MIT LICENSE, that licence is PandeLab's software licence and cannot grant rights 
  · evidence: https://www.acs.org/copyright.html

**`freesolv`** — Did not edit chem_mat_data/scripts/create_graph_datasets__freesolv.py. The researcher found two conflicting licences on the freesolv data, both from the same rights holder: CC-BY-4.0 stated in github.com/MobleyLab/FreeSolv's LICENSE file/README (added 2026-04-03, the source URL currently recorded for this dataset), versus CC-BY-NC-SA-4.0 recorded as the "rights" value on the canonical, citable eScholarship deposit of the same database (https://escholarship.org/uc/item/6sd403pz, ark:/13030/qt6sd403pz, last updated 2017) — a deposit the FreeSolv README itself names as the authoritative home ("maintained on the cite-able eScholarship repository ... Updated versions will be maintained there"). NC and SA each independently forbid what CC-BY permits (commercial redistribution; derivatives under any licence), so this is not a near-miss. The GitHub README also self-limits the grant to "the exten
  · evidence: https://github.com/MobleyLab/FreeSolv/blob/master/README.md

**`hiv`** — Do not apply — needs a maintainer decision. The HIV dataset (chem_mat_data/scripts/create_graph_datasets__hiv.py, loads MoleculeNet's HIV.csv from the DeepChem S3 bucket) has no license stated at any layer: no license on the S3 object, no license/terms on the NCI DTP AIDS Antiviral Screen wiki record that is the true upstream source, and no data-availability/licence statement in the MoleculeNet paper (whose CC-BY-NC-3.0 covers the article text, not the CSV). The researcher argues redistribution is nonetheless defensible because the data originate from a US federal agency (NCI/NIH, not subject to copyright under 17 U.S.C. §105) and NCI's site-wide reuse policy states its content is "free of copyright and may be reused without our permission" (credit requested) — but this is a statutory/policy argument, not a licence attached to the record, hence confidence is only "medium" and license_ide
  · evidence: https://www.cancer.gov/policies/copyright-reuse

**`tadf`** — tadf's recorded source (Nature Materials 15, 1120-1127 (2016), SI at nmat4717#Sec16) is paywalled and its only explicit rights statement is "© 2016 Macmillan Publishers Limited. All rights reserved." — no permissive licence exists. Worse, that SI contains only 179 SMILES while the redistributed tadf.csv (via ChemMatData's Nextcloud) has 460,199 rows, so the true source of the actual data file is unknown and untraceable (extensive search of Zenodo, Figshare, DataCite, GitHub, and the defunct Aspuru-Guzik group site all came back empty). The maintainer needs to decide: (1) whether to track down where tadf.csv on Nextcloud actually came from (predates commit caceeb4, Sep 2025) — likely candidates are direct author contact, a since-removed group website, or inheritance from aimat-lab/visual_graph_datasets, none of which carries a known licence; (2) whether to contact corresponding author Asp
  · evidence: https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fnmat4717/MediaObjects/41563_2016_BFnmat4717_MOESM45_ESM.pdf

---

## Follow-ups that are not licence questions

Recorded here because they surfaced during the licence review and would otherwise be lost.

### `boron1000` is misnamed and its provenance is wrong — fix before publishing

The script does **not** download BORON1000. Real BORON1000 is Reaxys-derived, proprietary
(Elsevier), and was deliberately never published as structures — only as Reaxys IDs. What
this script actually pulls (`github.com/ruslankotl/rxn-data-proc`,
`data/roche/classification/*`) is a byte-verified copy of the **Nippa et al. literature
borylation set** from `github.com/ETHmodlab/lsfml` (Nippa et al., *Nat. Chem.* 2024,
doi:10.1038/s41557-023-01360-5; Zenodo 10.5281/zenodo.8118845).

So `DATASET_NAME`, `DESCRIPTION` and `sources` all describe a different dataset from the
one the script produces. The licence (MIT, from the lsfml root LICENSE) is fine; the
identity is not. **Not published yet.** Fix the name, description and sources before it is.

### Three scripts cannot be re-run by anyone else

`synth_binary_global` and `synth_binary_local` read a hardcoded local path
(`/media/data/Downloads/zinc_250k.csv`); `zinc250k` pulls from ChemMatData's own Nextcloud
rather than any original source. All three should download from the ChemicalVAE repository
(`models/zinc/250k_rndm_zinc_drugs_clean_3.csv`), which is where the file provably comes
from — same 22,606,589 bytes, same git blob SHA-1.

### `pfas_classify` and `pfas_ttr_binding` share an upstream but were assessed apart

Both rest on the US EPA CompTox PFAS Master List. One agent recorded it as "explicitly
free of all copyright restrictions", the other as "not independently verified". The EPA
CompTox terms page has still not actually been opened by anyone. Worth doing once, and
recording centrally.

---

## How to work through this

The families are ordered roughly easiest-first. A decision on one family applies to every
dataset in it, so the four upstream families at the top (ZINC, GDB, EPI Suite, PubChem)
settle 17 datasets between them.

For each family the question is the same one you answered for ChEMBL: does the upstream's
licence permit redistribution, and if it carries an obligation, is the project willing to
pass that obligation on? The machinery to record the answer already exists - the `license`
field, ADR 009, and the `Settled` section of `rejected_datasets.md`.

**A caution about the research.** Confidence is self-reported by the agent that did the
work. Where it says `low`, treat the finding as a starting point rather than a conclusion:
in several cases the agent could not reach the source at all, and "no licence found" then
means "not found by this agent" rather than "none exists". The `high`-confidence rows
quote licence text they actually read.
