# License

ChemMatData redistributes other people's data. That is only legitimate when the terms
allow it, so the license stage exists to establish — with evidence — that they do.

This stage runs as a **fresh subagent** deliberately. By the time the pipeline reaches
it, real work has gone into making the dataset happen, and that is exactly the state of
mind in which an ambiguous license gets read charitably. An agent with no stake in the
outcome reads the terms as they are.

## The policy

The gate implements this table. It is deliberately strict: the cost of escalating a
license that would have been fine is a question to the user, while the cost of getting
it wrong is redistributing data the project has no right to.

| Verdict | Licenses |
|---|---|
| **PASS** | CC0 / public domain / Unlicense, CC-BY (any version), MIT, BSD (2- and 3-clause), Apache-2.0, ODC-BY |
| **ESCALATE** | CC-BY-NC and any NonCommercial variant; CC-BY-SA, ODbL and any ShareAlike; GPL/LGPL/AGPL and other copyleft; custom or bespoke terms; terms you could not pin down; **no license stated at all** |

Two things worth being explicit about:

**No license is not permission.** It is the most common case and the easiest to
misread. Data published without stated terms is, by default, not licensed for
redistribution — the absence of a prohibition is not a grant. It escalates.

**ShareAlike is not a near-miss of CC-BY.** `CC-BY-SA-4.0` shares a prefix with
`CC-BY-4.0` but carries a reciprocal obligation that propagates to everything the data
is combined with. The gate's matcher is written so the prefix cannot accidentally
match; do not talk yourself past it either.

Publisher supplementary information is a special case: SI files are typically covered
by the journal's copyright rather than by any open license, even when the article
itself is open access. Treat scraped SI as escalating unless the journal states
otherwise.

## The two traps

Both of these are real misses from testing this skill, not hypotheticals. The gate now
checks for both, but you should understand them rather than just satisfying the fields.

### Trap 1 — the upstream database

**A permissive license on the paper says nothing about a database the paper merely
aggregated.** Most property datasets are compilations of something else, and the
binding restriction usually lives upstream, not in the publication.

The case that caught this out: a CC-BY-4.0 paper republishing half-life data extracted
from enviPath, whose terms restrict use to "academic and non-commercial" purposes and
explicitly forbid anyone who would *"systematically extract, cache, reproduce, mirror,
republish, or redistribute substantial portions of the database"*. The paper is
genuinely CC-BY. Redistributing an aggregation of that database is still a question the
paper's license does not answer.

So the gate requires you to **state explicitly whether an upstream database exists**,
via the `upstream_database` field. Set it to `null` when the publication is the original
source. When a database is named, you must also record `upstream_terms_url` and
`upstream_license`, and anything non-permissive escalates.

Common upstreams worth checking for: enviPath, the CSD, Reaxys, SciFinder, ChEMBL,
PubChem, DrugBank, ZINC, the PDB, ICSD, Materials Project, NIST databases. Some are
open, several are not, and a few are open for use but not for redistribution.

### Trap 2 — preprint versus published version

The same work often exists as a preprint and as a journal article **under different
licenses**. In the case above the ChemRxiv preprint is CC-BY-**NC-ND**-4.0 while the
published ACS version is CC-BY-4.0 — so which file you actually downloaded determines
which license applies, and NoDerivatives would forbid redistributing a processed
derivative at all.

Establish the license against **the version of the file you actually downloaded**. If
you started from a preprint and licensed against the publication, or vice versa, either
switch the download to match or record the license for the version you used. The gate
warns when the evidence URL and the run's source URL are on different hosts, but it
cannot tell which file you took.

## What to research

Find, in this order of authority:

1. **The license on the data repository record.** Zenodo, Figshare, Dryad and OSF all
   display a license per record. This is the one that governs the data.
2. **A LICENSE / COPYING file inside the download**, or a license statement in the
   README.
3. **The upstream database's terms of use**, if the data was compiled from one.
4. **The article's Data Availability section**, which sometimes states terms directly.
5. **The journal's data policy**, for SI-only datasets.
6. **The repository's own terms**, for GitHub-hosted data (repository license file).

The article's license and the data's license are frequently different — a CC-BY article
can point at a CC-BY-NC dataset, and an all-rights-reserved article can point at a CC0
dataset. Report the license on **the data**.

## What to report

```json
{
  "identifier": "CC-BY-4.0",
  "evidence_url": "https://zenodo.org/records/1234567",
  "redistribution": "allowed",
  "attribution": "Cite Lopez et al., Sci. Data 3, 160086 (2016).",
  "upstream_database": null
}
```

and when the data was compiled from a database:

```json
{
  "identifier": "CC-BY-4.0",
  "evidence_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10569042/",
  "redistribution": "allowed",
  "attribution": "Cite Hafner, Fenner & Scheidegger, Environ. Sci. Technol. Lett. 2023.",
  "upstream_database": "enviPath / EAWAG-SOIL",
  "upstream_terms_url": "https://envipath.org/terms",
  "upstream_license": "academic and non-commercial use only; systematic redistribution prohibited"
}
```

- `identifier` — an SPDX-style identifier where one exists (`CC-BY-4.0`, `MIT`,
  `CC0-1.0`). Casing and separators do not matter; the gate normalises them. Omit it
  entirely if none was found rather than guessing.
- `evidence_url` — a page a human can open and check. Required. A finding without one
  fails the gate regardless of what it claims.
- `redistribution` — `allowed`, or a description of the restriction.
- `attribution` — what the license requires be credited, verbatim where possible. This
  belongs in the dataset's `sources` metadata.
- `upstream_database` — **required**. The database the data was compiled from, or
  `null` if the publication is the original source. Answering it wrongly is worse than
  answering it cautiously: name anything you are unsure about.
- `upstream_terms_url`, `upstream_license` — required whenever a database is named.
  Quote the restriction in your own words in `upstream_license` if it has no SPDX name.

Report **"none found"** rather than inferring a license from context. "The repo is
public so it must be fine" and "everyone else uses this dataset" are not license
findings.

## When it escalates

Ask the user with a concrete question: the license found, the evidence URL, what it
permits and what it forbids, and what redistributing under it would commit the project
to. Then record their decision verbatim:

```json
{
  "identifier": "CC-BY-NC-4.0",
  "evidence_url": "https://...",
  "redistribution": "allowed",
  "attribution": "Cite ...",
  "upstream_database": "enviPath",
  "upstream_terms_url": "https://envipath.org/terms",
  "upstream_license": "Academic and non-commercial use only; systematic redistribution forbidden.",
  "human_override": {
    "decision": "approved",
    "reason": "Non-commercial use only; ChemMatData is an academic project and the authors confirmed by email on 2026-08-01."
  }
}
```

**Send every field you already established, not just the override.** `cmmanage agent
set` replaces the whole `license` object, so an override submitted on its own deletes
`upstream_database`, `upstream_terms_url`, `upstream_license` and `attribution` — the
very evidence that justified escalating in the first place, leaving a record that says
a human approved something without saying what. Read the current value back with
`cmmanage agent status <slug>` and merge into it.

`decision` is `approved` or `rejected`, and an approval **must** carry a reason — the
gate rejects one without. Record the human's words verbatim rather than paraphrasing
them into your own analysis. The override records a human's decision; it is not a way
to unblock yourself. Never write one on your own judgement.

A rejected license ends the run. Say so plainly and stop; do not look for a workaround.

## Attribution follow-through

A CC-BY dataset requires credit. Make sure the attribution actually lands in the
dataset metadata rather than only in the manifest — the `sources` list in
`metadata_draft` is what a downstream user sees. If the license stage turns up an
attribution requirement the metadata stage did not capture, re-run the metadata stage
with it included. That marks the later stages stale, which is correct.
