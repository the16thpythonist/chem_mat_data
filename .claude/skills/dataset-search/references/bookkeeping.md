# Bookkeeping

Four files, two of them committed. What each is for, who writes it, and what must
never go in it.

| File | Committed | Written by | Scope |
|---|---|---|---|
| `discovery_stream.csv` (+ `.cursor`, `.ids`) | no, all three gitignored | `discover stream` | all runs, forever |
| `.dataset_search/<run>/` | no | this skill | one run |
| `docs/search_verdicts.md` | **yes** | `discover verdict` | all runs, forever |
| `MISSING_DATASETS.md` | **yes** | this skill (and screened as `listed`) | the standing list of leads |

Two more files are **read** and never written by this skill:
`chem_mat_data/metadata.yml` (the shipped catalog) and `docs/rejected_datasets.md`
(rejections made by a person).

`MISSING_DATASETS.md` is both read and written — it is the only file that is. It is
screened so a paper already on the leads list is reported as `listed`, and written
when a candidate is promoted.

## The run directory

```
.dataset_search/2026-08-19/
  screening.txt      the `discover check` table, as run
  screening.json     the same, as data: openalex_id, url, title, status, skip
  ids.txt            the OpenAlex ids of the working list
  titles.jsonl       titles and types back-filled from OpenAlex
  reviews/           one subagent report per paper, `<openalex_id>.md`
  candidates.md      the ranked shortlist, confirmed first
  findings.md        what this run learned
```

Gitignored, like `.dataset_work/`. One directory per run, named by date; a second
run on the same day gets `-2`.

`findings.md` is the half of the deliverable that survives. Append at the end of
**every stage**, not at the end of the run — a report that exists only as a chat
message dies with the session. Write down the sweep's counts, what the screening
removed and via which ledger, which subagents failed and how, and what you would
change next time. Concrete beats complete: "eleven flagged DOIs from one Brazilian
book-chapter publisher all 404, prefix not yet blacklisted" is an actionable hour
saved. "Verified the candidates" is nothing.

## The verdict ledger

`docs/search_verdicts.md` — one row per judged paper, appended by
`cmmanage discover verdict`, committed.

```markdown
| url | decision | name | record_url | why | date |
```

**Why it is committed and the sweep state is not.** The `.ids` sidecar stops a
paper being *fetched* twice. It says nothing about whether the paper was ever
*judged*. Without this ledger, a sweep run with a different threshold, a widened
year window, or a reset cursor re-verifies papers an earlier run already settled —
which is precisely the expensive half. The ledger is the memory of the judgement,
so it lives with the repository rather than in a run directory.

It is markdown rather than JSONL because people read it, the same reason
`rejected_datasets.md` and `MISSING_DATASETS.md` are markdown. Parsing is
tolerant — malformed lines are skipped, not fatal — so hand-edits and added prose
cannot break a sweep. Pipes inside a `why` are escaped and survive the round trip.

**What must not go in it:** licensing conclusions. Every row answers "does this
paper release a dataset". Nothing more.

## MISSING_DATASETS.md

The committed, standing list of leads — datasets believed to exist and not yet in
ChemMatData. It holds 86 numbered rows across five sections, and **the sections do
not share a header** — 5, 5, 4, 3 and 2 columns respectively. The widest:

```markdown
| # | Dataset / Paper | Data Type | Size | URL |
|---|-----------------|-----------|------|-----|
| 3 | Polyuniverse | Polymer SMILES + thermal/mechanical properties | >1M hypothetical polymers | https://doi.org/10.1039/D4DD00196F |
```

Match the header of the section you append to. Extra cells are silently dropped
when the table renders, and the URL is the last column — so an over-wide row loses
precisely the payload.

Promote **confirmed candidates only**.

**A hit here is not a skip.** `discover check` reports a paper already on this list
as `listed`, and it still gets a verification subagent. These rows are not decisions
the way `rejected`, `in_catalog` and `verified` are: they were confirmed by an LLM
from the *title alone* — no abstract, no paper fetch, no visit to a data record —
and 26% of that population was later measured as mislabelled
(`POSITIVES_VALIDATION.md`: 37 of 142 rejected on re-examination). Skipping them
would entrench a title-only guess and permanently deny it the record URL and entry
counts that only a subagent produces and that `/dataset-extraction` consumes.

Find the row with `grep -in '<the candidate DOI>' MISSING_DATASETS.md` — **`-i` is
required**, because OpenAlex lowercases DOIs and this file stores them as publishers
write them (eight rows carry uppercase). Screening normalises case and will say
`listed` where a case-sensitive grep finds nothing.

So on a hit, **update the existing row in place** with what your subagent found,
rather than appending a second one. A re-discovery is an upgrade of a weak lead, not
a duplicate.

When appending a genuinely new row, take whichever section fits — molecular
property, materials property, benchmark, lower priority — and **continue from the
global maximum across all sections**. Do not continue from the last row of the table
you are appending to: the numbers are non-contiguous upstream indices interleaved
across five sections, so a locally-next number collides. A candidate that fits none
of the sections gets a new one rather than being forced into a wrong one.

Fill `Data Type` and `Size` from what the subagent actually reported. "SMILES +
measured pKa" and "7,912 compounds" are what make the row worth having; "molecular
data" and "large" make it a URL with extra words.

Where an access barrier was noted, say so in `Data Type` — "(registration
required)" — so the person picking a lead knows before they start.

## Two blind spots

**The catalog.** `discover check` matches the shipped catalog on the `sources` URLs
each dataset declares. 13 catalog datasets declare none, so they cannot be matched
by URL at all, and `check` prints this caveat with their names every time it
runs.

**The identifier form.** Every source is matched on the bare DOI when the string
contains one, and on the normalised URL when it does not. A paper filed under a
publisher URL and later encountered as a `doi.org` link therefore does not match.
This is why stage 3 tells you to record verdicts under the URL exactly as the CSV
has it — the discipline is what closes this gap, since nothing detects it.

`new` therefore means **"no URL matched"**, not "definitely absent". Before
promoting a confirmed candidate, glance at its name and subject against
`cmdata list`. That is what catches the case the URL match structurally cannot —
a dataset already shipped under a name the catalog never linked to a paper.

If you find one, the fix is worth more than the catch: adding the paper URL to
that dataset's `sources` in `chem_mat_data/metadata.yml` closes the hole
permanently. Mention it to the user rather than editing the catalog mid-sweep.

## docs/rejected_datasets.md is not yours

That file records datasets investigated and refused **by a person**, with the
evidence and with what would have to change for the decision to be revisited. A
`discover check` hit against it is final.

Do not append to it. Do not re-research an entry because this sweep's classifier
scored it highly. If the user asks about a paper that hits, quote the entry and
let them decide — only a person can overturn one of these, which is the entire
reason the file is separate from the verdict ledger.

The distinction in one line: `search_verdicts.md` holds machine judgements about
*whether data exists*; `rejected_datasets.md` holds human judgements about
*whether we may or should ship it*.

## Why licensing appears nowhere

`/dataset-extraction` runs licence as its **first** gated stage, before anything is
downloaded, against a fixed vocabulary, escalating to a human on anything unclear —
and it treats "no licence at all", the most common case, as escalation rather than
permission.

If this skill also ruled on licences it would be making that call twice, in two
places, with only one of them gated, and the ungated one running first. A search
run that quietly dropped every CC-BY-NC record would also be hiding those papers
from the person who is allowed to decide.

So: observe, never act. If a subagent sees a licence on a record, it belongs in
the review file and in the candidate's notes as an observation. It must not change
a verdict, and it must not keep a paper off the shortlist.
