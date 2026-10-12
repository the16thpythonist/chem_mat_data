# Verification

One subagent, one work, one question. The brief itself lives in `SKILL.md` stage 3
and only there; this file is the reasoning behind it, the verdict vocabulary, and
the ways a verdict goes wrong.

## Why a subagent per paper

Three reasons, in descending order of importance.

**Context.** Reading a publication page, following its data availability statement
to a repository record and confirming the record is real takes several page
fetches and a lot of text. Doing that for thirty papers in one context means the
thirtieth paper is judged by an agent that has forgotten the first. A fresh agent
per paper gets the same attention for every paper.

**Bias.** By the time you are verifying, you want the sweep to have found
something. That is the state of mind in which a vague data-availability statement
reads as a deposit. An agent with no stake in the sweep's yield reads it as it is.
This is the same argument `/dataset-extraction` makes for running its licence
determination as a subagent, and it applies here for the same reason.

**Failure isolation.** A paper behind a bot check kills its own subagent and
nothing else. In a single-context loop it derails the whole run.

## Spawning

**No `name`, not in the background, six at a time.**

A named agent is routed through a mailbox: the spawn call returns a receipt, the
completion arrives later as a notification carrying no text, and the report never
arrives. `/dataset-extraction` documents this happening to three subagents in one
run — all three did their work and none of it reached the parent. Spawned unnamed
and in the foreground, the tool result *is* the report.

Six per batch, then write `candidates.md` and `findings.md` before the next batch.
A run that dies at paper 30 should leave 29 verdicts behind.

## The brief

**The brief itself lives in `SKILL.md`, stage 3, and only there.** It used to be
duplicated here as well, the two copies drifted, and the copy an agent was pointed
at was the stale one — so this file now covers the *reasoning* and the failure
modes, and never restates the words.

What this file adds to it:

- Give the subagent the title and the URL. Nothing else — not the classifier score,
  not what you hope it finds. A score of 0.94 in the brief is a thumb on the scale.
- A missing title is normal and not a blocker. Stage 2 back-fills titles from
  `openalex_id` where it can; where it cannot, brief on the bare URL and say so.
- Use the `general-purpose` agent type.

## The vocabulary

`--decision` takes exactly three values. They are fixed rather than free text for
the same reason `/dataset-extraction` fixes its licence fields: a later sweep has
to *match* against these, and a sentence explaining that the data is basically
available is indistinguishable to a string comparison from one explaining that it
is not.

### `confirmed` — the authors released data

The paper deposited or published data someone else could reuse. Includes:

- A Zenodo, Figshare, Dryad or institutional record with data files.
- A GitHub repository containing the data (not merely the model code).
- Supplementary information files carrying the actual tables.
- A database the paper introduces and makes available, even through a web portal.
- **Data behind a registration wall, a licence click-through, or "available from
  the authors on request".** Note the barrier in the evidence; it does not change
  the verdict.

### `rejected` — the authors released no data

- The paper only *uses* existing datasets. This is the single most common false
  positive, because "we train on 130k molecules from QM9" reads much like a
  release.
- A method, review or perspective paper with no deposit.
- The data availability statement says data are in the paper's own figures.
- The linked record does not exist, is empty, or contains only code.
- The paper is not about molecules or materials at all — the concept filter is
  broad and lets some through.

### `unclear` — the verification could not tell

- The publication page could not be reached (paywall, bot check, dead DOI) and no
  preprint was found.
- A record is referenced but its identifier is wrong or unresolvable.
- The statement is genuinely ambiguous and the record could not settle it.

`unclear` is a real answer and recording it is what stops the next sweep from
re-verifying the same unreachable paper. Say in `--why` *which* kind it was —
"ACS bot check, no preprint" and "Zenodo DOI 404s" lead to different next actions.
Do not round `unclear` down to `rejected`.

## How verdicts go wrong

**Trusting the data availability statement.** "Data are available in the Zenodo
repository" is a claim, not a record. Follow the link. Records that were never
published, embargoed indefinitely, or that hold only a README are common enough
that the follow-through is the whole job.

**Mistaking use for release.** Ask which sentence is doing the work. "We construct
a dataset of 12,000 measured solubilities and release it at …" is a release. "We
benchmark on ESOL, FreeSolv and Lipophilicity" is not, however many datasets it
names.

**Mistaking code for data.** A GitHub repository linked from a machine-learning
paper usually holds the model. Look for the data files, or for a download script
pointing somewhere real. A repository with a `data/` directory containing a loader
and no data is `rejected`.

**Counting a portal as absent.** A queryable web database with no bulk download is
still a release — `confirmed`, with the access barrier noted. Whether it can be
turned into a ChemMatData dataset is a later problem and not this skill's call.

**Letting the licence leak in.** A CC-BY-NC record is `confirmed`. So is one with
no licence at all. If a verdict's evidence sentence mentions redistribution, the
verdict has drifted out of scope — the whole shortlist is a set of leads, and
`/dataset-extraction`'s license stage is what rules on them.

**Judging one work's several DOIs separately.** A preprint and its published
version are the same dataset; so are a repository record's concept DOI and its
version DOIs, which is the far commoner case — Zenodo mints both, they differ by
one integer, and they score identically. `discover check` matches on DOI and cannot
collapse them, which is why stage 2 groups the working list by title first.

**Give a sibling the same decision as the work, not `rejected`.** Record it with
`--why 'duplicate DOI of <the other>'`. Filing the second DOI of a *confirmed*
release as `rejected` writes a permanent, committed row saying a paper that does
release data does not, and screening will honour it forever.

## Recording

```bash
cmmanage discover verdict '<paper url>' \
  --decision confirmed \
  --name '<proposed dataset name>' \
  --record-url '<data record landing page>' \
  --why '<one line of evidence>'
```

**Pass the URL exactly as the CSV has it**, not the publisher URL the subagent
resolved to. Verdicts are matched on the bare DOI when the string contains one and
on the URL as written when it does not, so a paper filed under
`nature.com/articles/…` will not match the `doi.org/10.1038/…` form the next sweep
presents — and it gets a second subagent for a question already answered. The
resolved record URL has its own field, `--record-url`, which is indexed too.

Record **every** verdict, rejections included. The rejections carry most of the
forward value: they are what stops the next sweep spending a subagent on the same
review article. A sweep that records only its wins leaves the next sweep to redo
all of its losses.

`--why` is what a person reads in six months. "no data" is useless; "method paper,
benchmarks on ESOL/FreeSolv, no deposit" is the whole answer at a glance.
