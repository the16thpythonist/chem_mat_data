---
name: dataset-search
description: Find publications that release reusable chemistry or materials datasets, by sweeping OpenAlex with the TinyLlama discovery classifier and verifying each flagged paper with its own subagent. Produces a ranked shortlist of candidates, promoted into MISSING_DATASETS.md, that /dataset-extraction then turns into processing scripts. Use when asked to find new datasets, look for datasets to add to ChemMatData, run a discovery sweep, or trawl the literature for dataset releases. Takes no input — the sweep decides what to look at.
---

# Dataset Search

Find publications that release datasets worth adding to ChemMatData.

**This skill takes no input.** It is the front half of the pipeline: it decides
*which* papers are worth extracting. `/dataset-extraction` is the back half and
takes a URL — the URLs in this skill's shortlist are what it consumes.

```
sweep     crawl OpenAlex, score each abstract    → discovery_stream.csv   [repo root]
screen    drop what is already answered          → cmmanage discover check
verify    one subagent per surviving paper       → reviews/<n>.md         [subagent]
shortlist rank, write up, promote                → MISSING_DATASETS.md
```

The sweep is cheap and the verification is expensive, so everything about this
skill is arranged to spend as few subagents as possible on papers whose answer is
already known.

## The one question this skill answers

**Does this paper release a dataset?** That is all.

It does **not** rule on whether the dataset may be redistributed. Licensing is
decided by `/dataset-extraction`'s license stage, which does it properly, with a
gate, against a fixed vocabulary, escalating to a human when it is unclear. A
shortlist entry is a *lead*, not an approval, and some leads will correctly die at
that stage.

Keeping these apart matters. A search run that started rejecting papers on licence
grounds would be making the same call twice, in two places, with only one of them
gated — and the ungated one running first. Note the licence you happen to see, if
you see one, as a plain observation on the candidate. Do not act on it.

## Two rules

1. **Never spend a subagent on a paper `discover check` says is settled.** The
   check is a local string match that costs milliseconds; a subagent is a full
   agent. The whole design is built around that ratio.
2. **When you cannot resolve something, stop and ask.** Same as every skill in
   this repository: do not guess, do not work around it. See *Escalation*.

---

## Setup

```bash
source .venv/bin/activate
cmmanage discover model-info
```

`model-info` must report the adapter **present** and the ML stack **OK**. If
either is missing, stop and tell the user what to run — do not attempt the sweep,
and do not attempt to install it yourself:

| What model-info says | What the user needs to run |
|---|---|
| `Adapter: missing` | `cmmanage discover model-download` |
| `ML stack: missing` | `uv pip install torch transformers peft safetensors accelerate` |

`model-info` prints `pip install chem_mat_database[dev]` for the missing stack,
which also works — `[dev]` is a superset that additionally pulls in
torch-geometric, jax and the docs toolchain. Offer the narrower command above and
mention the CLI's, so the difference does not read as one of them being wrong.

The classifier runs on GPU when one is visible and on CPU otherwise, at roughly
1–2 s per paper on CPU. A 300-paper sweep is therefore minutes on GPU and up to
ten on CPU. Neither needs babysitting; say which one is happening so the wait is
not a surprise.

**Make the run directory before anything else:**

```bash
run=$(date +%F); n=2
while [ -d ".dataset_search/$run" ]; do run="$(date +%F)-$n"; n=$((n+1)); done
mkdir -p ".dataset_search/$run/reviews" && echo "$run"
```

`$run` is the **bare directory name** — `2026-08-19`, not a path. Every use in this
skill writes `.dataset_search/<run>/…` and supplies the prefix itself, so putting
the prefix in `<run>` too produces `.dataset_search/.dataset_search/…`, which
fails a shell redirect outright and silently succeeds for the `Write` tool. A plain `mkdir -p` would silently reuse a directory when a run
already happened today, so the loop picks the next free `-2`, `-3` instead of
leaving the collision for you to notice.

---

## Stage 1 — sweep

```bash
cmmanage discover stream --limit 300
```

**Add `--include-title` only if the ledger already has a title column** — `head -1
discovery_stream.csv` tells you. Adding it to a ledger written without one is
refused (see below), and the existing rows keep no titles of their own. That is not
a dead end: titles are recoverable from `openalex_id` at any time, and stage 2 does
exactly that.

Defaults worth knowing rather than changing:

- **`--out` defaults to `discovery_stream.csv` at the repo root, and that is
  deliberate.** It is a shared, ever-growing ledger with `.cursor` and `.ids`
  sidecars beside it. Leaving it there is what makes the *next* sweep continue
  from where this one stopped instead of re-reading the newest papers again.
  **Do not redirect it into the run directory.**
- **`--limit 300`** is the default for a run of this skill, not the CLI's own
  default of 1000. At the flag rate this sweep actually produces — 115 of 1000,
  measured — 300 papers is about 35 flagged from *this* sweep. Use `--limit 1000`
  when the user asks for a big sweep, and tell them it is roughly three times the
  cost.

  **That is not the subagent count.** Stage 2 screens the whole ledger, so the
  papers you verify are this sweep's flags *plus every flagged row an earlier
  sweep left unjudged*. The number stage 2 prints is the real one — take it from
  there, never from this estimate.
- **`--include-title`** adds the title column, and stage 2 carries it through to
  the briefs. Take it on a fresh `--out`. On an existing file it must match what
  is already there: `stream` refuses to append a differently shaped row rather
  than making the CSV ragged, and names the flag when it does. If the shared
  ledger predates this flag, drop `--include-title` rather than starting the
  ledger over — the titles are a convenience, the de-duplication set is not.
- **`--threshold` defaults to 0.05** and should be left alone. It is tuned for
  recall on the upstream test set, so most flagged papers are *not* dataset
  releases and that is the intended trade. If a run flags a suspiciously round
  number of papers, read `references/sweep.md` before touching the threshold.

Copy the sweep's own summary line into `findings.md` when it finishes — how many
records were fetched, how many flagged, and whether it stopped early.

**If the sweep returns no new records**, the shared cursor has reached the end of
the useful tail. That is a real condition, not a failure; `references/sweep.md`
covers what to do about it. Do not delete the sidecars to "fix" it — that throws
away the de-duplication set for every future run.

---

## Stage 2 — screen

```bash
cmmanage discover check --from-csv discovery_stream.csv > .dataset_search/<run>/screening.txt
cmmanage discover check --from-csv discovery_stream.csv --new-only --json
```

The first writes the human-readable table for the record. The second is the working
list: `--new-only` selects the papers still worth verifying, `--json` gives you each
one's `openalex_id`, `url` and `title` — everything stage 3 needs. The two flags
compose; `--new-only` on its own prints bare URLs, which is what you want when
piping and not what you want when briefing.

**Plain `--json` without `--new-only` lists every flagged row, settled ones
included**, distinguished by a `skip` field. Do not iterate that array. Every paper falls into one of five states:

| status | meaning | what you do |
|---|---|---|
| `rejected` | a person investigated and refused it | **skip, and quote the entry** if the user asks about that paper |
| `in_catalog` | ChemMatData already distributes it | skip |
| `verified` | an earlier sweep already judged it | skip |
| `listed` | already a lead in `MISSING_DATASETS.md` | **verify it anyway**, then enrich that row instead of adding one |
| `new` | nobody has answered this one | verify it |

`new` and `listed` both get a subagent; the other three do not.

A `listed` paper is not a settled question. Those rows were confirmed from the
title alone, and 26% of that population was later measured as mislabelled
(`POSITIVES_VALIDATION.md`: 37 of 142 rejected on re-examination). They also lack
the record URL and entry counts that only a verification subagent produces and
that `/dataset-extraction` needs. Verifying one turns a weak lead into a real one.

### Back-fill the titles, then collapse the duplicates

Untitled rows are the norm on an older ledger, and both the skim below and the
briefs need titles. Resolve them from `openalex_id`, which `--new-only --json`
already gives you. OpenAlex takes 50 ids per request:

```bash
d=.dataset_search/<run>          # substitute <run>; shell state does not carry between commands
cmmanage discover check --from-csv discovery_stream.csv --new-only --json > $d/screening.json
python3 -c "
import json
ids=[r['openalex_id'] for r in json.load(open('$d/screening.json')) if r['openalex_id']]
open('$d/ids.txt','w').write('\n'.join(ids)+'\n')"

: > $d/titles.jsonl
[ -s $d/ids.txt ] || echo 'nothing to resolve'
split -l 50 $d/ids.txt $d/chunk_ 2>/dev/null
for f in $d/chunk_*; do
  [ -e "$f" ] || break
  curl -s "https://api.openalex.org/works?per-page=50&select=id,title,type,primary_location&filter=openalex_id:$(paste -sd'|' "$f")" \
   | python3 -c "
import json,sys
for w in json.load(sys.stdin).get('results',[]):
    loc = w.get('primary_location') or {}
    print(json.dumps({'openalex_id': w['id'].rsplit('/',1)[-1],
                      'title': w.get('title') or '', 'type': w.get('type') or '',
                      'landing_page_url': loc.get('landing_page_url') or ''}))
" >> $d/titles.jsonl
done
rm -f $d/chunk_*
```

That is a few seconds for the whole backlog. A handful will not resolve — one of
the current 115 has no API record at all — and those get briefed on their bare URL.

**Use `landing_page_url` for any row whose `url` is an `openalex.org/W…` page.**
A row with no DOI falls back to its OpenAlex landing page, and that page returns
**403 to every client, `crwl` included** — so briefing it guarantees an `unclear`,
and an `unclear` retires the paper from every future sweep. Three of the current 115
are in this state, and `primary_location` recovers a real page for two of them
(a HAL record and a handle.net one). Check with `curl -o /dev/null -w '%{http_code}'`
if you are unsure.

**Take `type` from the same call.** It costs nothing and it is the most useful
single signal about what a flagged row is: on the current ledger 110 of the 114
resolved rows are `report`, which is to say Zenodo-deposited PDFs rather than
journal articles.

**Then group the working list by title.** Repository records routinely appear under
several DOIs — Zenodo mints a concept DOI and one per version, differing by a single
integer and scoring identically — and `discover check` matches on DOI, so it cannot
collapse them. On the ledger as it stands, **115 flagged rows are 67 distinct
works**: 46 titles appear more than once. Brief **one subagent per distinct title**,
and record each sibling DOI with the same decision and
`--why 'duplicate DOI of <the other>'`. Never file a sibling of a confirmed release
as `rejected` — that writes a permanent row saying a paper that releases data does
not.

**Put the scale to the user before you spend anything.** This is the single most
expensive decision in the run, and the one place a person should see before it
happens. State three things: how many works you are about to verify, how many of
this sweep's own flags that is (a much larger number means an earlier sweep was
never verified), and what the titles and `type` values look like. Then **wait for a
go-ahead** — `AskUserQuestion` in an interactive session, a message back to whoever
invoked you otherwise. Only when there is no channel at all do you record the
numbers in `findings.md` and proceed: unattended runs are deliberate, silent ones
are not.

**Look at the titles before you spend anything.** The classifier scores language,
not substance, and the sweep is sorted newest-first, so a stretch of the ledger can
be dominated by one kind of document — self-published repository reports, a single
paper deposited several times under different DOIs, or a subject area that is not
chemistry at all. Skim the working list. If the titles are obviously not dataset
releases, or the same title appears repeatedly, say so to the user and agree the
scale before spawning ~100 subagents to produce ~100 rejections. Recording those
rejections is not wasted — they stop the next sweep re-asking — but it is a cost the
user should choose knowingly.

**Read the count before you plan the batches.** This working list is every flagged
paper from every sweep that has never been given a verdict — not just this sweep's
records. If it is far larger than this sweep's own flag count, an earlier sweep was
never verified. Say so in `findings.md`, and tell the user the number before you
start spawning.

**`new` means "no URL matched", not "definitely absent".** Two blind spots, and
`check` prints the first of them itself:

- 13 catalog datasets declare no `sources` URL, so the catalog check cannot see
  them. `check` prints the live figure and names them every time it runs.
- Every ledger is matched on the DOI or the URL as written. A paper recorded under
  one form and encountered as another will not match.

So when a candidate survives verification, glance at its name against `cmdata list`
before promoting it. That catches the `ames`-shaped case the URL match structurally
cannot.

Record in `findings.md` how many were screened out and by which ledger. It is the
cheapest measure of whether the sweep is finding new ground or re-treading old.

---

## Stage 3 — verify

**One fresh subagent per distinct work.** Not one subagent for a batch of papers,
not you reading abstracts. Each work gets an agent whose entire job is that work —
"distinct" because stage 2 has already collapsed the several DOIs a single work is
often published under, and a sibling DOI inherits its verdict rather than earning
one.

**Spawn them without a `name`, and not in the background.** Passing a `name`
routes the agent through a mailbox: the spawn returns a receipt, the completion
arrives as a notification carrying no text, and the report never reaches you. This
is documented in `/dataset-extraction` as something that actually happened there —
three subagents finished their work and none of it arrived. Spawned unnamed, the
tool result *is* the report.

Use the `general-purpose` agent type; these subagents need to read pages and write
a file, and no narrower type covers that.

**Run them in batches of six**, and after each batch append to `candidates.md` and
`findings.md` before starting the next. A run that dies at paper 30 should leave 29
verdicts on disk, not nothing.

### The brief

Give each subagent the paper's title, its URL, and this task. Title and
`openalex_id` both come from the `--new-only --json` output of stage 2.

**A missing title is normal, not a blocker.** Stage 2 back-fills what it can from
`openalex_id`; for anything still untitled, brief on the bare URL and say so — the
subagent resolves the URL regardless and reports the real title back.

> Determine whether this publication releases a dataset — that is, whether the
> authors deposited or published data that someone else could reuse, as opposed to
> only using or citing existing datasets.
>
> Read the publication page and its data availability statement. Follow any link
> to a data record (Zenodo, Figshare, GitHub, an institutional repository, a
> supplementary information file) and confirm the record exists and describes
> actual data.
>
> **If the URL you were given is itself a repository record** — a Zenodo, Figshare,
> OSTI or institutional-repository landing page — then judge that record directly.
> There is no separate publication page to find, and its absence is not evidence of
> anything. Ask the same question of the record: does it hold reusable data, or is
> it a document (a report, a review, a preprint) that merely discusses some? A
> deposited PDF is not a dataset.
>
> Report: `confirmed`, `rejected` or `unclear`; the data record URL if you found
> one; what the data actually is — how many entries, what kind of structures, what
> properties; and one sentence of evidence for the decision.
>
> **Access barriers are noted, not disqualifying.** A dataset behind a
> registration wall, available on request, or requiring a licence agreement is
> still `confirmed` — say so in the evidence and move on.
>
> **Do not evaluate the licence.** Whether the data may be redistributed is
> decided elsewhere. If a licence is stated on the record, report it as an
> observation, but it must not affect your verdict.
>
> **Read pages with `crwl` first**: `crwl '<url>' -o markdown-fit`. It drives a real
> Chromium and is the only reader that gets past a publisher bot check. Fall back to
> `defuddle parse '<url>' -m`, and to `WebFetch` only after both fail. Reaching for
> `WebFetch` first is how a readable page becomes a false `unclear`.
>
> Write your report to `.dataset_search/<run>/reviews/<openalex_id>.md`, beginning
> with a line reading `reviewer: subagent`.

Fill in both placeholders yourself before sending the brief. `<run>` is this run's
**bare directory name** from Setup (`2026-08-20`) — the `.dataset_search/` prefix is
already in the path, and putting it in `<run>` too writes every report to
`.dataset_search/.dataset_search/…`, which the `Write` tool creates without
complaint. `<openalex_id>` is the `openalex_id` field from stage 2's JSON. Never
pass either through literally, and never substitute the URL — a DOI contains a slash
and would scatter the reports into invented subdirectories.

**Give the subagent an absolute path.** The `Write` tool rejects a relative
`file_path`, so a brief carrying `.dataset_search/…` costs every subagent a failed
call before it guesses a root.

The reader preference is inside the brief above, where the subagent will actually
see it — it used to sit out here, addressed to a parent that never fetches a page.
`cmmanage agent tools` reports whether `crwl` and `defuddle` are installed.

### Recording each verdict

As each subagent returns:

```bash
cmmanage discover verdict '<paper url>' \
  --decision confirmed \
  --name '<proposed dataset name>' \
  --record-url '<data record landing page>' \
  --why '<one line of evidence>'
```

**Pass the URL exactly as it appears in the CSV**, not the resolved publisher URL
the subagent reports back. The ledger is matched on the DOI when the string has
one, and on the URL as written when it does not — so a verdict filed under
`nature.com/articles/…` is invisible to the next sweep, which will meet the same
paper as `doi.org/10.1038/…` and spend another subagent on it. The record URL
belongs in `--record-url`, which is indexed too.

**Record every verdict, including the rejections.** The rejections are most of the
value: they are what stops the next sweep spending a subagent on the same review
article. `--decision` is a fixed vocabulary — `confirmed`, `rejected`, `unclear` —
for the same reason the license gate fixes its own: prose cannot be matched
against by a later run.

`unclear` is a real answer. Use it when the subagent could not reach the paper or
could not tell, and say which in `--why`. Do not round it to `rejected`.

**To correct a verdict you already recorded, pass `--force`.** Without it `verdict`
refuses a paper that already has one and quotes the prior reason, which is the right
default — but a genuine correction (an `unclear` you later resolved) needs the flag.
The ledger keeps both rows and the newest one wins.

---

## Stage 4 — shortlist

Write `.dataset_search/<run>/candidates.md`: every verified paper, confirmed first
and ranked by how useful the dataset looks for graph learning, then the rejections
with their one-line reasons, then the screened-out papers as counts.

Rank on what the data *is* — entry count, whether entries carry structures
(SMILES/InChI/CIF/XYZ), how many properties, whether the properties are measured
or computed. A confirmed 40-entry dataset ranks below a confirmed 40,000-entry one
and the shortlist should say so rather than presenting them as equals.

Then promote the confirmed candidates into `MISSING_DATASETS.md`, the committed
list of leads — 86 numbered rows across five sections.

**The five sections do not share a header.** Match the one you are appending to:

| Section | Columns |
|---|---|
| High-Priority — Molecular Property | `# · Dataset / Paper · Data Type · Size · URL` |
| High-Priority — Materials Property | `# · Dataset / Paper · Data Type · Size · URL` |
| Medium-Priority — Benchmark/Interaction | `# · Dataset / Paper · Data Type · URL` |
| Lower Priority — Protein/Structural | `# · Dataset / Paper · URL` |
| DCP Probe Datasets | `# · Probe` |

A row with more cells than its header **silently loses the extra ones when
rendered** — and since the URL is last, a five-cell row appended to the
three-column section drops exactly the thing `/dataset-extraction` consumes. Count
the columns before you write.

**Check before you append.** Any candidate that screened `listed` is already in
that file, and one whose DOI appears there in a form the match missed will not
have — so grep for it:

```bash
grep -in '<the candidate DOI>' MISSING_DATASETS.md
```

**`-i` is not optional.** OpenAlex lowercases DOIs; publishers do not, and
`MISSING_DATASETS.md` stores them as written — eight of its rows carry uppercase
(`10.1039/D4DD00196F`). Screening normalises case and matches; a case-sensitive
grep does not, and you would append the duplicate row the `listed` status exists to
prevent. When `check` said `listed`, the row *is* there: use the grep to find the
line, not to decide whether it exists.

On a hit, **update that row in place** with the record URL and entry counts your
subagent found, rather than adding a second one, and say so in `findings.md`.

**If a `listed` paper came back `rejected`**, the lead is wrong — that is the 26%
this status exists to catch. Do not delete the row yourself: it is committed, and
a person put it there. Record the verdict as normal, and name the row in
`findings.md` and in your report to the user so they can remove it. The verdict
ledger will keep the next sweep from re-verifying it either way. Those
existing rows were confirmed from a title alone; yours has a verified record behind
it. Replacing a weak lead with a strong one is the point of the `listed` status.

Otherwise append to the table that fits — molecular property, materials property,
benchmark, lower priority. Do not renumber the existing rows. **Continue from the
global maximum across all sections**, not from the last row of the table you are
appending to: the numbers are non-contiguous upstream indices interleaved across
five sections, so "the next number" is meaningless locally. If a candidate fits
none of the existing tables, add a new section rather than forcing it into a wrong
one.

### Close the loop

Before writing the report, run stage 2's command once more:

```bash
cmmanage discover check --from-csv discovery_stream.csv --new-only
```

**Every paper on stage 2's working list must have disappeared from that list** —
not merely the ones you briefed. Sibling DOIs never get a subagent, so they are
exactly the rows that go missing when grouping is done but the siblings are not
recorded. Anything
still printed is a verdict that was never recorded — record it now, or name it in
`findings.md` as deliberately deferred. There are no gates in this skill; this is
the one check that stands in for them, it costs milliseconds, and it uses a command
you have already run twice.

Take the report's counts from the file rather than from memory:

```bash
grep -c '^| http' docs/search_verdicts.md
```

That is the **all-time** total. Run it once before stage 3 and once after; the
difference is this run's verdicts, and it is the difference you report.

### The run report

`findings.md` is the half of the deliverable that survives the session. Append to
it at the end of **every stage**, not at the end of the run.

Write down the sweep's counts, what the screening removed and via which ledger,
which papers surprised you, which subagents failed and how, and what you would
change about the next sweep. Be concrete: "the classifier flagged eleven Brazilian
book-chapter DOIs that all 404" is worth an hour to the next person; "verified the
candidates" is worth nothing.

Then tell the user: how many papers were swept, how many verified, what the
confirmed candidates are, and which one you would extract first.

---

## Escalation

Stop and ask the human when:

- `model-info` reports the classifier unavailable — that is theirs to install.
- The sweep returns nothing new and you cannot tell whether the tail is genuinely
  exhausted or the state is wrong.
- More than about a third of the subagents in a batch fail to reach their pages.
  One paywalled publisher is normal; a third of them means something is broken and
  burning subagents on it is waste.
- A paper looks confirmed but `discover check` flagged it `rejected` and the user
  is asking you to reconsider. Only a person may overturn a rejection.

Make the question concrete: what is blocking in one sentence, what you tried, what
you need. Use `AskUserQuestion` in an interactive session, or a message back to
whoever invoked you when running unattended. If you have no channel at all, write
the blocker into `findings.md` and stop — do not decide on their behalf because
asking was inconvenient.

---

## Resuming

There is no manifest and no gates here; the state is the files.

```bash
ls .dataset_search/                                  # which runs exist
cat .dataset_search/<run>/findings.md                # how far it got
grep -c '^| http' docs/search_verdicts.md            # verdicts recorded so far
cmmanage discover check --from-csv discovery_stream.csv --new-only
```

That last command is the resume point: papers already verified now screen as
`verified`, so whatever it prints is exactly the work left. Re-running it after an
interrupted run is how you pick up, not by re-reading `candidates.md`.

## The four files

| File | Committed? | Who writes it |
|---|---|---|
| `discovery_stream.csv` + `.cursor` + `.ids` | no | `discover stream`, shared across all runs |
| `.dataset_search/<run>/` | no | this skill, one directory per run |
| `docs/search_verdicts.md` | **yes** | `discover verdict`, one row per judged paper |
| `MISSING_DATASETS.md` | **yes** | this skill, confirmed candidates only |

## References

- `references/sweep.md` — operating the sweep: what the classifier is, tuning it,
  and what to do when it returns nothing.
- `references/verification.md` — why a subagent per work, worked examples of each
  verdict, and the failure modes that produce wrong ones. The brief itself is in
  stage 3 above, and only there.
- `references/bookkeeping.md` — the ledgers, the promotion format, and why
  licensing is deliberately absent from all of them.
