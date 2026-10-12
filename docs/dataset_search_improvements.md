# dataset-search: what to fix next, ranked

Scope: the skill at `.claude/skills/dataset-search/` (735 lines across `SKILL.md`,
`references/sweep.md`, `references/verification.md`, `references/bookkeeping.md`),
`chem_mat_data/discovery/ledger.py`, and the `discover check` / `discover verdict`
commands in `chem_mat_data/manage.py`. All line numbers are against `b336db6` on
`master`.

**Method, and what the method does not cover.** Four review lenses ran over the
skill — simulate (walk a run start to finish), contract (does the prose match what
the code does), design (is the arrangement right), failure (what happens when a
step goes wrong). **Each lens was capped at its four strongest findings.** Eleven
findings reached an independent skeptic whose job was to refute them; seven
survived, four were refuted. The seven collapse to **five distinct defects**,
because three lenses independently found the `MISSING_DATASETS.md` gap and two
found the subagent-budget mismatch. The cap is a cap and not a clean bill: every
lens that returned four had weaker findings it could not report, so absence from
this list is not evidence of absence.

Where the skeptic supplied a correction, the corrected version is what appears
below, and the correction is named.

**No live sweep was exercised.** Outbound HTTPS from Python fails in the review
environment with a self-signed-certificate error (a TLS-intercepting proxy), so
`discover stream`, `fetch` and `score` were never run. Everything below was
established against the 1000-record `discovery_stream.csv` already on disk, the CLI,
the ledger module, and isolated reproductions in a scratch directory. Nothing in
this document depends on network behaviour.

---

## The structural conclusion

**Every defect below sits on the same seam: the skill reasons about one run, and
the machinery it drives keeps state across all runs.**

- The budget line counts *this sweep's* flags; the screen returns *every* sweep's
  unjudged flags (finding 5).
- `--include-title` adds a column to a CSV that earlier runs already wrote a header
  for (finding 1).
- `MISSING_DATASETS.md` accumulates across runs and is the one cross-run file
  nothing screens against (finding 3).
- The verdict ledger is the cross-run memory, and nothing checks that this run
  finished writing to it (finding 4).
- `--force` is the only cross-run correction channel the tool advertises, and it
  is inert (finding 2).

Design decision 2 — no gates, state lives in files — is upheld by this review, and
three of the five fixes are one or two sentences of prose. But the state that lives
in files is *shared* state, and the skill currently documents it as though each run
owned it. That is the single edit that would prevent the next four findings of this
shape.

**None of the nine stated design decisions was overturned.** Five were attacked and
all five held: licensing out of scope (5), downloadability not required (6), the
shared sweep ledger (8, attacked twice), no gate machinery (2), and no input /
unattended runs (1). Two proposals were rejected *because* they would have broken a
decision that is correct — see the tail section, and the "do not do this" clauses in
findings 3 and 5.

---

## Defects

Ranked by severity. These are bugs and gaps; act on them directly.

### 1. `discover check` never shows a title, and `--include-title` makes the shared ledger ragged

**Anchor:** `.claude/skills/dataset-search/SKILL.md:100`, `chem_mat_data/manage.py:3020`.
**Lenses:** simulate, contract. **Severity: high** (on your machine; medium on a fresh clone).

Two separable defects that meet in the same flag.

**(a) No output of `discover check` carries a title.** SKILL.md:100 says
`--include-title` is worth taking because "the screening report and the subagent
briefs both read better with titles than with bare DOIs". The `--from-csv` loop
(manage.py:3343-3346) reads `row['flagged']` and `row['url']` and nothing else, so
the title is discarded at ingestion; `ScreenResult.to_dict()` (ledger.py:481) emits
`url/status/detail/source/skip`; the Rich table prints `status`, `url`, `matched`;
`--new-only` echoes the URL. I ran `cmmanage discover check --from-csv
discovery_stream.csv` and `--json` — no title in any mode, on any checkout. So
`screening.txt` has no titles, and stage 3's "Give each subagent the paper's title,
its URL, and this task" (SKILL.md:166, repeated at verification.md:64) names an input
no upstream stage hands over. The agent must invent an unprompted join back to the CSV.

**(b) That join is broken on this machine, and the flag is what breaks it.**
`discovery_stream.csv` here has the five-column header
`openalex_id,url,score,flagged,publication_year` and 1000 rows. `discover stream`
resumes by default; on an existing non-empty file it sets `write_header = False`
(manage.py:3028) while `columns` has already gained `title` (manage.py:3020), and
opens the file in append mode (manage.py:3083). The first run of this skill therefore
appends six-field rows under a five-field header. `pd.read_csv` on the result raises
`ParserError`; `csv.DictReader` strands the title under the `None` restkey. Screening
itself keeps working, because `flagged` and `url` sit before the ragged field.

**Scope, per the skeptic's correction.** `git check-ignore -v discovery_stream.csv` →
`.gitignore:47:*.csv`, so nothing ships and a fresh clone writes a correct six-column
header. (b) is a migration case — and the machine it bites is yours, with exactly that
five-column file, on the first invocation. The skeptic also lowered (a) from blocking
to degrading: the subagent resolves the URL regardless and returns the title in its
report, so `candidates.md` can be titled from the reports. What (a) costs is brief
readability and parent-side bookkeeping, not verification.

Not pinned by any test. `grep title` over both named test files returns nothing, and
`tests/test_discovery_stream.py:104` tests the row-*building* function on a fresh
record — the resume-plus-`--include-title` path through the CSV writer is untested.

**Change.**
(a) Carry `title` through `discover check --from-csv`: into the `--json` dicts, as a
column in the table, and after the URL in `--new-only`.
(b) Make `discover stream` compare its column set against an existing file's header
and exit with a message naming `--include-title` — "discovery_stream.csv has no
`title` column; re-run without `--include-title` or start a new `--out`" — rather than
appending a differently shaped row.
(c) Until (a) lands, drop the "screening report" half of SKILL.md:100 and say where
stage 3 is supposed to get the title from.

---

### 2. `verdict --force` reports success for a correction that changes nothing

**Anchor:** `chem_mat_data/discovery/ledger.py:379`.
**Lenses:** contract, failure. **Severity: medium.**

`VerdictLedger._index` does `self._by_key.setdefault(key, verdict)`, and `load()`
appends rows in file order, so the **earliest** row for a URL is what `lookup` returns
forever — and `append()` re-indexes in-process, so the correction is invisible without
even a reload. `STATUS_VERIFIED` is in `SKIP_STATUSES` (ledger.py:73) whatever the
decision was.

Reproduced in an isolated ledger: append `unclear` with `why='ACS bot check'`, then
append `confirmed`/`foo_db` for the same URL. Two rows land on disk;
`screen_many` returns `verified | unclear | skip True`. The corrected row is in the
committed file and is unreachable.

Meanwhile `cmmanage discover verdict --help` advertises
`--force  Append even though this paper already has a verdict`, and the duplicate
refusal path prints "pass --force to append a second verdict for the same paper". An
agent or human that hits the refusal is pointed by the tool at a flag that does
nothing, gets `OK recorded confirmed` for it, and moves on.
`grep -rn -- --force .claude/skills/dataset-search/` returns nothing — the skill never
mentions the flag at all.

**The skeptic's correction, and it matters.** Do not read this as "an `unclear` buries
a paper". Burial is deliberate and correct: verification.md:105 says outright that
"`unclear` is a real answer and recording it is what stops the next sweep from
re-verifying the same unreachable paper." What is not deliberate is that **the only
correction channel the tool offers is a silent no-op.** The trigger is narrow —
screening drops settled papers before stage 3, so the skill's own loop never revisits
an `unclear`; you need a human-initiated recheck, an in-run recovery after a subagent
failure, or a maintainer fixing the ledger by hand. Narrow, but entirely foreseeable,
and silent success is the worst failure class in a skill whose whole economy is "never
re-verify a settled paper".

Not pinned: `tests/test_discovery_ledger.py:359`
(`test_verdict_refuses_a_duplicate_and_quotes_the_prior_reason`) asserts only
`forced.exit_code == 0`. It never asserts which verdict wins afterwards.

**Change.** Make the last verdict win: `self._by_key[key] = verdict` at ledger.py:379.
Rows carry a `date` and the file is append-only, so file order is the intended
precedence. Keep the duplicate refusal so `--force` stays a deliberate act. Then add
to SKILL.md's recording section what `--force` now means: "`discover verdict` refuses a
paper that already has a verdict and quotes the prior reason. To supersede one — you
reached a page an earlier run recorded as `unclear` — re-run with `--force`; the newer
row wins."

If you want first-wins to stand instead, the equivalent fix is to make `--force` say
so: refuse it, or print "recorded as an annotation; screening still returns the earlier
`<decision>` — hand-edit docs/search_verdicts.md to supersede". Silently succeeding is
the part that cannot stay either way.

**Risk.** Last-write-wins over an append-only log is the natural semantics; the only
cross-key interaction is `_keys_for(record_url)`, and `check` is only ever fed paper
URLs (arguments, or the `url` column of the stream CSV), so a later row's `record_url`
cannot shadow an unrelated paper's screening result.

---

### 3. `MISSING_DATASETS.md` is written but never screened, and promotion has no duplicate check

**Anchor:** `.claude/skills/dataset-search/SKILL.md:231`, `chem_mat_data/discovery/ledger.py:492`.
**Lenses:** contract, design, failure (all three found it independently). **Severity: medium.**

`screen_many` (ledger.py:492-538) consults exactly three sources: `load_rejected()`,
`load_catalog()`, `VerdictLedger.load()`. `MISSING_DATASETS.md` is not among them,
and bookkeeping.md:13 states the read-only set as metadata.yml plus
rejected_datasets.md. The file holds **77 distinct DOIs** across 86 numbered rows —
the largest store of already-known leads in the repository, and the file this skill
writes into.

Verified: `cmmanage discover check --json 'https://doi.org/10.1039/D4DD00196F'
'https://doi.org/10.1186/s13321-019-0384-1'` returns `"status": "new", "skip": false`
for both. Those are rows 3 (Polyuniverse) and 7 (pKa QSAR) of `MISSING_DATASETS.md`.
The promotion instruction (SKILL.md:231-241, bookkeeping.md:73-76) says only "Append to
the table that fits… continue the numbering" — there is no instruction anywhere to check
whether the row already exists. verification.md:136-139 handles preprint-versus-published
duplicates *within* a run and nothing else.

**When it fires.** `discover stream` defaults to `--year-min 2010` and sorts
`publication_year:desc`, so ordinary continued pagination walks back into the years the
leads come from. Today's flagged set does not overlap (0 of 115) only because
`discovery_stream.csv` currently holds nothing but 2025 and 2026 records while the leads
span 2003-2025. `--year-min 2005` (sweep.md:105) widens it further. The cursor-reset
remedy (sweep.md:107) is *not* the trigger — `.ids` is deliberately preserved there. The
leads are also the classifier's own training positives, so they score high and get
flagged when reached.

**Consequence.** One wasted verification subagent per re-encountered lead, plus a
duplicate row appended to a committed file under a fresh number. Bounded, not
progressive: the same run records a verdict, after which the paper screens `verified`
forever, so the ceiling is roughly 77 subagents and 77 duplicate rows, once. The
duplicate rows are the part that persists — the next person picking a lead cannot tell
them from new ones.

**Change, and note that the skeptic reversed the direction here.** Do **not** add
`MISSING_DATASETS.md` as a fourth skip-ledger. Those rows are not decisions the way
`rejected` / `in_catalog` / `verified` are: `_identify_positives.py:23` confirmed them
with `identify_dataset(title)` — an LLM call on the title alone, no abstract, no paper
fetch, no data-record visit — and `POSITIVES_VALIDATION.md:97` measured **26% label
noise** on that same population ("Of 142 entries, 37 were rejected by the agent").
Skipping them would permanently entrench title-only guesses and deny them the
`record_url` and entry counts that only a verify subagent produces and that
`/dataset-extraction` consumes.

Do this instead:
(a) SKILL.md:231 — before appending, grep the candidate's DOI against
`MISSING_DATASETS.md`. On a hit, **update that row in place** with the record URL and
entry counts the subagent found, rather than adding a second row, and say so in
`findings.md`. That turns a re-discovery from a duplicate into an enrichment of a
weakly-sourced lead.
(b) Amend SKILL.md:136 and bookkeeping.md:85-99 to name *both* blind spots. Today they
name only the 13-of-46 catalog `sources` gap, which reads as exhaustive.
(c) Optional, and the better version of the code change: report a `MISSING_DATASETS.md`
hit as a **non-skipping** status (`listed`, outside `SKIP_STATUSES`), so the agent is
told rather than having to remember to grep. The paper still gets verified and still
gets a real verdict.

Also fix the numbering instruction while you are there: "continue the numbering" is
undefined against a file whose numbers are non-contiguous upstream indices interleaved
across five sections, with a global maximum of 142 over 86 rows. Say "continue from the
global maximum across all sections".

---

### 4. Nothing checks that the verdict ledger is complete when the run ends

**Anchor:** `.claude/skills/dataset-search/SKILL.md:254`.
**Lens:** design. **Severity: medium.**

SKILL.md:282: "There is no manifest and no gates here; the state is the files." The one
command that would close the loop — `cmmanage discover check --from-csv
discovery_stream.csv --new-only` — appears at stage 2 (SKILL.md:121) and under *Resuming*
(SKILL.md:288, "That last command is the resume point"), but never at the end of stage 4.
Stage 4 ends at "Then tell the user: how many papers were swept, how many verified…" with
the count coming from the agent's own memory. Every stage-4 deliverable — `candidates.md`,
`findings.md`, the promoted rows — can be written correctly whether or not
`discover verdict` was ever called.

Two ways a verdict goes missing: a batch that returns while the agent is writing
`candidates.md`, and an `unclear` the agent means to come back to. (The skeptic removed a
third mechanism the finding originally claimed: a `verdict` that exits 1 on a duplicate
cannot leave a gap, because the refusal only fires when `lookup` already matched a key,
and `_index` registers both `url` and `record_url` into the same map `screen_many` reads.)

**The skeptic also added the failure that makes this check worth more than a compliance
nag.** `_keys_for` mints a `doi:` key only when the string literally contains a
`10.dddd/…` DOI. I ran it:

```
_keys_for('https://www.nature.com/articles/s41597-023-02123-4')
  -> {'url:nature.com/articles/s41597-023-02123-4'}
_keys_for('https://doi.org/10.1038/s41597-023-02123-4')
  -> {'doi:10.1038/s41597-023-02123-4', 'url:doi.org/10.1038/s41597-023-02123-4'}
```

The CSV row for that paper is the `doi.org` form. So a verdict recorded exactly as
instructed — under the resolved publisher URL the subagent reported — is invisible to the
next sweep. That is a silent ledger row no amount of care about *remembering* to record
prevents.

**Consequence.** Bounded but silent: each dropped or mis-keyed verdict costs one
re-verification on the next run, which is precisely the cost the whole design exists to
avoid. It does not compound — the re-verification records a verdict — so the waste is
roughly (drop rate × flagged papers) per run, not a growing backlog.

**Change.**
(a) Close stage 4 with the command the skill already documents, run in the other
direction: re-run `cmmanage discover check --from-csv discovery_stream.csv --new-only`,
diff it against the list stage 2 produced, and require that every URL that was given a
subagent has disappeared. Anything still printed is a verdict that was not recorded —
record it now, or name it in `findings.md` as deliberately deferred.
(b) Take the run report's "how many verified" from
`grep -c '^| http' docs/search_verdicts.md` before and after, not from memory.
(c) One line in stage 3: pass `discover verdict` the URL **exactly as it appears in the
CSV**, not the resolved publisher URL the subagent reports.

This is the gate-equivalent, it costs milliseconds, and it introduces no gate machinery —
design decision 2 stands. It is one command the skill already teaches, invoked one stage
later.

---

## Design disagreement

One finding. Act on it differently: the mechanism is correct and the prose is what is
wrong, and the obvious "fix" would break a decision that is right.

### 5. The subagent budget counts this sweep's flags; stage 3's queue is every unjudged flag ever swept

**Anchor:** `.claude/skills/dataset-search/SKILL.md:96-99`.
**Lenses:** simulate, contract, design, failure. **Severity: medium.**

SKILL.md:96-99: "At the flag rate this sweep actually produces — 115 of 1000, measured —
300 papers is about 35 flagged, which is about 35 subagents." Stage 2's working list is
`cmmanage discover check --from-csv discovery_stream.csv --new-only` over a file that
SKILL.md:90-94 and sweep.md:85-88 both insist stays shared and ever-growing, and
`check --help` shows no row, limit, or year bound. So the working list is every flagged
row across all sweeps that has never received a verdict.

Run today:

```
$ cmmanage discover check --from-csv discovery_stream.csv
… 115 checked - 115 new -> 115 worth verifying
$ grep -c '^| http' docs/search_verdicts.md
0
```

A bare invocation on this repository therefore hands stage 3 roughly 150 papers — the
115-row backlog plus this sweep's ~35 — after the skill told the reader to expect 35. The
115 is the residue of the one-off 1000-record measurement sweep whose flag rate
SKILL.md:96 and sweep.md:61 both quote and which was never verified.

**What is and is not wrong.** The mechanism is right: in steady state the two numbers
coincide, because every verified paper gets a verdict, and they diverge only when a sweep
ran without verification or a run was interrupted. Nothing is stuck, either — the count is
printed by the stage-2 command before any subagent exists, SKILL.md:142 already tells the
agent to record it, and *Resuming* already frames outstanding `new` rows as the work left.
`discovery_stream.csv` is gitignored, so a fresh clone sees 35. What is wrong is that the
number a reader was promised is off by 4x on the author's own machine, with no sentence
anywhere saying it could be.

**Change — one clause each, no new flags.**
At SKILL.md:96-99: "…about 35 flagged from this sweep, plus any flagged rows earlier
sweeps left unjudged — the number stage 2 prints is the real subagent count."
At stage 2, after the `--new-only` command: "This is every flagged paper from every sweep
that has never been given a verdict, not just this sweep's records. If it is far larger
than this sweep's own flag count, an earlier sweep was never verified — say so in
`findings.md` and tell the user the batch count before spawning."
Optionally a fifth *Escalation* bullet: "`discover check` reports far more `new` papers
than this sweep flagged — say the number and confirm before spawning."

**Do not** add `--since-row`, `--tail` or `--limit` to `discover check`, and do not screen
only this run's rows. Both skeptics rejected it independently and for the same reason: it
permanently orphans every flagged paper an interrupted or pre-skill run never judged,
which is exactly what the shared sweep ledger plus the committed verdict ledger exist to
recover, and it inverts the recall-first trade the 0.05 threshold was chosen for
(sweep.md:62-63: "a false positive costs one subagent, a false negative loses a dataset
permanently"). **Do not** add a blocking confirmation gate either — design decision 1
makes unattended runs deliberate; report the number, do not stop on it.

---

## Considered and rejected

Each of these was raised by a lens and refuted by the skeptic.

**The subagent brief writes to a path built from values the skill refuses to give it
(`<run>`, `<openalex_id>`).** The parent has both: `openalex_id` is column 1 of
`discovery_stream.csv`, the file stage 2 already reads. verification.md:64's "give it the
title and the URL, nothing else" is a bias rule about what you may say *about the paper* —
it cannot forbid the path, which sits inside the blockquote the parent hands over. The
angle-bracket placeholder convention is used identically throughout both skills, including
in `/dataset-extraction`'s subagent briefs, which have survived real runs.

**Most of what the sweep flags is a repository deposit, not an article, and the brief's
first instruction does not apply.** The OpenAlex filter is
`type:journal-article|proceedings-article|report` plus `has_abstract:true`, so a bare
deposit cannot reach the classifier. The flagged Zenodo rows resolve as `type: report`
with full abstracts — Zenodo-published documents whose landing page *is* the publication
page. verification.md:79, the first `confirmed` bullet, is "A Zenodo, Figshare, Dryad or
institutional record with data files", which is exactly the rule said to be missing.

**The promoted lead drops the record URL and the licence observation, and ranks on size
alone.** All three limbs are false. bookkeeping.md:128 explicitly puts the licence
observation in the review file and the candidate's notes; `--record-url` is a column of
`docs/search_verdicts.md` and is indexed for screening (ledger.py:378); the ranking rule
names four criteria, not one, and bookkeeping.md:84 carries the access barrier into
`Data Type`. A mandatory licence field on the search subagent would systematically capture
the *article* licence and stamp it, ungated, onto a committed leads list — precisely the
failure design decision 5 exists to prevent.

**An interrupt between a batch returning and its verdicts being recorded loses the work,
and the review files cannot be recovered because their filenames have no usable key.** The
filename key is the join key: `awk -F, '$1=="W7153478152"{print $2}' discovery_stream.csv`
recovers the URL in one line. Verdicts are recorded as each subagent returns, so a resume
at any batch boundary loses nothing, and *Resuming*'s first two commands read the run
directory. (The residual risk here is covered properly by finding 4.)
