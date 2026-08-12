# dataset-extraction: what to fix next, ranked

Scope: the skill at `.claude/skills/dataset-extraction/`, its machinery in
`chem_mat_data/agent/`, and the `agent` command group in `chem_mat_data/manage.py`.
All line numbers are against `e39f0c4` on branch `dataset-extraction-skill`
(worktree `/tmp/claude-1000/-media-ssd2-Programming-chem-mat-data/202e8fa6-6fde-4211-8f71-ae295073d8a0/scratchpad/wt4`).
Run artifacts are `/tmp/claude-1000/-media-ssd2-Programming-chem-mat-data/202e8fa6-6fde-4211-8f71-ae295073d8a0/scratchpad/sandbox-elanos/.dataset_work/enalos_bp/`.
Nothing below is fixed by `e39f0c4`, which changed documentation only.

---

## The structural conclusion

**Every real defect either validation run found was found by a fresh subagent, and no
gate found any of them.** The verification subagent found four (a hook docstring that
described behaviour the code did not have, an `int(data.get('CID', -1))` that can never
reach its default, undocumented CID-0 passthrough, and the `.mpack`/`.csv` row-order
mismatch). The license subagent found a fabricated paper title with the wrong first
author — outside its brief, the most valuable catch of run 2. The gates passed every
version, including the one whose docstring lied.

That is not a gap the gates can close by getting stricter about the same things. It is a
division of labour, and the code half of it is currently mis-specified. The gates check
that *declared* facts are internally consistent with each other; almost every field they
compare is one the agent wrote against another field the same agent wrote. Where a gate
does have access to an independently-produced record, it does not read it: pycomex writes
`experiment_meta.json` and a byte-exact `experiment_code.py` into every results directory,
and `grep -rn 'experiment_meta|experiment_code|has_error|st_mtime' chem_mat_data/agent/`
returns nothing.

Two consequences drive the ranking:

1. **The cheapest wins are making the gates read records the agent did not author**
   (`experiment_meta.json`), and making the subagents' findings survive
   (`agent status` and the manifest currently throw them away).
2. **The second structural conclusion is smaller but more urgent: the skill now contains
   instructions that the code does not implement, and two of them were added yesterday.**
   `e39f0c4` added `cmmanage agent status <slug>  # shows the recorded license` at
   SKILL.md:342 and at license.md:167. `agent_status_command` (manage.py:1808-1857) prints
   the slug, the source URL, and the six stage lines. It does not print the license. That
   instruction exists to stop an override from deleting the evidence that justified
   escalating, and it cannot be followed. Documentation fixes that outrun the code are how
   this skill accumulates its worst defects, because an agent trusts them.

---

## Ranked

Ranked by severity of the underlying problem divided by cost. Confidence is marked per
item: **verified** = I reproduced it against the code in this session; **likely** = the
mechanism is verified but the harm is projected; **speculative** = design not yet probed.

---

### 1. `agent status` is blind, and the newest commit tells the agent to depend on it

**Confidence: verified. Cost: small (~30 lines + 4 doc lines).**

Three defects in one code path.

- `agent status` renders only stage symbols, blocker and notes. SKILL.md:342 and
  license.md:167 — both added by `e39f0c4` — instruct the agent to read the recorded
  license back with it before merging a `human_override`. It shows no license at all.
  The documented safeguard against an override deleting `upstream_database`,
  `upstream_terms_url`, `upstream_license` and `attribution` is unperformable, leaving
  "retype from memory" as the only option.
- `manage.py:1918-1921` writes `notes = f'{len(report.warnings)} warning(s)'` and discards
  the text. The real manifest records `notes: 1 warning(s)` for `prepare` and for
  `license`. The license line is the one place in the whole run where the difference
  between "benign human override" and "the upstream forbids derivative works" matters, and
  the manifest does not record which it was.
- `manage.py:1911-1916` records `report.errors[0].message` and drops errors 2..n. Blocked
  runs are exactly the ones resumed cold in a new session.

**Change.**
(a) Print the license record in `agent status` — identifier, evidence URL,
`upstream_database`/`upstream_license` when named, override decision and reason — plus the
recorded warning and blocker text under each stage line. This makes SKILL.md:342 true.
(b) Add `warnings: List[str]` and `blockers: List[str]` to `StageState` *alongside*
`notes` (not replacing it — `mark()` writes `notes='invalidated by a re-run of the …'` at
manifest.py:357, and tests/test_agent_pipeline.py:114 asserts on it), populate from
`report.warnings` / `report.errors`.
(c) Set `allow_unicode=True` on `yaml.dump` at manifest.py:298 in the same commit, or the
first warning quoting a `°C` target description writes `\xB0C` into the manifest.
(d) Add three lines to SKILL.md: `--no-record` and `--strict` exist
(manage.py:1867-1877), neither appears anywhere under `.claude/skills/`.

**Risk.** `agent status` currently renders blocker *or* notes as an either/or
(manage.py:1846-1849); a third field needs that branch restructured or a stale stage will
swallow its warnings.

**Note.** This closes known-outstanding item 3 outright, and the briefing's framing of
that item is wrong: `--no-record` already gives a read-only gate re-run. I confirmed all
six gates re-run in-process against the real manifest with the file md5 unchanged. It is
undocumented, not absent.

---

### 2. An approved license override waives every check, including the one added because of run 1

**Confidence: verified. Cost: small (one function restructure + 3 lines).**

`gates.py:388-409` returns from `gate_license` inside the override branch, before the
identifier, redistribution, attribution, upstream-database and host-divergence checks at
411-495. Probed against the real module:

- `{'evidence_url': ..., 'human_override': {'decision':'approved','reason':'looks fine to me'}}`
  → `ok=True`, one warning reading `license None approved by human override`.
- The real `enalos_bp` manifest with the override removed produces **two ERRORs**:
  `redistribution is 'allowed by the data record itself …; the upstream … forbids
  distributing or preparing derivative works'` and `upstream database "US EPA EPI Suite
  …" is under "No open license…" which is not permissive`. With the override, all of that
  collapses to the single green line `license 'CC-BY-4.0' approved by human override`.
  The gate report actively misrepresents the situation rather than merely under-describing
  it.

Separately, `gates.py:768` guards the final re-check on the override's *presence*:
`if not override and not is_permissive(...)`. Loading the real manifest and mutating it:
flipping the decision to `rejected` → `gate_verification ok=True`, zero issues; setting
`decision: maybe` → ok=True; deleting `evidence_url` → ok=True; changing the identifier to
`CC-BY-NC-4.0` → ok=True. The recheck cannot detect any change to the license record while
an override key is present. SKILL.md stage 6 tells the reader this gate confirms "the
license conclusion still holds".

Also: SKILL.md:318-319's bolded "`upstream_database` is **not optional** — omitting the key
fails the gate" is false whenever an override exists.

**Change.**
(a) Restructure `gate_license` to collect all issues first, then, when an approved override
is present, downgrade each ERROR to a WARNING prefixed verbatim
(`waived by human override: upstream database "US EPA EPI Suite" is under …`), followed by
the reason. Keep the verdict `ok=True` — the override is the designed escape hatch and
`test_human_override_still_wins_over_upstream` names that intent. Both existing override
tests assert only `.ok` and survive. Fix the `license None` rendering to
`no license identified` in the same edit.
(b) Replace `gates.py:765-772` with `for issue in gate_license(manifest).issues:
report.add(issue.level, issue.message)`. Three lines, and it subsumes rejected /
garbage-decision / deleted-evidence / changed-identifier / newly-restrictive-upstream at
once.

**Risk.** Waived warnings multiply the license stage's warning count (the real run goes 1
→ 3), so `agent gate license --strict` would then fail every overridden license. Decide
deliberately whether waived warnings are exempt from `--strict`. Re-running `gate_license`
inside `gate_verification` means any license ERROR now blocks the final gate — correct, but
it will block runs whose license record was edited after stage 3 passed.

**Do not** require an identifier before an override can be approved (license.md:20 makes
"no license stated at all" the documented escalate-to-human case, so identifier-absent +
override is the intended path, and the defect is the `None` rendering), and do not add a
`human_override.waives` list of issue identifiers (see rejected).

---

### 3. license.md instructs prose in the two fields the gate matches against fixed vocabularies

**Confidence: verified. Cost: small (doc + optional schema split).**

`license.md:127` says `redistribution` — "`allowed`, or a description of the restriction".
`license.md:133-134` says "Quote the restriction in your own words in `upstream_license`
if it has no SPDX name". `gates.py:427-432` errors on any `redistribution` not in
`('allowed','yes','permitted')`; `gates.py:477-483` runs `is_permissive()` — a set
membership test against 15 SPDX-ish tokens — on `upstream_license`.

So the documented way to fill these fields guarantees a false escalation. `redistribution:
'allowed with attribution'` — the natural CC-BY phrasing — ERRORs. `upstream_license:
'Public domain; PubChem places no restrictions on use or redistribution.'` ERRORs, directly
contradicting SKILL.md:300-304's claim that "a named upstream whose own license is
permissive passes without ceremony".

**Change.** Split each field: `redistribution` becomes an enum
(`allowed` | `restricted` | `unknown`) with free-text `redistribution_note`;
`upstream_license` becomes `upstream_license_id` (SPDX or `none`) plus free-text
`upstream_license_note`, with `is_permissive` applied only to the id. Update license.md's
"What to report" block and both SKILL.md examples. Fix SKILL.md:318-319 to say what is
true after item 2 lands.

**Risk.** A schema change to a persisted manifest section; old manifests carry the old
keys. Accept both for one release, or migrate on load.

---

### 4. `crwl -o markdown-fit` is a content-pruning mode, and there is no implausible-output check

**Confidence: verified. Cost: tiny (two string edits + one paragraph).**

SKILL.md:150 makes `crwl '<url>' -o markdown-fit` the reader for every page, and
troubleshooting.md:146 repeats it. `markdown-fit` is crawl4ai's pruning mode. Measured on
the elanos_bp source article: `-o markdown` = 234,604 bytes, `-o markdown-fit` = 160,678
bytes (31.5% pruned); `grep -c 'Full size table'` = 10 vs **0**. Those anchors are the only
route to `https://link.springer.com/article/10.1007/s11030-025-11196-5/tables/2`, which
reads `BP | − 88.60 | 548.00 | 188.98 ± 85.07` — degrees Celsius, i.e. the evidence that
disproves the shipped script's "in Kelvin". SKILL.md:475 tells the stage-6 subagent to
check "that the target values are in the range the paper reports"; the mandated reader mode
deletes the route to that range. `grep -rn -i table` across SKILL.md and all five
references returns no mention of publisher table pages at all.

Separately, `crwl 'https://db.chempharos.eu/datasets/Datasets.zul' -o markdown` returns
**1 byte, exit 0**, while `curl -sL -A 'Mozilla/5.0'` on the same URL returns 11,194 bytes
containing the upstream provenance the license gate requires. The skill's fallback ladder
(SKILL.md:148-163) is keyed on *failure*; a 1-byte exit-0 read is not a failure.

**Change.**
(a) Change both occurrences to `-o markdown`. Add one sentence: `markdown-fit` prunes
low-density blocks by design — it drops boilerplate but also drops tables, figure captions
and short link-only sections. Never for the first read of an article.
(b) Add an implausible-output rule to the tools section: after every page read, check the
byte count and check the output contains a token you expect from that page class (an
article: `availability` or `supporting`; a repository record: a filename or `download`).
Under ~2 KB, or missing the token, is a **failed** read — go to the next reader. Add
`curl -A 'Mozilla/5.0'` to the ladder as the reader that wins on Ajax/ZK apps where content
sits in hidden divs.
(c) Add a "publisher tables and figures live at their own URLs" note to data-sources.md with
the Springer `/article/<doi>/tables/<n>` form.

This is the whole of known-outstanding item 1, and it is the cheapest item on this list.

---

### 5. No gate reads the run's own record — a crashed run passes all six

**Confidence: verified. Cost: small-medium (~40 lines + test fixture update).**

pycomex writes `experiment_meta.json` (`name`, `status`, `has_error`, `start_time`, and
every parameter's actual value including `__TESTING__`) and `experiment_code.py` (a byte
copy of the executing module — sha256 of the real run's copy equals the on-disk script) into
every results directory. No gate opens either. `gate_execute` instead decides whether
testing mode was on by regexing the *current source text* (`gates.py:635`), which describes
the file now, not the run.

Reproduced consequences:

- A results directory with `status='failed'` / `has_error=True` and everything else intact
  passes `gate_script`, `gate_execute` and `gate_verification` with zero issues. The base
  template writes `.mpack`, `.mpack.gz` and `metadata.yml` before the final `after_save`
  hook — the hook the extraction agent routinely writes itself — so a run that crashes in
  the agent's own code ships.
- A script edited an hour after the run, whose `experiment_code.py` literally reads
  `OLD VERSION OF THE SCRIPT`, passes all three gates.
- `smoke_results_dir` and `results_dir` are the identical string in the real manifest
  (`…/create_graph_datasets__enalos_bp/debug`), because SKILL.md:419 and SKILL.md:437
  prescribe the same path — and `__DEBUG__` `shutil.rmtree`s that folder at the start of the
  next run, so by the time `gate_script` runs, the smoke output physically does not exist.
  `gate_script` on the real manifest returns `ok=True` with zero issues while inspecting a
  5432-compound `__TESTING__=False` full run and reporting it as the smoke run.
- Nothing ties a results directory to the script. Pointing `smoke_results_dir` at any other
  single-target dataset's debug folder passes.

**Change**, in this order of value per line:

1. `_check_run_provenance(results_dir, script_path)` used by `gate_execute` and
   `gate_verification`: ERROR when `has_error` is true or `status != 'done'`; ERROR when
   `meta['name'] != Path(script_path).stem`. Three checks, no false-positive surface, closes
   the crashed-run and wrong-directory holes outright. **Do this even if you do nothing
   else in this item.**
2. Missing/unparseable `experiment_meta.json` → WARNING, not ERROR. Fatal-with-no-override
   is a dead end for archived or relocated folders, and gates `sys.exit(1)` with no `--force`.
3. In `gate_execute`, replace the `__TESTING__` source regex with
   `meta['parameters']['__TESTING__']['value']`. The regex has a live false positive today:
   `__TESTING__ = False  # (was True during the smoke run)` hard-blocks the run.
4. Run the structural block (`gates.py:542-573`) against
   `<results_dir>/experiment_code.py` in `gate_execute`, extracted as
   `check_script_structure(source) -> list[VerificationIssue]`. Today those five checks run
   exactly once, on a file the workflow *guarantees* will be edited before the run
   (SKILL.md:407 mandates `__TESTING__ = True`, SKILL.md:428 mandates `False`). Of the five,
   `.dataset_work` is the one a stage-5 debugging edit plausibly introduces.
5. In `gate_verification` only, compare `experiment_code.py` against the working script and
   **warn** (not error) when they differ, printing both short hashes. Skip when
   `experiment_config.yml` is present — `save_code` writes a stub, not a copy, for
   config-based experiments.
6. Two SKILL.md lines: run the smoke as
   `python chem_mat_data/scripts/create_graph_datasets__<name>.py --__TESTING__ True --__DEBUG__ False`
   and record the unique timestamped folder pycomex prints. The script file is then never
   edited between smoke and full run, both runs' evidence survives, and
   `smoke_results_dir != results_dir` by construction. *(This step is **likely**, not
   verified: it depends on `ExperimentArgumentParser.parse` eval'ing CLI values, which one
   investigator confirmed but I did not re-run.)*

**Risk.** `write_results` (tests/test_agent_pipeline.py:565) emits neither file, so 15 tests
in `TestScriptGate`/`TestExecuteGate` break until the fixture emits an `experiment_code.py`
matching the script it wrote and a meta whose `name` equals the script stem. That is a
fixture-coupling refactor, and `test_testing_flag_left_on_is_an_error` changes meaning.
Do **not** hash the script against `smoke_results_dir/experiment_code.py` in `gate_script`
— the mandated `__TESTING__` flip makes that mismatch by design, and gates have no override.
Do **not** make the hash an ERROR anywhere: two of run 2's four verification fixes were
comment-only, and a full run "can take hours".

This closes known-outstanding item 2. Note that step 1 and step 6 are independent; step 6 is
what actually makes the smoke check non-vacuous, and it costs two documentation lines.

---

### 6. `source_url` is still settable in code, and fixing that re-arms a warning that will now fire on every run

**Confidence: verified. Cost: small.**

`e39f0c4` documented `source_url` as "set once by `init`" in SKILL.md:68-74. `agent set`
still accepts it (manage.py:1978), and manage.py:481-485's own help text still lists it
first among the freely settable scalars. The prose and the tool disagree, which is exactly
how the original defect happened: the real manifest records
`source_url: https://zenodo.org/records/14392754/files/BP.csv?download=1`.

The compound problem: with the download URL in `source_url`, the host-divergence warning
(`gates.py:486-495`) compares `zenodo.org` against `zenodo.org` and stays silent — I
confirmed it did **not** fire in run 2. Restore the correct Springer URL and it fires
immediately. So `e39f0c4`'s doc fix converts known-outstanding item 4 from a hypothetical
into a certainty: every ordinary paper-on-publisher / data-on-repository run will now warn.

The root cause is that the manifest has no field for the direct download URL, so an agent
that has just spent stage 1 establishing it has literally nowhere to record it.

**Change.** Remove `source_url` from the `scalars` dict and reject it with an error naming
`init --url`. Add `record_url` (the repository landing page) and `data_url` (the direct
download) as settable scalars. Have `gate_license` compare the evidence host against
`record_url` when present and `source_url` otherwise — which is the correct comparison
anyway, since the question the warning exists for is "is the file we downloaded covered by
the license we read", not "did the paper and the data live on the same host". Fix
manage.py:483.

**Risk.** Removing a settable key breaks any in-flight run that expects it; the error
message must name the replacement.

---

### 7. The final gate never compares `target_descriptions` or `dataset_name`

**Confidence: verified. Cost: small.**

`gate_verification` compares `('verbose','description')` at gates.py:741 and
`('tags','sources')` at gates.py:754. Nothing else. Probed on the real manifest: replacing
the draft's `target_descriptions` with `'BP - boiling point in KELVIN, log-transformed'`,
`dataset_type` with `'classification'` and `dataset_name` with `'something_else'` →
`ok=True`, **zero issues**. `target_descriptions` is the only per-target place units live,
and `dataset_name` is the identifier a downstream user types into `load_smiles_dataset`.
`gates.py:709-710`'s docstring claims the gate "confirms that what ended up in
`metadata.yml` actually matches what the metadata stage approved". (SKILL.md:383-385 already
states the truth correctly, so the docstring is the only place that overclaims.)

**Change.** Compare `target_descriptions` index-by-index after
`normalize_target_description_keys` at **WARNING** on divergence (matching how
`verbose`/`description` are already treated) and **ERROR** when an index approved in the
draft is missing or empty. **ERROR** when `draft['dataset_type']` is not in
`written['target_type']` — categorical, so a mismatch is unambiguous, and across all 45
catalog entries `target_type` is uniformly `['regression']` or `['classification']`.
**ERROR** when the `.mpack` stem does not equal `draft['dataset_name']`. Fix the docstring.
Document `--strict` at stage 6, which makes every warning blocking at the final gate for
zero code change.

**Be honest about what this buys.** It catches transcription drift between the approved
draft and the `METADATA` literal. It would **not** have caught the "in Kelvin" defect: that
script has no manifest and never went through the skill, and an agent confidently wrong
about units writes the same wrong string into both sides. See item 9 for the check that
would.

**Do not** promote the missing-`' - '`-separator warning to ERROR, and do not add a
`description[i].split(' - ')[0] == TARGET_COLUMNS[i]` check (both in rejected).

---

### 8. The `.mpack` and the `.csv` of the same run are a full permutation of each other

**Confidence: verified. Cost: medium.**

Measured on the real run: 5432 graphs, 5432 CSV rows, **5432/5432 positional mismatches**,
identical sorted multisets. `mpack[:3] = ['CC(C)CCC(CC)CC', 'CC(CC)(CC)C(O)=O',
'C[C@@H](N)C(C)(C)C']` against `csv[:3] = ['C=O', 'OC(=O)c1cc(Cl)ccc1Cl',
'CN(C)CN(C)C']`. Content is intact; only row order diverges. Anyone joining the two
artifacts by row index silently mismatches every compound.

Cause: `create_graph_datasets.py:357` feeds workers with `indices.pop()` (from the tail)
and appends graphs in *completion* order, while `save_csv` (lines 256-284) iterates the
dataset in insertion order. Second, independent bug: `save_csv` is applied at lines 324-326,
**before** the `__TESTING__` truncation at 333-337, so every smoke run writes a
full-length CSV next to a 50-graph mpack.

This is in the shared base template, so it affects every dataset in the repository, not just
skill-produced ones. `verify_results` (verification.py:186-190) checks only that both files
exist.

**Change.** Carry the source index through the worker and sort `graphs` back into dataset
order before saving; move the `save_csv` call after the `__TESTING__` truncation; set
`allow_unicode=True` on `yaml.dump` at create_graph_datasets.py:458 (this is where the
`\xB0C` escape in the known-outstanding list comes from — it is cosmetic, since
`yaml.safe_load` round-trips it correctly, but the file is meant to be human-readable).
Add a cross-artifact check to `verify_results`: CSV row count must equal `compounds`, and
the first N mpack SMILES must equal the corresponding CSV rows.

**Risk.** Changing row order changes the bytes of every dataset the next time it is
regenerated; if any downstream consumer has pinned an index-based split, it breaks. Worth
checking before shipping. The `verify_results` cross-check will fail on every existing
results directory until the template fix lands — sequence them.

---

### 9. Units have no check at all; record the source-stated range and check it

**Confidence: likely (the streaming implementation was measured; the manifest field is new design). Cost: medium.**

Units are the highest-consequence claim the pipeline makes and the only one with no
verification anywhere. Nothing in the manifest records what the paper says the values
should look like, so stage 6's instruction to check "that the target values are in the
range the paper reports" (SKILL.md:475) has no reference point recorded anywhere and no
mechanical backstop.

**Change.** Add `target_ranges` to `metadata_draft`: per target index, the min/max/mean the
source states, plus the URL it was read from (`null` with an explicit reason when the source
states none). `gate_metadata` requires the key present for regression datasets. In
`gate_execute`, stream the labels out of the `.mpack` with
`msgpack.Unpacker(f, ext_hook=…)` + `read_array_header()` — one graph at a time,
accumulating min/max/mean/n_unique per target index — and ERROR when the observed range
falls outside the declared range beyond tolerance, WARN when it is a suspiciously narrow
subset. Print the per-target summary statistics unconditionally, so the stage-6 subagent has
numbers to hold against the paper instead of having to compute them itself.

Measured: streaming is 0.08 s / 3 MB RSS on the real 29 MB mpack, versus 0.09 s / 69 MB for
`load_graphs`, and it scales to the 19.4 GB mpacks in this repo that `load_graphs` cannot
touch. **Never call `load_graphs` in a gate.**

This is the only proposal on the list that could have caught "in Kelvin" (declared −88.6 to
548.0 against a Kelvin claim), and it is the only one that catches a wrong `TARGET_COLUMNS`
(`D001`'s range is 0..1). It depends on item 4(c) to make publisher table pages reachable.

**Risk.** A new required field in `metadata_draft` blocks existing in-flight runs. Papers
that state no ranges need the explicit-`null`-with-reason path or the requirement becomes
an invitation to fabricate numbers, which is worse than no check.

---

### 10. `gate_prepare` accepts anything non-empty that does not start with `<html`

**Confidence: verified. Cost: small in the reduced form (~30 lines).**

`gates.py:211-218` checks `st_size > 0` and an HTML byte prefix, and nothing else. All of
the following pass with `ok=True`: `BP.csv` containing `Not Found`; a Zenodo JSON record
dump saved as `BP.csv`; `smiles,BP\n` and nothing else; `<?xml version="1.0"?><error>access
denied</error>`. `gates.py:231` requires only that the substring `http` appears somewhere in
the recipe, so `see the paper at http://doi.org/10.1000/xyz` passes while recording nothing
the script stage can build a download from. Conversely, a legitimate two-row
`license_compounds.csv` is **rejected** — `NON_DATA_NAMES` (gates.py:70) is matched as a
bare substring of the filename.

Severity is capped: the script must be self-contained, so a garbage scratch download surfaces
at the smoke run (`no .mpack dataset file found`, `smoke run produced 0 compounds`). The cost
is two wasted stages, not a shipped-wrong dataset. The one genuinely silent case is a
truncated-but-parseable download: the agent types `input_rows` from the partial copy, the
script downloads the full file, and `gate_execute` only *warns* when `compounds >
input_rows` — so an unnoticed truncation converts the execute gate's one quantitative
safeguard into a no-op.

**Change.**
(a) Directory-level, not per-file: ERROR unless **at least one** file in the downloads dir
parses as a table with ≥1 data row (`sep=None, engine='python'`; no ≥2-columns requirement —
a single-column SMILES list is legitimate). This catches every demonstrated case while
leaving ancillary JSON and prose files alone.
(b) Match `NON_DATA_NAMES` against the filename stem with word boundaries, and list the
excluded files in the "no data files" error so it is diagnosable.
(c) Promote `compounds > input_rows` from WARNING to ERROR (gates.py:653-658), with the same
`max_drop_rate`-style escape hatch the drop check has. This is the actual fix for the silent
case.
(d) Recipe check: require at least one URL whose host is not `doi.org`/`dx.doi.org`.
(e) Reword the "no README captured — units will have to be inferred" warning to "no schema
documentation captured", and fire it only when neither a README nor a schema section in the
recipe is present. In the real run the agent's own `download_recipe.md` documented the 798
columns, the units and the value range better than any README would, and the gate warned
anyway. (Known-outstanding item 5.)

**Do not** add a per-suffix parse dispatch, a JSON "repository record" heuristic, or a rule
matching URL final segments against downloaded filenames — all three block legitimate runs
(see rejected).

---

### 11. A blocked run cannot record why

**Confidence: verified mechanism, projected harm. Cost: small.**

SKILL.md:78-113 teaches the agent to compose a good escalation — what is blocking, what was
tried with specific URLs and tools, what is needed — and then SKILL.md:105 tells it to persist
that by running `cmmanage agent gate <stage> <slug>`, which records the gate's mechanical
first error instead. On a paywalled paper the manifest ends up saying `no data files in the
downloads directory (only README/recipe files were found)`. `agent set` rejects `blocker`
and `notes` outright (manage.py:1985-1996), so the information has nowhere to go.

The sharper defect, which matters more than the missing prose: **`BLOCKED` is not a
human-escalation marker.** `--record` defaults on, so *any* failed gate marks BLOCKED,
including ones the agent fixes thirty seconds later. A resuming reader cannot distinguish
"needs a human with an ACS subscription" from "the agent hadn't written the draft yet".

**Change.** With item 1(b) done (all errors recorded, not just the first), add one command:
`cmmanage agent block <stage> <slug> --why <one line> --tried <text> --need <text>`, which
sets `needs_human: true` on the stage state so `agent status` can render
`✗ blocked — needs a human: <why>` versus `✗ blocked — gate failure, resumable`. Run it
*after* the gate, and say so at SKILL.md:102-105: the gate records the symptom, `agent block`
records the diagnosis.

Also fix `Manifest.mark` (manifest.py:343) to preserve `blocker` and `notes` unless new ones
are passed, instead of constructing a fresh `StageState`. Today `agent block` followed by
`agent gate` — the ordering the original proposal prescribed — silently erases everything the
block command wrote. A status change is not a reason to forget why.

**Risk.** `needs_human` on `StageState` touches `to_dict`/`from_dict`; old manifests must load
with it absent (manifest.py:109-116 already tolerates missing keys). Do not add `agent note`
— no demonstrated demand.

---

### 12. Manifest durability, and no way to find a run again

**Confidence: verified. Cost: small.**

Three separate small things in the state layer, cheap enough to do in one session:

- `manifest.py:297-298` rewrites the file in place with a plain truncating `open`, no temp
  file, no backup — on every `agent set` and every `agent gate`. `load()` then does the worst
  possible thing with damage: `StageState.from_dict` returns a pending state for any non-dict
  (manifest.py:109-110) and absent stages are filled in as pending (manifest.py:235-238), so a
  manifest truncated to `slug: g\nsource_url: https://x` loads **without error** reporting
  that nothing has been done. A resuming agent re-downloads, re-licenses and re-scripts a
  completed run, losing the human's override. Fix: temp + `os.replace`, a `.bak` copy, and
  raise on a missing `slug`/`created_at`/`stages` mapping rather than defaulting to pending.
- The agent group registers only `venv, tools, verify, init, status, gate, set`
  (manage.py:825-832). There is no way to enumerate `.dataset_work/`, so a human returning to
  a blocked run must already know its slug. Add `agent list`: slug, dataset name, next stage,
  status, blocker first line, age, sorted blocked-first.
- `init --force` (manage.py:1782-1789) constructs a fresh manifest over the old one, deleting a
  recorded `license.human_override` — a human's verbatim words, often the product of an email
  exchange — while leaving the downloads that made the run cheap to resume. Rename the existing
  file to `manifest.<created_at>.yml` instead, and refuse outright when `human_override` is
  present unless a separate explicit flag is given, quoting the recorded decision in the error.

Also worth noting and not fixing here: `agent set` writes the entire manifest payload with
**zero** staleness invalidation (manage.py:1956-2020 never calls `mark()`). Replacing
`metadata_draft`, `results_dir` and `script_path` on a fully-passed run leaves all six stages
`passed`. The staleness model guards `gate`, but `set` is where state actually changes. This
is the mechanism by which an ERROR added anywhere is trivially cleared by rewriting the draft
to match whatever the script says — worth knowing before relying on any new ERROR.

---

### 13. Two smaller instruction defects

**Confidence: verified. Cost: tiny.**

- **SKILL.md:52 and SKILL.md:485 say "re-run" a stale stage.** For `execute` — whose own text
  warns "This can take hours" — that reads as an instruction to reprocess the dataset. Gates
  execute nothing; they read files. Measured on the real completed run: `script` 2.5 ms,
  `execute` 2.0 ms, `verification` 3.6 ms, all `ok=True`. And license.md:181 routes every
  CC-BY dataset with an attribution requirement — the most common passing license — straight
  into that cascade by design. Replace with: "Re-run its *gate* first — the results on disk are
  untouched and it normally re-passes in well under a second. Only redo the stage's work if the
  gate then fails."
- **SKILL.md:385-387 documents a failure mode that does not exist.** It says putting
  `description` only in `METADATA` and not in `DESCRIPTION` "is the way to fail that gate at the
  very last stage". `create_graph_datasets.py:453` does `metadata.update(e.METADATA)`, so
  `METADATA['description']` lands in `metadata.yml` and matches the draft — it passes. The
  instruction trains the reader to guard against the wrong thing. Replace with the true
  statement: the template merges `METADATA` over the derived fields, so whichever of
  `DESCRIPTION` / `METADATA['description']` you set must match the approved draft verbatim.
- **`examples/xyz_3d.py` contradicts the stage-4 non-negotiable it illustrates.** SKILL.md:371-373
  says `load_dataset` must download from the original URL and must not read `.dataset_work/`.
  The example sets `SOURCE_PATH = ~/Downloads/qm9.xyz_bundle`, contains no download code at all
  (`grep 'requests|urlretrieve|FileDownloadSource'` → nothing), and passes `gate_script`'s
  self-containment check only because `re.search(r'https?://', source)` (gates.py:569) is
  satisfied by the two DOIs sitting in `METADATA['sources']`. Either give the example real
  download logic or label it as the one archetype where the data is fetched by the intermediate
  base.

Also worth two lines in SKILL.md while editing: the skill has **zero** coverage of a paper that
publishes several datasets (`grep -rn -i 'multiple dataset|several dataset|each dataset'` across
all nine skill files returns nothing), while the elanos_bp source paper publishes nine across
nine repository records, and the manifest is single-valued throughout. State the rule: one slug
per shippable dataset; enumerate them first and say so to the user. And specify `input_rows`
properly — SKILL.md:206's trailing comment "rows in the raw source, for the drop check" is its
entire specification, for the number that drives the pipeline's only defence against silent data
loss.

---

### 14. Resolve citations against the DOI registry

**Confidence: verified that the mechanism works; the gate integration is design. Cost: medium.**

The pipeline's output is a citable scientific artifact whose provenance is entirely
agent-authored prose, and no gate reads any of it. In run 2 the agent invented a paper title
with the wrong first author in its own download recipe, and the license subagent caught it
*outside its brief* — by luck. `gate_metadata` requires `sources` to be a non-empty list
(gates.py:301-303) and nothing more, so a hallucinated DOI ships.

One endpoint covers both registries:
`curl -sL -H 'Accept: application/vnd.citationstyles.csl+json' https://doi.org/<doi>` returns
CSL-JSON via Crossref for the Springer DOI (title, first author `Koutroumpa, Nikoletta-Maria`),
via DataCite for the Zenodo DOI, and HTML for a fabricated one. ~1 s per DOI.

**Change.** `cmmanage agent cite <slug> <doi>` fetches and stores the registry record verbatim
under `citations[<doi>]`. `gate_metadata` then requires every DOI-shaped entry in `sources` to
have a resolved record; `gate_license` requires `evidence_url`'s DOI, when it has one, to be
among them; the skill instructs the agent to compose citation prose *from* the stored records
rather than from memory. Add `--offline` degrading to a warning.

Cheaper alternative if that is too much machinery: widen the stage-3 subagent's brief from
"license" to "provenance and license" and make citation verification an explicit deliverable.
That converts run 2's luckiest catch into a process, for the cost of one paragraph — and given
the structural conclusion at the top of this document, it may be the better trade.

---

## Considered and rejected

**Reconcile the produced dataset against the raw source instead of counting rows.** Provably
cannot detect what it exists to detect: it reads `SMILES_COLUMN`/`TARGET_COLUMNS` out of the
run's own `experiment_meta.json` and then reads those same columns from `raw_data.csv`, so the
sets match by construction. Demonstrated by building three genuine datasets from the real
`raw_data.csv` (correct columns; `TARGET_COLUMNS=['D001']`; `SMILES_COLUMN='Canonical SMILES'`)
— absent-fraction 0.00% for all three. It also hard-fails every classification dataset built
with the repo's one-hot pattern (15 of 71 scripts) and every log/unit-transformed target (6
scripts), `raw_data.csv` is present in 0 of 51 real results dirs, and `load_graphs` would need
~65 GB peak on the largest mpack. Replaced by item 9, which sources the expectation from the
paper rather than from the run.

**Split the manifest into source-scoped and dataset-scoped state with a `DEPENDS_ON` graph.**
The motivating scenario is structurally unreachable — `run_gate` refuses to gate `script` while
`license` has not passed, so the "human approves an override days later and invalidates the
multi-hour run" event cannot happen. When re-gating is reachable it costs 2-4 ms, not hours
(measured). And the proposed dependency map omits `prepare → license`, although `gate_license`
reads `manifest.source_url` — so a prepare re-run landing on a different record would leave a
license established against the old record marked `passed`, reintroducing the run-1 failure
class. Positional invalidation over-invalidates cheaply and never under-invalidates a
safety-relevant edge; keep it. Fix the wording instead (item 13).

**Cache the license finding across a paper's sibling datasets.** license.md:6-9 runs the license
stage as a fresh subagent *deliberately*: "an agent with no stake in the outcome reads the terms
as they are." Both runs' best catches came from that independence. Nine independent
investigations are nine replications of the only safety check, not nine chances to disagree; the
disagreement is the signal. Also the license is not source-scoped in the motivating example —
the paper's nine datasets live in nine separate Zenodo records with per-record license fields.
If the rediscovery cost matters, `agent init <slug> --from <sibling>` handing the sibling's
conclusion to the fresh subagent as **prior art it must agree or disagree with** gets the saving
without losing the independence.

**Pin `script_sha256` in the manifest.** By the proposer's own reasoning the digest always
differs at the execute gate (the mandated `__TESTING__` flip), so the conditional carries no
information. It also requires gates to mutate the manifest, which today happens exclusively in
`agent_gate_command`. `experiment_code.py` is already a byte-exact record written by the runner
rather than by the agent being policed — use it (item 5).

**Require an identifier before an override can be approved, and a `human_override.waives` list
of issue identifiers.** license.md:124 says to omit `identifier` entirely rather than guess, and
license.md:20 makes "no license stated at all" the documented escalate-to-human case — so
identifier-absent + override is the intended path, and the defect is the `None` rendering.
`VerificationIssue` is `level` + free-text `message` with 69 `report.add()` sites; adding a code
taxonomy makes persisted manifest schema that can never be reworded, and the extraction agent
would author both the license blob and the `waives` list, so it buys no oversight.

**Promote the missing-`' - '`-separator warning to ERROR.** 43 of 484 shipped descriptions lack
the separator, whole datasets at a time (tmqmg 20/20, tmqm 8/8, kulik_spin 7/7). It is style, and
`test_description_without_separator_is_a_warning` asserts the current behaviour by name.
`--strict` already covers anyone who wants it blocking.

**Have `gate_execute` assert `description[i].split(' - ')[0] == TARGET_COLUMNS[i]`.** Breaks six
shipped scripts, including `half_life` — the script the skill itself produced in validation run 1
— and both binary-classification scripts, whose one-hot pattern produces more descriptions than
`TARGET_COLUMNS` entries. It also contradicts metadata.md:82, which teaches a semantic label as
the head for classification targets.

**Derive `input_rows` from the downloaded scratch file.** The denominator would come from a copy
that may itself be truncated, arriving labelled "derived, not typed" and trusted accordingly.
Promote `compounds > input_rows` to ERROR instead (item 10c) — that compares the typed denominator
against the *script's* own acquisition of the source.

**Demote the preprint/published host-divergence warning.** It reverses part of `e39f0c4`, and a
"both hosts are publication venues" test needs a hardcoded venue allowlist that would suppress the
paper-on-Springer / SI-elsewhere case. Compare against `record_url` instead (item 6) — that is the
comparison the warning should have been making from the start.

**Per-format parse dispatch in `gate_prepare`, a JSON "repository record" heuristic, and matching
URL final segments against downloaded filenames.** The JSON rule would have blocked the only
end-to-end run the skill has ever passed — that run's downloads directory legitimately contains
`zenodo_record_14392754.json` as provenance. The URL rule is wrong by construction for figshare
(`ndownloader/files/<id>`), OSF and Dryad, three of the eight hosts documented in
data-sources.md. `pd.read_csv(nrows=5)` with a ≥2-column requirement hard-errors on
semicolon-separated CSV, single-column SMILES lists and prose `.txt` documentation. A false ERROR
here is more expensive than a false pass, because the pass is caught at the smoke run while the
ERROR stops a correct run dead.

**`agent note` (an append-only timestamped log in the manifest).** No demonstrated demand from
either validation run. With all gate errors and warnings persisted (item 1), nothing remains for
it to carry.
