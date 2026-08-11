---
name: dataset-extraction
description: Turn a scientific publication into a working ChemMatData processing script, through six gated stages — prepare (locate and download the data), metadata (assemble the descriptive fields), license (check redistribution is permitted), script (write and smoke-test), execute (full run), verification (final independent pass). Each stage ends at a code-enforced gate and blocks to the human when it cannot proceed. Use when asked to add a dataset to ChemMatData, process a paper into a dataset, extract a dataset from a publication, or write a create_graph_datasets script. Input is a publication URL or a concrete paper title.
---

# Dataset Extraction

Turn a publication into a committed, verified `create_graph_datasets__<name>.py`.

The input is a **publication URL or a concrete paper title** — deciding whether a
paper is worth extracting happens before this skill runs.

```
prepare       locate + download the raw data        → .dataset_work/<slug>/downloads/
metadata      assemble the descriptive fields       → manifest.metadata_draft
license       is redistribution permitted?          → manifest.license   [subagent]
script        write + smoke-test on 50 elements     → chem_mat_data/scripts/
execute       full run + drop-rate analysis         → results/<script>/debug/
verification  final independent pass                → verdict            [subagent]
```

Two rules govern the whole run:

1. **A stage is done when its gate passes, not when you think it went well.**
   `cmmanage agent gate <stage> <slug>` is the arbiter. It exits non-zero and records
   the stage as blocked on failure.
2. **When you cannot fix something, stop and ask the human.** Do not work around a
   blocker, do not guess, do not proceed to the next stage. See *Escalation* below.

The run ends at a verified local script. Uploading to the remote share is a separate,
deliberate human step (`cmmanage metadata upload`) and never part of this skill.

## Setup

```bash
source .venv/bin/activate
cmmanage agent init <slug> --url "<publication url or title>"
```

`<slug>` is a short identifier, normally the eventual dataset name. This creates
`.dataset_work/<slug>/` with a `manifest.yml` and an empty `downloads/`.

**Resuming.** State survives between sessions. If a slug already exists:

```bash
cmmanage agent status <slug>
```

That prints each stage as `pending`, `passed`, `blocked` or `stale`, plus any recorded
blocker, plus the next stage to run. Start from there rather than beginning again. A
`stale` stage passed once but an earlier stage has since been re-run, so its result no
longer reflects what was approved — re-run it.

## Recording state

Each stage hands its conclusions to the next through the manifest:

```bash
cmmanage agent set <slug> <key> <value>
```

Scalar keys: `source_url`, `dataset_name`, `script_path`, `smoke_results_dir`,
`results_dir`, `input_rows`, `max_drop_rate`, `drop_rate_reason`.
JSON-object keys: `metadata_draft`, `license`.

The gates read the manifest. A stage that does not record its findings will fail its
own gate.

---

## Escalation

When a stage hits something you cannot resolve — a paywall, an ambiguous license, a
dataset that turns out not to exist, a format you cannot parse — **stop and ask the
user with AskUserQuestion**, in the same session, rather than failing silently or
inventing a workaround.

Make the question concrete and give real options. State plainly:

- **What** is blocking, in one sentence, without jargon.
- **What you tried** — the specific URLs, tools and fallbacks.
- **What you need** from them to continue.

For example, on a paywalled file: explain that the article's supplementary data sits
behind an ACS login, that you tried `crwl` on the article page (403), searched for an
open-access preprint (none found), and tried the direct SI URL (login redirect); then
offer the options of downloading it manually into
`.dataset_work/<slug>/downloads/`, supplying an alternative URL, or abandoning this
paper.

Before asking, record the blocker so it survives the session:

```bash
cmmanage agent gate <stage> <slug>     # marks the stage blocked with the first error
```

If the user resolves it, re-run the gate and continue. If they choose to abandon, stop
cleanly and say what was left behind.

**Do not escalate for things you can determine yourself.** A missing README means you
read the data more carefully; it is not a blocker. Escalate when progress genuinely
requires a decision or an access right that is theirs.

---

## Stage 1 — prepare

Locate and download the raw data, and write the recipe the script stage will need.

Read the publication and look in "Data Availability", "Data Records", "Data and Code
Availability", "Supporting Information", "Associated Content", and author notes.
Follow repository links to their landing pages. `references/data-sources.md` has the
direct-download URL pattern for each major host — **the landing-page URL is almost
never the download URL**, and saving a landing page as `data.csv` is the most common
way this stage fails while appearing to succeed. The gate checks for it.

If the paper yields nothing, search for `"<title> dataset download"` or
`"<first author> <year> dataset zenodo OR figshare OR github"`. If the article is
paywalled, look for a preprint — you only need the Data Availability section.

### Which files to take

- **Prefer tabular**: `.csv`, `.tsv`, `.xlsx`, `.json`, `.sdf`, `.smi` over archives.
- **No duplicates** — each logical dataset once; skip the `.zip` if the `.csv` inside
  is separately available.
- **Multiple files when genuinely distinct** (train/test splits, separate property
  sets), not alternate encodings of the same table.
- **One README** if it documents the schema. The later stages depend on this more than
  on anything else.
- **A LICENSE file if one exists** — stage 3 needs it.
- **Skip** standalone PDB/CIF/coordinate-only files unless they are the only format.
- **Size guard**: over ~2 GB, or hundreds of small files, take the primary data plus
  the README.

### Tools

| Need | Use | Fallback |
|---|---|---|
| Read a page | `defuddle parse '<url>' -m` | `crwl '<url>' -o markdown-fit`, then `WebFetch` |
| Search | `WebSearch` | — |
| Download | `curl -L -o '<abs path>' '<url>'` | `mcp__file-download__download_file` |

Check a tool exists before relying on it — all three page readers have failed
simultaneously in testing:

- `defuddle` needs `npm install -g defuddle`. Preferred: fast, no browser needed.
- `crwl` needs `uv tool install crawl4ai` **and** `playwright install chromium`. The
  browser is a separate step, and the failure without it (`Executable doesn't exist at
  .../chromium-*`) does not say so.
- `WebFetch` is built in but is 403'd by several publishers, ChemRxiv among them.

For downloads that 403, retry with `curl -L -A 'Mozilla/5.0'`. **Publisher sites are
often unreachable by any scripted client.** Before concluding a file cannot be had, try
the Europe PMC supplementary route in `references/data-sources.md` — it serves the same
publisher-deposited files without a bot check, and is usually faster than fighting the
block.

Save into `.dataset_work/<slug>/downloads/`, descriptive filenames, no subdirectories.
Sanity-check each download — `file <path>` must not say "HTML document".

### Write the download recipe

`.dataset_work/<slug>/downloads/download_recipe.md` must record, per file: the direct
URL; the right Python approach (`requests.get(url, timeout=300)`, streamed with
chunked writes for large files, or `FileDownloadSource` for Figshare and anything
needing `ssl_verify=False`); whether it is an archive and which members matter; any
special headers or SSL settings; approximate size.

### Gate

```bash
cmmanage agent set <slug> input_rows <n>      # rows in the raw source, for the drop check
cmmanage agent gate prepare <slug>
```

Checks: data files exist and are non-empty; none is a saved HTML page; the recipe
exists and records a URL. Warns when no README was captured.

---

## Stage 2 — metadata

One deliberate pass assembling the descriptive metadata, before any code is written.
Doing this first means the script is written against fields that have already been
agreed, and the final gate can check that what shipped matches what was approved.

Read `references/metadata.md` for the field spec, the established tag vocabulary, and
how to write a description and target descriptions well.

Only the **human-authored** fields belong here. `compounds`, `targets` and
`target_type` are derived by the base template at save time and are checked later.

Work out from the README and the data itself: the target columns in the order they
will appear in the vector, what each one means, and **its units**. Do not guess units
when the README states them; if nothing states them, say so in the description rather
than inventing them.

```bash
cmmanage agent set <slug> metadata_draft '{
  "dataset_name": "demo_ds",
  "dataset_type": "regression",
  "verbose": "Human Readable Name",
  "description": "Two to five sentences: what, how many, measured or computed how, used for what.",
  "tags": ["Molecules", "SMILES", "..."],
  "sources": ["https://doi.org/...", "https://zenodo.org/records/..."],
  "target_count": 2,
  "target_descriptions": {
    "0": "col_a - what it is, with units",
    "1": "col_b - what it is, with units"
  }
}'
cmmanage agent gate metadata <slug>
```

Checks: `dataset_name` is lowercase snake_case; `dataset_type` is regression or
classification; `verbose`, `description`, `tags`, `sources` present;
`target_descriptions` covers every index from 0 to `target_count - 1` with non-empty
values. Warns on a short description and on descriptions missing the
`<column> - <meaning>` form.

---

## Stage 3 — license

**Run this as a fresh subagent.** Spawn a general-purpose agent whose only job is to
determine the license, giving it the publication URL, the repository URL and the
downloads folder. Do not do it yourself.

The reason is bias, not capacity. By this point you have invested real work in making
this dataset happen, which is exactly the state of mind in which an ambiguous license
gets read charitably. An agent with no stake reads it as it is.

Brief the subagent to find: the license on the data repository record (not the
article's own license — they frequently differ), any LICENSE file in the download, the
journal's data policy, and any explicit terms-of-use. Tell it to report the license
identifier, a URL that a human can check, whether redistribution is permitted, and
what attribution is required — and to report "none found" rather than inferring a
license from context.

The policy is in `references/license.md`. In short: public domain, CC-BY, MIT/BSD/
Apache pass. **NonCommercial, ShareAlike, copyleft, and anything not clearly
identified escalate to the human** — including no license at all, which is the most
common case and is not the same as permission.

```bash
cmmanage agent set <slug> license '{
  "identifier": "CC-BY-4.0",
  "evidence_url": "https://zenodo.org/records/...",
  "redistribution": "allowed",
  "attribution": "Cite Lopez et al., Sci. Data 3, 160086 (2016)."
}'
cmmanage agent gate license <slug>
```

If the gate escalates, **ask the user** — present the license found, the evidence URL,
what it permits and what it forbids, and let them decide. Record their decision:

```bash
cmmanage agent set <slug> license '{
  "identifier": "CC-BY-NC-4.0",
  "evidence_url": "https://...",
  "human_override": {"decision": "approved", "reason": "<their reason, verbatim>"}
}'
```

An override without a reason fails the gate. Never write one on your own judgement —
it exists to record a human decision, not to unblock yourself.

If the attribution requirement is non-trivial, fold it into the metadata `sources` and
re-run the metadata stage.

---

## Stage 4 — script

Write the script and prove it works on 50 elements before committing to a full run.

Read `references/script-anatomy.md` for the template, hook contract and parameters, and
pick the archetype:

| Data looks like | Extends | Example |
|---|---|---|
| CSV/Excel with SMILES + target columns | `create_graph_datasets.py` | `examples/smiles_csv.py` |
| Archive or custom text format | `create_graph_datasets.py` | `examples/archive_extract.py` |
| A bundle of `.xyz` files (3D geometry) | `create_graph_datasets__from_xyz.py` | `examples/xyz_3d.py` |
| Transition-metal complexes needing decomposition | *nothing* — standalone | `examples/tmc.py` |

Non-negotiables, all gate-enforced:

- Write it to `chem_mat_data/scripts/create_graph_datasets__<name>.py`. Because it
  lives there, `Experiment.extend('create_graph_datasets.py')` is a **relative** path.
- **Self-contained**: `load_dataset` downloads from the original URL with caching. It
  must not read `.dataset_work/` — that is scratch, not a dependency.
- The script must carry through the approved `metadata_draft`, but **the draft's fields
  do not map one-to-one onto `METADATA`**:

  | Draft field | Goes into |
  |---|---|
  | `verbose`, `tags`, `sources`, `target_descriptions` | the `METADATA` dict |
  | `description` | the **`DESCRIPTION`** parameter — the template writes it into `metadata.yml` from there, *not* from `METADATA` |
  | `dataset_name` | the `DATASET_NAME` parameter |
  | `dataset_type` | the `DATASET_TYPE` parameter |
  | `target_count` | nothing — it is derived from the data at save time |

  The verification gate compares `verbose`, `description`, `tags` and `sources` in the
  written `metadata.yml` against the draft. Putting `description` only in `METADATA`
  and not in `DESCRIPTION` is the way to fail that gate at the very last stage.
- Filtering: skip only rows with missing/unparseable SMILES or missing targets,
  preferring the `-1` sentinel. No dropping multi-component SMILES, no atom-count
  thresholds, no de-duplication.

### Smoke run

Set `__TESTING__ = True` and run:

```bash
cmmanage agent venv                              # once per session
.dataset_work/.venv/bin/python chem_mat_data/scripts/create_graph_datasets__<name>.py
```

**Three attempts.** On failure, read the traceback, fix, retry.
`references/troubleshooting.md` covers the common failures — check it before spending
an attempt. After the third failure, stop and escalate with the last error and what you
tried.

```bash
cmmanage agent set <slug> script_path chem_mat_data/scripts/create_graph_datasets__<name>.py
cmmanage agent set <slug> smoke_results_dir chem_mat_data/scripts/results/create_graph_datasets__<name>/debug
cmmanage agent gate script <slug>
```

Checks: the script is in `chem_mat_data/scripts/`, defines `load_dataset`, ends with
`run_if_main()`, uses a relative extend path, contains a download URL, does not
reference `.dataset_work`; and the smoke run produced a valid `.mpack` and
`metadata.yml` with a target count matching the approved metadata.

---

## Stage 5 — execute

Set `__TESTING__ = False` and run the full dataset. This can take hours; say so before
starting rather than going quiet.

```bash
.dataset_work/.venv/bin/python chem_mat_data/scripts/create_graph_datasets__<name>.py
cmmanage agent set <slug> results_dir chem_mat_data/scripts/results/create_graph_datasets__<name>/debug
cmmanage agent gate execute <slug>
```

Checks everything the final verify does, plus that `__TESTING__` is off, plus the
**drop rate**: how many of the `input_rows` you recorded in stage 1 survived
processing. Above 2% it warns; above 10% it escalates.

The drop rate matters because molecules that fail conversion are logged and skipped
rather than raising — a run can complete cleanly having discarded most of its input.
When it escalates, investigate before asking:

```bash
grep -c 'error processing' chem_mat_data/scripts/results/create_graph_datasets__<name>/debug/experiment_out.log
```

Usually the SMILES column is wrong, holds InChI, or needs unescaping. If the loss is
genuine and explained by the source — tmQM ships ~7% of rows with no SMILES at all —
record that rather than hiding it:

```bash
cmmanage agent set <slug> max_drop_rate 0.10
cmmanage agent set <slug> drop_rate_reason "7,700 of 108,000 source rows have no SMILES"
```

Raising the threshold is a judgement call about the data. If you cannot explain the
loss, do not raise it — escalate.

---

## Stage 6 — verification

**Run this as a fresh subagent too**, for the same reason as the license stage: the
agent that wrote the script is the worst-placed judge of whether it worked.

Brief the subagent with the slug, the results directory and the manifest, and have it
run `cmmanage agent gate verification <slug>` and independently sanity-check the
output — that the compound count is plausible against the paper's stated dataset size,
that the target values are in the range the paper reports, that the units in
`metadata.yml` match the source, and that a few SMILES from the raw file actually
appear in the processed CSV. Have it report findings, not reassurance.

The gate itself checks: every earlier stage passed; the results verify; **what landed
in `metadata.yml` matches what the metadata stage approved** (the script carries its
own `METADATA` literal, so the two can silently diverge); and the license conclusion
still holds.

Act on what the subagent reports. If it finds a real problem, fix it and re-run the
affected stage — which will mark the later stages stale, as it should.

### Finishing

Report: dataset name, compound count, target count and what they are, the drop rate,
the license and its attribution requirement, the script path, and the results
directory. Then tell the user that uploading is theirs to trigger via
`cmmanage metadata upload`.

---

## Reference files

Read on demand, not up front:

- `references/script-anatomy.md` — script template, hooks, parameters, extension hierarchy.
- `references/metadata.md` — `METADATA` spec, tag vocabulary, writing descriptions.
- `references/license.md` — the redistribution policy and how to research a license.
- `references/data-sources.md` — direct-download URL patterns per host.
- `references/troubleshooting.md` — common failures and fixes.
