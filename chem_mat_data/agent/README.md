# Agent helpers

This package no longer contains an agent. The workflow that turns a scientific
publication into a ChemMatData processing script now lives as a **Claude Code skill**
at [`.claude/skills/dataset-extraction/`](../../.claude/skills/dataset-extraction/),
where the instructions, the script template, the metadata specification and the four
worked examples are all editable as prose rather than as prompt strings in Python.

To use it, open Claude Code in this repository and invoke the skill:

```
/dataset-extraction https://www.nature.com/articles/sdata201686
```

The skill covers three stages - deciding whether a paper publishes usable data,
downloading the dataset files, and writing plus running plus verifying the processing
script. It stops at a verified local script; uploading to the remote share stays a
deliberate step (`cmmanage metadata upload`).

## What is left here

Helpers that should not depend on a model's judgement, all exposed through the
`cmmanage agent` command group:

| Module | Command | Purpose |
|---|---|---|
| `tools.py` | `cmmanage agent tools [--install]` | Report whether the external page readers are installed, and install the missing ones. |
| `environment.py` | `cmmanage agent venv` | Build a clean `uv` virtual environment with the project installed, so generated scripts run in isolation. |
| `verification.py` | `cmmanage agent verify <results-dir>` | Assert the hard post-conditions of a run: the `.mpack` and `metadata.yml` exist, the compound and target counts are sane, and every target index has a description. |

`verify` exits non-zero on failure and is what the skill treats as the definition of
success - a processing script can exit cleanly while still having produced an unusable
dataset, because molecules that fail conversion are logged and skipped rather than
raising.

## External page readers

Stage 1 of the skill reads publication pages with
[crawl4ai](https://github.com/unclecode/crawl4ai) (`crwl`), falling back to `defuddle`
and then `WebFetch`. `crwl` leads because it drives a real Chromium and so is the only
one of the three that gets past a publisher bot check - ChemRxiv 403s the other two.

Neither reader is a Python dependency, so nothing about installing this project brings
them along. Install and check them with:

```bash
cmmanage agent tools --install
```

Note that `crwl` needs its browser downloaded as a separate step (`crawl4ai-setup`);
without it, it fails on every page with a message that does not say so. The command
reports that case as the missing browser rather than as a missing `crwl`.
