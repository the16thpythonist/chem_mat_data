# Troubleshooting

You get three run attempts. Check here before spending one — most failures below are
recognizable from the traceback alone.

## Experiment / pycomex

**`FileNotFoundError` on the base template, or `Experiment.extend` fails**

The script is not in `chem_mat_data/scripts/`, so the relative
`'create_graph_datasets.py'` does not resolve. Move the script there rather than
switching to an absolute path — an absolute path breaks for every other user of the
repo.

**Parameters seem ignored — the run uses template defaults**

`Experiment.extend(..., glob=globals())` must come *after* the module-level parameter
assignments. Declare `DATASET_NAME`, `METADATA`, etc. at the top, then call `extend`.

**A hook does not take effect**

In a sub-experiment always use `@experiment.hook('name', default=False, replace=True)`.
`default=True, replace=False` registers a fallback that an existing implementation
wins over.

**Results land somewhere unexpected**

`__DEBUG__ = True` (the template default) writes to
`results/<script_name>/debug/` and overwrites the previous run. That is intended —
verify against that path.

## Processing

**Run "succeeds" but `compounds` is far below the input row count**

Workers drop molecules that throw, printing `! error processing ...` to the log and
continuing. Check `experiment_out.log`:

```bash
grep -c 'error processing' chem_mat_data/scripts/results/create_graph_datasets__<name>/debug/experiment_out.log
```

Usually the SMILES column is wrong, contains InChI rather than SMILES, or the values
need unescaping. Investigate rather than accepting the loss.

**`KeyError: 'targets'` or `'smiles'`**

Every value dict from `load_dataset` needs both keys. `data['smiles'] = data['smiles']`
is a no-op that appears in older scripts — it only works when the column is already
literally named `smiles`. Assign from the actual column.

**Target values arrive as strings**

`csv.DictReader` yields strings, and `pd.read_csv` will too for a column with any
non-numeric entry. Cast explicitly and handle the sentinel:

```python
def _to_float(value) -> float:
    if value is None or value == '' or pd.isna(value):
        return -1.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return -1.0
```

**Ragged target vectors**

Every record needs the same `len(targets)`; the metadata's `targets` count comes from
the first graph alone, so a mismatch corrupts the dataset silently. Pad with `-1.0`.

**RDKit floods the console**

```python
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
```

**Hangs, or memory blows up**

The template starts one worker per CPU before loading data, deliberately. If you moved
worker startup after the load, you are copying the whole dataset into every child
process — put it back. For a genuinely large dataset, run with `__TESTING__ = True`
first to confirm the pipeline before committing to the full pass.

## Downloads

**Parse error on a file that downloaded "fine"**

You saved an HTML landing page. `file <path>` and `head -c 300 <path>` will show it.
See `data-sources.md` for the direct-URL pattern of that host.

**SSL / certificate errors**

Common on Figshare. Use `FileDownloadSource(url, verbose=True, ssl_verify=False)` from
`chem_mat_data.connectors`.

**403 / bot blocking**

```bash
curl -L -A 'Mozilla/5.0' -o <path> '<url>'
```

If that fails, in this order:

1. **Europe PMC**, for anything with a PMC ID — it serves publisher supplementary files
   as a plain ZIP with no bot check. This is the single most useful escape hatch and it
   resolves most publisher 403s. See `data-sources.md`.
2. A **published version** if you started from a preprint, or a **preprint** if you
   started from a paywalled article. Note that the two may carry different licenses, so
   re-check `license.md` if you switch.
3. Only then, report the URL for manual download rather than burning attempts.

Do not reach for a Playwright click-through to get past a block. It may fetch the file
once, but the generated script has to download the data unaided — a route that needs a
scripted browser is not a route the script can use.

**All three page readers failing at once**

`defuddle: command not found`, `crwl` dying with `Executable doesn't exist at
.../chromium-*`, and `WebFetch` returning 403 is a real combination, not bad luck. One
command diagnoses and fixes it:

```bash
cmmanage agent tools --install
```

It reports each reader separately and distinguishes a missing `crwl` from a `crwl`
whose browser was never downloaded — the second is by far the more common and its own
error message does not say which command fixes it.

If `defuddle` was working yesterday and is `command not found` today, check whether the
active Node version changed: `npm install -g` installs into the *current* nvm Node, so
`nvm use` of another version hides it. Reinstall, or point a shim at an absolute path.

**`defuddle` returns 403 but the page loads in a browser**

Expected on a bot-checking host — ChemRxiv and most publishers. `defuddle` and
`WebFetch` are both plain HTTP clients and fail together. This is a property of the
host, not a broken install; do not retry `defuddle` with different flags.

It also means you started from the wrong reader. `crwl` is the default precisely
because it does not have this failure mode:

```bash
crwl '<url>' -o markdown-fit
```

`defuddle` is worth reaching for only when `crwl` is unavailable, or when a page came
back badly rendered and a cheap second opinion helps. On a publisher host, go straight
to `crwl`.

**Excel file parses into nonsense**

SI spreadsheets often carry title rows and notes above the table, and multiple sheets.
Inspect first:

```python
xl = pd.ExcelFile(path)
print(xl.sheet_names)
print(pd.read_excel(path, sheet_name=0, nrows=15, header=None))
```

Then use `sheet_name=` and `skiprows=` accordingly.

## Verification

**`cmmanage agent verify` reports missing `target_descriptions`**

Keys must cover `0 .. len(targets)-1` as strings — `'0'`, `'1'`, … See `metadata.md`.

**verify reports a missing `.mpack`**

The run did not reach the save step. Read `experiment_out.log` from the bottom; the
real error is usually an exception during processing that the `finally` block masked
behind worker-shutdown noise.

**Everything passes but the count is 50**

`__TESTING__` is still `True`. Set it to `False` and do the real run.

## Environment

**`ModuleNotFoundError` for `chem_mat_data` in the run venv**

Recreate it — `cmmanage agent venv` builds a fresh venv and installs the project
editable with dev extras.

**`cmmanage: command not found`**

`source .venv/bin/activate` first.
