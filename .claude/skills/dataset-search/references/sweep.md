# The sweep

What the first stage actually does, what its defaults mean, and what to do when it
misbehaves. Read this before changing any flag.

## What the classifier is

A **TinyLlama-1.1B-Chat** sequence classifier with a fine-tuned QLoRA adapter
(~13 MB), trained upstream at
[aimat-lab/Dataset-Discovery-Pipeline](https://github.com/aimat-lab/Dataset-Discovery-Pipeline)
on ~2,000 manually labelled chemistry and materials papers. The labelled corpus
lives in this repository at `scripts/discovery_{positives,negatives,training_data,test_set}.jsonl`.

Every paper is rendered into one fixed prompt:

```
Classify whether the following scientific paper introduces a dataset
of molecules or materials that is released for reuse.

Title: {title}

Abstract: {abstract}

Answer:
```

One forward pass gives two logits; a softmax over them gives the positive-class
probability written to the `score` column.

The reason a language model is used at all rather than keyword matching: the
distinction that matters is between *releasing* a dataset ("we release a benchmark
of 50,000 molecules") and *using* one ("we evaluate on the QM9 benchmark"). Those
two sentences share most of their vocabulary. A bag-of-words classifier cannot
separate them; a 1.1B language model can.

## What gets fetched

`discover stream` walks the OpenAlex `/works` endpoint with cursor pagination,
sorted newest-first, filtered to:

- **Concepts** — chemistry (`C185592680`) and materials science (`C192562407`).
- **Types** — `journal-article`, `proceedings-article`, `report`. Book chapters
  and review articles are excluded because they almost never release reusable
  data and dominated the false positives when the stream was last validated.
- **Year** — `--year-min`, default 2010.

Two further filters run before the classifier sees anything: papers with no title
or no abstract are skipped, and a blacklist of DOI prefixes drops publishers known
to produce only spam, placeholder records or content outside chemistry entirely
(ResearchHub, several Turkish and Brazilian book-chapter publishers). Both are in
`chem_mat_data/discovery/openalex.py` and both mark the record as seen, so a
skipped paper is never re-examined.

## The threshold

`--threshold` defaults to **0.05** — a positive-class probability of five percent
is enough to flag. That is deliberately, aggressively low: it is the point of best
F1 on the upstream test set, and it is chosen for **recall over precision**.

The consequence is that most flagged papers are not dataset releases. In the
sweep currently on disk, 115 of 1000 records were flagged, and the flag rate you
should expect is around 10–12%. That is the intended trade: a false positive costs
one subagent, a false negative loses a dataset permanently.

**Do not raise the threshold to reduce the verification workload.** Reduce
`--limit` instead. Raising the threshold discards exactly the marginal papers the
low threshold exists to catch — genuine releases do score below 0.2. If you want
fewer papers, look at fewer papers.

## The shared state

Three files at the repository root, and they belong together:

| File | Holds |
|---|---|
| `discovery_stream.csv` | every scored record, appended across runs |
| `discovery_stream.csv.cursor` | the OpenAlex `next_cursor` token, rewritten each batch |
| `discovery_stream.csv.ids` | every OpenAlex ID already written, append-only |

They are plain text on purpose — `wc -l` tells you how far a run got, and a broken
run can be repaired with an editor. The cursor is rewritten after every batch and
the IDs file is flushed per record, so killing the process mid-batch costs at most
one re-fetched page.

**Leave the output at the default path.** Redirecting `--out` into a run directory
gives that run its own cursor, which means it restarts pagination at the newest
papers and re-scores everything the previous run already saw. The shared ledger is
the only thing making successive sweeps cumulative.

**The column set must match the file you are appending to.** Resuming appends rows
without rewriting the header, so adding `--include-title` to a ledger written
without it would put six-field rows under a five-field header. `pandas.read_csv`
then raises `ParserError` and a spreadsheet mis-aligns every later column — and
because the ledger is cumulative, that damage is permanent. `stream` refuses the
run rather than doing it, and names the flag responsible. Take the refusal at face
value: drop the flag, or start a separate `--out`. Do not reach for `--no-resume`,
which fixes the header by discarding the de-duplication set.

`--no-resume` ignores the sidecars and starts over. It is almost never what you
want.

## When the sweep returns nothing new

The fetcher stops after 20 consecutive pages in which every record was already
seen — at 200 records per page, that is 4,000 already-seen records crossed before
it concludes the tail is exhausted. Hitting that is the normal end state of a
ledger that has caught up with the index.

What to do, in order:

1. **Wait.** OpenAlex indexes new chemistry and materials papers continuously;
   sorted newest-first, a caught-up ledger genuinely has nothing until more are
   published.
2. **Widen the window** — `--year-min 2005` opens older papers the current ledger
   has never paginated into.
3. **Delete only the cursor**, keeping `.ids`. This restarts pagination from the
   newest paper while preserving the de-duplication set, so nothing is
   re-classified but the whole ordering is walked again.

**The CLI's own hint contradicts this file, and this file wins.** When a sweep
returns nothing, `discover stream` suggests `--no-resume` and deleting the
`.cursor`/`.ids` sidecars. That advice is written for a single throwaway output
file, not for the shared cumulative ledger the skill uses. Follow the order above
instead.

**Never delete `.ids` to force new results.** It is the memory of everything ever
seen; deleting it makes the next sweep re-score thousands of records it has
already judged, and re-flag papers already in the verdict ledger.

## Device and speed

The loader detects capability in three steps: CUDA plus `bitsandbytes` gives 4-bit
NF4 quantization (~1 GB of VRAM); CUDA alone gives float16; otherwise CPU float32.
Override with `--device` or disable quantization with `--no-quantize`.

Budget roughly 1–2 s per paper on CPU and well under that on a GPU. A 300-paper
sweep is minutes either way; the verification stage, not the sweep, is what costs.

If the model fails to load, re-run with `--verbose`. Loading output is suppressed
by default through five layers (environment variables, logger levels, warning
filters, stdout/stderr redirection, and offline mode when the base model is
cached), and a genuine error can be hidden along with the noise. `--verbose`
disables all of it.

## Two-step alternative

`cmmanage discover fetch` writes a JSONL of raw records, and
`cmmanage discover score` classifies it.
Use them instead of `cmmanage discover stream` when you want to inspect what was fetched before
spending time scoring, re-score at a different threshold without re-fetching, or
fetch on one machine and score on another. They take the same classifier flags, so
nothing has to be relearned.
