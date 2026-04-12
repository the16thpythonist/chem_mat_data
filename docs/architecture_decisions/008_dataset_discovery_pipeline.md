# Dataset Discovery Pipeline

## Status

implemented

## Context

The ChemMatData package provides a growing catalog of chemistry and materials-science datasets for
graph neural network research. Two bottlenecks limit the expansion of this catalog (see also ADR 005):

1. **Finding** new datasets in the scientific literature.
2. **Processing** each dataset's raw files into the unified graph format.

ADR 005 addresses the second bottleneck with the ``cmmanage agent`` workflow: given a publication URL,
an LLM agent reads the paper, downloads the data, and generates a processing script. That workflow is
effective but expensive — each invocation browses a paper, visits repository links, and runs multiple
LLM reasoning steps. At the scale of the full OpenAlex index (hundreds of millions of works, tens of
thousands new per week in chemistry and materials science alone), running the agent on every paper is
neither practical nor economical.

What is needed is a cheap upstream filter that can scan the literature at scale and produce a short
list of papers *likely* to publish a reusable dataset. Only those flagged papers are then worth feeding
to the expensive agent pipeline for final verification.

The upstream [Dataset-Discovery-Pipeline](https://github.com/aimat-lab/Dataset-Discovery-Pipeline)
repository provides exactly this filter: it walks the OpenAlex API for chemistry and materials-science
papers and applies a fine-tuned TinyLlama-1.1B sequence classifier to the title and abstract of each
work, flagging those whose language suggests a dataset release. The classifier is a QLoRA adapter
(~13 MB) over the TinyLlama/TinyLlama-1.1B-Chat-v1.0 base model and was trained on a manually labeled
corpus of ~2,000 papers.

The question is how to integrate this filter into ChemMatData as a composable, offline-capable CLI
tool that fits the existing ``cmmanage`` architecture.

## Decision

### Two-Stage Filter Architecture

The discovery pipeline is positioned as a **cheap upstream filter** that feeds into the **expensive
downstream agent** (ADR 005). The pipeline intentionally over-includes (default threshold 0.05,
tuned for recall over precision on the upstream test set) because missing a genuine dataset release
is more costly than sending a few false positives to the agent step, which can reject them quickly.

The boundary between the two stages is a simple file: the discovery pipeline outputs a list of
flagged OpenAlex IDs and DOI URLs, and the agent's ``identify`` command takes a URL as input.
No shared state, no API coupling — just a file hand-off.

### CLI Surface: Three Command Shapes

The pipeline is exposed under ``cmmanage discover`` with five commands organized around two
usage patterns:

**Two-step pattern** (``fetch`` → ``score``): Separates data acquisition from classification.
``fetch`` writes a JSONL file of raw records; ``score`` reads that JSONL file, loads the classifier,
and writes both an ``_all`` and a ``_flagged`` output. This is useful when the user wants to inspect
the fetched data before scoring, re-score with a different threshold without re-fetching, or run the
two steps on different machines (e.g., fetch on a server with network access, score on a GPU node).

**Single-pass pattern** (``stream``): Interleaves fetching and classification in one command, writing
scored results to a CSV as they come in with live terminal progress. This is the convenient default for
most users who just want to run the pipeline end-to-end. The CSV columns (``openalex_id``, ``url``,
``score``, ``flagged``, ``publication_year``) are designed to be directly openable in a spreadsheet
for manual review.

**Utility commands** (``model-info``, ``model-download``): Diagnostic and setup helpers. ``model-info``
is read-only — it shows the cache location and dependency status without touching the filesystem.
``model-download`` pre-fetches the adapter archive, useful for offline environments or CI warmup.

All three fetch-capable commands (``fetch``, ``score``, ``stream``) share the same classifier options
(``--device``, ``--no-quantize``, ``--max-length``, ``--model-url``) so users can switch between the
two-step and single-pass workflows without learning different flags.

### Flat Sidecar Files for Resumable State

Pagination and de-duplication state is stored in two plain-text sidecar files next to the output:

- ``<output>.cursor`` — a single-line file containing the OpenAlex ``next_cursor`` token.
- ``<output>.ids`` — a newline-delimited set of OpenAlex IDs already written to the output.

This design was chosen over SQLite, JSON, or an in-memory-only approach for several reasons:

- **Inspectable.** A user can ``cat`` or ``wc -l`` the files to understand exactly where a run
  stopped and how many records were processed. When a run goes wrong, the state can be edited
  by hand (e.g., deleting the cursor file to restart pagination while keeping the seen-ID set).
- **Atomic.** The cursor file is rewritten after every batch; the IDs file is append-only with
  an immediate ``flush()`` after each entry. If the process is killed mid-batch, the worst case
  is re-fetching a single page — no corruption, no write-ahead log, no recovery procedure.
- **No dependencies.** No SQLite driver, no schema migrations, no connection management. The
  files are just strings.

The ``FetchState`` class encapsulates this logic and supports both one-shot usage (``load`` +
``update_cursor``) and streaming usage as a context manager with ``mark_seen`` flushing to disk
immediately.

### Model Distribution via Nextcloud

The classifier adapter is distributed as a ``tar.gz`` archive hosted on the same Nextcloud instance
that serves the dataset files. The URL is configured under ``[discovery] classifier_url`` in the
config file, following the same pattern as the dataset file share URL.

This was chosen over HuggingFace Hub hosting because:

- The existing Nextcloud infrastructure is already in place and trusted by the team.
- The adapter is a single static artifact (~13 MB) that changes rarely — it does not benefit
  from Hub features like model cards, versioning, or community contributions.
- Downloading from Nextcloud requires no authentication and no HuggingFace account.

The base model (TinyLlama-1.1B-Chat) is still fetched from HuggingFace Hub on first load, since
it is a public model maintained by the TinyLlama team and benefits from Hub caching. The
``ensure_classifier_model`` function downloads the archive to a ``.tar.gz.part`` temporary file
and renames it atomically on success, so a failed download never leaves a corrupt cache.

### TinyLlama-1.1B as the Classifier Backbone

The upstream pipeline chose TinyLlama-1.1B-Chat-v1.0 as the classifier backbone for a specific
set of trade-offs:

- **Small enough for CPU inference.** At 1.1B parameters the model runs on a laptop CPU in
  ~1-2 seconds per paper, which is acceptable for batch sizes of hundreds to low thousands.
  This makes the pipeline accessible without a GPU.
- **Large enough for language understanding.** The classifier needs to distinguish dataset-release
  language ("we release a benchmark of 50,000 molecules") from dataset-usage language ("we evaluate
  on the QM9 benchmark"). A bag-of-words or TF-IDF classifier struggles with this distinction;
  a language model handles it well.
- **QLoRA adapter.** The fine-tuned weights are a ~13 MB LoRA adapter, not a full model copy.
  This keeps the download small and the memory overhead low. On CUDA with bitsandbytes, 4-bit
  quantization further reduces memory to ~1 GB.

The threshold of 0.05 (positive-class probability) is deliberately low. At this threshold the
upstream test set achieved the best F1 score, trading many false positives for very few false
negatives. This is the right trade-off for a filter whose output feeds into a more expensive
verification step.

### CPU-First with Optional Quantization

The classifier loading logic follows a capability-detection pattern:

1. If CUDA is available and bitsandbytes is installed, load with 4-bit NF4 quantization.
2. If CUDA is available but bitsandbytes is not, load in float16.
3. Otherwise, load on CPU in float32.

The user can force a specific device with ``--device`` or disable quantization with
``--no-quantize``. This layered approach ensures the pipeline works everywhere — from a CI
runner with no GPU to a workstation with an A100 — without requiring the user to understand
quantization configuration.

### ML Dependencies in Dev Extras

The ML stack (``torch``, ``transformers``, ``peft``, ``safetensors``, ``accelerate``, and
optionally ``bitsandbytes``) is placed in the ``[dev]`` extras group rather than the core
dependencies. This keeps the base ``pip install chem_mat_database`` lightweight for users who
only need the dataset loading and graph processing functionality.

A runtime check (``is_ml_stack_available()``) gates the classifier commands: if the ML
dependencies are missing, the commands print a clear error message pointing to
``pip install chem_mat_database[dev]`` rather than crashing with an obscure import error. The
``model-info`` command always works regardless of ML availability, since its purpose is
precisely to diagnose what is and isn't installed.

### Noise Suppression

Loading a HuggingFace model for the first time produces a substantial amount of terminal noise:
download progress bars, ``LOAD REPORT`` diagnostics from the model class, deprecation warnings
from transformers internals, and unauthenticated-request warnings from the ``hf_xet`` Rust
extension. In a CLI tool this noise obscures the actual pipeline output.

The suppression strategy is layered:

1. **Environment variables.** ``HF_HUB_DISABLE_PROGRESS_BARS=1`` and
   ``TRANSFORMERS_VERBOSITY=error`` before any HuggingFace imports.
2. **Python logging.** ``huggingface_hub``, ``transformers``, ``accelerate``, and ``peft``
   loggers set to ``ERROR``.
3. **Warnings filter.** ``warnings.filterwarnings('ignore')`` for known deprecation categories.
4. **stdout/stderr redirection.** ``contextlib.redirect_stdout/stderr`` into a ``StringIO``
   buffer during model loading, surfaced only if loading fails.
5. **Offline mode.** When the base model is already in the HuggingFace cache,
   ``HF_HUB_OFFLINE=1`` and ``TRANSFORMERS_OFFLINE=1`` are set before loading. This avoids
   the ``hf_xet`` Rust extension's unauthenticated-request warning, which writes directly to
   file descriptor 2 and bypasses all Python-level redirection.

All suppression is disabled when the user passes ``--verbose``.

## Consequences

### Advantages

**Scalable filtering.** The pipeline can process thousands of papers per hour on a laptop CPU,
narrowing the full OpenAlex chemistry index down to a short list of plausible dataset releases.
This makes the expensive agent workflow (ADR 005) practical at scale — instead of running the
agent on every paper, it only runs on the few hundred flagged by the classifier.

**Composable workflow.** The two-step and single-pass patterns serve different use cases without
code duplication. The shared ``FetchState``, ``build_csv_row``, and classifier infrastructure
is reused across commands. The file-based hand-off to the agent pipeline requires no coupling
between the two systems.

**Offline-capable.** After the initial model download (``model-download``), the entire pipeline
runs without network access to HuggingFace. Only the OpenAlex API requires connectivity, and
even that is resumable across network interruptions.

**Accessible.** CPU-first design with no mandatory GPU dependency means the pipeline runs on
any developer machine, CI runner, or cloud instance. Optional quantization provides a performance
boost when hardware is available without making it a requirement.

**Inspectable state.** Plain-text sidecar files make it trivial to audit, debug, and manually
repair pipeline state. No database tooling, no binary formats, no recovery procedures.

### Disadvantages

**Base model download.** The TinyLlama-1.1B base model (~2 GB) is downloaded from HuggingFace
Hub on first use. This is a one-time cost cached by the Hub, but it can be surprising for users
who expected only the ~13 MB adapter download. The ``model-info`` command helps diagnose this,
and the offline mode detection avoids repeated download checks.

**Noise suppression fragility.** The layered suppression strategy works today but is coupled to
implementation details of HuggingFace libraries (logger names, warning categories, the ``hf_xet``
Rust extension's behavior). Library upgrades may introduce new noise sources that require
additional suppression layers. The ``--verbose`` flag provides an escape hatch when suppression
hides a genuine error.

**Threshold sensitivity.** The default threshold of 0.05 was tuned on the upstream training
corpus. On a different distribution of papers (e.g., a subfield with unusual terminology for
dataset releases) the recall/precision trade-off may shift. Users can adjust via ``--threshold``
but have no built-in way to evaluate the classifier on their own labeled data.

**Two command patterns.** Having both ``fetch``+``score`` and ``stream`` for the same underlying
operation adds surface area to the CLI. Users must understand when to use which. The help text
and group-level documentation mitigate this, but it is still more to learn than a single command.

**Dev extras scope.** Placing ML dependencies in ``[dev]`` alongside development tools (pytest,
ruff, etc.) means a user who only wants to run the discovery pipeline also installs testing
infrastructure. A dedicated ``[discovery]`` extras group would be more precise but would add
yet another extras key to maintain.
