import os
import sys
import json
import yaml
import time
import difflib
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, Optional

import rich_click as click
from prettytable import PrettyTable, TableStyle
from rich.style import Style
from rich.text import Text
from rich.pretty import pprint
from rich.console import Console, ConsoleOptions
from rich.panel import Panel
from rich.syntax import Syntax
from pycomex.functional.experiment import Experiment
from pycomex.utils import dynamic_import

from chem_mat_data.config import Config
from chem_mat_data.web import NextcloudFileShare
from chem_mat_data.utils import get_version
from chem_mat_data.utils import PATH
from chem_mat_data.utils import DOCS_PATH
from chem_mat_data.utils import PAGES_PATH
from chem_mat_data.utils import PROJECT_PATH
from chem_mat_data.utils import METADATA_PATH
from chem_mat_data.utils import TEMPLATE_ENV
from chem_mat_data.agent.manifest import STAGES
from chem_mat_data.utils import CsvListType
from chem_mat_data.utils import open_file_in_editor
from chem_mat_data.utils import RichMixin

# Re-exports of the discovery helpers, aliased so that tests can patch them
# via ``chem_mat_data.manage.<name>`` without importing the submodule.
from chem_mat_data.discovery.openalex import (
    fetch_works as discovery_openalex_fetch_works,
)
from chem_mat_data.discovery.io import (
    FetchState as DiscoveryFetchState,
    iter_jsonl as discovery_iter_jsonl,
    write_jsonl_line as discovery_write_jsonl_line,
    build_csv_row as discovery_build_csv_row,
    STREAM_CSV_BASE_COLUMNS as DISCOVERY_STREAM_CSV_BASE_COLUMNS,
)
from chem_mat_data.discovery.classifier import (
    ensure_classifier_model as discovery_ensure_classifier_model,
    load_classifier as discovery_load_classifier,
    is_ml_stack_available as discovery_is_ml_stack_available,
)

# The path to the "scripts" folder of the experiment modules
SCRIPTS_PATH: str = os.path.join(PATH, 'scripts')
# The path to the "results" folder of the experiment module executions.
RESULTS_PATH: str = os.path.join(PATH, 'scripts', 'results')


class RichDiffDisplay(RichMixin):
    """
    Rich display element that shows the diff between two metadata.yml files.
    """

    def __init__(
        self,
        local_file: str,
        remote_file: str,
        diff_lines: List[str],
        changed_lines: int,
    ):
        self.local_file = local_file
        self.remote_file = remote_file
        self.diff_lines = diff_lines
        self.changed_lines = changed_lines

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> Any:

        yield ''

        # Show file paths being compared
        from rich.table import Table

        comparison_table = Table(box=None)
        comparison_table.add_column('File', style='magenta', no_wrap=True)
        comparison_table.add_column('Path', style='cyan', no_wrap=False)
        comparison_table.add_row('Local', self.local_file)
        comparison_table.add_row('Remote', 'metadata.yml (from server)')

        yield comparison_table

        # Show the actual diff if there are changes
        if self.changed_lines > 0 and self.diff_lines:
            yield ''

            # Limit diff output to first 50 lines to avoid overwhelming display
            display_lines = self.diff_lines[:50]
            if len(self.diff_lines) > 50:
                display_lines.append(
                    f'... ({len(self.diff_lines) - 50} more lines truncated)'
                )

            # Create a syntax-highlighted diff
            diff_text = '\n'.join(display_lines)
            syntax = Syntax(diff_text, 'diff', theme='monokai', line_numbers=False)

            panel = Panel(
                syntax,
                title=f'File Differences (showing first {min(50, len(self.diff_lines))} lines)',
                expand=True,
            )

            yield panel

        # Show diff summary
        yield ''
        if self.changed_lines == 0:
            yield Text('✅ Files are identical', style='green')
        else:
            yield Text(
                f'📊 {self.changed_lines} lines differ between local and remote files',
                style='yellow',
            )


CODE_STYLE = Style(color='grey66', bgcolor='grey19')


def render_help(text: str) -> Text:
    """
    Convert a help string with backtick code spans and rich markup into a styled
    Rich ``Text`` object.

    Backtick-delimited spans (e.g. \`metadata.yml\`) are rendered as pills with a
    subtle dark background. Rich markup like ``[bold]...[/bold]`` in the remaining
    text is preserved.

    :param text: The help text string, optionally containing \`code\` spans and
        ``[bold]`` rich markup.

    :returns: A Rich ``Text`` object ready for console rendering.
    """
    import re

    result = Text()
    parts = re.split(r'`([^`]+)`', text)
    for i, part in enumerate(parts):
        if i % 2 == 0:
            result.append_text(Text.from_markup(part))
        else:
            result.append(f' {part} ', style=CODE_STYLE)

    return result


class RichHelpCommand(click.RichCommand):
    """
    A custom command class that uses a Rich widget instead of the default help text
    rendering. This avoids the issue where rich_click dims paragraphs after the first one.

    Pass a ``rich_help`` widget (any Rich renderable) via the ``cls`` and ``rich_help``
    keyword arguments of the ``@click.command`` decorator. The widget is rendered between
    the usage line and the options panel, constrained to ``help_max_width`` characters.
    """

    def __init__(self, *args, rich_help=None, help_max_width: int = 120, **kwargs):
        super().__init__(*args, **kwargs)
        self.rich_help_widget = rich_help
        self.help_max_width = help_max_width

    def format_help(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        self.format_usage(ctx, formatter)

        if self.rich_help_widget:
            help_width = min(Console().width, self.help_max_width)
            help_console = Console(width=help_width)
            help_console.print(self.rich_help_widget)

        self.format_options(ctx, formatter)
        self.format_epilog(ctx, formatter)


class RichHelpGroup(click.RichGroup):
    """
    A custom group class that uses a Rich widget for help text rendering,
    analogous to :class:`RichHelpCommand` but for command groups.

    The widget is rendered between the usage line and the options/commands
    panels. Commands are listed in registration order (not alphabetical).
    """

    def __init__(self, *args, rich_help=None, help_max_width: int = 120, **kwargs):
        super().__init__(*args, **kwargs)
        self.rich_help_widget = rich_help
        self.help_max_width = help_max_width

    def format_help(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        self.format_usage(ctx, formatter)

        if self.rich_help_widget:
            help_width = min(Console().width, self.help_max_width)
            help_console = Console(width=help_width)
            help_console.print(self.rich_help_widget)

        self.format_options(ctx, formatter)

    def list_commands(self, ctx: click.Context) -> List[str]:
        """Return commands in registration order, not alphabetical."""
        return list(self.commands.keys())


class RichDocsCompileHelp(RichMixin):
    """
    Rich display element that renders the help text for the ``docs compile`` command.
    """

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> Any:

        yield render_help(
            'Fetch dataset metadata from the remote file share server and compile it into a '
            'JSON file that powers the interactive dataset explorer on the landing page.\n'
        )

        yield render_help(
            'The command connects to the Nextcloud file share, downloads the current '
            '`metadata.yml`, and transforms each dataset entry into a structured JSON object '
            'containing the dataset name, category, number of compounds, target types, tags, '
            'and source references. Internal datasets (names starting with `_`) are excluded '
            'automatically.\n'
        )

        yield render_help(
            'The resulting `datasets.json` file is written into the `pages/` directory so that '
            'the static landing page can load it at runtime. It also embeds a download base '
            'URL so the explorer can link directly to the raw dataset files on the server.\n'
        )

        yield render_help(
            'Run this command before `cmmanage docs deploy` whenever the dataset metadata on '
            'the server has changed, so that the landing page reflects the latest state.\n'
        )


class RichManageHelp(RichMixin):
    """
    Rich display element that renders the extended help text for the cmmanage CLI.

    The help text consists of an introductory paragraph about ChemMatData, a paragraph
    explaining the cmmanage interface, and one paragraph for each command category
    (Metadata, Dataset, Docs, Agent) with the category name bolded.
    """

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> Any:

        yield render_help(
            'ChemMatData is a Python package that provides unified access to chemistry and material science '
            'datasets for machine learning applications. It is specifically designed for training graph neural '
            'networks (GNNs) by offering both raw tabular data (CSV/pandas) and processed graph representations. '
            'The package connects to a remote Nextcloud file share server to distribute datasets and uses a '
            'local caching system to avoid redundant downloads.\n'
        )

        yield render_help(
            'The `cmmanage` command line interface is the internal management tool for the ChemMatData package. '
            'Unlike the `cmdata` command, which is intended for end users to browse and download datasets, `cmmanage` '
            'is aimed at developers and maintainers of the package and the dataset collection. It exposes commands '
            'for maintaining metadata, creating and uploading datasets, managing documentation, and running an '
            'LLM-based dataset discovery agent. The commands are organized into the following categories:\n'
        )

        yield render_help(
            '[bold]Metadata[/bold] commands manage the `metadata.yml` file that describes all datasets in the collection. '
            'This file is the central registry that maps dataset names to their properties such as the number of '
            'elements, target labels, and dataset category. The metadata commands allow you to collect metadata from '
            'individual experiment results, upload the compiled file to the remote server, edit it locally, compare '
            'local and remote versions with a diff view, and remove individual dataset entries.\n'
        )

        yield render_help(
            '[bold]Dataset[/bold] commands handle the creation and distribution of dataset files. Datasets are '
            'produced by running experiment modules (Python scripts) that fetch raw data, process molecular or '
            'crystal structures into the graph format, and package the results. The dataset commands let you '
            'trigger these experiment scripts to create new datasets locally and then upload the resulting files '
            'to the remote file share server for distribution.\n'
        )

        yield render_help(
            '[bold]Docs[/bold] commands assist with building and deploying the package documentation. This includes '
            'compiling dataset metadata into structured formats for the documentation site and the landing page, '
            'as well as deploying the full documentation to GitHub Pages. These commands ensure that the public '
            'documentation stays in sync with the current state of the dataset collection and makes it easy to '
            'preview changes locally before publishing.\n'
        )

        yield render_help(
            '[bold]Agent[/bold] commands are deterministic helpers supporting the `dataset-extraction` '
            'skill, which turns a scientific publication into a working processing script. The skill itself lives '
            'in `.claude/skills/dataset-extraction/` and is invoked from Claude Code; these commands provide the '
            'two pieces that should not be left to a model - building a clean virtual environment in which to run '
            'a generated script, and verifying that a finished run actually produced a valid dataset.\n'
        )


class RichAgentGroupHelp(RichMixin):
    """
    Rich display element that renders the help text for the ``agent`` command group.
    """

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> Any:

        yield render_help(
            'The `agent` commands are deterministic helpers that support the '
            '`dataset-extraction` skill. The skill itself - the instructions for finding '
            'a dataset in a publication, downloading it, and writing a processing script '
            '- lives in `.claude/skills/dataset-extraction/` and is invoked from Claude '
            'Code rather than from this CLI.\n'
        )

        yield render_help(
            'Three helpers remain here because they should not depend on a model\'s '
            'judgement. `cmmanage agent tools` reports whether the external page readers '
            'the prepare stage needs are installed, and installs them with `--install`. '
            '`cmmanage agent venv` builds a clean virtual environment with the '
            'project installed, so that generated scripts run in isolation from the '
            'development environment. `cmmanage agent verify` checks a finished results '
            'folder against the hard post-conditions: that a message pack dataset and a '
            '`metadata.yml` exist, that the compound and target counts are sane, and that '
            'every target index carries a description.\n'
        )

        yield render_help(
            'A processing run can exit cleanly and still produce unusable output, because '
            'molecules that fail conversion are logged and skipped rather than raising. '
            'Treat a clean `verify` as the definition of success, not the absence of a '
            'traceback.\n'
        )


class RichAgentVenvHelp(RichMixin):
    """
    Rich display element that renders the help text for the ``agent venv`` command.
    """

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> Any:

        yield render_help(
            'Create a clean virtual environment for running generated dataset processing '
            'scripts. The environment is built with `uv` and the project is installed into '
            'it in editable mode together with its development dependencies.\n'
        )

        yield render_help(
            'The venv is created at `.dataset_work/.venv` by default and is recreated from '
            'scratch on every invocation, so a broken environment is always one command '
            'away from being fixed. The path of the resulting Python interpreter is printed '
            'on the last line of output so that it can be captured by a calling script.\n'
        )


class RichAgentToolsHelp(RichMixin):
    """
    Rich display element that renders the help text for the ``agent tools`` command.
    """

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> Any:

        yield render_help(
            'Report whether the external page-reading tools the prepare stage depends on '
            'are installed, and install the missing ones with `--install`. Checking is the '
            'default because both installs are slow and reach the network.\n'
        )

        yield render_help(
            '`crwl` is the default reader. It drives a real Chromium, which makes it the '
            'only one that gets past a publisher bot check - ChemRxiv returns 403 to '
            '`defuddle` and to `WebFetch` while serving `crwl` the full article. '
            '`defuddle` is the fallback: about three seconds quicker per page because it '
            'starts no browser, but it fails outright on any host that bot-checks.\n'
        )

        yield render_help(
            'The ordering is about asymmetric cost, not about which tool is better. '
            'Reaching for `crwl` where `defuddle` would have done wastes a few seconds; '
            'reaching for `defuddle` on a defended host costs a 403, a diagnosis and a '
            'retry. A default that always works beats one that is usually faster.\n'
        )

        yield render_help(
            'A `crwl` without a downloaded Chromium fails on every page with an error that '
            'does not name the fix, so the browser is checked separately and reported as '
            'the missing piece it is.\n'
        )


class RichAgentInitHelp(RichMixin):
    """
    Rich display element that renders the help text for the ``agent init`` command.
    """

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> Any:

        yield render_help(
            'Start a new staged extraction run. Creates `.dataset_work/<slug>/` with a '
            '`manifest.yml` recording every stage as pending, and an empty `downloads/` '
            'folder for the prepare stage to fill.\n'
        )

        yield render_help(
            'The slug is a short identifier for the run and is usually the eventual '
            'dataset name. The `--url` should be the publication URL, or a concrete '
            'enough paper title that it can be found by search; deciding whether a '
            'paper is worth extracting at all happens before this point.\n'
        )


class RichAgentStatusHelp(RichMixin):
    """
    Rich display element that renders the help text for the ``agent status`` command.
    """

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> Any:

        yield render_help(
            'Show where an extraction run stands: the status of each of the six stages, '
            'any recorded blockers, and which stage comes next.\n'
        )

        yield render_help(
            'A stage is `pending` before it runs, `passed` once its gate is clean, '
            '`blocked` when it hit something needing a human, and `stale` when an '
            'earlier stage has since been re-run and its result can no longer be '
            'trusted. This is the first thing to run when picking up a run in a new '
            'session.\n'
        )


class RichAgentGateHelp(RichMixin):
    """
    Rich display element that renders the help text for the ``agent gate`` command.
    """

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> Any:

        yield render_help(
            'Check a stage against its post-conditions and record the outcome in the '
            'manifest. A stage whose predecessors have not passed fails immediately, '
            'because checking a script against metadata that was never agreed produces '
            'a meaningless verdict.\n'
        )

        yield render_help(
            'The gates check concrete things rather than asking whether the stage went '
            'well: whether a downloaded file is secretly an HTML landing page, whether '
            '`Experiment.extend` was handed an absolute path, whether the script reaches '
            'into the scratch directory instead of downloading its own data, and how '
            'many input rows survived the full run.\n'
        )

        yield render_help(
            'Exits non-zero when a gate fails, and marks the stage `blocked` with the '
            'first error as the blocker. Use `--no-record` to check without writing to '
            'the manifest, and `--strict` to fail on warnings too.\n'
        )


class RichAgentSetHelp(RichMixin):
    """
    Rich display element that renders the help text for the ``agent set`` command.
    """

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> Any:

        yield render_help(
            'Record a value into the run manifest. This is how each stage hands its '
            'conclusions to the next one and to the gates.\n'
        )

        yield render_help(
            'Scalar keys take a plain value: `record_url`, `dataset_name`, '
            '`script_path`, `smoke_results_dir`, `results_dir`, `input_rows`, '
            '`max_drop_rate`, `drop_rate_reason`. The structured keys `metadata_draft` '
            'and `license` take a JSON object, so a whole finding can be written in one '
            'call.\n'
        )

        yield render_help(
            '`source_url` is not settable. It is written once by `agent init --url` and '
            'records where the run began, which is what lets the license gate ask '
            'whether the file you downloaded is covered by the license you read. Record '
            'the data repository landing page as `record_url` instead.\n'
        )


class RichAgentVerifyHelp(RichMixin):
    """
    Rich display element that renders the help text for the ``agent verify`` command.
    """

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> Any:

        yield render_help(
            'Verify that a dataset processing run produced complete and usable output. '
            'Point this at the results folder of a `create_graph_datasets__<name>.py` run, '
            'which for a default `__DEBUG__` run is '
            '`chem_mat_data/scripts/results/create_graph_datasets__<name>/debug`.\n'
        )

        yield render_help(
            'Errors mean the results are not usable: a missing or empty `.mpack` file, a '
            'missing or unparseable `metadata.yml`, a zero compound count, or a '
            '`target_descriptions` mapping that does not cover every target index. '
            'Warnings flag results that are valid but incomplete, such as missing tags, '
            'sources, or a compound count low enough to suggest that `__TESTING__` was '
            'left enabled.\n'
        )

        yield render_help(
            'The command exits with a non-zero status when any error is found, so it can '
            'gate a workflow. Use `--strict` to have warnings fail the check as well.\n'
        )


class RichDiscoveryGroupHelp(RichMixin):
    """
    Rich display element that renders the help text for the ``discover`` command group.
    """

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> Any:

        yield render_help(
            'The discovery system is the cheap upstream filter of the dataset-discovery '
            'pipeline. It walks the OpenAlex index of chemistry and materials-science '
            'papers and applies a fine-tuned TinyLlama-1.1B classifier to the '
            '`title + abstract` of each work to flag those likely to release a reusable '
            'molecular or materials dataset.\n'
        )

        yield render_help(
            'The typical workflow is two steps. First, `cmmanage discover fetch` queries '
            'the OpenAlex `/works` endpoint with cursor pagination, reconstructs each '
            'abstract from its inverted-index form, and streams the results into a JSONL '
            'file (with a sibling `.cursor` and `.ids` file so the run can be '
            'resumed or extended later). Second, `cmmanage discover score` loads the '
            'classifier adapter from the user config directory and writes both an '
            '`_all` and an `_flagged` JSONL — the flagged file is the short list '
            'worth feeding to `cmmanage agent identify` for final LLM verification.\n'
        )

        yield render_help(
            'Use `cmmanage discover stream` to interleave the fetch and the classifier '
            'in a single pass, writing scored results to a CSV as they come in. '
            'Use `cmmanage discover model-download` to pre-fetch the classifier archive '
            'to a fresh machine (useful for offline or CI warmup), and '
            '`cmmanage discover model-info` to inspect where the model is cached and '
            'whether the required ML dependencies are installed.\n'
        )

        yield render_help(
            'The classifier is a ~13 MB LoRA adapter over TinyLlama/TinyLlama-1.1B-Chat-v1.0 '
            'and is downloaded once from the URL configured under `[discovery] '
            'classifier_url` in the config file. The ML stack (torch, transformers, peft, '
            'optionally bitsandbytes) ships with the `dev` extras: install via '
            '`pip install chem_mat_database[dev]`.\n'
        )


class RichDiscoveryModelInfoHelp(RichMixin):
    """
    Rich display element that renders the help text for the ``discover model-info`` command.
    """

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> Any:

        yield render_help(
            'Show where the TinyLlama LoRA adapter is cached on disk, whether the '
            'adapter files are present, and whether the required ML dependencies '
            '(torch, transformers, peft, optionally bitsandbytes) are importable.\n'
        )

        yield render_help(
            'This is a read-only diagnostic command — it never downloads anything '
            'or modifies the filesystem. Use it to verify that a previous '
            '`cmmanage discover model-download` succeeded or to debug import errors '
            'when the ML stack is not fully installed.\n'
        )


class RichDiscoveryModelDownloadHelp(RichMixin):
    """
    Rich display element that renders the help text for the ``discover model-download``
    command.
    """

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> Any:

        yield render_help(
            'Download the TinyLlama LoRA adapter archive from the Nextcloud URL '
            'configured under `[discovery] classifier_url` in the config file and '
            'extract it into the user config directory. The archive is a ~13 MB '
            '`tar.gz` containing the adapter weights, tokenizer, and configuration.\n'
        )

        yield render_help(
            'Subsequent runs of `cmmanage discover score` or '
            '`cmmanage discover stream` will use the cached model automatically, '
            'so this command only needs to run once per machine (or after clearing '
            'the cache). Use `--force` to re-download even when the adapter is '
            'already present.\n'
        )

        yield render_help(
            'Use `--model-url` to override the configured URL, for example to test '
            'a different adapter version without editing the config file.\n'
        )


class RichDiscoveryFetchHelp(RichMixin):
    """
    Rich display element that renders the help text for the ``discover fetch`` command.
    """

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> Any:

        yield render_help(
            'Query the OpenAlex `/works` endpoint for chemistry and materials-science '
            'papers, reconstruct each abstract from its inverted-index representation, '
            'and stream the results into a JSONL file. Each line contains the fields '
            '`openalex_id`, `doi`, `title`, `abstract`, and `publication_year`.\n'
        )

        yield render_help(
            'Pagination state is persisted in sibling `.cursor` and `.ids` files next '
            'to the output, so a run can be interrupted and resumed exactly where it '
            'left off. The `.ids` file also de-duplicates across runs — records '
            'already fetched in a previous invocation are skipped automatically. Use '
            '`--reset` to discard any existing state and start from scratch.\n'
        )

        yield render_help(
            'The output JSONL is the input expected by `cmmanage discover score`. For '
            'a combined fetch-and-score workflow in a single pass, use '
            '`cmmanage discover stream` instead.\n'
        )


class RichDiscoveryScoreHelp(RichMixin):
    """
    Rich display element that renders the help text for the ``discover score`` command.
    """

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> Any:

        yield render_help(
            'Read an input JSONL file (typically produced by `cmmanage discover fetch`), '
            'load the TinyLlama classifier adapter, and score each record\'s '
            '`title + abstract` for the likelihood of describing a reusable dataset '
            'release. Two output JSONL files are written: `--out-all` contains every '
            'record with an added `prediction` field, and `--out-flagged` contains only '
            'the records whose score exceeds the threshold.\n'
        )

        yield render_help(
            'The flagged file is the short list worth feeding to '
            '`cmmanage agent identify` for final LLM verification. The default '
            'threshold of 0.05 matched the best test-set performance in the upstream '
            'Dataset-Discovery-Pipeline and tends to over-include rather than miss '
            'genuine hits — this is intentional, as the downstream agent step is the '
            'precision filter.\n'
        )

        yield render_help(
            'The classifier runs on CPU by default and uses 4-bit quantization when '
            'CUDA and bitsandbytes are available. Use `--device` to force a specific '
            'device and `--no-quantize` to disable quantization.\n'
        )


class RichDiscoveryStreamHelp(RichMixin):
    """
    Rich display element that renders the help text for the ``discover stream`` command.
    """

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> Any:

        yield render_help(
            'Combine `fetch` and `score` into a single streaming pass: each paper is '
            'fetched from OpenAlex, immediately classified, and written to a CSV output '
            'file. This avoids the intermediate JSONL step and provides live terminal '
            'feedback as papers are processed.\n'
        )

        yield render_help(
            'The CSV columns are `openalex_id`, `url`, `score`, `flagged`, and '
            '`publication_year`. The `url` column contains the DOI link when available '
            'and falls back to the OpenAlex landing page otherwise. Use `--include-title` '
            'to append a trailing `title` column, and `--flagged-only` to restrict output '
            'to papers above the threshold.\n'
        )

        yield render_help(
            'Resumable state (`.cursor` and `.ids` sidecars) is maintained next to the '
            'output file, so a run can be interrupted and continued later without '
            're-processing already-seen papers. Use `--no-resume` to discard any existing '
            'state and start fresh.\n'
        )


class RichCommandGroups(RichMixin):
    """
    Rich display element that shows command groups in separate panels.
    Each panel contains all commands of that command group.
    """

    def __init__(self, command_groups: Dict[str, click.Group]):
        self.command_groups = command_groups

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> Any:
        from rich.table import Table

        for group_name, group in self.command_groups.items():
            # Create a table for this command group with fixed width for command column
            table = Table(
                title=None,
                box=None,
                show_header=False,
                padding=(0, 1),
                expand=True,
            )
            # Force fixed width by setting both min and max to the same value
            table.add_column(
                'Command', style='bold cyan', min_width=25, max_width=25, no_wrap=True
            )
            table.add_column('Description', style='white', ratio=1)

            # Add all subcommands of this group to the table
            for cmd_name, cmd in group.commands.items():
                # Try to get help text from various sources
                short_help = cmd.short_help
                if not short_help and cmd.help:
                    # Extract first meaningful line from the full help text
                    help_lines = [
                        line.strip() for line in cmd.help.split('\n') if line.strip()
                    ]
                    short_help = help_lines[0] if help_lines else ''
                if not short_help:
                    short_help = ''

                # Show composite command (e.g., "metadata collect" instead of just "collect")
                composite_cmd = f'{group_name} {cmd_name}'
                table.add_row(composite_cmd, short_help)

            # Wrap the table in a panel with the group name as title. A group
            # may override the displayed title by setting a ``display_title``
            # attribute — used e.g. to keep the panel heading as "Discovery"
            # while the CLI verb is ``discover``.
            display_title = getattr(group, 'display_title', None) or group_name.capitalize()
            panel = Panel(
                table,
                title=f'[bold]{display_title}[/bold]',
                title_align='left',
                style='bright_black',
            )

            yield panel


def is_experiment_archive(folder_path: str) -> bool:
    """
    Helper function to determine if a given ``folder_path`` represents a valid experiment archive.
    """
    if not os.path.isdir(folder_path):
        return False

    data_path = os.path.join(folder_path, 'experiment_data.json')
    meta_path = os.path.join(folder_path, 'experiment_meta.json')
    code_path = os.path.join(folder_path, 'experiment_code.py')
    metadata_path = os.path.join(folder_path, 'metadata.yml')
    if (
        not os.path.exists(data_path)
        or not os.path.exists(meta_path)
        or not os.path.exists(code_path)
        or not os.path.exists(metadata_path)
    ):
        return False

    return True


class CLI(click.RichGroup):
    # This list contains the string identifiers for the possible strategies that can be used for
    # the metadata collect method, which will collect the individual metadata files from the
    # experiment results.
    EXPERIMENT_COLLECT_STRATEGIES: List[str] = ['recent']

    def __init__(self, **kwargs):
        super(CLI, self).__init__(invoke_without_command=True, **kwargs)

        # This config file is a global singleton instance which allows access to the most
        # important config parameter of the project such as the URL address of the fileshare
        # server from which the datasets will be downloaded.
        self.config = Config()
        self.file_share = NextcloudFileShare(
            self.config.get_fileshare_url(),
            **self.config.get_fileshare_parameters('nextcloud'),
        )
        self.cache = self.config.cache

        self.help = ''

        ## -- Adding commands --

        # metadata command group
        self.add_command(self.metadata_group)
        self.metadata_group.add_command(self.metadata_collect_command)
        self.metadata_group.add_command(self.metadata_upload_command)
        self.metadata_group.add_command(self.metadata_edit_command)
        self.metadata_group.add_command(self.metadata_diff_command)
        self.metadata_group.add_command(self.metadata_remove_command)

        # dataset command group
        self.add_command(self.dataset_group)
        self.dataset_group.add_command(self.dataset_create_command)
        self.dataset_group.add_command(self.dataset_upload_command)

        # docs command group
        self.add_command(self.docs_group)
        self.docs_group.add_command(self.docs_collect_datasets_command)
        self.docs_group.add_command(self.docs_compile_command)
        self.docs_group.add_command(self.docs_deploy_command)

        # agent command group
        self.add_command(self.agent_group)
        self.agent_group.add_command(self.agent_venv_command)
        self.agent_group.add_command(self.agent_tools_command)
        self.agent_group.add_command(self.agent_verify_command)
        self.agent_group.add_command(self.agent_init_command)
        self.agent_group.add_command(self.agent_status_command)
        self.agent_group.add_command(self.agent_gate_command)
        self.agent_group.add_command(self.agent_set_command)

        # discovery command group (CLI verb is `discover`, keeping the
        # user-visible panel title as "Discovery" via display_title override).
        self.discovery_group.display_title = 'Discovery'
        self.add_command(self.discovery_group)
        self.discovery_group.add_command(self.discovery_model_info_command)
        self.discovery_group.add_command(self.discovery_model_download_command)
        self.discovery_group.add_command(self.discovery_fetch_command)
        self.discovery_group.add_command(self.discovery_score_command)
        self.discovery_group.add_command(self.discovery_stream_command)

    # The maximum width (in characters) for the help text paragraphs. This ensures
    # that the text wraps at a reasonable width even on very wide terminals.
    HELP_MAX_WIDTH: int = 120

    def format_help(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        """
        Custom help formatting that displays the usage line, extended help text,
        options, and command groups in that order.
        """
        self.format_usage(ctx, formatter)

        # Display the extended help text with a max width constraint
        help_width = min(Console().width, self.HELP_MAX_WIDTH)
        help_console = Console(width=help_width)
        help_console.print(RichManageHelp())

        self.format_options(ctx, formatter)

        # Collect and display command groups
        command_groups = {}
        for name, cmd in self.commands.items():
            if isinstance(cmd, click.Group):
                command_groups[name] = cmd

        if command_groups:
            click.echo(RichCommandGroups(command_groups))

    def format_commands(
        self, ctx: click.Context, formatter: click.HelpFormatter
    ) -> None:
        # Override parent behavior - command groups are rendered in format_help instead.
        pass

    ## == METADATA COMMAND GROUP ==
    # This command group is used to manage the metadata.yml file both locally and on the remote server

    @click.group(
        'metadata', help='Commands for managing the metadata YML of the remote server.'
    )
    @click.pass_obj
    def metadata_group(
        self,
    ):
        """
        This command group contains commands that can be used to manage the metadata.yml file on the
        remote file server. This metadata file contains all the necessary information about the
        datasets and for example determines what information will be shown when using the
        ``list`` command.
        """
        pass

    def collect_dataset_archives_map(
        self, results_path: str
    ) -> Dict[str, List[Experiment]]:
        """
        A helper function which creates a ``dataset_archives_map`` data structure if given the absolute
        string ``results_path`` of where to search the experiment archives. Returns a dict data structure
        whose string keys are the dataset names and the values are the corresponding Experiment instances
        imported from the experiment archives.

        :param results_path: The absolute path to the folder where the experiment archives are stored.

        :return: A dict data structure mapping dataset names to Experiment instances.
        """
        click.secho('collecting experiment archives...', fg='bright_black')

        # ~ collecting experiment archives
        # We first want to generally collect all the experiment archives here for each dataset.
        experiments: List[Experiment] = []
        for root, folders, files in os.walk(results_path):
            for folder in folders:
                folder_path = os.path.join(root, folder)
                if is_experiment_archive(folder_path):
                    try:
                        experiment = Experiment.load(folder_path)
                    except Exception:
                        continue

                    # for now we just add the loaded experiment to the list and will then later
                    # sort out the metadata from it.
                    experiments.append(experiment)

                    experiment_name = experiment.metadata['name']
                    experiment_id = os.path.basename(folder_path)
                    click.secho(
                        f' * collecting experiment {experiment_name}/{experiment_id}',
                        fg='bright_black',
                    )

        click.secho(f'collected {len(experiments)} experiments')

        # Now that we have all the experiments we can sort the experiments into lists of which dataset they belong to.
        # best case there is inly one dataest per experiment, but there are good chances that this will not be the
        # case. And if there are multiple datasets per experiment, we will have to decide which one to use to get
        # the metadata from based on the strategy.
        dataset_archives_map: Dict[str, List[Experiment]] = defaultdict(list)
        for experiment in experiments:
            dataset = experiment.parameters['DATASET_NAME']
            dataset_archives_map[dataset].append(experiment)

        return dataset_archives_map

    @click.command('collect')
    @click.option(
        '--blank',
        is_flag=True,
        help=(
            'If this flag is set, the initial metadata.yml file will be created from scratch.'
            'Otherwise, the remote metadata.yml file will be downloaded and used as a base.'
        ),
    )
    @click.option(
        '-p',
        '--path',
        help='The path at which to save the new metadata file',
        default=METADATA_PATH,
        show_default=True,
    )
    @click.option(
        '-s',
        '--source',
        help='The source path at which to search for dataset metadata.',
        default=RESULTS_PATH,
    )
    @click.option(
        '--strategy',
        help='The strategy to use for collecting the metadata.',
        type=click.Choice(EXPERIMENT_COLLECT_STRATEGIES),
        default='recent',
    )
    @click.option(
        '--verbose',
        is_flag=True,
        help='If this flag is set, the command will print additional information.',
    )
    @click.option(
        '--datasets',
        type=CsvListType(),
        help=(
            'A list of dataset names to collect metadata for. If not provided, all datasets will be collected.'
        ),
    )
    @click.pass_obj
    def metadata_collect_command(
        self,
        blank: bool,
        path: str,
        source: str,
        strategy: str,
        verbose: bool,
        datasets: List[str],
    ) -> None:
        """
        This command collects the metadata of all datasets that are currently stored on the LOCAL system
        and bundles that information into a single metadata.yml file. Will use the current version of the
        remote file share metadata as a base unless the `--blank` flag is set.
        """
        time_start: float = time.time()

        ## --- Creating the Base ---
        # If the blank flag is NOT explicitly set, we are going to try and download the metadata.yml file from the remote file
        # share server and use that as a base version which is then updated with the local metadata.
        metadata_all: dict = {'datasets': []}
        if not blank:
            click.echo(
                'downloading the metadata.yml file from the remote file share server...'
            )
            metadata_all.update(self.file_share.fetch_metadata())

        ## --- Collecting the Metadata ---

        # This method will return a dict data structure that maps the dataset names to lists of Experiment instances
        # that have been loaded from the results folder of completed experiment. For each dataset name key the corresponding
        # value list consists of all experiment archives related to that dataset.
        dataset_archives_map: Dict[str, List[Experiment]] = (
            self.collect_dataset_archives_map(source)
        )

        if datasets:
            dataset_archives_map = {
                name: experiments
                for name, experiments in dataset_archives_map.items()
                if name in datasets
            }

        click.echo(
            f'updating information on {len(dataset_archives_map)} datasets: {" ".join(dataset_archives_map.keys())}'
        )

        # This data structure will hold the metadata for each dataset that we have collected from the experiments.
        dataset_metadata_map: Dict[str, dict] = {}
        for dataset, experiments in dataset_archives_map.items():
            if strategy == 'recent':
                experiment = sorted(experiments, key=lambda e: e.metadata['end_time'])[
                    -1
                ]

            # The actual experiment metadata that we are interested in will be stored as an additional artifact
            # called "metadata.yml" in the experiment folder. We will load this file and then add the metadata
            # to the dataset_metadata_map.
            metadata_path: str = os.path.join(experiment.path, 'metadata.yml')
            with open(metadata_path, mode='r') as file:
                metadata: dict = yaml.load(file, Loader=yaml.FullLoader)

            dataset_metadata_map[dataset] = metadata

        # --- merging the metadata ---
        # The structure of the overall metadata is such that there is an additional key for each dataset and the
        # corresponding value is again a dict-like object that contains the metadata for that dataset.
        click.echo('merging the metadata...')
        for dataset, metadata in dataset_metadata_map.items():
            metadata_all['datasets'][dataset] = metadata

        # --- saving the file ---
        # In the end we can save the collected metadata into a file and place it at the specified path.
        path = os.path.expanduser(path)
        click.echo(f'saving the metadata file @ {path}...')
        with open(path, 'w') as file:
            yaml.dump(metadata_all, file, sort_keys=True, indent=4)

        time_end: float = time.time()

        click.secho(
            f'✅ collected metadata.yml file in {time_end - time_start:.2f} seconds',
            fg='green',
        )

        if verbose:
            pprint(metadata_all, max_string=100)

        click.secho()

    @click.command('upload')
    @click.option(
        '-p',
        '--path',
        help='The path to the metadata file that should be uploaded.',
        default=METADATA_PATH,
        show_default=True,
    )
    @click.pass_obj
    def metadata_upload_command(self, path: str) -> None:
        """
        Uploads a local metadata.yml file to the remore file share server. Note that this will overwrite the
        version of the metadata file that is currently stored on the server.
        """
        if not os.path.exists(path):
            click.secho(
                f'Error: The specified metadata file "{path}" does not exist.', fg='red'
            )
            sys.exit(1)

        click.secho('uploading the metadata file to the remote file share server...')
        self.file_share.upload('metadata.yml', path)
        click.secho('✅ uploaded metadata.yml', fg='green')
        click.secho()

    @click.command('edit')
    @click.option(
        '-p',
        '--path',
        help='The path to the metadata file that should be uploaded.',
        default='./metadata.yml',
    )
    @click.pass_obj
    def metadata_edit_command(
        self,
        path: str,
    ) -> None:
        """
        Opens the metadata.yml file in the default editor of the system.
        """
        if not os.path.exists(path):
            click.secho(
                f'Error: The specified metadata file "{path}" does not exist.', fg='red'
            )
            sys.exit(1)

        open_file_in_editor(path)

    @click.command('diff', short_help='Compare local metadata.yml with remote version')
    @click.option(
        '--local-file',
        '-f',
        type=click.Path(exists=True, dir_okay=False, readable=True),
        default=METADATA_PATH,
        show_default=True,
        help='Path to the local metadata.yml file to compare',
    )
    @click.pass_obj
    def metadata_diff_command(self, local_file: str) -> None:
        """
        Compares a local metadata.yml file with the remote metadata.yml file from the server.
        Shows the number of different lines and displays a visual diff of the changes.

        By default, uses the package metadata.yml file for comparison.
        """

        ## --- Fetch Remote Metadata ---
        try:
            # Try to fetch the remote metadata
            self.file_share.fetch_metadata(force=True)

            if (
                not hasattr(self.file_share, 'metadata')
                or self.file_share.metadata is None
            ):
                click.secho('❌ Could not fetch remote metadata.yml file', fg='red')
                sys.exit(1)

        except Exception as e:
            click.secho(f'❌ Error fetching remote metadata: {str(e)}', fg='red')
            sys.exit(1)

        ## --- Read Local File ---
        try:
            with open(local_file, 'r', encoding='utf-8') as f:
                local_content = f.read()
        except Exception as e:
            click.secho(f'❌ Error reading local file: {str(e)}', fg='red')
            sys.exit(1)

        ## --- Normalize Both Files for Comparison ---
        try:
            # Parse local file to normalize it
            local_data = yaml.safe_load(local_content)
            # Convert both to the same YAML format for fair comparison
            local_normalized = yaml.dump(
                local_data, default_flow_style=False, sort_keys=True, indent=2
            )
            remote_normalized = yaml.dump(
                self.file_share.metadata,
                default_flow_style=False,
                sort_keys=True,
                indent=2,
            )
        except Exception as e:
            click.secho(f'❌ Error normalizing YAML data: {str(e)}', fg='red')
            sys.exit(1)

        ## --- Calculate Diff ---
        local_lines = local_normalized.splitlines(keepends=False)
        remote_lines = remote_normalized.splitlines(keepends=False)

        # Generate unified diff
        diff_lines = list(
            difflib.unified_diff(
                local_lines,
                remote_lines,
                fromfile=f'local/{os.path.basename(local_file)}',
                tofile='remote/metadata.yml',
                lineterm='',
            )
        )

        # Count changed lines (lines starting with + or - but not +++ or ---)
        changed_lines = sum(
            1
            for line in diff_lines
            if (line.startswith('+') and not line.startswith('+++'))
            or (line.startswith('-') and not line.startswith('---'))
        )

        ## --- Display Results ---
        rich_diff = RichDiffDisplay(
            local_file=local_file,
            remote_file='remote/metadata.yml',
            diff_lines=diff_lines,
            changed_lines=changed_lines,
        )
        click.echo(rich_diff)

    @click.command(
        'remove', short_help='Remove a dataset entry from the local metadata.yml file'
    )
    @click.argument('dataset_name', type=str)
    @click.option(
        '--metadata-path',
        '-f',
        type=click.Path(exists=True, dir_okay=False, readable=True, writable=True),
        default=METADATA_PATH,
        show_default=True,
        help='Path to the local metadata.yml file to modify',
    )
    @click.option(
        '--backup',
        is_flag=True,
        default=True,
        show_default=True,
        help='Create a backup copy before modifying the file',
    )
    @click.pass_obj
    def metadata_remove_command(
        self, dataset_name: str, metadata_path: str, backup: bool
    ) -> None:
        """
        Removes a dataset entry from the local metadata.yml file.

        DATASET_NAME: Name of the dataset to remove from the metadata file
        """

        ## --- Read and Parse Local File ---
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata_content = f.read()
            metadata = yaml.safe_load(metadata_content)
        except Exception as e:
            click.secho(f'❌ Error reading metadata file: {str(e)}', fg='red')
            sys.exit(1)

        ## --- Check if Dataset Exists ---
        if 'datasets' not in metadata:
            click.secho('❌ No datasets section found in metadata file', fg='red')
            sys.exit(1)

        if dataset_name not in metadata['datasets']:
            click.secho(
                f'❌ Dataset "{dataset_name}" not found in metadata file', fg='red'
            )
            available_datasets = list(metadata['datasets'].keys())
            if available_datasets:
                click.secho(
                    f'Available datasets: {", ".join(sorted(available_datasets))}',
                    fg='yellow',
                )
            sys.exit(1)

        ## --- Create Backup if Requested ---
        if backup:
            backup_path = f'{metadata_path}.backup'
            try:
                import shutil

                shutil.copy2(metadata_path, backup_path)
                click.secho(f'📋 Created backup at: {backup_path}', fg='bright_black')
            except Exception as e:
                click.secho(
                    f'⚠️ Warning: Could not create backup: {str(e)}', fg='yellow'
                )

        ## --- Remove Dataset ---
        removed_dataset = metadata['datasets'].pop(dataset_name)

        ## --- Write Updated Metadata ---
        try:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                yaml.dump(
                    metadata, f, default_flow_style=False, sort_keys=True, indent=4
                )
            click.secho(
                f'✅ Successfully removed dataset "{dataset_name}" from {metadata_path}',
                fg='green',
            )

            # Show some info about what was removed
            compounds = removed_dataset.get('compounds', 'Unknown')
            targets = removed_dataset.get('targets', 'Unknown')
            click.secho(
                f'   Removed dataset had {compounds} compounds and {targets} targets',
                fg='bright_black',
            )

        except Exception as e:
            click.secho(f'❌ Error writing updated metadata file: {str(e)}', fg='red')
            sys.exit(1)

    ## == DATASET COMMAND GROUP ==
    # all the commands related to the interaction with the local datasets - such as the (re)creation of
    # datasets or the uploading to the remote file share server.

    @click.group(
        'dataset', help=('Commands for managing the datasets on the local system.')
    )
    @click.pass_obj
    def dataset_group(
        self,
    ) -> None:
        pass

    def collect_dataset_experiment_map(experiments_path: str) -> Dict[str, Experiment]:
        """
        A helper function which creates a ``dataset_experiment_map`` data structure if given
        the absolute string ``experiments_path`` of where to search the experiment modules.
        Returns a dict data structure whose string keys are the dataset names and the values
        are the corresponding Experiment instances imported from the experiment modules.

        :param experiments_path: The absolute path to the folder where the experiment modules are stored.

        :return: A dict data structure mapping dataset names to Experiment instances.
        """
        # In this data structure we are going to map the string dataset names to the imported
        # Experiment instances that can potentially be executed to create the corresponding
        # datasets.
        dataset_experiment_map: Dict[str, Experiment] = {}

        file_names: List[str] = os.listdir(experiments_path)
        for file_name in file_names:
            file_path = os.path.join(experiments_path, file_name)
            if os.path.isfile(file_path) and file_name.endswith('.py'):
                # Each python module we find, we'll import and check if it has the "DATASET_NAME"
                # attribute. If it does, we'll add it to the list of available experiment modules.
                module = dynamic_import(file_path)
                if hasattr(module, 'DATASET_NAME') and hasattr(
                    module, '__experiment__'
                ):
                    experiment = getattr(module, '__experiment__')
                    dataset_name = experiment.parameters['DATASET_NAME']
                    dataset_experiment_map[dataset_name] = experiment

        return dataset_experiment_map

    @click.command(
        'create', short_help='Creates a new dataset using the experiment modules.'
    )
    @click.option(
        '--scripts-path', help='The path to the scripts folder.', default=SCRIPTS_PATH
    )
    @click.option(
        '--all',
        help='If this flag is set, all available datasets will be created.',
        is_flag=True,
    )
    @click.argument('name', type=str)
    @click.pass_obj
    def dataset_create_command(
        self,
        scripts_path: str,
        name: str,
        all: bool,
    ) -> None:
        """
        This command can be used to trigger the execution of one or multiple experiment modules which
        will re-create a dataset.

        The NAME argument may either be the string name of a single dataset to be created or a
        comma separated list of dataset names.
        """
        # ~ collecting experiment modules
        # First of all we construct a data structure that tells us which experiment modules are actually
        # available to create which datasets.
        click.secho('collecting available experiment modules...', color='bright_black')

        # This method creates a dict data structure that maps the dataset names to the Experiment instances
        # that are imported from the experiment modules.
        dataset_experiment_map: Dict[str, Experiment] = (
            self.collect_dataset_experiment_map(scripts_path)
        )
        click.secho(f'found {len(dataset_experiment_map)} experiment modules')

        # ~ parsing the name argument
        # The name argument may be a single dataset name or a comma separated list of dataset names.
        if all:
            dataset_names: List[str] = list(dataset_experiment_map.keys())
        else:
            dataset_names: List[str] = [n.replace(' ', '') for n in name.split(',')]

        # ~ executing the experiments
        # We can fetch the experiment instances from the map that we previously constructed and
        # we can forcefully execute these experiements by using the "run" method.
        click.secho(
            'running the experiment modules. Beware that this will take some time...'
        )
        for dataset_name in dataset_names:
            click.secho(f'⭐ running "{dataset_name}"', fg='yellow')
            experiment = dataset_experiment_map[dataset_name]
            experiment.run()

        click.secho('✅ all experiments have been executed', fg='green')

    @click.command(
        'upload', short_help='Uploads local datasets to the remote file share server.'
    )
    @click.argument('name', type=str, required=False)
    @click.option(
        '--results-path', help='The path to the results folder.', default=RESULTS_PATH
    )
    @click.option(
        '-p', '--path', help='The path to the dataset file that should be uploaded.'
    )
    @click.option(
        '--strategy',
        help='The strategy to use for selecting the experiment to use for the upload.',
        type=click.Choice(EXPERIMENT_COLLECT_STRATEGIES),
        default='recent',
    )
    @click.option(
        '--all',
        help='If this flag is set, all available datasets will be uploaded.',
        is_flag=True,
    )
    @click.pass_obj
    def dataset_upload_command(
        self,
        name: str,
        results_path: str,
        path: str,
        strategy: str,
        all: bool,
    ) -> None:
        """
        This command will upload the graph dataset MPACK file and the raw CSV file for a given dataset
        NAME to the remote file share server. If the `--all` flag is set, all available datasets will be
        uploaded. Note that this will overwrite any existing dataset files on the remote server.

        The NAME argument may either be the string name of a single dataset to be uploaded or a
        comma-separated list of dataset names.
        """

        # This method will return a dict data structure that maps the dataset names to lists of Experiment instances
        # that have been loaded from the results folder of completed experiment. For each dataset name key the corresponding
        # value list consists of all experiment archives related to that dataset.
        dataset_archives_map: Dict[str, List[Experiment]] = defaultdict(list)
        dataset_archives_map.update(self.collect_dataset_archives_map(results_path))
        click.secho(f'found {len(dataset_archives_map)} datasets')

        # ~ parsing the name argument
        # The name argument may be a single dataset name or a comma separated list of dataset names.
        if all:
            dataset_names: List[str] = list(dataset_archives_map.keys())
        elif name:
            dataset_names: List[str] = [n.replace(' ', '') for n in name.split(',')]
        else:
            click.secho(
                '❌ Error: Must specify either dataset NAME(s) or use --all flag',
                fg='red',
            )
            sys.exit(1)

        # ~ uploading the datasets
        # Now we can go through the dataset archives map and upload the dataset files to the file share server.
        click.secho('uploading the datasets...')
        for dataset_name in dataset_names:
            experiments = dataset_archives_map[dataset_name]
            if len(experiments) == 0:
                click.secho(
                    f'❗ no experiment archives found for dataset "{dataset_name}"',
                    fg='red',
                )
                continue

            click.secho(f'⭐ dataset "{dataset_name}"', fg='yellow')

            # First we need to select a single experiment to use as the basis of the upload for each dataset
            # based on the selected strategy.
            if strategy == 'recent':
                experiment = sorted(experiments, key=lambda e: e.metadata['end_time'])[
                    -1
                ]

            # The actual dataset file that we are interested in will be stored as an additional artifact
            # called "{dataset_name}.xxx" in the experiment folder. We will load this file and then
            # upload it to the file share server.

            mpack_file_name = f'{dataset_name}.mpack.gz'
            mpack_file_path: str = os.path.join(
                experiment.path, f'{dataset_name}.mpack.gz'
            )
            if os.path.exists(mpack_file_path):
                file_size = os.path.getsize(mpack_file_path)
                file_size_mb = file_size / (1024 * 1024)
                click.secho(f'   uploading "{mpack_file_name}" [{file_size_mb:.1f} MB]')
                self.file_share.upload(mpack_file_name, mpack_file_path)

            csv_file_name = f'{dataset_name}.csv.gz'
            csv_file_path: str = os.path.join(experiment.path, f'{dataset_name}.csv.gz')
            if os.path.exists(csv_file_path):
                file_size = os.path.getsize(csv_file_path)
                file_size_mb = file_size / (1024 * 1024)
                click.secho(f'   uploading "{csv_file_name}" [{file_size_mb:.1f} MB]')
                self.file_share.upload(csv_file_name, csv_file_path)

        click.secho('✅ datasets uploaded', fg='green')
        click.secho()

    ## == AGENT COMMAND GROUP ==
    # Commands to interact with the LLM-powered agent workflow for automated
    # dataset discovery and processing script generation.

    @click.group(
        'agent',
        cls=RichHelpGroup,
        rich_help=RichAgentGroupHelp(),
        short_help='Deterministic helpers for the dataset-extraction skill.',
    )
    @click.pass_obj
    def agent_group(self) -> None:
        pass

    @click.command(
        'venv',
        cls=RichHelpCommand,
        rich_help=RichAgentVenvHelp(),
        short_help='Create a clean virtual environment for running generated scripts.',
    )
    @click.option(
        '--path',
        type=click.Path(file_okay=False, path_type=str),
        default=None,
        help='Where to create the venv. Defaults to `.dataset_work/.venv`.',
    )
    @click.pass_obj
    def agent_venv_command(
        self,
        path: Optional[str],
    ) -> None:
        from pathlib import Path as _Path
        from rich.console import Console

        from chem_mat_data.agent.environment import (
            create_agent_venv,
            EnvironmentSetupError,
        )

        console = Console()
        venv_path = _Path(path) if path else None

        console.print('[cyan]Creating virtual environment (this may take a while)...[/cyan]')
        try:
            python_path = create_agent_venv(venv_path)
        except EnvironmentSetupError as exc:
            click.secho(f'❌ {exc}', fg='red')
            sys.exit(1)

        console.print('[green]✓ virtual environment ready[/green]')
        # Printed bare on the last line so a calling script can capture it.
        click.echo(str(python_path))

    @click.command(
        'tools',
        cls=RichHelpCommand,
        rich_help=RichAgentToolsHelp(),
        short_help='Check (or install) the external page-reading tools.',
    )
    @click.option(
        '--install',
        is_flag=True,
        default=False,
        help='Install the tools that are missing instead of only reporting them.',
    )
    @click.pass_obj
    def agent_tools_command(
        self,
        install: bool,
    ) -> None:
        from rich.console import Console

        from chem_mat_data.agent.environment import EnvironmentSetupError
        from chem_mat_data.agent.tools import check_page_readers, install_missing

        console = Console()
        statuses = check_page_readers()

        if install and any(not status.available for status in statuses):
            console.print('[cyan]Installing missing tools (this may take a while)...[/cyan]')
            console.print()
            try:
                statuses = install_missing(statuses)
            except EnvironmentSetupError as exc:
                click.secho(f'❌ {exc}', fg='red')
                sys.exit(1)
            console.print()

        console.print()
        console.print('[bold]Page-reading tools[/bold]')
        console.print()

        for status in statuses:
            name = status.spec.name
            if status.available:
                version = f' [dim]{status.version}[/dim]' if status.version else ''
                console.print(f'  [bold green]✓[/bold green] {name}{version}')
                console.print(f'      [dim]{status.spec.purpose}[/dim]')
            else:
                console.print(f'  [bold red]✗[/bold red] {name}')
                for problem in status.problems:
                    console.print(f'      [red]{problem}[/red]')
                console.print(
                    f'      [dim]install: {" ".join(status.spec.install_cmd)}[/dim]'
                )
            console.print()

        missing = [status for status in statuses if not status.available]
        if missing:
            if not install:
                console.print(
                    f'[bold red]{len(missing)} missing[/bold red] - re-run with '
                    '`--install` to install them.'
                )
            else:
                console.print(
                    f'[bold red]{len(missing)} still missing[/bold red] after installing.'
                )
            sys.exit(1)

        console.print('[bold green]✓[/bold green] all page-reading tools available')

    @click.command(
        'verify',
        cls=RichHelpCommand,
        rich_help=RichAgentVerifyHelp(),
        short_help='Check that a processing run produced a valid dataset.',
    )
    @click.argument(
        'results_dir',
        type=click.Path(path_type=str),
    )
    @click.option(
        '--strict',
        is_flag=True,
        default=False,
        help='Treat warnings as failures as well.',
    )
    @click.pass_obj
    def agent_verify_command(
        self,
        results_dir: str,
        strict: bool,
    ) -> None:
        from rich.console import Console

        from chem_mat_data.agent.verification import verify_results

        console = Console()
        report = verify_results(results_dir)

        console.print()
        console.print(f'[bold]Verifying[/bold] [dim]{report.results_dir}[/dim]')
        console.print()

        if report.dataset_name:
            console.print(f'  dataset    [bold]{report.dataset_name}[/bold]')
        if report.compounds is not None:
            console.print(f'  compounds  [bold]{report.compounds}[/bold]')
        if report.targets is not None:
            console.print(f'  targets    [bold]{report.targets}[/bold]')
        console.print()

        for issue in report.errors:
            console.print(f'  [bold red]✗[/bold red] {issue.message}')
        for issue in report.warnings:
            console.print(f'  [yellow]![/yellow] {issue.message}')

        if report.ok and not report.warnings:
            console.print('  [bold green]✓[/bold green] all checks passed')

        console.print()

        if not report.ok:
            count = len(report.errors)
            console.print(
                f'[bold red]FAILED[/bold red] - {count} error(s). '
                'The results are not usable as they stand.'
            )
            sys.exit(1)

        if strict and report.warnings:
            count = len(report.warnings)
            console.print(
                f'[bold red]FAILED[/bold red] - {count} warning(s) with --strict.'
            )
            sys.exit(1)

        if report.warnings:
            console.print(
                '[green]PASSED[/green] with warnings - the dataset is valid but '
                'incomplete.'
            )
        else:
            console.print('[bold green]PASSED[/bold green]')

    ## -- staged extraction pipeline --
    # The dataset-extraction skill walks a publication through six gated stages. The
    # commands below own the state that has to survive between those stages, and the
    # gates that decide whether a stage may be considered done.

    @staticmethod
    def _render_status_license(console, manifest) -> None:
        """
        Print a scannable summary of the recorded license conclusion.

        The status output previously showed only that the license stage had passed,
        which reads identically whether the license is clean or was waived over an
        upstream that forbids redistribution. Those are the two cases a human most
        needs told apart, so the override decision and the upstream are surfaced here.

        Deliberately a summary, not a dump: the free-text fields run to whole
        paragraphs, and a status display that has to be scrolled is one nobody reads.
        The verbatim record lives in ``.dataset_work/<slug>/manifest.yml``, which is
        where the skill sends an agent that needs to merge into it.

        :param console: A Rich Console.
        :param manifest: The Manifest whose license should be rendered.
        """
        license_info = manifest.license or {}
        if not license_info:
            return

        def short(value: Any, limit: int = 96) -> str:
            """Collapse a possibly-paragraph-length field to one scannable line."""
            text = ' '.join(str(value).split())
            return text if len(text) <= limit else text[: limit - 1] + '…'

        console.print()
        console.print('  [bold]license[/bold]')

        identifier = license_info.get('identifier') or '[dim]none recorded[/dim]'
        console.print(f'      identifier   {identifier}')

        for key, label in (
            ('evidence_url', 'evidence'),
            ('redistribution', 'redistrib'),
            ('attribution', 'attribution'),
        ):
            value = license_info.get(key)
            if value:
                console.print(f'      {label:<12} [dim]{short(value)}[/dim]')

        upstream = license_info.get('upstream_database')
        if upstream in (None, '', 'none', 'null'):
            console.print(
                '      upstream     [dim]none - the publication is the source[/dim]'
            )
        else:
            console.print(f'      upstream     [bold]{short(upstream)}[/bold]')
            for key, label in (
                ('upstream_terms_url', 'terms'),
                ('upstream_license', 'its license'),
            ):
                value = license_info.get(key)
                if value:
                    console.print(f'      {label:<12} [dim]{short(value)}[/dim]')

        override = license_info.get('human_override') or {}
        if override:
            decision = str(override.get('decision', '')).lower()
            colour = 'green' if decision == 'approved' else 'red'
            console.print(
                f'      override     [bold {colour}]{decision or "unspecified"}'
                f'[/bold {colour}] - [dim]'
                f'{short(override.get("reason") or "no reason recorded")}[/dim]'
            )
            console.print(
                '      [dim]full record: .dataset_work/'
                f'{manifest.slug}/manifest.yml[/dim]'
            )

    @staticmethod
    def _render_gate_report(console, report) -> None:
        """
        Print a gate report to the console.

        :param console: A Rich Console.
        :param report: The VerificationReport to render.
        """
        if report.dataset_name:
            console.print(f'  dataset    [bold]{report.dataset_name}[/bold]')
        if report.compounds is not None:
            console.print(f'  compounds  [bold]{report.compounds}[/bold]')
        if report.targets is not None:
            console.print(f'  targets    [bold]{report.targets}[/bold]')
        if report.dataset_name or report.compounds is not None:
            console.print()

        # The observed target statistics are printed whether or not anything is wrong.
        # Stage 6 is asked to confirm the values match what the paper reports, and this
        # is the only place those numbers appear without someone computing them.
        if getattr(report, 'target_statistics', None):
            for stat in report.target_statistics:
                if stat.minimum is None:
                    console.print(f'  target {stat.index}   [dim]no finite values[/dim]')
                    continue
                distinct = f'{stat.unique}{"+" if stat.unique_capped else ""} distinct'
                console.print(
                    f'  target {stat.index}   '
                    f'[bold]{stat.minimum:g}[/bold] … [bold]{stat.maximum:g}[/bold]'
                    f'   mean {stat.mean:g}'
                    f'   [dim]n={stat.count}, {distinct}'
                    + (f', {stat.missing} missing' if stat.missing else '')
                    + '[/dim]'
                )
            console.print()

        for issue in report.errors:
            console.print(f'  [bold red]✗[/bold red] {issue.message}')
        for issue in report.warnings:
            console.print(f'  [yellow]![/yellow] {issue.message}')

        if report.ok and not report.warnings:
            console.print('  [bold green]✓[/bold green] all checks passed')

    @click.command(
        'init',
        cls=RichHelpCommand,
        rich_help=RichAgentInitHelp(),
        short_help='Start a new staged extraction run.',
    )
    @click.argument('slug', type=str)
    @click.option(
        '--url',
        '-u',
        type=str,
        default=None,
        help='Publication URL or a concrete paper title to extract from.',
    )
    @click.option(
        '--force',
        is_flag=True,
        default=False,
        help='Overwrite an existing manifest for this slug.',
    )
    @click.pass_obj
    def agent_init_command(
        self,
        slug: str,
        url: Optional[str],
        force: bool,
    ) -> None:
        from rich.console import Console

        from chem_mat_data.agent.manifest import Manifest, work_dir_for

        console = Console()

        if Manifest.exists(slug) and not force:
            click.secho(
                f'❌ a manifest for "{slug}" already exists. '
                f'Use --force to start over, or `cmmanage agent status {slug}` '
                'to see where it stands.',
                fg='red',
            )
            sys.exit(1)

        manifest = Manifest.create(slug, source_url=url)
        manifest_path = manifest.save()

        work = work_dir_for(slug)
        (work / 'downloads').mkdir(parents=True, exist_ok=True)

        console.print()
        console.print(f'[green]✓[/green] initialised extraction run [bold]{slug}[/bold]')
        console.print(f'  manifest   [dim]{manifest_path}[/dim]')
        console.print(f'  downloads  [dim]{work / "downloads"}[/dim]')
        console.print()
        console.print('  next stage [bold]prepare[/bold]')
        console.print()

    @click.command(
        'status',
        cls=RichHelpCommand,
        rich_help=RichAgentStatusHelp(),
        short_help='Show the stage status of an extraction run.',
    )
    @click.argument('slug', type=str)
    @click.pass_obj
    def agent_status_command(
        self,
        slug: str,
    ) -> None:
        from rich.console import Console

        from chem_mat_data.agent.manifest import (
            Manifest,
            STAGES,
            PASSED,
            BLOCKED,
            STALE,
        )

        console = Console()

        try:
            manifest = Manifest.load(slug)
        except FileNotFoundError as exc:
            click.secho(f'❌ {exc}', fg='red')
            sys.exit(1)

        symbols = {
            PASSED: '[bold green]✓[/bold green]',
            BLOCKED: '[bold red]✗[/bold red]',
            STALE: '[yellow]~[/yellow]',
        }

        console.print()
        console.print(f'[bold]{manifest.slug}[/bold]')
        if manifest.source_url:
            console.print(f'[dim]source  {manifest.source_url}[/dim]')
        if manifest.record_url:
            console.print(f'[dim]record  {manifest.record_url}[/dim]')
        console.print()

        for stage in STAGES:
            state = manifest.state(stage)
            symbol = symbols.get(state.status, '[dim]·[/dim]')
            console.print(f'  {symbol} {stage:<13} [dim]{state.status}[/dim]')

            # notes, errors and warnings are not alternatives - a blocked stage has a
            # blocker and may also have warnings worth reading before deciding what to
            # do about it, and rendering only one of them is how the useful one gets
            # hidden.
            if state.notes:
                console.print(f'      [dim]{state.notes}[/dim]')
            if state.errors:
                for message in state.errors:
                    console.print(f'      [bold red]✗[/bold red] {message}')
            elif state.blocker:
                console.print(f'      [bold red]✗[/bold red] {state.blocker}')
            for message in state.warnings:
                console.print(f'      [yellow]![/yellow] {message}')

        self._render_status_license(console, manifest)

        console.print()

        if manifest.complete:
            console.print('[bold green]All stages passed.[/bold green]')
        else:
            console.print(f'  next stage [bold]{manifest.next_stage()}[/bold]')
        console.print()

    @click.command(
        'gate',
        cls=RichHelpCommand,
        rich_help=RichAgentGateHelp(),
        short_help='Check a stage against its post-conditions.',
    )
    @click.argument('stage', type=click.Choice(STAGES))
    @click.argument('slug', type=str)
    @click.option(
        '--strict',
        is_flag=True,
        default=False,
        help='Treat warnings as failures as well.',
    )
    @click.option(
        '--record/--no-record',
        default=True,
        help='Record the outcome in the manifest. On by default.',
    )
    @click.pass_obj
    def agent_gate_command(
        self,
        stage: str,
        slug: str,
        strict: bool,
        record: bool,
    ) -> None:
        from rich.console import Console

        from chem_mat_data.agent.manifest import Manifest, PASSED, BLOCKED
        from chem_mat_data.agent.gates import run_gate

        console = Console()

        try:
            manifest = Manifest.load(slug)
        except FileNotFoundError as exc:
            click.secho(f'❌ {exc}', fg='red')
            sys.exit(1)

        report = run_gate(stage, manifest)

        console.print()
        console.print(f'[bold]Gate:[/bold] {stage}  [dim]({slug})[/dim]')
        console.print()
        self._render_gate_report(console, report)
        console.print()

        failed = (not report.ok) or (strict and report.warnings)

        if record:
            # The full text is recorded, not a count. A stage that warned once and a
            # stage that warned once about the upstream forbidding derivative works
            # are the same line otherwise, and the second one has to survive into the
            # session that resumes this run.
            warnings = [issue.message for issue in report.warnings]
            errors = [issue.message for issue in report.errors]
            if failed:
                blocker = (
                    report.errors[0].message
                    if report.errors
                    else 'warnings treated as failures under --strict'
                )
                manifest.mark(
                    stage, BLOCKED, blocker=blocker, warnings=warnings, errors=errors
                )
            else:
                manifest.mark(stage, PASSED, warnings=warnings)
            manifest.save()

        if failed:
            count = len(report.errors)
            console.print(
                f'[bold red]BLOCKED[/bold red] at stage [bold]{stage}[/bold] '
                f'- {count} error(s).'
            )
            console.print(
                '[dim]This needs a human decision. Explain the problem plainly, '
                'say what was tried, and say what you need.[/dim]'
            )
            console.print()
            sys.exit(1)

        next_stage = manifest.next_stage()
        if report.warnings:
            console.print('[green]PASSED[/green] with warnings.')
        else:
            console.print(f'[bold green]PASSED[/bold green] - stage {stage} complete.')
        if next_stage:
            console.print(f'  next stage [bold]{next_stage}[/bold]')
        console.print()

    @click.command(
        'set',
        cls=RichHelpCommand,
        rich_help=RichAgentSetHelp(),
        short_help='Record a value into the run manifest.',
    )
    @click.argument('slug', type=str)
    @click.argument('key', type=str)
    @click.argument('value', type=str)
    @click.pass_obj
    def agent_set_command(
        self,
        slug: str,
        key: str,
        value: str,
    ) -> None:
        from rich.console import Console

        from chem_mat_data.agent.manifest import Manifest

        console = Console()

        try:
            manifest = Manifest.load(slug)
        except FileNotFoundError as exc:
            click.secho(f'❌ {exc}', fg='red')
            sys.exit(1)

        # Scalar fields are set directly; the structured sections take a JSON blob so
        # that the whole metadata draft or license finding can be written in one go.
        structured = {'metadata_draft', 'license'}
        # `source_url` is deliberately absent. It records where the run began and is
        # the fixed point the license gate compares its evidence against; overwriting
        # it with the download URL discovered in stage 1 silently disables that check,
        # which is exactly what happened before `record_url` existed to receive it.
        readonly = {
            'source_url': (
                'source_url is set once by `cmmanage agent init <slug> --url ...` and '
                'records where the run began. Overwriting it with a URL found later '
                'disables the license gate\'s preprint/published check, which compares '
                'against it. Record the data repository landing page as `record_url`, '
                'and the direct download URL in downloads/download_recipe.md.'
            ),
        }
        scalars = {
            'record_url': str,
            'dataset_name': str,
            'script_path': str,
            'smoke_results_dir': str,
            'results_dir': str,
            'drop_rate_reason': str,
            'input_rows': int,
            'max_drop_rate': float,
        }

        if key in readonly:
            click.secho(f'❌ {readonly[key]}', fg='red')
            sys.exit(1)

        if key in structured:
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as exc:
                click.secho(f'❌ {key} expects JSON: {exc}', fg='red')
                sys.exit(1)
            if not isinstance(parsed, dict):
                click.secho(f'❌ {key} expects a JSON object', fg='red')
                sys.exit(1)
            setattr(manifest, key, parsed)

        elif key in scalars:
            try:
                setattr(manifest, key, scalars[key](value))
            except ValueError:
                click.secho(
                    f'❌ {key} expects a {scalars[key].__name__} value, got {value!r}',
                    fg='red',
                )
                sys.exit(1)

        else:
            valid = ', '.join(sorted(structured | set(scalars)))
            click.secho(f'❌ unknown key {key!r}. Valid keys: {valid}', fg='red')
            sys.exit(1)

        manifest.save()
        console.print(f'[green]✓[/green] {slug}: set [bold]{key}[/bold]')

    ## == DISCOVERY COMMAND GROUP ==
    # This command group implements the lightweight first stage of the dataset
    # discovery pipeline: walking OpenAlex for chemistry/materials papers and
    # scoring their title+abstract with a fine-tuned TinyLlama classifier to
    # flag likely dataset releases.

    @click.group(
        'discover',
        cls=RichHelpGroup,
        rich_help=RichDiscoveryGroupHelp(),
        short_help='Crawl OpenAlex and classify papers that release datasets.',
    )
    @click.pass_obj
    def discovery_group(self) -> None:
        pass

    @click.command(
        'fetch',
        cls=RichHelpCommand,
        rich_help=RichDiscoveryFetchHelp(),
        short_help='Fetch titles and abstracts of candidate papers from OpenAlex.',
    )
    @click.option(
        '--out',
        '-o',
        type=click.Path(dir_okay=False, path_type=str),
        default='discovery_papers.jsonl',
        show_default=True,
        help='Output JSONL path. A sibling `.cursor` and `.ids` file tracks resumable state.',
    )
    @click.option(
        '--limit',
        '-n',
        type=int,
        default=1000,
        show_default=True,
        help='Maximum number of new records to fetch in this run.',
    )
    @click.option(
        '--year-min',
        type=int,
        default=2010,
        show_default=True,
        help='Only include works published from this year onward.',
    )
    @click.option(
        '--reset/--resume',
        default=False,
        help='Delete the existing cursor/ids state files before fetching.',
    )
    @click.pass_obj
    def discovery_fetch_command(
        self,
        out: str,
        limit: int,
        year_min: int,
        reset: bool,
    ) -> None:
        from rich.console import Console

        console = Console()
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if reset:
            for suffix in ('.cursor', '.ids'):
                p = Path(str(out_path) + suffix)
                if p.exists():
                    p.unlink()

        state = DiscoveryFetchState(out_path)
        console.print(
            f'[cyan]Fetching up to {limit} new papers from OpenAlex '
            f'(year ≥ {year_min})…[/cyan]'
        )

        fetched = 0
        try:
            with state, out_path.open('a', encoding='utf-8') as out_fp:
                for record in discovery_openalex_fetch_works(
                    state=state,
                    limit=limit,
                    year_lower_bound=year_min,
                ):
                    discovery_write_jsonl_line(out_fp, record)
                    fetched += 1
                    if fetched % 50 == 0:
                        console.print(
                            f'[dim]  {fetched}/{limit} fetched…[/dim]'
                        )
        except KeyboardInterrupt:
            console.print('[yellow]Interrupted — progress is persisted.[/yellow]')

        console.print(
            f'[green]Wrote {fetched} new records to {out_path}[/green]'
        )

    @click.command(
        'score',
        cls=RichHelpCommand,
        rich_help=RichDiscoveryScoreHelp(),
        short_help='Score papers with the TinyLlama classifier and flag likely dataset releases.',
    )
    @click.option(
        '--in',
        'input_path',
        type=click.Path(exists=True, dir_okay=False, path_type=str),
        required=True,
        help='Input JSONL file with records containing title and abstract fields.',
    )
    @click.option(
        '--out-all',
        type=click.Path(dir_okay=False, path_type=str),
        default='discovery_scored_all.jsonl',
        show_default=True,
        help='Output JSONL where every record gains a `prediction` field.',
    )
    @click.option(
        '--out-flagged',
        type=click.Path(dir_okay=False, path_type=str),
        default='discovery_scored_flagged.jsonl',
        show_default=True,
        help='Output JSONL containing only records with `prediction` above the threshold.',
    )
    @click.option(
        '--threshold',
        '-t',
        type=float,
        default=0.05,
        show_default=True,
        help=(
            'Probability threshold above which a paper is flagged. 0.05 matched '
            'the best test-set performance in the upstream Dataset-Discovery-Pipeline.'
        ),
    )
    @click.option(
        '--max-length',
        type=int,
        default=512,
        show_default=True,
        help='Max tokenized prompt length passed to the classifier.',
    )
    @click.option(
        '--device',
        type=click.Choice(['auto', 'cpu', 'cuda']),
        default='auto',
        show_default=True,
        help='Device to run inference on. `auto` uses CUDA if available.',
    )
    @click.option(
        '--no-quantize',
        is_flag=True,
        default=False,
        help='Disable 4-bit quantization even when CUDA + bitsandbytes are available.',
    )
    @click.option(
        '--model-url',
        type=str,
        default=None,
        help=(
            'Override the Nextcloud URL of the classifier archive. '
            'Defaults to `[discovery] classifier_url` from the config file.'
        ),
    )
    @click.pass_obj
    def discovery_score_command(
        self,
        input_path: str,
        out_all: str,
        out_flagged: str,
        threshold: float,
        max_length: int,
        device: str,
        no_quantize: bool,
        model_url: Optional[str],
    ) -> None:
        from rich.console import Console

        console = Console()

        if not discovery_is_ml_stack_available():
            click.secho(
                '❌ Required ML packages (torch, transformers, peft) are not installed.\n'
                '   Install them via: pip install chem_mat_database[dev]',
                fg='red',
            )
            sys.exit(1)

        resolved_url = model_url or self.config.get_discovery_classifier_url()
        model_dir = self.config.get_discovery_model_dir()

        console.print(
            f'[cyan]Loading classifier (device={device}, threshold={threshold})…[/cyan]'
        )
        try:
            discovery_ensure_classifier_model(
                url=resolved_url, target_dir=model_dir
            )
        except ValueError as exc:
            click.secho(f'❌ {exc}', fg='red')
            sys.exit(1)

        effective_device = None if device == 'auto' else device
        quantize = False if no_quantize else None
        classifier = discovery_load_classifier(
            model_dir=model_dir,
            device=effective_device,
            quantize=quantize,
            max_length=max_length,
        )

        in_path = Path(input_path)
        all_path = Path(out_all)
        flagged_path = Path(out_flagged)
        all_path.parent.mkdir(parents=True, exist_ok=True)
        flagged_path.parent.mkdir(parents=True, exist_ok=True)

        total = 0
        flagged = 0
        with (
            all_path.open('w', encoding='utf-8') as all_fp,
            flagged_path.open('w', encoding='utf-8') as flagged_fp,
        ):
            for record in discovery_iter_jsonl(in_path):
                score = classifier.predict_score(
                    title=record.get('title'),
                    abstract=record.get('abstract'),
                )
                record = {**record, 'prediction': float(score)}
                discovery_write_jsonl_line(all_fp, record)
                total += 1
                if score > threshold:
                    discovery_write_jsonl_line(flagged_fp, record)
                    flagged += 1

        console.print(
            f'[green]Scored {total} records; flagged {flagged} above '
            f'threshold {threshold}.[/green]'
        )
        console.print(f'[dim]  all     → {all_path}[/dim]')
        console.print(f'[dim]  flagged → {flagged_path}[/dim]')

    @click.command(
        'stream',
        cls=RichHelpCommand,
        rich_help=RichDiscoveryStreamHelp(),
        short_help='Fetch OpenAlex + classify in a single streaming pass, writing a CSV.',
    )
    @click.option(
        '--out',
        '-o',
        type=click.Path(dir_okay=False, path_type=str),
        default='discovery_stream.csv',
        show_default=True,
        help='CSV output file. Sibling `.cursor` and `.ids` files track resumable state.',
    )
    @click.option(
        '--limit',
        '-n',
        type=int,
        default=1000,
        show_default=True,
        help='Maximum number of new OpenAlex records to process in this run.',
    )
    @click.option(
        '--year-min',
        type=int,
        default=2010,
        show_default=True,
        help='Only include works published from this year onward.',
    )
    @click.option(
        '--threshold',
        '-t',
        type=float,
        default=0.05,
        show_default=True,
        help=(
            'Probability threshold above which a paper is flagged. 0.05 matched '
            'the best test-set performance in the upstream Dataset-Discovery-Pipeline.'
        ),
    )
    @click.option(
        '--max-length',
        type=int,
        default=512,
        show_default=True,
        help='Max tokenized prompt length passed to the classifier.',
    )
    @click.option(
        '--device',
        type=click.Choice(['auto', 'cpu', 'cuda']),
        default='auto',
        show_default=True,
        help='Device to run inference on. `auto` uses CUDA if available.',
    )
    @click.option(
        '--no-quantize',
        is_flag=True,
        default=False,
        help='Disable 4-bit quantization even when CUDA + bitsandbytes are available.',
    )
    @click.option(
        '--model-url',
        type=str,
        default=None,
        help=(
            'Override the Nextcloud URL of the classifier archive. '
            'Defaults to `[discovery] classifier_url` from the config file.'
        ),
    )
    @click.option(
        '--resume/--no-resume',
        default=True,
        help='Resume from any existing `.cursor`/`.ids` sidecars next to --out.',
    )
    @click.option(
        '--include-title',
        is_flag=True,
        default=False,
        help='Also write the paper title as a trailing CSV column.',
    )
    @click.option(
        '--flagged-only/--all',
        default=False,
        help='Only write rows above the threshold instead of every scored record.',
    )
    @click.option(
        '--verbose',
        '-v',
        is_flag=True,
        default=False,
        help=(
            'Show the full classifier load output (transformers warnings, LOAD REPORT, '
            'HF download progress). Off by default for a quieter CLI.'
        ),
    )
    @click.pass_obj
    def discovery_stream_command(
        self,
        out: str,
        limit: int,
        year_min: int,
        threshold: float,
        max_length: int,
        device: str,
        no_quantize: bool,
        model_url: Optional[str],
        resume: bool,
        include_title: bool,
        flagged_only: bool,
        verbose: bool,
    ) -> None:
        import contextlib
        import csv
        import io
        import os as _os
        import warnings

        from rich.console import Console
        from rich.panel import Panel
        from rich.progress import (
            BarColumn,
            Progress,
            TextColumn,
            TimeRemainingColumn,
        )

        console = Console()

        if not discovery_is_ml_stack_available():
            click.secho(
                '❌ Required ML packages (torch, transformers, peft) are not installed.\n'
                '   Install them via: pip install chem_mat_database[dev]',
                fg='red',
            )
            sys.exit(1)

        # Ensure the classifier is downloaded before we touch OpenAlex, so a
        # misconfigured URL fails fast instead of half-way through a fetch.
        resolved_url = model_url or self.config.get_discovery_classifier_url()
        model_dir = self.config.get_discovery_model_dir()
        try:
            discovery_ensure_classifier_model(url=resolved_url, target_dir=model_dir)
        except ValueError as exc:
            click.secho(f'❌ {exc}', fg='red')
            sys.exit(1)

        console.print()
        console.print(
            f'  [cyan]Loading classifier[/cyan] '
            f'[dim](device={device}, threshold={threshold})[/dim]'
        )

        effective_device = None if device == 'auto' else device
        quantize = False if no_quantize else None

        # ── Noise suppression during model load ──────────────────────────
        # In quiet mode (the default) we mute four overlapping sources of
        # pre-stream output so the progress bar has a clean canvas to paint on:
        # 1. HF hub's own tqdm download progress.
        # 2. transformers' own logger (the benign LOAD REPORT from our last turn).
        # 3. Python deprecation warnings such as the `torch_dtype` / `use_return_dict`
        #    ones emitted from inside transformers.
        # 4. The `hf_xet` Rust extension's unauthenticated-requests warning,
        #    which writes directly to fd 2 and escapes Python-level capture.
        #    We dodge it by putting transformers into offline mode whenever the
        #    base model is already in the local HF cache — no Hub call, no
        #    Rust code path, no warning. On a first-time load the warning will
        #    appear once and subsequent runs are clean.
        # As a last-line defence we also redirect stdout/stderr into a buffer so
        # any Python-level C-extension noise is captured. The buffer is surfaced
        # on error so we never eat a real stack trace.
        load_noise = io.StringIO()

        def _maybe_enable_offline_mode() -> None:
            """Set ``HF_HUB_OFFLINE=1`` if the base model is already cached."""
            try:
                import json as _json
                from huggingface_hub import try_to_load_from_cache

                adapter_cfg = Path(model_dir) / 'adapter_config.json'
                if not adapter_cfg.exists():
                    return
                with adapter_cfg.open(encoding='utf-8') as _f:
                    base_ckpt = _json.load(_f).get('base_model_name_or_path')
                if not base_ckpt:
                    return
                cached = try_to_load_from_cache(
                    repo_id=base_ckpt, filename='config.json'
                )
                if cached is not None:
                    _os.environ['HF_HUB_OFFLINE'] = '1'
                    _os.environ['TRANSFORMERS_OFFLINE'] = '1'
            except Exception:
                # Best effort — on failure we just accept the one-time warning.
                pass

        def _load_classifier():
            return discovery_load_classifier(
                model_dir=model_dir,
                device=effective_device,
                quantize=quantize,
                max_length=max_length,
            )

        try:
            if verbose:
                classifier = _load_classifier()
            else:
                import logging as _logging

                _os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
                _os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')
                _maybe_enable_offline_mode()
                # huggingface_hub prints a plain warning about sending
                # unauthenticated requests via its own logger — silence it
                # alongside the transformers one.
                for _name in ('huggingface_hub', 'transformers', 'accelerate', 'peft'):
                    _logging.getLogger(_name).setLevel(_logging.ERROR)

                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=DeprecationWarning)
                    warnings.filterwarnings('ignore', category=FutureWarning)
                    warnings.filterwarnings('ignore', category=UserWarning)
                    try:
                        import transformers.utils.logging as _tf_log

                        _tf_log.set_verbosity_error()
                    except Exception:
                        pass
                    with (
                        contextlib.redirect_stdout(load_noise),
                        contextlib.redirect_stderr(load_noise),
                    ):
                        classifier = _load_classifier()
        except Exception:
            if not verbose and load_noise.getvalue():
                sys.stderr.write(load_noise.getvalue())
            raise

        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if not resume:
            for suffix in ('.cursor', '.ids'):
                p = Path(str(out_path) + suffix)
                if p.exists():
                    p.unlink()

        columns = list(DISCOVERY_STREAM_CSV_BASE_COLUMNS)
        if include_title:
            columns.append('title')

        # When resuming into an existing CSV, seed seen_ids from the rows
        # already written so that deleting the .ids sidecar by hand still
        # produces correct skip behaviour.
        existing_short_ids: set[str] = set()
        write_header = True
        if resume and out_path.exists() and out_path.stat().st_size > 0:
            write_header = False
            try:
                with out_path.open('r', encoding='utf-8', newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        sid = row.get('openalex_id')
                        if sid:
                            existing_short_ids.add(sid)
            except Exception:
                # Corrupt or partial CSV — fall back to cursor/ids sidecars.
                pass

        state = DiscoveryFetchState(out_path)

        # Surface the resume state so the user understands whether they are
        # starting fresh or picking up mid-traversal. Computed here before we
        # enter the fetch loop so we can show it on the first frame.
        preload_state = DiscoveryFetchState(out_path)
        preload_state.load()
        resume_seen_count = len(preload_state.seen_ids) + len(
            existing_short_ids - {
                _short.rsplit('/', 1)[-1] for _short in preload_state.seen_ids
            }
        )
        if resume and (resume_seen_count > 0 or preload_state.cursor != '*'):
            cursor_preview = (
                (preload_state.cursor[:32] + '…')
                if len(preload_state.cursor) > 32
                else preload_state.cursor
            )
            console.print(
                f'  [dim]Resuming:[/dim] '
                f'[bold]{resume_seen_count}[/bold] seen IDs, '
                f'cursor [dim]{cursor_preview}[/dim]'
            )

        progress = Progress(
            TextColumn('  [bold cyan]Streaming[/bold cyan]'),
            BarColumn(bar_width=None),
            TextColumn('[bold]{task.completed}[/bold]/{task.total}'),
            TextColumn('• flagged [bold green]{task.fields[flagged]}[/bold green]'),
            TimeRemainingColumn(),
            console=console,
            expand=True,
        )

        written = 0
        flagged_count = 0
        scored_count = 0

        console.print()
        try:
            with (
                progress,
                state,
                out_path.open('a', encoding='utf-8', newline='') as csv_fp,
            ):
                task_id = progress.add_task('stream', total=limit, flagged=0)

                # Extend seen_ids with any short IDs recovered from the existing
                # CSV, so the fetcher skips them even if the .ids sidecar is gone.
                if existing_short_ids:
                    for short in existing_short_ids:
                        state.seen_ids.add(f'https://openalex.org/{short}')

                writer = csv.DictWriter(csv_fp, fieldnames=columns)
                if write_header:
                    writer.writeheader()
                    csv_fp.flush()

                for record in discovery_openalex_fetch_works(
                    state=state,
                    limit=limit,
                    year_lower_bound=year_min,
                ):
                    score = classifier.predict_score(
                        title=record.get('title'),
                        abstract=record.get('abstract'),
                    )
                    scored_count += 1
                    row = discovery_build_csv_row(
                        record=record,
                        score=score,
                        threshold=threshold,
                        include_title=include_title,
                    )
                    is_flagged = bool(row['flagged'])
                    if is_flagged or not flagged_only:
                        writer.writerow(row)
                        csv_fp.flush()
                        written += 1

                    if is_flagged:
                        flagged_count += 1
                        title_display = (record.get('title') or '').strip()
                        if len(title_display) > 72:
                            title_display = title_display[:71] + '…'
                        # Toast printed *above* the pinned progress bar.
                        # Rich's Progress auto-rewrites the bar after the print.
                        progress.console.print(
                            f'  [yellow]{float(score):.3f}[/yellow]  '
                            f'[dim]{row["openalex_id"]:<14}[/dim]  '
                            f'{title_display}',
                            highlight=False,
                        )

                    progress.update(task_id, advance=1, flagged=flagged_count)

        except KeyboardInterrupt:
            console.print()
            console.print(
                '  [yellow]Interrupted — partial results have been flushed to disk.[/yellow]'
            )

        written_label = 'flagged-only' if flagged_only else 'all rows'
        console.print()
        console.print(
            Panel(
                (
                    f'Scored:  [bold]{scored_count}[/bold]\n'
                    f'Written: [bold]{written}[/bold]  [dim]({written_label})[/dim]\n'
                    f'Flagged: [bold green]{flagged_count}[/bold green]  '
                    f'[dim](threshold {threshold})[/dim]\n'
                    f'CSV:     [dim]{out_path}[/dim]'
                ),
                title='[bold green]Stream complete[/bold green]',
                title_align='left',
                style='green',
                expand=False,
                padding=(1, 2),
            )
        )

        # If the run found nothing and we were resuming, the most common
        # cause is stale sidecar state that has run past the useful tail of
        # OpenAlex. Tell the user how to start over.
        if scored_count == 0 and resume and resume_seen_count > 0:
            console.print(
                '  [yellow]No new records were fetched.[/yellow] '
                'The stored cursor may be past the end of the useful tail of '
                'OpenAlex or stuck on fully-seen pages.\n'
                f'  Try [bold]cmmanage discover stream --no-resume[/bold] to start '
                'fresh, or delete [dim]{0}.cursor[/dim] and [dim]{0}.ids[/dim].'.format(
                    out_path
                )
            )
        console.print()

    @click.command(
        'model-download',
        cls=RichHelpCommand,
        rich_help=RichDiscoveryModelDownloadHelp(),
        short_help='Download the classifier archive from Nextcloud into the user config folder.',
    )
    @click.option(
        '--model-url',
        type=str,
        default=None,
        help='Override the URL configured under `[discovery] classifier_url`.',
    )
    @click.option(
        '--force/--no-force',
        default=False,
        help='Re-download even when the classifier is already cached locally.',
    )
    @click.pass_obj
    def discovery_model_download_command(
        self,
        model_url: Optional[str],
        force: bool,
    ) -> None:
        from rich.console import Console

        console = Console()
        resolved_url = model_url or self.config.get_discovery_classifier_url()
        target = self.config.get_discovery_model_dir()

        try:
            path = discovery_ensure_classifier_model(
                url=resolved_url, target_dir=target, force=force
            )
        except ValueError as exc:
            click.secho(f'❌ {exc}', fg='red')
            sys.exit(1)
        except Exception as exc:
            click.secho(f'❌ Download failed: {exc}', fg='red')
            sys.exit(1)

        console.print(f'[green]Classifier ready at {path}[/green]')

    @click.command(
        'model-info',
        cls=RichHelpCommand,
        rich_help=RichDiscoveryModelInfoHelp(),
        short_help='Show the classifier cache location and ML dependency status.',
    )
    @click.pass_obj
    def discovery_model_info_command(self) -> None:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        console = Console()
        model_dir = Path(self.config.get_discovery_model_dir())
        url = self.config.get_discovery_classifier_url()
        adapter_present = (model_dir / 'adapter_config.json').exists()
        ml_ok = discovery_is_ml_stack_available()

        table = Table(
            show_header=False,
            box=None,
            padding=(0, 2, 0, 0),
            expand=True,
        )
        table.add_column('label', style='dim', no_wrap=True)
        table.add_column('value', overflow='fold', ratio=1)

        table.add_row('Model dir:', str(model_dir))
        table.add_row(
            'Adapter:',
            Text(
                'present' if adapter_present else 'missing (run model-download)',
                style='green' if adapter_present else 'yellow',
            ),
        )
        table.add_row(
            'Classifier URL:',
            Text(url or '<unset>', style='' if url else 'yellow'),
        )
        table.add_row(
            'ML stack:',
            Text(
                'torch/transformers/peft OK'
                if ml_ok
                else 'missing (pip install chem_mat_database[dev])',
                style='green' if ml_ok else 'red',
            ),
        )

        console.print(
            Panel(
                table,
                title='[bold]Discovery model info[/bold]',
                title_align='left',
                style='bright_blue',
                expand=True,
                padding=(1, 2),
            )
        )

    ## == DOCS COMMAND GROUP ==
    # This command group is used to manage the documentation of the package.

    @click.group('docs', help='Commands for managing the documentation of the package.')
    @click.pass_obj
    def docs_group(
        self,
    ) -> None:
        """
        This command group contains commands that can be used to manage the documentation of the package.
        This includes commands to build the documentation, open it in a browser, or edit it.
        """
        pass

    @click.command(
        'collect-datasets',
        help='Compiles the datasets for the documentation from the metadata file.',
    )
    @click.option(
        '--metadata-path',
        help='The path to the metadata file to use for compiling the datasets.',
        default=METADATA_PATH,
        show_default=True,
    )
    @click.option(
        '--output-path',
        help='The path to the output file where the compiled datasets will be written.',
        default=os.path.join(DOCS_PATH, 'datasets.md'),
        show_default=True,
    )
    @click.pass_obj
    def docs_collect_datasets_command(
        self,
        metadata_path: str,
        output_path: str,
    ) -> None:
        """
        This command will dynamically create the overview of the datasets that are available in the
        remote file share server using the metadata.yml file. By default, this will collect the dataset information
        from the results of the experiments that have been executed the most recently.
        """

        ## -- Getting Metadata --
        # This dictionary contains all the metadata information stored on the remote file share server.
        click.secho('fetching metadata from remote fileshare server...')
        metadata: dict = self.file_share.fetch_metadata(force=True)

        ## -- Creating Table --
        # Using prettytable to create the table in the markdown format
        click.secho('compiling datasets overview...')
        table = PrettyTable()
        table.align = 'l'
        table.field_names = [
            'Name',
            'Category',
            'Description',
            'No. Elements',
            'Target Type',
        ]
        for dataset_name, dataset_info in metadata['datasets'].items():
            table.add_row(
                [
                    f'**{dataset_name}**',
                    dataset_info.get('category', 'organic'),
                    dataset_info.get('verbose', '-'),
                    str(dataset_info.get('compounds', 0)),
                    ', '.join(dataset_info.get('target_type', [])),
                ]
            )

        # This is important to set the style to markdown so that the string that is created from this is
        # actually a valid markdown syntax.
        table.set_style(TableStyle.MARKDOWN)
        datasets_table: str = table.get_string()

        ## -- Writing the Docs File --
        # Here we use the jinja2 template to actually render the content of the documentation file
        # mainly centered around the just created table of the datasets.
        click.secho('writing datasets overview to file...')
        template = TEMPLATE_ENV.get_template('docs_dataset.md.j2')
        content = template.render(
            file_share_url=self.file_share.url,
            datasets_table=datasets_table,
        )
        with open(output_path, 'w') as file:
            file.write(content)

        click.secho('✅ datasets overview written to file', fg='green')
        click.secho('')

    @click.command(
        'compile',
        cls=RichHelpCommand,
        rich_help=RichDocsCompileHelp(),
        short_help='Compile dataset metadata into JSON for the landing page.',
    )
    @click.option(
        '--output-path',
        default=os.path.join(PAGES_PATH, 'datasets.json'),
        show_default=True,
        help='The path to the output JSON file.',
    )
    @click.pass_obj
    def docs_compile_command(
        self,
        output_path: str,
    ) -> None:
        from datetime import datetime, timezone
        from rich.console import Console
        from rich.panel import Panel

        console = Console()

        console.print()
        console.print(
            Panel(
                '[bold]Compile Dataset Metadata[/bold]\n'
                f'[dim]Output: {output_path}[/dim]',
                style='bright_blue',
                expand=False,
            )
        )
        console.print()

        # Step 1: Fetch metadata from remote
        with console.status(
            '[bold cyan]Fetching metadata from remote server...', spinner='dots'
        ):
            try:
                metadata: dict = self.file_share.fetch_metadata(force=True)
            except Exception as e:
                console.print(
                    '[red]  [bold]\\[1/3][/bold] Failed to fetch metadata[/red]'
                )
                console.print(Panel(str(e), title='Error', style='red'))
                return

        num_total = len(metadata.get('datasets', {}))
        console.print(
            f'[green]  [bold]\\[1/3][/bold] Fetched metadata ({num_total} datasets found)[/green]'
        )

        # Step 2: Transform metadata into JSON structure
        with console.status(
            '[bold cyan]Transforming dataset metadata...', spinner='dots'
        ):
            datasets = []
            for dataset_name, dataset_info in metadata.get('datasets', {}).items():
                # Skip internal/test datasets prefixed with underscore
                if dataset_name.startswith('_'):
                    continue

                # Build a clean, fallback-safe dataset object
                verbose_fallback = dataset_name.replace('_', ' ').title()
                datasets.append(
                    {
                        'name': dataset_name,
                        'verbose': dataset_info.get('verbose', verbose_fallback),
                        'category': dataset_info.get('category', 'organic'),
                        'compounds': dataset_info.get('compounds', 0),
                        'targets': dataset_info.get('targets', 0),
                        'target_type': dataset_info.get('target_type', []),
                        'description': dataset_info.get('description', ''),
                        'tags': dataset_info.get('tags', []),
                        'sources': dataset_info.get('sources', []),
                        'target_descriptions': dataset_info.get(
                            'target_descriptions', {}
                        ),
                        'raw': dataset_info.get('raw', []),
                        'min_version': dataset_info.get('min_version', None),
                        'notes': dataset_info.get('notes', []),
                    }
                )

            # Collect unique categories for the summary
            categories = sorted(set(d['category'] for d in datasets))

        console.print(
            f'[green]  [bold]\\[2/3][/bold] Transformed {len(datasets)} public datasets[/green]'
        )

        # Step 3: Write JSON file
        with console.status('[bold cyan]Writing datasets.json...', spinner='dots'):
            # Construct the download base URL so the landing page can link directly to raw files
            download_base = f'{self.file_share.base_url}/public.php/dav/files/{self.file_share.share_token}'

            output = {
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'version': get_version(),
                'download_base': download_base,
                'datasets': datasets,
            }

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)

        console.print(f'[green]  [bold]\\[3/3][/bold] Written to {output_path}[/green]')

        console.print()
        console.print(
            Panel(
                f'[bold green]Compilation complete![/bold green]\n\n'
                f'  Datasets:    {len(datasets)}\n'
                f'  Categories:  {", ".join(categories)}\n'
                f'  Output:      {output_path}\n\n'
                f"[dim]Run 'cmmanage docs deploy' to publish the updated site.[/dim]",
                style='green',
                expand=False,
            )
        )

    @click.command(
        'deploy', short_help='Build and deploy the documentation site to GitHub Pages.'
    )
    @click.option(
        '--preview',
        is_flag=True,
        help='Build the site locally and open it in the browser without deploying.',
    )
    @click.option(
        '--port',
        default=8000,
        show_default=True,
        help='The port to use for the local preview server.',
    )
    @click.option(
        '--build-dir',
        default=os.path.join(PROJECT_PATH, '_site'),
        show_default=True,
        help='The directory to build the site into.',
    )
    @click.pass_obj
    def docs_deploy_command(
        self,
        preview: bool,
        port: int,
        build_dir: str,
    ) -> None:
        """
        Build and deploy the full documentation site to GitHub Pages. This command performs all
        the steps required for a complete deployment: building the MkDocs documentation, copying
        the landing page, and pushing the result to the ``gh-pages`` branch.

        The site is composed of two parts: a custom landing page served at the root (``/``) and
        the MkDocs documentation served under ``/docs/``.

        Use ``--preview`` to build the site locally and open it in the browser for inspection
        without pushing anything to the remote.
        """
        import shutil
        import subprocess
        import webbrowser
        import http.server

        from rich.console import Console
        from rich.panel import Panel

        console = Console()
        mkdocs_dir = os.path.join(build_dir, 'docs')
        pages_dir = PAGES_PATH
        project_dir = PROJECT_PATH
        errors = []

        console.print()
        mode_label = 'Preview' if preview else 'Deploy'
        console.print(
            Panel(
                f'[bold]GitHub Pages {mode_label}[/bold]\n'
                f'[dim]Building site into {build_dir}[/dim]',
                style='bright_blue',
                expand=False,
            )
        )
        console.print()

        # Step 1: Clean previous build
        with console.status('[bold cyan]Cleaning previous build...', spinner='dots'):
            if os.path.exists(build_dir):
                shutil.rmtree(build_dir)
            os.makedirs(build_dir, exist_ok=True)
        console.print('[green]  [bold]\\[1/4][/bold] Cleaned build directory[/green]')

        # Step 2: Build MkDocs
        with console.status(
            '[bold cyan]Building MkDocs documentation...', spinner='dots'
        ):
            result = subprocess.run(
                ['mkdocs', 'build', '-d', mkdocs_dir],
                cwd=project_dir,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                errors.append(f'mkdocs build failed:\n{result.stderr}')

        if errors:
            console.print('[red]  [bold]\\[2/4][/bold] MkDocs build failed[/red]')
            console.print(Panel(errors[-1], title='Error', style='red'))
            return

        console.print(
            '[green]  [bold]\\[2/4][/bold] Built MkDocs documentation[/green]'
        )

        # Step 3: Copy landing page
        with console.status('[bold cyan]Copying landing page...', spinner='dots'):
            if not os.path.isdir(pages_dir):
                errors.append(f'Landing page directory not found: {pages_dir}')
            else:
                for item in os.listdir(pages_dir):
                    src = os.path.join(pages_dir, item)
                    dst = os.path.join(build_dir, item)
                    if os.path.isdir(src):
                        shutil.copytree(src, dst, dirs_exist_ok=True)
                    else:
                        shutil.copy2(src, dst)

                # Add .nojekyll to prevent GitHub Pages Jekyll processing
                nojekyll_path = os.path.join(build_dir, '.nojekyll')
                with open(nojekyll_path, 'w') as f:
                    pass

        if errors:
            console.print('[red]  [bold]\\[3/4][/bold] Landing page copy failed[/red]')
            console.print(Panel(errors[-1], title='Error', style='red'))
            return

        console.print('[green]  [bold]\\[3/4][/bold] Copied landing page[/green]')

        # Step 4: Deploy or preview
        if preview:
            console.print(
                f'[green]  [bold]\\[4/4][/bold] Starting preview server on port {port}[/green]'
            )
            console.print()

            url = f'http://localhost:{port}'
            console.print(
                Panel(
                    f'[bold green]Preview server running[/bold green]\n\n'
                    f'  Landing page:   [link={url}]{url}[/link]\n'
                    f'  Documentation:  [link={url}/docs/]{url}/docs/[/link]\n\n'
                    f'[dim]Press Ctrl+C to stop the server[/dim]',
                    style='green',
                    expand=False,
                )
            )

            webbrowser.open(url)

            handler = lambda *args, **kwargs: http.server.SimpleHTTPRequestHandler(
                *args, directory=build_dir, **kwargs
            )
            server = http.server.HTTPServer(('localhost', port), handler)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                server.shutdown()
                console.print('\n[dim]Preview server stopped.[/dim]')
        else:
            with console.status('[bold cyan]Deploying to gh-pages...', spinner='dots'):
                result = subprocess.run(
                    ['ghp-import', '-n', '-p', '-f', build_dir],
                    cwd=project_dir,
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    errors.append(f'ghp-import failed:\n{result.stderr}')

            if errors:
                console.print('[red]  [bold]\\[4/4][/bold] Deployment failed[/red]')
                console.print(Panel(errors[-1], title='Error', style='red'))

                # Check if ghp-import is installed
                ghp_check = shutil.which('ghp-import')
                if not ghp_check:
                    console.print()
                    console.print(
                        Panel(
                            '[bold yellow]ghp-import is not installed.[/bold yellow]\n\n'
                            'Install it with: [bold]pip install ghp-import[/bold]',
                            style='yellow',
                            expand=False,
                        )
                    )
                return

            console.print('[green]  [bold]\\[4/4][/bold] Deployed to gh-pages[/green]')
            console.print()
            console.print(
                Panel(
                    '[bold green]Deployment complete![/bold green]\n\n'
                    '  The site should be live shortly at:\n'
                    '  [link=https://the16thpythonist.github.io/chem_mat_data/]https://the16thpythonist.github.io/chem_mat_data/[/link]',
                    style='green',
                    expand=False,
                )
            )

        # Clean up build directory after deploy (not for preview, user may want to inspect)
        if not preview and os.path.exists(build_dir):
            shutil.rmtree(build_dir)


@click.group(cls=CLI)
@click.option('-v', '--version', is_flag=True, help='Print the version of the package')
@click.option(
    '--no-verify',
    is_flag=True,
    help='turn off ssl verification for the fileshare connection',
)
@click.pass_context
def cli(
    ctx: Any,
    version: bool,
    no_verify: bool,
) -> None:

    # 07.06.24
    # This is actually required to make the CLI class work as it is currently implemented.
    # The problem is that all the commands which are also methods need to be decorated with the
    # @pass_obj in the position of the "self" argument and here we set the object to be passed
    # as the CLI instance to make that happen.
    ctx.obj = ctx.command
    ctx.obj.file_share.verfiy = not no_verify

    if version:
        version = get_version()
        click.echo(Text(version, style=Style(bold=True)))
        sys.exit(0)

    # 07.06.24
    # There was previously a bug here which caused the help text to be rendered for every
    # command. After adding the additional condition "not ctx.invoked_subcommand" this was
    # fixed.

    # This section will implement the small convenience feature that if no arguments are passed
    # to the CLI at all, it will simply show the help information.
    elif not ctx.args and not ctx.invoked_subcommand:
        ctx.command.format_help(ctx, ctx.formatter_class())


if __name__ == '__main__':
    cli()  # pragma: no cover
