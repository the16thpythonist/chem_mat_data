"""
Per-stage gates for the dataset extraction pipeline.

Each of the six stages ends at a gate: a set of post-conditions that must hold before
the next stage may begin. The gates live in Python rather than in the skill's prose
for the same reason the final verification does - they are the points where a
plausible-looking result is separated from a correct one, and that judgement should
not depend on how invested an agent is in its own work.

The checks are deliberately concrete. A gate does not ask whether the agent believes
the stage went well; it asks whether the downloaded file is HTML rather than CSV,
whether ``Experiment.extend`` was given an absolute path, whether the script reaches
back into the scratch directory instead of downloading its own data, and whether the
compound count survived the full run.

Every gate returns a :class:`~chem_mat_data.agent.verification.VerificationReport`, so
errors and warnings are reported uniformly and a caller can act on the distinction.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

import yaml

from chem_mat_data.utils import PROJECT_PATH
from chem_mat_data.agent.manifest import (
    Manifest,
    STAGES,
    work_dir_for,
)
from chem_mat_data.agent.verification import (
    ERROR,
    WARNING,
    VerificationIssue,
    VerificationReport,
    normalize_target_description_keys,
    stream_target_statistics,
    verify_results,
)


#: Licenses that permit redistribution with at most attribution. Anything outside this
#: set is escalated to a human rather than decided by the pipeline.
PERMISSIVE_LICENSES: set = {
    'cc0',
    'cc-0',
    'cc0-1.0',
    'publicdomain',
    'public-domain',
    'pd',
    'unlicense',
    'cc-by',
    'cc-by-1.0',
    'cc-by-2.0',
    'cc-by-2.5',
    'cc-by-3.0',
    'cc-by-4.0',
    'mit',
    'bsd',
    'bsd-2-clause',
    'bsd-3-clause',
    'apache',
    'apache-2.0',
    'odc-by',
}

#: Values of ``redistribution`` that the gate reads as "yes".
REDISTRIBUTION_ALLOWED: tuple = ('allowed', 'yes', 'permitted')

#: Values of ``redistribution`` that the gate reads as "no". Recognised explicitly so
#: that a stated refusal is reported as a refusal, rather than as a formatting problem.
REDISTRIBUTION_DENIED: tuple = (
    'forbidden',
    'no',
    'denied',
    'prohibited',
    'restricted',
    'unclear',
    'unknown',
)

#: Names of files in the downloads folder that are documentation rather than data.
NON_DATA_NAMES: tuple = ('readme', 'download_recipe', 'license', 'licence', 'manifest')

#: Byte prefixes that indicate an HTML page was saved instead of a data file.
HTML_MARKERS: tuple = (b'<!doctype html', b'<html', b'<!DOCTYPE HTML')

#: Drop rate above which the execute gate warns.
DROP_RATE_WARN: float = 0.02

#: Drop rate above which the execute gate escalates, unless overridden per dataset.
DROP_RATE_ERROR: float = 0.10


def normalize_license(identifier: str) -> str:
    """
    Reduce a license identifier to a comparable form.

    Strips punctuation and casing differences so that ``CC BY 4.0``, ``cc-by-4.0``
    and ``CC_BY_4.0`` all compare equal.

    :param identifier: The raw license string.

    :returns: The normalised identifier.
    """
    text = identifier.strip().lower()
    text = re.sub(r'\s+', '-', text)
    text = re.sub(r'[_\s]+', '-', text)
    text = re.sub(r'-+', '-', text)
    return text.strip('-')


def looks_like_identifier(value: Optional[str]) -> bool:
    """
    Return whether a value is a license identifier rather than a sentence about one.

    The gate matches license fields against a fixed list, so prose in one of them is
    not merely untidy - it is unreadable to the check. "No open license; EPA grants
    only personal use" and "CC0-1.0 applies to the whole record" both fail
    :func:`is_permissive`, the first correctly and the second not, and the gate cannot
    tell those two cases apart. Detecting prose lets it say so instead of silently
    treating every sentence as a restriction.

    The test is deliberately loose: identifiers are short and wordless, descriptions
    are neither.

    :param value: The candidate string.

    :returns: True when the value is shaped like an identifier.
    """
    if not value:
        return False

    text = str(value).strip()
    if len(text) > 40:
        return False
    if len(text.split()) > 4:
        return False
    # Sentence punctuation is the giveaway that survives length limits.
    return not re.search(r'[.;:!?]\s', text)


def is_permissive(identifier: Optional[str]) -> bool:
    """
    Return whether a license permits redistribution under the project's policy.

    The policy is deliberately strict: public domain, CC-BY and the common permissive
    software licenses pass. NonCommercial, ShareAlike, copyleft, and anything not
    clearly identified escalate to a human.

    :param identifier: The license identifier, or ``None``.

    :returns: True when the license is on the permissive list.
    """
    if not identifier:
        return False

    normalized = normalize_license(identifier)
    if normalized in PERMISSIVE_LICENSES:
        return True

    # A trailing version on an otherwise permissive family, e.g. 'cc-by-sa-4.0' must
    # NOT match 'cc-by', so only strip a numeric version suffix.
    stripped = re.sub(r'-\d+(\.\d+)*$', '', normalized)
    return stripped in PERMISSIVE_LICENSES


def _host(url: Optional[str]) -> Optional[str]:
    """
    Return the lowercased hostname of a URL, without any ``www.`` prefix.

    :param url: The URL, or ``None``.

    :returns: The hostname, or ``None`` when the value is absent or not a URL.
    """
    if not url or not str(url).startswith(('http://', 'https://')):
        return None

    host = urlparse(str(url)).netloc.lower()
    return host[4:] if host.startswith('www.') else host or None


def _data_files(downloads_dir: Path) -> List[Path]:
    """
    Return the downloaded files that are actual data rather than documentation.

    :param downloads_dir: The downloads folder.

    :returns: Sorted list of data file paths.
    """
    if not downloads_dir.exists():
        return []

    return sorted(
        path
        for path in downloads_dir.rglob('*')
        if path.is_file() and not is_documentation_name(path)
    )


def is_documentation_name(path: Path) -> bool:
    """
    Return whether a filename marks the file as documentation rather than data.

    The whole stem must be the name: ``LICENSE`` and ``README.md`` are documentation,
    ``license_compounds.csv`` is data about licensed compounds. A substring test
    excluded the latter, so the gate rejected legitimate downloads while accepting a
    file containing the words "Not Found".

    Erring towards *including* a file is safe here. A documentation file that slips
    through is checked for emptiness and for being a saved web page, which it passes,
    and the directory-level table check only requires that *something* present is
    tabular.

    :param path: The downloaded file.

    :returns: True when the name marks it as documentation.
    """
    return path.stem.lower() in NON_DATA_NAMES


def parses_as_table(path: Path, max_bytes: int = 1_000_000) -> bool:
    """
    Return whether a file reads as a table with at least one data row.

    Only the first ``max_bytes`` are examined. The failures this separates - an error
    page saved as a CSV, a JSON record dump, a header row with nothing under it - all
    show up immediately, and reading a multi-gigabyte archive in full to prove it is
    data would cost more than the check is worth.

    Deliberately permissive about shape: a single-column list of SMILES is a legitimate
    dataset, so no minimum column count is imposed. A file too large to have been read
    in full, or in a format pandas cannot sniff, is *not* treated as a failure - the
    question asked here is only whether at least one file in the directory is plainly
    tabular.

    :param path: The file to inspect.
    :param max_bytes: How much of the file to read.

    :returns: True when the prefix parses as a table with a data row.
    """
    try:
        with open(path, 'rb') as file:
            prefix = file.read(max_bytes)
    except OSError:
        return False

    if not prefix.strip():
        return False

    text = prefix.decode('utf-8', errors='replace')
    # A partial read almost always ends mid-line; dropping the tail avoids rejecting a
    # good file over a truncated final row.
    if len(prefix) == max_bytes and '\n' in text:
        text = text[: text.rindex('\n')]

    stripped = text.lstrip()
    if stripped.startswith(('{', '[', '<')):
        return False

    try:
        import pandas as pd
        from io import StringIO

        frame = pd.read_csv(
            StringIO(text), sep=None, engine='python', nrows=8, on_bad_lines='skip'
        )
    except Exception:
        return False

    return len(frame) >= 1 and len(frame.columns) >= 1


def _looks_like_html(path: Path) -> bool:
    """
    Return whether a file begins with an HTML document marker.

    Saving a repository landing page instead of the file behind it is the single most
    common way the prepare stage fails while appearing to succeed.

    :param path: The file to inspect.

    :returns: True when the file starts with an HTML marker.
    """
    try:
        with path.open('rb') as file:
            head = file.read(1024).lstrip().lower()
    except OSError:
        return False

    return any(head.startswith(marker.lower()) for marker in HTML_MARKERS)


def gate_prepare(manifest: Manifest) -> VerificationReport:
    """
    Check that the prepare stage produced usable raw data.

    Verifies that data files were downloaded, that none of them is a saved HTML page,
    and that a download recipe recording the direct URLs exists for the script stage
    to build self-contained download logic from.

    :param manifest: The run manifest.

    :returns: The gate report.
    """
    work = work_dir_for(manifest.slug)
    report = VerificationReport(results_dir=work)

    if not manifest.source_url:
        report.add(ERROR, 'no source_url recorded in the manifest')

    downloads = manifest.downloads_dir
    if not downloads.exists():
        report.add(ERROR, f'no downloads directory at {downloads}')
        return report

    data_files = _data_files(downloads)
    if not data_files:
        excluded = sorted(
            path.name for path in downloads.rglob('*') if path.is_file()
        )
        report.add(
            ERROR,
            'no data files in the downloads directory - everything present was treated '
            f'as documentation: {excluded or "the directory is empty"}',
        )
    else:
        for path in data_files:
            if path.stat().st_size == 0:
                report.add(ERROR, f'downloaded file is empty: {path.name}')
            elif _looks_like_html(path):
                report.add(
                    ERROR,
                    f'{path.name} is an HTML page, not a data file - a landing page '
                    'was saved instead of the file behind it',
                )

        # Asked of the directory rather than of each file, so that ancillary JSON
        # records and notes alongside the real data stay welcome. What it rules out is
        # a downloads folder where *nothing* is tabular: an error page saved as a CSV,
        # a repository record dump, a header row with no rows under it.
        if not any(parses_as_table(path) for path in data_files):
            names = ', '.join(path.name for path in data_files)
            report.add(
                ERROR,
                f'nothing in the downloads directory parses as a table with data rows '
                f'({names}). A saved error page, a repository record or a header-only '
                'file looks like a download but has nothing in it.',
            )

    recipe = downloads / 'download_recipe.md'
    if not recipe.exists():
        report.add(
            ERROR,
            'no download_recipe.md - the script stage needs the direct URLs to build '
            'self-contained download logic',
        )
    else:
        text = recipe.read_text(errors='replace')
        if not text.strip():
            report.add(ERROR, 'download_recipe.md is empty')
        else:
            # A DOI resolves to a landing page, not to a file, so a recipe containing
            # only DOIs records nothing the script stage can download from - and
            # "records a URL" was previously satisfied by the substring "http".
            hosts = {
                _host(url)
                for url in re.findall(r'https?://[^\s<>)"\']+', text)
            }
            downloadable = {
                host
                for host in hosts
                if host and host not in ('doi.org', 'dx.doi.org')
            }
            if not hosts:
                report.add(ERROR, 'download_recipe.md records no URL')
            elif not downloadable:
                report.add(
                    ERROR,
                    'download_recipe.md records only DOI links, which resolve to '
                    'landing pages. The script stage needs the direct file URL it '
                    'should download from.',
                )

    ## -- is the schema documented anywhere? --
    # A README is one way; the recipe describing the columns is another, and in
    # practice a better one. Warning about a missing README while the agent's own
    # recipe documents every column is noise that trains the reader to skim.
    readme_found = any(
        'readme' in path.name.lower() for path in downloads.rglob('*') if path.is_file()
    )
    schema_documented = readme_found
    if not schema_documented and recipe.exists():
        recipe_text = recipe.read_text(errors='replace').lower()
        schema_documented = any(
            marker in recipe_text
            for marker in ('schema', 'column', 'field', 'units', 'unit)')
        )

    if not schema_documented:
        report.add(
            WARNING,
            'no schema documentation captured - neither a README nor a description of '
            'the columns in download_recipe.md. Column meanings and units will have to '
            'be inferred, which is where units go wrong.',
        )

    return report


def gate_metadata(manifest: Manifest) -> VerificationReport:
    """
    Check that the metadata stage assembled a complete, well-formed draft.

    Only the human-authored fields are checked here. The derived fields (``compounds``,
    ``targets``, ``target_type``) are produced by the base template at save time and
    are checked by the execute and verification gates instead.

    :param manifest: The run manifest.

    :returns: The gate report.
    """
    report = VerificationReport(results_dir=work_dir_for(manifest.slug))
    draft = manifest.metadata_draft or {}

    if not draft:
        report.add(ERROR, 'no metadata_draft recorded in the manifest')
        return report

    dataset_name = draft.get('dataset_name')
    if not dataset_name:
        report.add(ERROR, 'metadata_draft is missing "dataset_name"')
    elif not re.fullmatch(r'[a-z0-9][a-z0-9_]*', str(dataset_name)):
        report.add(
            ERROR,
            f'"dataset_name" must be lowercase snake_case: {dataset_name!r}',
        )

    dataset_type = draft.get('dataset_type')
    if dataset_type not in ('regression', 'classification'):
        report.add(
            ERROR,
            f'"dataset_type" must be "regression" or "classification", '
            f'got {dataset_type!r}',
        )

    description = str(draft.get('description', '')).strip()
    if not description:
        report.add(ERROR, 'metadata_draft is missing "description"')
    elif len(description) < 120:
        report.add(
            WARNING,
            f'"description" is only {len(description)} characters - it should say '
            'what was measured, on what, by what method, and what it is used for',
        )

    if not str(draft.get('verbose', '')).strip():
        report.add(ERROR, 'metadata_draft is missing "verbose" (display name)')

    tags = draft.get('tags') or []
    if not tags:
        report.add(ERROR, 'metadata_draft is missing "tags"')
    elif 'Molecules' not in tags:
        report.add(WARNING, 'tags do not include "Molecules"')

    sources = draft.get('sources') or []
    if not sources:
        report.add(ERROR, 'metadata_draft is missing "sources" (provenance URLs)')

    ## -- target descriptions --
    descriptions = draft.get('target_descriptions') or {}
    target_count = draft.get('target_count')

    if not descriptions:
        report.add(ERROR, 'metadata_draft is missing "target_descriptions"')
    elif not isinstance(descriptions, dict):
        report.add(ERROR, '"target_descriptions" is not a mapping')
    else:
        normalized = normalize_target_description_keys(descriptions)
        if not normalized:
            report.add(ERROR, '"target_descriptions" has no usable integer keys')
        else:
            expected = (
                target_count if isinstance(target_count, int) else len(normalized)
            )
            missing = [index for index in range(expected) if index not in normalized]
            if missing:
                indices = ', '.join(str(index) for index in missing)
                report.add(
                    ERROR,
                    f'"target_descriptions" is missing index/indices {indices} '
                    f'(expected {expected} targets)',
                )

            for index, value in sorted(normalized.items()):
                text = str(value).strip()
                if not text:
                    report.add(ERROR, f'target description {index} is empty')
                elif ' - ' not in text:
                    report.add(
                        WARNING,
                        f'target description {index} does not follow '
                        f'"<column> - <meaning with units>": {text!r}',
                    )

    return report


#: Units whose scale makes a negative value impossible, matched against the target
#: description. Deliberately tiny and conservative: a false positive here teaches the
#: reader to ignore the check, which costs more than the misses. Anything that might be
#: log-transformed, a difference, or a gauge measurement is excluded by the `log` guard
#: below and by simply not being listed - most chemical targets legitimately go
#: negative.
NON_NEGATIVE_UNITS: tuple = (
    (r'\bkelvin\b|\(\s*K\s*\)', 'kelvin, an absolute temperature scale'),
    (r'\bg/mol\b|\bmolecular weight\b|\bmolar mass\b', 'a molecular mass'),
)


def check_target_plausibility(
    report: VerificationReport,
    statistics: List,
    draft: dict,
) -> None:
    """
    Check observed target values against what their descriptions claim they are.

    This is the one check that catches a *mislabelled unit*, which no comparison of
    declared against observed can: an agent confidently wrong about units writes the
    same wrong string on both sides, so the two agree and prove nothing. Physics does
    not agree with it. A target described in kelvin cannot hold negative values, and
    `elanos_bp` shipped for months describing a boiling point "in Kelvin" whose
    minimum is -88.6.

    Ranges are checked only when the metadata draft states them, and are optional by
    design. A required field would be an invitation to fabricate numbers for the many
    papers that state no range, which is worse than having no check.

    :param report: The report to add issues to.
    :param statistics: Observed :class:`TargetStatistics`, one per target index.
    :param draft: The approved metadata draft.

    :returns: Nothing; issues are added to ``report``.
    """
    descriptions = normalize_target_description_keys(
        draft.get('target_descriptions') or {}
    )
    declared = draft.get('target_ranges') or {}

    for stat in statistics:
        key = str(stat.index)
        # normalize_target_description_keys returns integer keys, whatever the draft
        # used, so look up by the index rather than by its string form.
        description = str(descriptions.get(stat.index, '') or '')

        ## -- a unit that cannot be negative, holding negative values --
        if stat.minimum is not None and stat.minimum < 0 and description:
            # A log-transformed anything is routinely negative, whatever the underlying
            # unit is, so the mention of a log takes the target out of scope entirely.
            if not re.search(r'\blog', description, re.IGNORECASE):
                for pattern, label in NON_NEGATIVE_UNITS:
                    if re.search(pattern, description, re.IGNORECASE):
                        report.add(
                            ERROR,
                            f'target {key} is described as {label} but its minimum is '
                            f'{stat.minimum:g}. Either the unit in the description is '
                            'wrong or the values are not what the description says '
                            f'they are. Description: {description!r}',
                        )
                        break

        ## -- a constant column --
        if stat.count and stat.unique == 1:
            report.add(
                ERROR,
                f'target {key} has the same value in every row ({stat.minimum:g}) - '
                'this is almost always the wrong column',
            )

        ## -- missing values --
        if stat.missing:
            report.add(
                WARNING,
                f'target {key} has {stat.missing} missing or non-finite value(s) out '
                f'of {stat.missing + stat.count}',
            )

        ## -- the declared range, when one was recorded --
        bounds = declared.get(key) or declared.get(stat.index)
        if not isinstance(bounds, dict):
            continue

        low = bounds.get('min')
        high = bounds.get('max')
        if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
            continue
        if stat.minimum is None or stat.maximum is None:
            continue

        span = float(high) - float(low)
        tolerance = abs(span) * 0.01 if span else 1e-6

        if stat.minimum < low - tolerance or stat.maximum > high + tolerance:
            report.add(
                ERROR,
                f'target {key} runs {stat.minimum:g}..{stat.maximum:g} but the source '
                f'states {low:g}..{high:g}. Values outside the published range mean '
                'the wrong column, the wrong units, or the wrong file.',
            )
        elif span and (stat.maximum - stat.minimum) < abs(span) * 0.1:
            report.add(
                WARNING,
                f'target {key} covers {stat.minimum:g}..{stat.maximum:g}, under a tenth '
                f'of the published {low:g}..{high:g} - check this is the whole dataset '
                'rather than one split',
            )


def check_experiment_record(
    report: VerificationReport,
    results_dir: Optional[str],
    script_path: Optional[str] = None,
    label: str = '',
) -> None:
    """
    Check a results folder against the record pycomex wrote about the run.

    Every other check in this module compares one thing the agent declared against
    another thing the same agent declared. This one does not: ``experiment_meta.json``
    is written by the experiment framework, so it can contradict the agent, and it is
    the only independent witness available to the gates.

    What it catches that nothing else does: a run that raised partway through still
    leaves a plausible-looking folder behind, with a message pack, a metadata file and
    sane-looking counts, because molecules that fail conversion are logged and skipped
    rather than raising. Such a folder passes every structural check. ``status`` and
    ``has_error`` say plainly that it should not.

    Comparing ``name`` against the script filename additionally catches a results
    directory left over from a different script - stale output being read as though it
    belonged to this run.

    :param report: The report to add issues to.
    :param results_dir: The results folder to inspect.
    :param script_path: Path of the script that should have produced it, if known.
    :param label: Prefix for messages, e.g. ``'smoke run'``.

    :returns: Nothing; issues are added to ``report``.
    """
    prefix = f'{label}: ' if label else ''

    if not results_dir:
        return

    meta_path = Path(results_dir) / 'experiment_meta.json'
    if not meta_path.is_file():
        report.add(
            WARNING,
            f'{prefix}no experiment_meta.json in the results folder - the run record '
            'cannot be checked, so a crashed run would look like a successful one',
        )
        return

    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, ValueError) as exc:
        report.add(WARNING, f'{prefix}experiment_meta.json could not be read: {exc}')
        return

    if not isinstance(meta, dict):
        report.add(WARNING, f'{prefix}experiment_meta.json is not a mapping')
        return

    ## -- did the run actually finish? --
    # Asserted positively against pycomex's own definition of success rather than by
    # listing failure values, so a run killed while still 'running' is caught too.
    status = str(meta.get('status', '')).strip().lower()
    if status != 'done':
        detail = f' ({meta["error_type"]})' if meta.get('error_type') else ''
        report.add(
            ERROR,
            f'{prefix}the experiment record reports status {status or "unknown"!r}'
            f'{detail}, not "done" - these results come from a run that did not '
            'finish. A partial run still leaves a complete-looking folder behind.',
        )

    if meta.get('has_error'):
        error_type = meta.get('error_type') or 'unknown error'
        report.add(
            ERROR,
            f'{prefix}the experiment record reports has_error=True ({error_type}) - '
            'the run raised, and its output cannot be trusted',
        )

    ## -- do these results belong to this script? --
    if script_path:
        expected = Path(script_path).stem
        recorded = str(meta.get('name', '')).strip()
        if recorded and expected and recorded != expected:
            report.add(
                ERROR,
                f'{prefix}the results were produced by {recorded!r} but the recorded '
                f'script is {expected!r} - this folder belongs to a different run',
            )


def _classify_override(override: dict) -> dict:
    """
    Work out what a recorded ``human_override`` means, without acting on it yet.

    Separated from applying it so that the automatic checks can run first: a human
    approving a license for one stated reason should still be *told* what else the
    gate found, rather than having those findings silently discarded.

    :param override: The ``human_override`` mapping, possibly empty.

    :returns: A mapping with ``state`` - one of ``absent``, ``approved``, ``rejected``,
        ``malformed``, ``unreasoned`` - and ``reason``.
    """
    if not override:
        return {'state': 'absent', 'reason': ''}

    decision = str(override.get('decision', '')).lower()
    reason = str(override.get('reason', '') or '').strip()

    if decision not in ('approved', 'rejected'):
        return {'state': 'malformed', 'reason': reason}
    if decision == 'rejected':
        return {'state': 'rejected', 'reason': reason}
    if not reason:
        return {'state': 'unreasoned', 'reason': ''}
    return {'state': 'approved', 'reason': reason}


def _apply_override(
    report: VerificationReport,
    override_state: dict,
    identifier: Optional[str],
) -> None:
    """
    Apply a human license decision to an already-populated report.

    An approval downgrades the errors found to warnings, each prefixed so that what was
    waived stays legible in every later gate run and in the manifest. It does not erase
    them: the human approved the dataset, not the disappearance of the reasons.

    A rejection is final and adds an error of its own; a malformed or unreasoned
    override is a defect in the record rather than a decision, and cannot waive
    anything.

    :param report: The report to modify in place.
    :param override_state: The mapping returned by :func:`_classify_override`.
    :param identifier: The license identifier, for the message.
    """
    state = override_state['state']

    if state == 'absent':
        return

    if state == 'malformed':
        report.add(
            ERROR,
            'license human_override must record decision: approved | rejected',
        )
        return

    if state == 'unreasoned':
        report.add(ERROR, 'an approved license override must record a reason')
        return

    if state == 'rejected':
        report.add(
            ERROR,
            'the license was reviewed and rejected: '
            f'{override_state["reason"] or "no reason recorded"}',
        )
        return

    ## -- approved: downgrade, do not delete --
    waived = 0
    for index, issue in enumerate(report.issues):
        if issue.level == ERROR:
            waived += 1
            report.issues[index] = VerificationIssue(
                level=WARNING,
                message=f'waived by human override: {issue.message}',
            )

    summary = (
        f'license {identifier!r} approved by human override: '
        f'{override_state["reason"]}'
    )
    if waived:
        summary += f' [{waived} check(s) waived - listed below]'
    report.issues.insert(0, VerificationIssue(level=WARNING, message=summary))


def gate_license(manifest: Manifest) -> VerificationReport:
    """
    Check that the license stage reached a defensible, evidenced conclusion.

    Passing requires a named license that is on the permissive list, backed by a URL
    that a human can check. Anything else - NonCommercial, ShareAlike, copyleft, no
    license at all, or terms the agent could not pin down - fails the gate and has to
    be resolved by a human, who records their decision as an explicit override.

    The license on the paper is not the end of the question. Datasets are routinely
    aggregations of an upstream database that carries its own, usually stricter, terms:
    a CC-BY paper can republish data from a source that forbids redistribution outright.
    The gate therefore requires the stage to state explicitly whether an upstream
    database exists, and escalates whenever one does unless its terms are themselves
    permissive.

    It also compares the host of the evidence URL against the host the run started from,
    because the same work often exists as a preprint and a published version under
    *different* licenses - and establishing the license against one while downloading
    the other is a silent and easy mistake.

    :param manifest: The run manifest.

    :returns: The gate report.
    """
    report = VerificationReport(results_dir=work_dir_for(manifest.slug))
    license_info = manifest.license or {}

    if not license_info:
        report.add(ERROR, 'no license information recorded in the manifest')
        return report

    identifier = license_info.get('identifier')
    evidence = license_info.get('evidence_url')

    # The override is classified now but applied at the *end*, after every automatic
    # check has run. Returning here - which is what this did - meant an approval given
    # for one reason silently waived every other check too, including the upstream
    # provenance check that exists because a previous run shipped a dataset whose
    # upstream forbade redistribution.
    override_state = _classify_override(license_info.get('human_override') or {})

    if not evidence:
        report.add(
            ERROR,
            'no "evidence_url" - the license finding must point at a page a human '
            'can check',
        )

    if not identifier:
        report.add(
            ERROR,
            'no license identified. A dataset with no stated license is not clearly '
            'redistributable and needs a human decision.',
        )
    elif not is_permissive(identifier):
        report.add(
            ERROR,
            f'license {identifier!r} is not on the permissive list (public domain, '
            'CC-BY, MIT/BSD/Apache). NonCommercial, ShareAlike and copyleft terms '
            'need a human decision recorded as a license override.',
        )

    redistribution = str(license_info.get('redistribution', '')).strip().lower()
    if redistribution and redistribution not in REDISTRIBUTION_ALLOWED:
        if redistribution in REDISTRIBUTION_DENIED:
            report.add(
                ERROR,
                f'the license stage concluded redistribution is {redistribution!r}',
            )
        else:
            # Anything else is a sentence about redistribution rather than a verdict on
            # it. Saying so beats reporting a paragraph back as though it were a value.
            report.add(
                ERROR,
                '"redistribution" must be one of '
                f'{"/".join(REDISTRIBUTION_ALLOWED)} or '
                f'{"/".join(REDISTRIBUTION_DENIED)} - the gate matches it exactly. '
                f'Got prose: {redistribution[:60]!r}... Put the reasoning in '
                '"attribution" or in the human override reason, and record the verdict '
                'here.',
            )

    if not str(license_info.get('attribution', '')).strip():
        report.add(
            WARNING,
            'no attribution requirement recorded - CC-BY style licenses require the '
            'source to be credited in the dataset metadata',
        )

    ## -- upstream provenance --
    # The paper's license governs the paper. It does not govern a database the paper
    # merely aggregated, which is where the binding restriction usually lives.
    if 'upstream_database' not in license_info:
        report.add(
            ERROR,
            'the license stage must state "upstream_database": the name of the '
            'database the data was aggregated from, or null if the publication is '
            'the original source. A permissive paper license does not override the '
            'terms of an upstream database.',
        )
    else:
        upstream = license_info.get('upstream_database')
        named = bool(upstream) and str(upstream).strip().lower() not in (
            'none',
            'null',
            'n/a',
        )

        if named:
            upstream_license = license_info.get('upstream_license')
            if not str(license_info.get('upstream_terms_url', '')).strip():
                report.add(
                    ERROR,
                    f'data derives from "{upstream}" but no "upstream_terms_url" was '
                    'recorded - its terms of use must be cited for a human to check',
                )

            if not upstream_license:
                report.add(
                    ERROR,
                    f'data derives from "{upstream}" whose license was not '
                    'established. Upstream databases frequently restrict '
                    'redistribution regardless of the publication license, so this '
                    'needs a human decision.',
                )
            elif not looks_like_identifier(upstream_license):
                # Prose here fails is_permissive for the wrong reason: a sentence
                # saying the upstream is CC0 escalates exactly like one saying it
                # forbids redistribution, so an escalation stops meaning anything.
                report.add(
                    ERROR,
                    f'"upstream_license" must be a license identifier - CC-BY-4.0, '
                    'MIT, CC0-1.0, "proprietary", or "none" when there is no stated '
                    'license - because the gate matches it against a fixed list. Got '
                    f'prose: {str(upstream_license)[:60]!r}... A sentence is '
                    'indistinguishable from a restriction here, so quote the terms in '
                    'the override reason and record the identifier itself.',
                )
            elif not is_permissive(upstream_license):
                report.add(
                    ERROR,
                    f'upstream database "{upstream}" is under {upstream_license!r}, '
                    'which is not permissive. Redistributing an aggregation of it '
                    'needs a human decision recorded as a license override.',
                )

    ## -- preprint vs published version --
    # Compared against the *record* the files came from, not the publication. A paper
    # on a publisher and its data on Zenodo is the ordinary arrangement and says
    # nothing; the question worth asking is whether the license was read on the same
    # host that served the download - a ChemRxiv preprint against its published ACS
    # version being the case that motivated this. Falls back to source_url for runs
    # that never recorded a record_url.
    compared_url = manifest.record_url or manifest.source_url
    compared_label = 'the data record' if manifest.record_url else 'the run source'
    compared_host = _host(compared_url)
    evidence_host = _host(evidence)
    if compared_host and evidence_host and compared_host != evidence_host:
        report.add(
            WARNING,
            f'the license was established on {evidence_host} but {compared_label} is '
            f'on {compared_host}. The same work often exists in two versions under '
            'different licenses - confirm the files actually downloaded are covered by '
            'the license recorded here.',
        )

    ## -- the human decision, applied to everything found above --
    _apply_override(report, override_state, identifier)

    return report


def gate_script(manifest: Manifest) -> VerificationReport:
    """
    Check the generated script's structure and its smoke run.

    The structural checks catch the two failure modes that survive a successful run
    but break for everyone else: an absolute path to the base template, and a script
    that reads its data out of the scratch directory instead of downloading it. The
    smoke run then proves the pipeline end to end on 50 elements before any time is
    spent on the full dataset.

    :param manifest: The run manifest.

    :returns: The gate report.
    """
    report = VerificationReport(results_dir=work_dir_for(manifest.slug))

    if not manifest.script_path:
        report.add(ERROR, 'no script_path recorded in the manifest')
        return report

    script_path = Path(PROJECT_PATH) / manifest.script_path
    if not script_path.exists():
        report.add(ERROR, f'script not found: {manifest.script_path}')
        return report

    expected_dir = Path(PROJECT_PATH) / 'chem_mat_data' / 'scripts'
    if script_path.parent.resolve() != expected_dir.resolve():
        report.add(
            ERROR,
            f'script must live in chem_mat_data/scripts/, found at '
            f'{script_path.parent}',
        )

    if not script_path.name.startswith('create_graph_datasets__'):
        report.add(
            WARNING,
            f'script name does not follow create_graph_datasets__<name>.py: '
            f'{script_path.name}',
        )

    source = script_path.read_text(errors='replace')

    if 'run_if_main()' not in source:
        report.add(ERROR, 'script does not end with experiment.run_if_main()')

    if 'load_dataset' not in source:
        report.add(ERROR, 'script does not define a load_dataset hook')

    ## -- the base template path must be relative --
    extend_match = re.search(r'Experiment\.extend\(\s*[\'"]([^\'"]+)[\'"]', source)
    if extend_match:
        template = extend_match.group(1)
        if template.startswith('/') or re.match(r'^[A-Za-z]:[\\/]', template):
            report.add(
                ERROR,
                f'Experiment.extend uses an absolute path ({template}) - it must be '
                'relative, since the script sits next to the base template',
            )
    elif 'Experiment(' not in source:
        report.add(ERROR, 'script neither extends a template nor builds an Experiment')

    ## -- the script must be self-contained --
    if '.dataset_work' in source:
        report.add(
            ERROR,
            'script references .dataset_work - it must download its own data from the '
            'original URL rather than reading the scratch directory',
        )

    if not re.search(r'https?://', source):
        report.add(
            ERROR,
            'script contains no download URL - it is not self-contained',
        )

    ## -- the smoke run --
    if not manifest.smoke_results_dir:
        report.add(ERROR, 'no smoke_results_dir recorded - the smoke run did not run')
        return report

    check_experiment_record(
        report,
        manifest.smoke_results_dir,
        script_path=manifest.script_path,
        label='smoke run',
    )

    smoke_report = verify_results(manifest.smoke_results_dir)
    for issue in smoke_report.issues:
        # A 50-element smoke run is meant to look like a testing run, so the low
        # compound count warning is expected here rather than informative.
        if '__TESTING__' in issue.message:
            continue
        report.add(issue.level, f'smoke run: {issue.message}')

    if smoke_report.compounds is not None and smoke_report.compounds == 0:
        report.add(ERROR, 'smoke run produced 0 compounds')

    ## -- the smoke run must agree with the approved metadata --
    draft = manifest.metadata_draft or {}
    expected_targets = draft.get('target_count')
    if not isinstance(expected_targets, int):
        normalized = normalize_target_description_keys(
            draft.get('target_descriptions') or {}
        )
        expected_targets = len(normalized) or None

    if expected_targets and smoke_report.targets:
        if smoke_report.targets != expected_targets:
            report.add(
                ERROR,
                f'smoke run produced {smoke_report.targets} targets but the approved '
                f'metadata describes {expected_targets}',
            )

    return report


def gate_execute(manifest: Manifest) -> VerificationReport:
    """
    Check the full run, including how many input rows survived processing.

    Molecules that fail conversion are logged and skipped rather than raising, so a
    run can complete cleanly having quietly discarded a large part of the dataset.
    The drop rate is therefore checked explicitly: above 2% it is recorded, above 10%
    it escalates, unless the manifest carries an explained per-dataset override.

    :param manifest: The run manifest.

    :returns: The gate report.
    """
    report = VerificationReport(results_dir=work_dir_for(manifest.slug))

    if not manifest.results_dir:
        report.add(ERROR, 'no results_dir recorded - the full run did not run')
        return report

    check_experiment_record(
        report, manifest.results_dir, script_path=manifest.script_path
    )

    ## -- testing mode must be off --
    if manifest.script_path:
        script_path = Path(PROJECT_PATH) / manifest.script_path
        if script_path.exists():
            source = script_path.read_text(errors='replace')
            if re.search(r'^__TESTING__\s*[:=].*True', source, re.MULTILINE):
                report.add(
                    ERROR,
                    '__TESTING__ is still True in the script - the full run only '
                    'processed 50 elements',
                )

    run_report = verify_results(manifest.results_dir)
    for issue in run_report.issues:
        report.add(issue.level, issue.message)

    report.dataset_name = run_report.dataset_name
    report.compounds = run_report.compounds
    report.targets = run_report.targets

    ## -- what the target values actually look like --
    # Computed unconditionally and carried on the report, so the stage-6 subagent is
    # handed the numbers to hold against the paper rather than having to derive them,
    # and so a human reading the gate output can see the target at a glance.
    mpack_paths = sorted(Path(manifest.results_dir).glob('*.mpack'))
    if mpack_paths:
        report.target_statistics = stream_target_statistics(mpack_paths[0])
        check_target_plausibility(
            report, report.target_statistics, manifest.metadata_draft or {}
        )

    ## -- drop rate --
    if manifest.input_rows and run_report.compounds:
        dropped = manifest.input_rows - run_report.compounds
        if dropped < 0:
            # More output than input means input_rows does not describe the file the
            # script actually downloaded - most often because the scratch copy was
            # truncated and counted, while the script fetched the whole thing. That is
            # the one prepare-stage failure nothing else catches, and left as a warning
            # it silently turns the drop-rate check below into a no-op: the denominator
            # is wrong, so no drop can ever look large.
            explained = bool(str(manifest.drop_rate_reason or '').strip())
            message = (
                f'the run produced more compounds ({run_report.compounds}) than the '
                f'recorded input_rows ({manifest.input_rows}). input_rows does not '
                'describe the file the script downloaded - most likely the copy in '
                'downloads/ was truncated - so the drop-rate check below is meaningless.'
            )
            if explained:
                # An explanation was recorded, so this is a known difference rather
                # than an undetected truncation. It stays visible as a warning: the
                # drop-rate check really is uninformative here, whatever the reason.
                report.add(
                    WARNING,
                    f'{message} Recorded reason: {manifest.drop_rate_reason}',
                )
            else:
                report.add(
                    ERROR,
                    f'{message} Recount against the real source and re-run the prepare '
                    'stage, or record drop_rate_reason if the difference is understood.',
                )
        else:
            rate = dropped / manifest.input_rows
            threshold = (
                manifest.max_drop_rate
                if manifest.max_drop_rate is not None
                else DROP_RATE_ERROR
            )
            summary = (
                f'{dropped} of {manifest.input_rows} rows dropped '
                f'({rate:.1%}) during processing'
            )

            if rate > threshold:
                report.add(
                    ERROR,
                    f'{summary} - above the {threshold:.0%} escalation threshold. '
                    'Check that the SMILES column is correct and inspect '
                    '"error processing" lines in experiment_out.log.',
                )
            elif rate > DROP_RATE_WARN:
                note = (
                    f' (expected: {manifest.drop_rate_reason})'
                    if manifest.drop_rate_reason
                    else ''
                )
                report.add(WARNING, f'{summary}{note}')
    elif not manifest.input_rows:
        report.add(
            WARNING,
            'no input_rows recorded, so the drop rate could not be checked',
        )

    return report


def gate_verification(manifest: Manifest) -> VerificationReport:
    """
    Final pass over the whole run.

    Re-runs the full results check, re-runs the license gate, and compares what ended up
    in ``metadata.yml`` against what the metadata stage approved. The two can diverge
    silently, because the script carries its own ``METADATA`` literal and nothing before
    this point compares it against the agreed draft.

    Specifically compared: ``verbose``, ``description``, ``tags``, ``sources``, the
    per-target ``target_descriptions``, ``dataset_type`` against the written
    ``target_type``, and ``dataset_name`` against the message pack that was written.
    Wording differences warn; a missing approved field, a changed task type or a
    renamed dataset are errors.

    What this does **not** do is confirm that the approved metadata was right in the
    first place. An agent confidently wrong about a unit writes the same wrong string
    into the draft and into the script, and both sides agree. The observed-value checks
    in :func:`gate_execute` are what catch that; this gate catches drift between what
    was agreed and what shipped.

    :param manifest: The run manifest.

    :returns: The gate report.
    """
    report = VerificationReport(results_dir=work_dir_for(manifest.slug))

    ## -- every earlier stage must have passed --
    for stage in STAGES[: STAGES.index('verification')]:
        state = manifest.state(stage)
        if state.status != 'passed':
            report.add(
                ERROR,
                f'stage "{stage}" is {state.status}, not passed'
                + (f': {state.blocker}' if state.blocker else ''),
            )

    if not manifest.results_dir:
        report.add(ERROR, 'no results_dir recorded')
        return report

    check_experiment_record(
        report, manifest.results_dir, script_path=manifest.script_path
    )

    run_report = verify_results(manifest.results_dir)
    for issue in run_report.issues:
        report.add(issue.level, issue.message)

    report.dataset_name = run_report.dataset_name
    report.compounds = run_report.compounds
    report.targets = run_report.targets

    ## -- written metadata must match the approved draft --
    metadata_path = Path(manifest.results_dir) / 'metadata.yml'
    draft = manifest.metadata_draft or {}
    if metadata_path.exists() and draft:
        try:
            with metadata_path.open() as file:
                written = yaml.safe_load(file) or {}
        except yaml.YAMLError:
            written = {}

        for key in ('verbose', 'description'):
            approved = str(draft.get(key, '')).strip()
            actual = str(written.get(key, '')).strip()
            if approved and actual and approved != actual:
                report.add(
                    WARNING,
                    f'"{key}" in metadata.yml differs from the approved draft',
                )
            elif approved and not actual:
                report.add(
                    ERROR, f'"{key}" was approved but is missing from metadata.yml'
                )

        for key in ('tags', 'sources'):
            approved_list = draft.get(key) or []
            actual_list = written.get(key) or []
            missing = [item for item in approved_list if item not in actual_list]
            if missing:
                report.add(
                    ERROR,
                    f'{key} approved in the metadata stage but absent from '
                    f'metadata.yml: {missing}',
                )

        ## -- per-target descriptions --
        # The only per-target place units live, and the script carries its own METADATA
        # literal, so this is where the approved draft and what actually shipped can
        # silently diverge. Divergence warns, to match how verbose/description are
        # treated; an approved index going missing entirely is an error, because a
        # target with no description is unusable to whoever downloads it.
        approved_descriptions = normalize_target_description_keys(
            draft.get('target_descriptions') or {}
        )
        written_descriptions = normalize_target_description_keys(
            written.get('target_descriptions') or {}
        )
        for index in sorted(approved_descriptions):
            approved_text = str(approved_descriptions[index] or '').strip()
            actual_text = str(written_descriptions.get(index, '') or '').strip()
            if not approved_text:
                continue
            if not actual_text:
                report.add(
                    ERROR,
                    f'target {index} was described in the approved metadata but has no '
                    'description in metadata.yml',
                )
            elif approved_text != actual_text:
                report.add(
                    WARNING,
                    f'the description of target {index} in metadata.yml differs from '
                    f'the approved draft: {actual_text!r} against {approved_text!r}',
                )

        ## -- the task type --
        # Categorical, so a mismatch is unambiguous rather than a wording difference.
        approved_type = str(draft.get('dataset_type', '')).strip().lower()
        written_types = [
            str(item).strip().lower() for item in (written.get('target_type') or [])
        ]
        if approved_type and written_types and approved_type not in written_types:
            report.add(
                ERROR,
                f'the metadata stage approved a {approved_type!r} dataset but '
                f'metadata.yml declares {written_types} - a downstream user will load '
                'this with the wrong task',
            )

    ## -- the dataset name --
    # The identifier a user types into load_smiles_dataset. Checked against the file
    # that was actually written rather than against the draft alone, and only when a
    # message pack exists, so a partial run degrades instead of raising.
    approved_name = str((manifest.metadata_draft or {}).get('dataset_name', '')).strip()
    if approved_name:
        mpack_paths = sorted(Path(manifest.results_dir).glob('*.mpack'))
        if mpack_paths and mpack_paths[0].stem != approved_name:
            report.add(
                ERROR,
                f'the dataset was saved as {mpack_paths[0].stem!r} but the metadata '
                f'stage approved the name {approved_name!r} - this is the identifier '
                'a user passes to load_smiles_dataset',
            )

    ## -- the license conclusion must still hold --
    # Re-run the real gate rather than re-implementing a weaker version of it. The
    # previous check asked only whether the top-level identifier was permissive *or*
    # an override key existed, which meant a `rejected` override read as an approval
    # (a rejection is a non-empty dict, so `not override` was False), and a permissive
    # paper license waved through an upstream database that forbids redistribution.
    license_report = gate_license(manifest)
    for issue in license_report.errors:
        report.add(ERROR, f'license: {issue.message}')
    for issue in license_report.warnings:
        report.add(WARNING, f'license: {issue.message}')

    return report


#: Gate functions by stage name.
GATES: dict = {
    'prepare': gate_prepare,
    'metadata': gate_metadata,
    'license': gate_license,
    'script': gate_script,
    'execute': gate_execute,
    'verification': gate_verification,
}


def run_gate(stage: str, manifest: Manifest) -> VerificationReport:
    """
    Run the gate for a stage, after confirming its prerequisites are met.

    A stage whose predecessors have not passed fails immediately: checking the script
    against metadata that was never agreed produces a meaningless verdict.

    :param stage: The stage name.
    :param manifest: The run manifest.

    :returns: The gate report.
    :raises ValueError: If the stage name is not recognised.
    """
    if stage not in GATES:
        valid = ', '.join(STAGES)
        raise ValueError(f'unknown stage {stage!r}. Valid stages: {valid}')

    unmet = manifest.unmet_prerequisites(stage)
    if unmet:
        report = VerificationReport(results_dir=work_dir_for(manifest.slug))
        report.add(
            ERROR,
            f'cannot gate "{stage}" - earlier stage(s) have not passed: '
            f'{", ".join(unmet)}',
        )
        return report

    return GATES[stage](manifest)
