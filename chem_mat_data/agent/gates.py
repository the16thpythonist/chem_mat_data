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
    VerificationReport,
    normalize_target_description_keys,
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
        if path.is_file()
        and not any(name in path.name.lower() for name in NON_DATA_NAMES)
    )


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
        report.add(
            ERROR,
            'no data files in the downloads directory (only README/recipe files '
            'were found)',
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
        elif 'http' not in text:
            report.add(ERROR, 'download_recipe.md records no URL')

    readme_found = any(
        'readme' in path.name.lower() for path in downloads.rglob('*') if path.is_file()
    )
    if not readme_found:
        report.add(
            WARNING,
            'no README captured - column meanings and units will have to be inferred',
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
    override = license_info.get('human_override') or {}

    if not evidence:
        report.add(
            ERROR,
            'no "evidence_url" - the license finding must point at a page a human '
            'can check',
        )

    ## -- an explicit human decision overrides the automatic policy --
    if override:
        decision = str(override.get('decision', '')).lower()
        if decision not in ('approved', 'rejected'):
            report.add(
                ERROR,
                'license human_override must record decision: approved | rejected',
            )
        elif decision == 'rejected':
            report.add(
                ERROR,
                'the license was reviewed and rejected: '
                f'{override.get("reason", "no reason recorded")}',
            )
        elif not str(override.get('reason', '')).strip():
            report.add(ERROR, 'an approved license override must record a reason')
        else:
            report.add(
                WARNING,
                f'license {identifier!r} approved by human override: '
                f'{override.get("reason")}',
            )
        return report

    if not identifier:
        report.add(
            ERROR,
            'no license identified. A dataset with no stated license is not clearly '
            'redistributable and needs a human decision.',
        )
        return report

    if not is_permissive(identifier):
        report.add(
            ERROR,
            f'license {identifier!r} is not on the permissive list (public domain, '
            'CC-BY, MIT/BSD/Apache). NonCommercial, ShareAlike and copyleft terms '
            'need a human decision recorded as a license override.',
        )

    redistribution = str(license_info.get('redistribution', '')).lower()
    if redistribution and redistribution not in ('allowed', 'yes', 'permitted'):
        report.add(
            ERROR,
            f'the license stage concluded redistribution is {redistribution!r}',
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
            elif not is_permissive(upstream_license):
                report.add(
                    ERROR,
                    f'upstream database "{upstream}" is under {upstream_license!r}, '
                    'which is not permissive. Redistributing an aggregation of it '
                    'needs a human decision recorded as a license override.',
                )

    ## -- preprint vs published version --
    source_host = _host(manifest.source_url)
    evidence_host = _host(evidence)
    if source_host and evidence_host and source_host != evidence_host:
        report.add(
            WARNING,
            f'the license was established on {evidence_host} but the run started from '
            f'{source_host}. Preprint and published versions of the same work often '
            'carry different licenses - confirm the file actually used is covered by '
            'the license recorded here.',
        )

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

    ## -- drop rate --
    if manifest.input_rows and run_report.compounds:
        dropped = manifest.input_rows - run_report.compounds
        if dropped < 0:
            report.add(
                WARNING,
                f'more compounds ({run_report.compounds}) than recorded input rows '
                f'({manifest.input_rows}) - is input_rows correct?',
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

    Re-runs the full results check and additionally confirms that what ended up in
    ``metadata.yml`` actually matches what the metadata stage approved. The two can
    diverge silently: the script carries its own ``METADATA`` literal, and nothing
    before this point compares it against the draft that was agreed.

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

    ## -- the license conclusion must still hold --
    license_info = manifest.license or {}
    override = license_info.get('human_override') or {}
    if not override and not is_permissive(license_info.get('identifier')):
        report.add(
            ERROR,
            'the recorded license is not permissive and carries no human override',
        )

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
