"""
Post-condition checks for a dataset processing run.

After a ``create_graph_datasets__<name>.py`` script finishes, the results folder is
supposed to contain a message pack dataset file and a ``metadata.yml`` describing it.
A script can, however, exit cleanly while still having produced something unusable:
the processing workers drop molecules that raise an exception, and the metadata is
assembled from whatever the script happened to declare.

These checks exist so that "the run succeeded" is a verifiable fact rather than a
judgement call. They are deliberately implemented in Python rather than left as
instructions to an agent, because they are the point at which a plausible-looking run
is separated from a correct one.

Used by the ``cmmanage agent verify`` command and by the ``dataset-extraction`` skill.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


#: Issue level indicating a problem that makes the results unusable.
ERROR: str = 'error'

#: Issue level indicating something suboptimal that does not invalidate the results.
WARNING: str = 'warning'


@dataclass
class VerificationIssue:
    """
    A single problem found while verifying a results folder.

    :param level: Either :data:`ERROR` or :data:`WARNING`.
    :param message: Human-readable description of the problem.
    """

    level: str
    message: str


@dataclass
class VerificationReport:
    """
    The outcome of verifying a results folder.

    Collects every issue found rather than failing on the first one, so that a single
    verification run tells the caller everything that needs fixing.

    :param results_dir: The folder that was inspected.
    :param issues: All issues found, in the order they were detected.
    :param dataset_name: Name derived from the message pack file, if one was found.
    :param compounds: Number of compounds reported by the metadata.
    :param targets: Number of targets reported by the metadata.
    """

    results_dir: Path
    issues: List[VerificationIssue] = field(default_factory=list)
    dataset_name: Optional[str] = None
    compounds: Optional[int] = None
    targets: Optional[int] = None

    @property
    def errors(self) -> List[VerificationIssue]:
        """All issues at :data:`ERROR` level."""
        return [issue for issue in self.issues if issue.level == ERROR]

    @property
    def warnings(self) -> List[VerificationIssue]:
        """All issues at :data:`WARNING` level."""
        return [issue for issue in self.issues if issue.level == WARNING]

    @property
    def ok(self) -> bool:
        """``True`` when no error-level issues were found."""
        return not self.errors

    def add(self, level: str, message: str) -> None:
        """
        Append an issue to the report.

        :param level: Either :data:`ERROR` or :data:`WARNING`.
        :param message: Description of the problem.
        """
        self.issues.append(VerificationIssue(level=level, message=message))


def normalize_target_description_keys(
    target_descriptions: Dict[Any, Any],
) -> Dict[int, Any]:
    """
    Convert ``target_descriptions`` keys to integers.

    Both string keys (``'0'``) and integer keys (``0``) occur in the existing metadata
    - string dominates - so both are accepted here and normalised for comparison.
    Keys that are neither are dropped, which surfaces as a missing index.

    :param target_descriptions: The raw mapping as loaded from the metadata.

    :returns: A mapping with integer keys.
    """
    normalized: Dict[int, Any] = {}
    for key, value in target_descriptions.items():
        try:
            normalized[int(key)] = value
        except (TypeError, ValueError):
            continue
    return normalized


def verify_results(results_dir: str | Path) -> VerificationReport:
    """
    Verify that a dataset processing run produced complete, usable output.

    Checks, in order:

    - The folder exists and contains a ``.mpack`` dataset file.
    - A ``metadata.yml`` exists, parses, and is a mapping.
    - ``compounds`` is present and greater than zero.
    - ``targets`` is present and greater than zero.
    - ``target_descriptions`` covers every target index from ``0`` to ``targets - 1``
      whenever the dataset has more than one target.
    - Descriptive fields (``description``, ``tags``, ``sources``, ``verbose``) are
      present, and the compressed dataset and raw CSV exist. These are warnings.

    Example:

    .. code-block:: python

        report = verify_results('chem_mat_data/scripts/results/create_graph_datasets__esol/debug')
        if not report.ok:
            for issue in report.errors:
                print(issue.message)

    :param results_dir: Path to the experiment results folder to inspect.

    :returns: A :class:`VerificationReport` describing everything that was found.
    """
    results_path = Path(results_dir)
    report = VerificationReport(results_dir=results_path)

    if not results_path.exists():
        report.add(ERROR, f'results directory does not exist: {results_path}')
        return report

    if not results_path.is_dir():
        report.add(ERROR, f'not a directory: {results_path}')
        return report

    ## -- dataset file --
    # The uncompressed .mpack is the actual dataset. Its absence means the run never
    # reached the save step, whatever the exit code said.
    mpack_paths = sorted(results_path.glob('*.mpack'))
    if not mpack_paths:
        report.add(
            ERROR,
            'no .mpack dataset file found - the run did not reach the save step. '
            'Check experiment_out.log for the underlying error.',
        )
    else:
        mpack_path = mpack_paths[0]
        report.dataset_name = mpack_path.stem

        if mpack_path.stat().st_size == 0:
            report.add(ERROR, f'dataset file is empty: {mpack_path.name}')

        if len(mpack_paths) > 1:
            names = ', '.join(path.name for path in mpack_paths)
            report.add(
                WARNING,
                f'multiple .mpack files found ({names}); verifying against '
                f'{mpack_path.name}',
            )

        if not (results_path / f'{mpack_path.stem}.mpack.gz').exists():
            report.add(
                WARNING,
                f'no compressed {mpack_path.stem}.mpack.gz - is COMPRESS set to False?',
            )

        if not (results_path / f'{mpack_path.stem}.csv').exists():
            report.add(
                WARNING,
                f'no raw {mpack_path.stem}.csv export found',
            )

    ## -- metadata --
    metadata_path = results_path / 'metadata.yml'
    if not metadata_path.exists():
        report.add(ERROR, 'no metadata.yml found')
        return report

    try:
        with metadata_path.open() as file:
            metadata = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        report.add(ERROR, f'metadata.yml is not valid YAML: {exc}')
        return report

    if not isinstance(metadata, dict):
        report.add(ERROR, 'metadata.yml does not contain a mapping')
        return report

    ## -- compound count --
    compounds = metadata.get('compounds')
    if compounds is None:
        report.add(ERROR, 'metadata.yml is missing the "compounds" field')
    elif not isinstance(compounds, int):
        report.add(ERROR, f'"compounds" is not an integer: {compounds!r}')
    elif compounds == 0:
        report.add(
            ERROR,
            'the dataset contains 0 compounds - every molecule failed processing',
        )
    else:
        report.compounds = compounds
        if compounds <= 50:
            report.add(
                WARNING,
                f'only {compounds} compounds - if this is unexpected, __TESTING__ is '
                'probably still set to True',
            )

    ## -- targets --
    targets = metadata.get('targets')
    if targets is None:
        report.add(ERROR, 'metadata.yml is missing the "targets" field')
    elif not isinstance(targets, int):
        report.add(ERROR, f'"targets" is not an integer: {targets!r}')
    elif targets == 0:
        report.add(ERROR, 'the dataset has 0 targets')
    else:
        report.targets = targets

    ## -- target descriptions --
    # Every target index needs a description: this is the metadata a downstream user
    # relies on to know what they are training against, and it cannot be recovered
    # from the dataset file itself.
    raw_descriptions = metadata.get('target_descriptions')
    if report.targets is not None:
        if not raw_descriptions:
            level = WARNING if report.targets == 1 else ERROR
            report.add(
                level,
                f'no "target_descriptions" for {report.targets} target(s)',
            )
        elif not isinstance(raw_descriptions, dict):
            report.add(
                ERROR,
                f'"target_descriptions" is not a mapping: {type(raw_descriptions).__name__}',
            )
        else:
            descriptions = normalize_target_description_keys(raw_descriptions)
            missing = [
                index for index in range(report.targets) if index not in descriptions
            ]
            if missing:
                indices = ', '.join(str(index) for index in missing)
                report.add(
                    ERROR,
                    f'"target_descriptions" is missing index/indices {indices} '
                    f'(dataset has {report.targets} targets)',
                )

            empty = [
                index
                for index, value in sorted(descriptions.items())
                if not str(value).strip()
            ]
            if empty:
                indices = ', '.join(str(index) for index in empty)
                report.add(
                    ERROR, f'"target_descriptions" is empty for index/indices {indices}'
                )

            extra = [index for index in descriptions if index >= report.targets]
            if extra:
                indices = ', '.join(str(index) for index in sorted(extra))
                report.add(
                    WARNING,
                    f'"target_descriptions" has entries beyond the target count: {indices}',
                )

    ## -- descriptive fields --
    # Not fatal, but a dataset published without these is significantly less useful.
    if not str(metadata.get('description', '')).strip():
        report.add(WARNING, 'no "description" set')

    if not metadata.get('tags'):
        report.add(WARNING, 'no "tags" set')

    if not metadata.get('sources'):
        report.add(WARNING, 'no "sources" set - provenance URLs should be recorded')

    if not str(metadata.get('verbose', '')).strip():
        report.add(WARNING, 'no "verbose" display name set')

    if not metadata.get('target_type'):
        report.add(WARNING, 'no "target_type" set')

    return report
