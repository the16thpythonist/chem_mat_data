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

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import msgpack
import numpy as np
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
    :param acknowledged: Whether a human has already decided about this. Such an issue
        stays visible - that is the whole point of downgrading a waived error rather
        than deleting it - but ``--strict`` must not treat it as a failure, or a
        recorded decision becomes a permanent block and the human is asked the same
        question forever.
    """

    level: str
    message: str
    acknowledged: bool = False


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
    #: Observed per-target statistics, when they were computed. Carried on the report so
    #: that whoever reads it - a human, or the stage-6 subagent - has the actual numbers
    #: to hold against the paper rather than having to derive them.
    target_statistics: List['TargetStatistics'] = field(default_factory=list)

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

    @property
    def unacknowledged_warnings(self) -> List[VerificationIssue]:
        """Warnings that no human has already ruled on - what ``--strict`` acts on."""
        return [issue for issue in self.warnings if not issue.acknowledged]

    def add(self, level: str, message: str, acknowledged: bool = False) -> None:
        """
        Append an issue to the report.

        :param level: Either :data:`ERROR` or :data:`WARNING`.
        :param message: Description of the problem.
        :param acknowledged: Set when a human has already decided about this.
        """
        self.issues.append(
            VerificationIssue(
                level=level, message=message, acknowledged=acknowledged
            )
        )


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


#: Distinct-value counting stops here. The count exists only to tell a continuous
#: target from a constant or a handful of classes, and an unbounded set would
#: reintroduce the memory cost that streaming exists to avoid.
UNIQUE_CAP: int = 64


@dataclass
class TargetStatistics:
    """
    Observed statistics for one target index across a whole dataset.

    :param index: Position of the target in the ``graph_labels`` vector.
    :param count: Number of finite values seen.
    :param minimum: Smallest finite value.
    :param maximum: Largest finite value.
    :param mean: Arithmetic mean of the finite values.
    :param unique: Number of distinct finite values, which distinguishes a real-valued
        target from a constant or a handful of classes mislabelled as regression.
        Counting stops at :data:`UNIQUE_CAP`; see :attr:`unique_capped`.
    :param unique_capped: True when counting hit the cap, so ``unique`` is a floor
        rather than a total. Reported separately because a capped count printed as an
        exact one is a wrong number, and the whole purpose of these statistics is to
        be trustworthy at a glance.
    :param missing: Number of entries that were absent or not finite.
    """

    index: int
    count: int = 0
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    mean: Optional[float] = None
    unique: int = 0
    unique_capped: bool = False
    missing: int = 0


def _ext_hook(code: int, data: bytes) -> Any:
    """
    Decode the numpy extension type used by the dataset message pack format.

    Deliberately a local copy of ``chem_mat_data.data.ext_hook`` rather than an import
    of it: that module pulls in RDKit, which costs about a second, and the gates that
    call this run in milliseconds. The wire format is four lines and does not move.

    :param code: The msgpack extension type code.
    :param data: The packed payload.

    :returns: A numpy array for code 1, the raw payload otherwise.
    """
    if code == 1:
        unpacked = msgpack.unpackb(data)
        return np.frombuffer(
            unpacked['data'], dtype=unpacked['dtype']
        ).reshape(unpacked['shape'])
    return data


def stream_target_statistics(mpack_path: str | Path) -> List[TargetStatistics]:
    """
    Compute per-target statistics without holding the dataset in memory.

    Reads one graph at a time out of the message pack. This matters more than it
    sounds: ``load_graphs`` materialises the entire dataset, which is 69 MB of resident
    memory for a 29 MB file and simply impossible for the multi-gigabyte datasets in
    this collection. Streaming the same file costs about 3 MB and a tenth of a second,
    so a gate can afford to do it on every run.

    Non-finite and missing entries are counted rather than skipped silently, since a
    target that is half NaN is a finding in itself.

    Example:

    .. code-block:: python

        for stat in stream_target_statistics('results/debug/esol.mpack'):
            print(stat.index, stat.minimum, stat.maximum)

    :param mpack_path: Path to the uncompressed ``.mpack`` dataset file.

    :returns: One :class:`TargetStatistics` per target index, in order. Empty when the
        file cannot be read as a dataset.
    """
    accumulators: Dict[int, Dict[str, Any]] = {}

    try:
        with open(mpack_path, 'rb') as file:
            unpacker = msgpack.Unpacker(file, ext_hook=_ext_hook, raw=False)
            count = unpacker.read_array_header()
            for _ in range(count):
                graph = unpacker.unpack()
                if not isinstance(graph, dict):
                    continue
                labels = graph.get('graph_labels')
                if labels is None:
                    continue

                for index, raw in enumerate(labels):
                    entry = accumulators.setdefault(
                        index,
                        {'total': 0.0, 'count': 0, 'missing': 0, 'capped': False,
                         'min': None, 'max': None, 'values': set()},
                    )
                    try:
                        value = float(raw)
                    except (TypeError, ValueError):
                        entry['missing'] += 1
                        continue
                    if not math.isfinite(value):
                        entry['missing'] += 1
                        continue

                    entry['count'] += 1
                    entry['total'] += value
                    entry['min'] = value if entry['min'] is None else min(entry['min'], value)
                    entry['max'] = value if entry['max'] is None else max(entry['max'], value)
                    # Capped: the count only has to distinguish "continuous" from "a
                    # few classes", and an unbounded set on a 3-million-row dataset
                    # would defeat the point of streaming.
                    if len(entry['values']) < UNIQUE_CAP:
                        entry['values'].add(value)
                    else:
                        entry['capped'] = True
    except (OSError, ValueError, msgpack.exceptions.UnpackException):
        return []

    statistics: List[TargetStatistics] = []
    for index in sorted(accumulators):
        entry = accumulators[index]
        statistics.append(
            TargetStatistics(
                index=index,
                count=entry['count'],
                minimum=entry['min'],
                maximum=entry['max'],
                mean=(entry['total'] / entry['count']) if entry['count'] else None,
                unique=len(entry['values']),
                unique_capped=entry['capped'],
                missing=entry['missing'],
            )
        )
    return statistics


#: How many rows to compare between the two artifacts. The failure this catches is a
#: whole-file permutation, not a subtle one, so a prefix settles it - and reading the
#: entire CSV of a multi-gigabyte dataset to prove it would be its own problem.
ARTIFACT_SAMPLE: int = 64


def check_artifacts_agree(
    report: VerificationReport,
    mpack_path: Path,
    csv_path: Path,
) -> None:
    """
    Check that the message pack and the CSV of one run describe the same rows.

    They are produced from the same dataset by different code paths, and used to
    disagree: graphs came off a multiprocessing queue in completion order while the CSV
    was written in source order, so the two were a full permutation of each other -
    5432 of 5432 rows mismatched on a real run, with identical contents. Nothing
    detected it, because both files existed and both had the right length. Anyone
    joining them by row index got a different compound in every row.

    Compares the row count and the first :data:`ARTIFACT_SAMPLE` SMILES. On results
    produced before the ordering fix this reports an error, correctly: those two files
    really do disagree.

    :param report: The report to add issues to.
    :param mpack_path: The message pack dataset file.
    :param csv_path: The raw CSV export of the same run.

    :returns: Nothing; issues are added to ``report``.
    """
    try:
        import csv as csv_module

        with open(csv_path, newline='') as file:
            reader = csv_module.reader(file)
            header = next(reader, None)
            if not header or 'smiles' not in header:
                return
            column = header.index('smiles')
            csv_smiles = [row[column] for row in reader if len(row) > column]
    except (OSError, ValueError, StopIteration):
        return

    mpack_smiles: List[str] = []
    total = 0
    try:
        with open(mpack_path, 'rb') as file:
            unpacker = msgpack.Unpacker(file, ext_hook=_ext_hook, raw=False)
            total = unpacker.read_array_header()
            for position in range(total):
                graph = unpacker.unpack()
                if position < ARTIFACT_SAMPLE and isinstance(graph, dict):
                    mpack_smiles.append(str(graph.get('graph_repr', '')))
    except (OSError, ValueError, msgpack.exceptions.UnpackException):
        return

    if total != len(csv_smiles):
        report.add(
            ERROR,
            f'{mpack_path.name} holds {total} graphs but {csv_path.name} has '
            f'{len(csv_smiles)} rows - the two artifacts of this run disagree about '
            'how large the dataset is',
        )
        return

    mismatches = [
        position
        for position, (left, right) in enumerate(zip(mpack_smiles, csv_smiles))
        if left and left != right
    ]
    if mismatches:
        position = mismatches[0]
        report.add(
            ERROR,
            f'{mpack_path.name} and {csv_path.name} are not in the same order - '
            f'{len(mismatches)} of the first {len(mpack_smiles)} rows differ, first at '
            f'row {position} ({mpack_smiles[position]!r} against '
            f'{csv_smiles[position]!r}). Joining them by row index pairs the wrong '
            'compound with the wrong values.',
        )


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

        csv_path = results_path / f'{mpack_path.stem}.csv'
        if not csv_path.exists():
            report.add(
                WARNING,
                f'no raw {mpack_path.stem}.csv export found',
            )
        else:
            check_artifacts_agree(report, mpack_path, csv_path)

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
