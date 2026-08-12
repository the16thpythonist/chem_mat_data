"""
Persistent state for the staged dataset extraction pipeline.

The ``dataset-extraction`` skill walks a publication through six gated stages
(prepare, metadata, license, script, execute, verification). Each stage produces
something the next one consumes, and each is separated from the next by a gate that
has to pass before work continues.

That only works if the state survives between stages - and between sessions, since a
stage can block on something only a human can resolve (a paywalled file, a
NonCommercial license). This module owns that state: a ``manifest.yml`` per dataset
under ``.dataset_work/<slug>/``, recording what each stage concluded, what is
blocking, and the metadata drafted along the way.

A stage that is re-run invalidates everything downstream of it: re-deciding the
metadata after the script was written means the script no longer matches what was
approved, so it has to be re-gated. :meth:`Manifest.mark` handles that automatically.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from chem_mat_data.utils import PROJECT_PATH


#: The pipeline stages, in execution order. A stage may only run once every stage
#: before it has passed.
STAGES: List[str] = [
    'prepare',
    'metadata',
    'license',
    'script',
    'execute',
    'verification',
]

#: Stage has not been attempted yet.
PENDING: str = 'pending'

#: Stage completed and its gate passed.
PASSED: str = 'passed'

#: Stage hit something it cannot resolve on its own and needs a human.
BLOCKED: str = 'blocked'

#: Stage passed previously but an earlier stage has since been re-run, so its result
#: can no longer be trusted.
STALE: str = 'stale'

#: Root directory for all extraction working state, gitignored.
WORK_DIR: Path = Path(PROJECT_PATH) / '.dataset_work'


def work_dir_for(slug: str) -> Path:
    """
    Return the working directory for a given dataset slug.

    :param slug: The dataset slug.

    :returns: Path to ``.dataset_work/<slug>/``.
    """
    return WORK_DIR / slug


def _string_list(value: Any) -> List[str]:
    """
    Coerce a loaded YAML value into a list of strings.

    A hand-edited manifest is a normal thing to encounter - the whole point of the
    file is that a human can read and repair it - so a scalar where a list belongs
    should not crash a resume.

    :param value: Whatever was loaded from the manifest.

    :returns: A list of strings, empty when there is nothing usable.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if item is not None]
    return [str(value)]


@dataclass
class StageState:
    """
    The recorded outcome of a single pipeline stage.

    :param status: One of :data:`PENDING`, :data:`PASSED`, :data:`BLOCKED`,
        :data:`STALE`.
    :param at: ISO timestamp of the last status change.
    :param blocker: Plain-language description of what is blocking, when blocked.
        The headline reason only; see :attr:`errors` for the full set.
    :param notes: Free-form notes the stage wants to carry forward, such as why a
        previously passed stage was invalidated.
    :param warnings: Every warning message the gate emitted, verbatim. Recorded
        because a count is not actionable: the difference between "a human waived
        this for a stated reason" and "the upstream forbids derivative works" is a
        warning that both render as ``1 warning(s)``.
    :param errors: Every error message the gate emitted, verbatim. A blocked stage is
        the one most likely to be resumed cold in a later session, which is exactly
        when keeping only the first error costs the most.
    """

    status: str = PENDING
    at: Optional[str] = None
    blocker: Optional[str] = None
    notes: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialise to a plain dict, dropping empty fields."""
        data: Dict[str, Any] = {'status': self.status}
        if self.at:
            data['at'] = self.at
        if self.blocker:
            data['blocker'] = self.blocker
        if self.notes:
            data['notes'] = self.notes
        if self.warnings:
            data['warnings'] = list(self.warnings)
        if self.errors:
            data['errors'] = list(self.errors)
        return data

    @classmethod
    def from_dict(cls, data: Any) -> 'StageState':
        """
        Build a :class:`StageState` from loaded YAML.

        :param data: The mapping loaded from the manifest, or anything else, in which
            case a pending state is returned.

        :returns: The stage state.
        """
        if not isinstance(data, dict):
            return cls()
        return cls(
            status=data.get('status', PENDING),
            at=data.get('at'),
            blocker=data.get('blocker'),
            notes=data.get('notes'),
            warnings=_string_list(data.get('warnings')),
            errors=_string_list(data.get('errors')),
        )


@dataclass
class Manifest:
    """
    The full persistent state of one dataset extraction run.

    Example:

    .. code-block:: python

        manifest = Manifest.create('hopv15', source_url='https://...')
        manifest.mark('prepare', PASSED)
        manifest.save()

        manifest = Manifest.load('hopv15')
        print(manifest.next_stage())   # 'metadata'

    :param slug: Short identifier for this extraction run; also the working directory
        name.
    :param source_url: The publication URL or title the run started from.
    :param dataset_name: The eventual ``DATASET_NAME``, once decided.
    :param stages: Per-stage state, keyed by stage name.
    :param metadata_draft: The human-authored metadata fields assembled by the
        metadata stage and consumed by the script stage.
    :param license: What the license stage concluded.
    :param input_rows: Number of rows in the raw source, used for the drop-rate check.
    :param max_drop_rate: Optional per-dataset override of the escalation threshold,
        as a fraction. Set when a large drop is known and explained.
    :param drop_rate_reason: Why an elevated drop rate is expected, when overridden.
    :param script_path: Repository-relative path of the generated script.
    :param smoke_results_dir: Results folder of the ``__TESTING__`` smoke run.
    :param results_dir: Results folder of the full run.
    :param created_at: ISO timestamp of creation.
    :param updated_at: ISO timestamp of the last save.
    """

    slug: str
    source_url: Optional[str] = None
    dataset_name: Optional[str] = None
    stages: Dict[str, StageState] = field(default_factory=dict)
    metadata_draft: Dict[str, Any] = field(default_factory=dict)
    license: Dict[str, Any] = field(default_factory=dict)
    input_rows: Optional[int] = None
    max_drop_rate: Optional[float] = None
    drop_rate_reason: Optional[str] = None
    script_path: Optional[str] = None
    smoke_results_dir: Optional[str] = None
    results_dir: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    # -- construction ---------------------------------------------------------------

    @classmethod
    def create(cls, slug: str, source_url: Optional[str] = None) -> 'Manifest':
        """
        Create a new manifest with every stage pending.

        :param slug: Short identifier for the run.
        :param source_url: The publication URL or title.

        :returns: The new manifest, not yet written to disk.
        """
        now = _timestamp()
        return cls(
            slug=slug,
            source_url=source_url,
            stages={stage: StageState() for stage in STAGES},
            created_at=now,
            updated_at=now,
        )

    @classmethod
    def path_for(cls, slug: str) -> Path:
        """
        Return the manifest path for a slug.

        :param slug: The dataset slug.

        :returns: Path to ``.dataset_work/<slug>/manifest.yml``.
        """
        return work_dir_for(slug) / 'manifest.yml'

    @classmethod
    def exists(cls, slug: str) -> bool:
        """
        Return whether a manifest exists for the given slug.

        :param slug: The dataset slug.

        :returns: True when the manifest file is present.
        """
        return cls.path_for(slug).exists()

    @classmethod
    def load(cls, slug: str) -> 'Manifest':
        """
        Load an existing manifest from disk.

        Stages absent from the file are filled in as pending, so a manifest written by
        an older version stays loadable.

        :param slug: The dataset slug.

        :returns: The loaded manifest.
        :raises FileNotFoundError: If no manifest exists for the slug.
        """
        path = cls.path_for(slug)
        if not path.exists():
            raise FileNotFoundError(
                f'no manifest for "{slug}" at {path}. '
                f'Run `cmmanage agent init {slug} --url <url>` first.'
            )

        with path.open() as file:
            data = yaml.safe_load(file) or {}

        stages = {
            stage: StageState.from_dict((data.get('stages') or {}).get(stage))
            for stage in STAGES
        }

        return cls(
            slug=data.get('slug', slug),
            source_url=data.get('source_url'),
            dataset_name=data.get('dataset_name'),
            stages=stages,
            metadata_draft=data.get('metadata_draft') or {},
            license=data.get('license') or {},
            input_rows=data.get('input_rows'),
            max_drop_rate=data.get('max_drop_rate'),
            drop_rate_reason=data.get('drop_rate_reason'),
            script_path=data.get('script_path'),
            smoke_results_dir=data.get('smoke_results_dir'),
            results_dir=data.get('results_dir'),
            created_at=data.get('created_at'),
            updated_at=data.get('updated_at'),
        )

    # -- persistence ----------------------------------------------------------------

    def save(self) -> Path:
        """
        Write the manifest to disk, creating the working directory if needed.

        :returns: The path written to.
        """
        path = self.path_for(self.slug)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.updated_at = _timestamp()

        data: Dict[str, Any] = {
            'slug': self.slug,
            'source_url': self.source_url,
            'dataset_name': self.dataset_name,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'stages': {
                stage: self.stages.get(stage, StageState()).to_dict()
                for stage in STAGES
            },
        }

        # Only write the optional sections that actually carry something, so the file
        # stays readable by a human deciding whether a blocked run is worth resuming.
        for key in (
            'metadata_draft',
            'license',
            'input_rows',
            'max_drop_rate',
            'drop_rate_reason',
            'script_path',
            'smoke_results_dir',
            'results_dir',
        ):
            value = getattr(self, key)
            if value not in (None, {}, []):
                data[key] = value

        with path.open('w') as file:
            yaml.dump(data, file, default_flow_style=False, sort_keys=False)

        return path

    # -- stage state ----------------------------------------------------------------

    def state(self, stage: str) -> StageState:
        """
        Return the state of a stage.

        :param stage: The stage name.

        :returns: The stage state, pending if never recorded.
        :raises ValueError: If the stage name is not recognised.
        """
        _check_stage(stage)
        return self.stages.setdefault(stage, StageState())

    def mark(
        self,
        stage: str,
        status: str,
        blocker: Optional[str] = None,
        notes: Optional[str] = None,
        warnings: Optional[List[str]] = None,
        errors: Optional[List[str]] = None,
    ) -> None:
        """
        Record the outcome of a stage.

        Marking a stage as passed invalidates every later stage that had previously
        passed, setting it to :data:`STALE`. Re-deciding the metadata after the script
        was written means the script no longer reflects what was approved, and that
        has to be visible rather than silently accepted.

        :param stage: The stage name.
        :param status: One of :data:`PENDING`, :data:`PASSED`, :data:`BLOCKED`,
            :data:`STALE`.
        :param blocker: Plain-language description of the blocker, when blocked.
        :param notes: Free-form notes to carry forward.
        :param warnings: Every warning the gate emitted, verbatim.
        :param errors: Every error the gate emitted, verbatim.

        :raises ValueError: If the stage or status is not recognised.
        """
        _check_stage(stage)
        if status not in (PENDING, PASSED, BLOCKED, STALE):
            raise ValueError(f'unknown status: {status!r}')

        self.stages[stage] = StageState(
            status=status,
            at=_timestamp(),
            blocker=blocker,
            notes=notes,
            warnings=list(warnings or []),
            errors=list(errors or []),
        )

        if status == PASSED:
            for later in STAGES[STAGES.index(stage) + 1 :]:
                later_state = self.stages.get(later, StageState())
                if later_state.status == PASSED:
                    self.stages[later] = StageState(
                        status=STALE,
                        at=_timestamp(),
                        notes=f'invalidated by a re-run of the {stage} stage',
                    )

    def next_stage(self) -> Optional[str]:
        """
        Return the first stage that still needs work.

        :returns: The stage name, or ``None`` when the whole pipeline has passed.
        """
        for stage in STAGES:
            if self.state(stage).status != PASSED:
                return stage
        return None

    def blockers(self) -> Dict[str, str]:
        """
        Return every currently blocked stage and its blocker text.

        :returns: Mapping of stage name to blocker description.
        """
        return {
            stage: self.state(stage).blocker or 'blocked, no detail recorded'
            for stage in STAGES
            if self.state(stage).status == BLOCKED
        }

    def unmet_prerequisites(self, stage: str) -> List[str]:
        """
        Return the stages before ``stage`` that have not passed.

        Used to stop a stage running on the output of one that never completed - the
        failure mode where a script is written against metadata that was never agreed.

        :param stage: The stage about to be attempted.

        :returns: Names of the preceding stages that are not passed, in order.
        :raises ValueError: If the stage name is not recognised.
        """
        _check_stage(stage)
        return [
            earlier
            for earlier in STAGES[: STAGES.index(stage)]
            if self.state(earlier).status != PASSED
        ]

    # -- convenience ----------------------------------------------------------------

    @property
    def downloads_dir(self) -> Path:
        """The folder holding the files downloaded by the prepare stage."""
        return work_dir_for(self.slug) / 'downloads'

    @property
    def complete(self) -> bool:
        """``True`` when every stage has passed."""
        return self.next_stage() is None


def _check_stage(stage: str) -> None:
    """
    Validate a stage name.

    :param stage: The stage name to check.
    :raises ValueError: If the stage is not one of :data:`STAGES`.
    """
    if stage not in STAGES:
        valid = ', '.join(STAGES)
        raise ValueError(f'unknown stage {stage!r}. Valid stages: {valid}')


def _timestamp() -> str:
    """Return the current UTC time as an ISO-8601 string, to second resolution."""
    return (
        datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()
    )
