"""
Tests for recording that a stage is waiting on a person.

The skill teaches the agent to compose a careful escalation - what is blocking, what was
tried, what is needed - and then told it to persist that by running the gate, which
records the gate's own mechanical error instead. `agent set` rejected `blocker` and
`notes`, so the diagnosis had nowhere to go and the manifest kept only the symptom.

The sharper problem was that `blocked` does not distinguish anything. Every failed gate
records it, including ones the agent fixes moments later, so a reader resuming cold
could not tell "needs a person with a subscription" from "the draft was not written
yet".
"""

import pytest

from chem_mat_data.agent import manifest as manifest_module
from chem_mat_data.agent.manifest import BLOCKED, PASSED, PENDING, Manifest, StageState


@pytest.fixture
def work_root(tmp_path, monkeypatch):
    """Redirect the pipeline working directory into a temporary folder."""
    monkeypatch.setattr(manifest_module, 'WORK_DIR', tmp_path / '.dataset_work')
    return tmp_path / '.dataset_work'


def test_escalate_marks_the_stage_and_records_the_reason(work_root):
    manifest = Manifest.create('demo')

    manifest.escalate(
        'prepare',
        why='the supplementary data is behind an ACS login',
        detail='crwl 403; no preprint found; SI URL redirects to login',
    )

    state = manifest.state('prepare')
    assert state.status == BLOCKED
    assert state.needs_human is True
    assert state.escalation == 'the supplementary data is behind an ACS login'
    assert 'crwl 403' in state.escalation_detail


def test_detail_is_optional(work_root):
    manifest = Manifest.create('demo')

    manifest.escalate('license', why='the upstream terms are ambiguous')

    assert manifest.state('license').escalation_detail is None


def test_an_escalation_must_say_something(work_root):
    """An empty reason is the failure mode this exists to prevent."""
    manifest = Manifest.create('demo')

    with pytest.raises(ValueError):
        manifest.escalate('prepare', why='   ')


def test_an_unknown_stage_raises(work_root):
    manifest = Manifest.create('demo')

    with pytest.raises(ValueError):
        manifest.escalate('nonsense', why='x')


# -- the interaction with gate runs, which is where this used to break --

def test_a_later_gate_run_does_not_erase_the_escalation(work_root):
    """
    The regression.

    `mark` built a fresh StageState, so recording a blocker and then running the gate
    - the order the skill prescribes - discarded the diagnosis and kept the symptom.
    """
    manifest = Manifest.create('demo')
    manifest.escalate('prepare', why='paywalled', detail='tried four routes')

    manifest.mark('prepare', BLOCKED, blocker='no data files in the downloads directory')

    state = manifest.state('prepare')
    assert state.needs_human is True
    assert state.escalation == 'paywalled'
    assert state.escalation_detail == 'tried four routes'
    # The gate's own output still replaces the previous blocker: it describes this run.
    assert state.blocker == 'no data files in the downloads directory'


def test_gate_output_is_replaced_rather_than_accumulated(work_root):
    manifest = Manifest.create('demo')
    manifest.mark('prepare', BLOCKED, blocker='first', errors=['first', 'second'])
    manifest.mark('prepare', BLOCKED, blocker='third', errors=['third'])

    state = manifest.state('prepare')
    assert state.blocker == 'third'
    assert state.errors == ['third']


def test_passing_clears_the_escalation(work_root):
    """A resolved blocker is resolved; a stale one would be its own quiet lie."""
    manifest = Manifest.create('demo')
    manifest.escalate('prepare', why='paywalled', detail='tried four routes')

    manifest.mark('prepare', PASSED)

    state = manifest.state('prepare')
    assert state.needs_human is False
    assert state.escalation is None
    assert state.escalation_detail is None


def test_an_ordinary_gate_failure_is_not_marked_as_needing_a_human(work_root):
    """`blocked` alone must stay distinguishable from `blocked and waiting on someone`."""
    manifest = Manifest.create('demo')

    manifest.mark('metadata', BLOCKED, blocker='dataset_name is not snake_case')

    assert manifest.state('metadata').needs_human is False
    assert manifest.state('metadata').escalation is None


# -- persistence --

def test_the_escalation_survives_a_save_and_load(work_root):
    manifest = Manifest.create('demo')
    manifest.escalate('prepare', why='paywalled', detail='tried four routes')
    manifest.save()

    reloaded = Manifest.load('demo')

    state = reloaded.state('prepare')
    assert state.needs_human is True
    assert state.escalation == 'paywalled'
    assert state.escalation_detail == 'tried four routes'


def test_a_manifest_without_the_fields_still_loads(work_root):
    """Manifests written before these fields existed must resume unchanged."""
    state = StageState.from_dict({'status': BLOCKED, 'blocker': 'something'})

    assert state.needs_human is False
    assert state.escalation is None


def test_clean_stages_do_not_write_the_fields(work_root):
    """The manifest is read by people; absent flags should not clutter it."""
    data = StageState(status=PENDING).to_dict()

    assert 'needs_human' not in data
    assert 'escalation' not in data
