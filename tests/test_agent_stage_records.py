"""
Tests for the stage record fields that carry gate output forward.

A gate's warnings and errors used to be reduced to a count and a first-error string at
the moment they were recorded. These tests pin the behaviour that replaced that, since
the whole value of the change is that the text survives into a later session.
"""

from chem_mat_data.agent import manifest as manifest_module
from chem_mat_data.agent.manifest import (
    BLOCKED,
    PASSED,
    Manifest,
    StageState,
    _string_list,
)


def test_stage_state_defaults_to_empty_lists():
    state = StageState()

    assert state.warnings == []
    assert state.errors == []


def test_stage_state_omits_empty_lists_when_serialised():
    """The manifest is read by humans; empty keys are noise."""
    data = StageState(status=PASSED).to_dict()

    assert 'warnings' not in data
    assert 'errors' not in data


def test_stage_state_round_trips_warnings_and_errors():
    state = StageState(
        status=BLOCKED,
        blocker='first problem',
        warnings=['a warning'],
        errors=['first problem', 'second problem'],
    )

    restored = StageState.from_dict(state.to_dict())

    assert restored.warnings == ['a warning']
    assert restored.errors == ['first problem', 'second problem']
    assert restored.blocker == 'first problem'


def test_mark_records_every_error_not_just_the_first():
    """A blocked stage is the one most likely to be resumed cold."""
    manifest = Manifest.create('probe')

    manifest.mark(
        'prepare',
        BLOCKED,
        blocker='no data files',
        errors=['no data files', 'the recipe records no URL'],
    )

    assert manifest.state('prepare').errors == [
        'no data files',
        'the recipe records no URL',
    ]


def test_mark_records_warning_text_on_a_passing_stage():
    manifest = Manifest.create('probe')

    manifest.mark('prepare', PASSED, warnings=['no README captured'])

    assert manifest.state('prepare').warnings == ['no README captured']


def test_warnings_survive_a_save_and_load_cycle(monkeypatch, tmp_path):
    """The point of the change: the text outlives the session that produced it."""
    monkeypatch.setattr(manifest_module, 'WORK_DIR', tmp_path)

    manifest = Manifest.create('probe')
    manifest.mark(
        'license',
        PASSED,
        warnings=["license 'CC-BY-4.0' approved by human override: academic use"],
    )
    manifest.save()

    reloaded = Manifest.load('probe')

    assert reloaded.state('license').warnings == [
        "license 'CC-BY-4.0' approved by human override: academic use"
    ]


def test_errors_survive_a_save_and_load_cycle(monkeypatch, tmp_path):
    """A run resumed cold needs every error, not the headline."""
    monkeypatch.setattr(manifest_module, 'WORK_DIR', tmp_path)

    manifest = Manifest.create('probe')
    manifest.mark(
        'prepare',
        BLOCKED,
        blocker='no data files',
        errors=['no data files', 'the recipe records no URL'],
    )
    manifest.save()

    reloaded = Manifest.load('probe')

    assert reloaded.state('prepare').errors == [
        'no data files',
        'the recipe records no URL',
    ]
    assert reloaded.state('prepare').blocker == 'no data files'


def test_staleness_does_not_inherit_the_previous_warnings():
    """
    A stage invalidated by a re-run has no findings of its own any more.

    Carrying the old warnings forward would present conclusions about a run state that
    no longer exists as though they still applied.
    """
    manifest = Manifest.create('probe')
    manifest.mark('metadata', PASSED, warnings=['a stale warning'])
    manifest.mark('prepare', PASSED)

    invalidated = manifest.state('metadata')
    assert invalidated.warnings == []
    assert 'invalidated' in (invalidated.notes or '')


def test_notes_still_carry_the_staleness_message():
    """`notes` was kept alongside the new fields rather than replaced."""
    manifest = Manifest.create('probe')
    manifest.mark('script', PASSED)
    manifest.mark('metadata', PASSED)

    assert 'metadata' in (manifest.state('script').notes or '')


# -- defensive loading --

def test_string_list_accepts_a_bare_scalar():
    """A hand-edited manifest should not crash a resume."""
    assert _string_list('one thing') == ['one thing']


def test_string_list_handles_none_and_junk():
    assert _string_list(None) == []
    assert _string_list([]) == []
    assert _string_list(['a', None, 'b']) == ['a', 'b']
    assert _string_list(7) == ['7']


def test_from_dict_survives_a_scalar_where_a_list_belongs():
    state = StageState.from_dict(
        {'status': BLOCKED, 'errors': 'someone typed one string'}
    )

    assert state.errors == ['someone typed one string']
