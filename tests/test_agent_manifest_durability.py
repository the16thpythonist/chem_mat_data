"""
Tests for surviving a damaged manifest, and for `set` invalidating what it changes.

The manifest was rewritten in place with a plain truncating open on every `set` and
every `gate`, and `load` filled every gap with a default - so a file truncated by an
interruption loaded *without error* as a run where nothing had been done. A resuming
agent would re-download, re-license and re-script a finished run, and the human's
recorded licence decision would be gone.

Separately, `set` performed no staleness invalidation at all. Replacing the approved
metadata draft on a fully passed run left all six stages passed, which made every gate
that compares the manifest against reality satisfiable by editing the manifest.
"""

import pytest
import yaml

from chem_mat_data.agent import manifest as manifest_module
from chem_mat_data.agent.manifest import (
    BLOCKED,
    PASSED,
    STAGES,
    STALE,
    Manifest,
    ManifestDamagedError,
)


@pytest.fixture
def work_root(tmp_path, monkeypatch):
    """Redirect the pipeline working directory into a temporary folder."""
    monkeypatch.setattr(manifest_module, 'WORK_DIR', tmp_path / '.dataset_work')
    return tmp_path / '.dataset_work'


def completed(slug='demo') -> Manifest:
    """Build and save a manifest with every stage passed."""
    manifest = Manifest.create(slug, source_url='https://example.org/paper')
    for stage in STAGES:
        manifest.mark(stage, PASSED)
    manifest.license = {
        'identifier': 'CC-BY-4.0',
        'human_override': {'decision': 'approved', 'reason': 'the human said so'},
    }
    manifest.save()
    return manifest


# -- damage is refused rather than silently absorbed --

def test_a_truncated_manifest_is_refused(work_root):
    """The regression: this used to load as a run where nothing had been done."""
    manifest = completed()
    manifest.path_for('demo').write_text('slug: demo\nsource_url: https://example.org/p')

    with pytest.raises(ManifestDamagedError) as exc:
        Manifest.load('demo')

    assert 'truncated' in str(exc.value)


def test_unparseable_yaml_is_refused(work_root):
    completed()
    manifest_module.Manifest.path_for('demo').write_text('slug: [unclosed\n')

    with pytest.raises(ManifestDamagedError):
        Manifest.load('demo')


def test_a_non_mapping_document_is_refused(work_root):
    completed()
    Manifest.path_for('demo').write_text('- just\n- a list\n')

    with pytest.raises(ManifestDamagedError):
        Manifest.load('demo')


def test_stages_that_are_not_a_mapping_are_refused(work_root):
    completed()
    path = Manifest.path_for('demo')
    data = yaml.safe_load(path.read_text())
    data['stages'] = 'all of them'
    path.write_text(yaml.dump(data))

    with pytest.raises(ManifestDamagedError):
        Manifest.load('demo')


def test_an_individual_missing_stage_still_defaults_to_pending(work_root):
    """Forward compatibility must survive: a new stage in an old file is not damage."""
    completed()
    path = Manifest.path_for('demo')
    data = yaml.safe_load(path.read_text())
    del data['stages']['verification']
    path.write_text(yaml.dump(data))

    loaded = Manifest.load('demo')

    assert loaded.state('verification').status == 'pending'
    assert loaded.state('prepare').status == PASSED


def test_the_error_points_at_the_backup_when_one_exists(work_root):
    manifest = completed()
    manifest.save()  # a second save leaves a .bak of the first
    Manifest.path_for('demo').write_text('slug: demo\n')

    with pytest.raises(ManifestDamagedError) as exc:
        Manifest.load('demo')

    assert 'manifest.yml.bak' in str(exc.value)


# -- writes are atomic and backed up --

def test_saving_twice_leaves_a_backup(work_root):
    manifest = completed()
    manifest.save()

    assert Manifest.path_for('demo').with_suffix('.yml.bak').exists()


def test_the_backup_holds_the_previous_contents(work_root):
    manifest = completed()
    manifest.dataset_name = 'changed'
    manifest.save()

    backup = yaml.safe_load(
        Manifest.path_for('demo').with_suffix('.yml.bak').read_text()
    )

    assert backup.get('dataset_name') is None


def test_no_temporary_files_are_left_behind(work_root):
    completed()

    leftovers = list((work_root / 'demo').glob('.manifest-*'))

    assert leftovers == []


# -- invalidation --

@pytest.mark.parametrize(
    'stage,expected',
    [
        (STAGES[0], STAGES),
        ('prepare', STAGES[STAGES.index('prepare'):]),
        ('license', STAGES[STAGES.index('license'):]),
        ('metadata', STAGES[STAGES.index('metadata'):]),
        ('execute', STAGES[STAGES.index('execute'):]),
    ],
)
def test_invalidate_from_marks_the_stage_and_everything_after(
    work_root, stage, expected
):
    manifest = completed()

    invalidated = manifest.invalidate_from(stage, reason='because')

    assert invalidated == list(expected)
    for name in expected:
        assert manifest.state(name).status == STALE


def test_invalidate_from_leaves_earlier_stages_alone(work_root):
    manifest = completed()

    manifest.invalidate_from('script', reason='because')

    assert manifest.state('prepare').status == PASSED
    assert manifest.state('metadata').status == PASSED


def test_invalidate_from_does_not_touch_stages_that_had_not_passed(work_root):
    manifest = Manifest.create('demo')
    manifest.mark('prepare', PASSED)
    manifest.mark('metadata', BLOCKED, blocker='not done')

    manifest.invalidate_from('prepare', reason='because')

    assert manifest.state('prepare').status == STALE
    assert manifest.state('metadata').status == BLOCKED


def test_the_reason_is_recorded(work_root):
    manifest = completed()

    manifest.invalidate_from('license', reason='invalidated by a change to license')

    assert 'change to license' in manifest.state('license').notes
