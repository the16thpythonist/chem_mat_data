"""
Tests that the final gate compares what shipped against what was approved.

`gate_verification` claimed to confirm that `metadata.yml` matches the approved draft
while comparing only four fields. Probed on a real manifest, replacing the draft's
`target_descriptions`, `dataset_type` and `dataset_name` produced zero issues.

What this catches is transcription drift: the draft is approved, and then the script's
own `METADATA` literal diverges from it. It does *not* confirm the draft was right -
an agent wrong about a unit writes the same wrong string on both sides. The observed
value checks in `gate_execute` cover that.
"""

import yaml

import pytest

from chem_mat_data.agent import manifest as manifest_module
from chem_mat_data.agent.manifest import PASSED, Manifest
from chem_mat_data.agent.gates import gate_verification


@pytest.fixture
def work_root(tmp_path, monkeypatch):
    """Redirect the pipeline working directory into a temporary folder."""
    monkeypatch.setattr(manifest_module, 'WORK_DIR', tmp_path / '.dataset_work')
    return tmp_path / '.dataset_work'


def draft() -> dict:
    """A metadata draft that matches what write_results writes."""
    return {
        'dataset_name': 'demo_ds',
        'dataset_type': 'regression',
        'verbose': 'Demo Dataset',
        'description': 'A synthetic dataset used to exercise the pipeline end to end.',
        'tags': ['Molecules', 'SMILES'],
        'sources': ['https://doi.org/10.0000/demo'],
        'target_count': 1,
        'target_descriptions': {'0': 'mpC - melting point in degrees Celsius'},
    }


def write_results(path, name='demo_ds', **metadata_overrides):
    """Write a results folder whose metadata matches the draft unless overridden."""
    path.mkdir(parents=True, exist_ok=True)
    smiles = ['CCO', 'CCC']
    (path / f'{name}.mpack').write_bytes(b'\x00' * 32)
    (path / f'{name}.mpack.gz').write_bytes(b'\x00' * 16)
    (path / f'{name}.csv').write_text(
        'smiles,mpC\n' + ''.join(f'{s},1.0\n' for s in smiles)
    )

    base = draft()
    metadata = {
        'compounds': 1000,
        'targets': 1,
        'target_type': [base['dataset_type']],
        'description': base['description'],
        'verbose': base['verbose'],
        'raw': ['csv'],
        'tags': base['tags'],
        'sources': base['sources'],
        'target_descriptions': base['target_descriptions'],
    }
    metadata.update(metadata_overrides)
    (path / 'metadata.yml').write_text(yaml.dump(metadata, allow_unicode=True))
    return path


def build(tmp_path, name='demo_ds', draft_overrides=None, **metadata_overrides):
    """Build a complete manifest pointing at a results folder."""
    manifest = Manifest.create('demo')
    manifest.metadata_draft = draft()
    if draft_overrides:
        manifest.metadata_draft.update(draft_overrides)
    manifest.license = {
        'identifier': 'CC-BY-4.0',
        'evidence_url': 'https://example.org',
        'attribution': 'cite',
        'upstream_database': None,
    }
    for stage in ('prepare', 'metadata', 'license', 'script', 'execute'):
        manifest.mark(stage, PASSED)
    results = tmp_path / 'results' / 'debug'
    write_results(results, name=name, **metadata_overrides)
    manifest.results_dir = str(results)
    return manifest


def test_a_matching_run_passes(work_root, tmp_path):
    report = gate_verification(build(tmp_path))

    assert report.ok, [i.message for i in report.errors]


# -- target descriptions --

def test_a_changed_target_description_warns(work_root, tmp_path):
    """The regression: this produced no issue at all."""
    manifest = build(
        tmp_path,
        target_descriptions={'0': 'mpC - boiling point in KELVIN, log-transformed'},
    )

    report = gate_verification(manifest)

    assert report.ok
    assert any('description of target 0' in i.message for i in report.warnings)


def test_a_missing_target_description_is_an_error(work_root, tmp_path):
    manifest = build(tmp_path, target_descriptions={})

    report = gate_verification(manifest)

    assert not report.ok
    assert any('has no description' in i.message for i in report.errors)


def test_integer_and_string_keys_compare_equal(work_root, tmp_path):
    """Both forms occur in the existing metadata; neither should read as drift."""
    manifest = build(
        tmp_path,
        target_descriptions={0: 'mpC - melting point in degrees Celsius'},
    )

    report = gate_verification(manifest)

    assert report.ok
    assert not [i for i in report.warnings if 'description of target' in i.message]


# -- task type --

def test_a_changed_task_type_is_an_error(work_root, tmp_path):
    manifest = build(tmp_path, target_type=['classification'])

    report = gate_verification(manifest)

    assert not report.ok
    assert any('wrong task' in i.message for i in report.errors)


def test_task_type_casing_does_not_matter(work_root, tmp_path):
    manifest = build(tmp_path, target_type=['Regression'])

    assert gate_verification(manifest).ok


# -- dataset name --

def test_a_renamed_dataset_is_an_error(work_root, tmp_path):
    """The identifier a user passes to load_smiles_dataset."""
    manifest = build(tmp_path, name='something_else')

    report = gate_verification(manifest)

    assert not report.ok
    assert any('load_smiles_dataset' in i.message for i in report.errors)


def test_no_mpack_does_not_raise(work_root, tmp_path):
    """A partial run should degrade, not crash, on the name check."""
    manifest = build(tmp_path)
    for path in (tmp_path / 'results' / 'debug').glob('*.mpack'):
        path.unlink()

    report = gate_verification(manifest)

    assert not report.ok
    assert not any('load_smiles_dataset' in i.message for i in report.errors)
