"""
Tests for how a human license decision interacts with the automatic policy.

Two defects motivated these. An approved override used to return from ``gate_license``
before the identifier, redistribution, attribution, upstream-provenance and
host-divergence checks had run, so an approval given for one reason silently waived
every other finding. And the final verification gate asked only whether an override key
*existed*, which meant a ``rejected`` decision - a non-empty dict - read as an approval
and a rejected dataset passed the last gate.
"""

from pathlib import Path

import pytest
import yaml

from chem_mat_data.agent import manifest as manifest_module
from chem_mat_data.agent.manifest import PASSED, Manifest
from chem_mat_data.agent.gates import gate_license, gate_verification


def valid_draft() -> dict:
    """Return a metadata draft that passes the metadata gate."""
    return {
        'dataset_name': 'demo_ds',
        'dataset_type': 'regression',
        'verbose': 'Demo Dataset',
        'description': (
            'A synthetic demonstration dataset of organic compounds with '
            'experimentally measured melting points in degrees Celsius, used to '
            'exercise the extraction pipeline from end to end.'
        ),
        'tags': ['Molecules', 'SMILES', 'Melting Point'],
        'sources': ['https://doi.org/10.0000/demo'],
        'target_count': 1,
        'target_descriptions': {'0': 'mpC - Experimental melting point (Celsius)'},
    }


def write_results(path: Path, name: str = 'demo_ds') -> Path:
    """
    Write a synthetic experiment results folder that passes verification.

    :param path: Folder to populate.
    :param name: Dataset base name.

    :returns: The populated folder.
    """
    path.mkdir(parents=True, exist_ok=True)
    (path / f'{name}.mpack').write_bytes(b'\x00' * 32)
    (path / f'{name}.mpack.gz').write_bytes(b'\x00' * 16)
    (path / f'{name}.csv').write_text('smiles,mpC\nCCO,-114\n')

    draft = valid_draft()
    (path / 'metadata.yml').write_text(
        yaml.dump(
            {
                'compounds': 1000,
                'targets': 1,
                'target_type': ['regression'],
                'description': draft['description'],
                'verbose': draft['verbose'],
                'raw': ['csv'],
                'tags': draft['tags'],
                'sources': draft['sources'],
                'target_descriptions': draft['target_descriptions'],
            }
        )
    )
    return path


@pytest.fixture
def work_root(tmp_path, monkeypatch):
    """
    Redirect the pipeline working directory into a temporary folder.

    :param tmp_path: pytest temporary directory.
    :param monkeypatch: pytest monkeypatch fixture.

    :returns: The temporary working root.
    """
    monkeypatch.setattr(manifest_module, 'WORK_DIR', tmp_path / '.dataset_work')
    return tmp_path / '.dataset_work'


def licence(**overrides) -> dict:
    """
    Build a license record whose *upstream* is non-permissive.

    This is the shape that matters: the paper's own license is fine, and the binding
    restriction lives upstream - exactly the enviPath and EPI Suite cases from the two
    validation runs.

    :param overrides: Fields to add or replace.

    :returns: The license mapping.
    """
    record = {
        'identifier': 'CC-BY-4.0',
        'evidence_url': 'https://zenodo.org/records/1',
        'redistribution': 'allowed',
        'attribution': 'Cite the paper.',
        'upstream_database': 'US EPA EPI Suite',
        'upstream_terms_url': 'https://example.gov/terms',
        'upstream_license': 'No open license; forbids derivative works.',
    }
    record.update(overrides)
    return record


def build(work_root, license_info, complete=False, tmp_path=None) -> Manifest:
    """
    Build a manifest carrying the given license.

    With ``complete=True`` the manifest is furnished well enough to reach the license
    section of the verification gate: every earlier stage passed, an approved metadata
    draft, and a real results folder. Without those the gate returns early on
    ``no results_dir recorded`` and never evaluates the license at all.

    :param work_root: The redirected working root fixture.
    :param license_info: The license mapping to record.
    :param complete: Whether to furnish a complete, verifiable run.
    :param tmp_path: Required when ``complete`` is set, for the results folder.

    :returns: The manifest.
    """
    manifest = Manifest.create('demo')
    manifest.license = license_info

    if complete:
        manifest.metadata_draft = valid_draft()
        for stage in ('prepare', 'metadata', 'license', 'script', 'execute'):
            manifest.mark(stage, PASSED)
        results = tmp_path / 'results' / 'debug'
        write_results(results)
        manifest.results_dir = str(results)

    return manifest


# -- an approval must not waive checks silently --

def test_approval_does_not_hide_what_it_waived(work_root):
    """The regression: this used to return before the upstream check ever ran."""
    manifest = build(
        work_root,
        licence(human_override={'decision': 'approved', 'reason': 'data record only'}),
    )

    report = gate_license(manifest)

    assert report.ok
    waived = [i.message for i in report.warnings if i.message.startswith('waived')]
    assert any('EPA EPI Suite' in message for message in waived), (
        'the upstream finding must survive the approval as a warning'
    )


def test_approval_reports_how_many_checks_it_waived(work_root):
    manifest = build(
        work_root,
        licence(human_override={'decision': 'approved', 'reason': 'data record only'}),
    )

    report = gate_license(manifest)

    summary = report.warnings[0].message
    assert 'approved by human override' in summary
    assert 'waived' in summary


def test_approval_of_a_clean_license_waives_nothing(work_root):
    manifest = build(
        work_root,
        licence(
            upstream_database=None,
            human_override={'decision': 'approved', 'reason': 'belt and braces'},
        ),
    )

    report = gate_license(manifest)

    assert report.ok
    assert not [i for i in report.warnings if i.message.startswith('waived')]


def test_approval_still_reports_a_missing_evidence_url(work_root):
    """An approval is a decision about terms, not a licence to omit the evidence."""
    record = licence(human_override={'decision': 'approved', 'reason': 'fine'})
    record.pop('evidence_url')

    report = gate_license(build(work_root, record))

    assert any('evidence_url' in i.message for i in report.warnings)


# -- a rejection is final --

def test_rejection_fails_the_license_gate(work_root):
    manifest = build(
        work_root, licence(human_override={'decision': 'rejected', 'reason': 'no'})
    )

    report = gate_license(manifest)

    assert not report.ok
    assert any('rejected' in i.message for i in report.errors)


def test_rejection_is_not_waived_by_its_own_presence(work_root):
    """A rejection must never downgrade the findings into warnings."""
    manifest = build(
        work_root, licence(human_override={'decision': 'rejected', 'reason': 'no'})
    )

    report = gate_license(manifest)

    assert not any(i.message.startswith('waived') for i in report.warnings)


@pytest.mark.parametrize(
    'override',
    [
        {'decision': 'maybe', 'reason': 'unclear'},
        {'reason': 'no decision recorded'},
        {'decision': 'approved'},
    ],
)
def test_malformed_overrides_cannot_waive_anything(work_root, override):
    report = gate_license(build(work_root, licence(human_override=override)))

    assert not report.ok
    assert not any(i.message.startswith('waived') for i in report.warnings)


# -- the final gate must re-check, not rubber-stamp --

def test_verification_rejects_a_rejected_license(work_root, tmp_path):
    """
    The headline regression.

    `if not override` is False for a rejection, because a rejection is a non-empty
    dict, so the final gate skipped the check entirely and a refused dataset passed.
    """
    manifest = build(
        work_root,
        licence(human_override={'decision': 'rejected', 'reason': 'not ours'}),
        complete=True,
        tmp_path=tmp_path,
    )

    report = gate_verification(manifest)

    assert not report.ok
    assert any('rejected' in i.message for i in report.errors)


def test_verification_rechecks_the_upstream_not_just_the_identifier(work_root, tmp_path):
    """
    A permissive paper license used to satisfy the final gate on its own.

    The upstream forbids derivative works and there is no override, so the run must
    not be allowed to finish.
    """
    manifest = build(work_root, licence(), complete=True, tmp_path=tmp_path)

    report = gate_verification(manifest)

    assert not report.ok
    assert any('EPA EPI Suite' in i.message for i in report.errors)


def test_verification_surfaces_the_waiver_rather_than_hiding_it(work_root, tmp_path):
    """An approved run finishes, but the final report still says what was waived."""
    manifest = build(
        work_root,
        licence(human_override={'decision': 'approved', 'reason': 'data record only'}),
        complete=True,
        tmp_path=tmp_path,
    )

    report = gate_verification(manifest)

    assert any('waived by human override' in i.message for i in report.warnings)
