"""
Tests for the separation of the run's source URL from the data record's URL.

`source_url` is the fixed point a run is judged against, written once by `init`. It used
to be a freely settable scalar, and the elanos_bp run overwrote it with the direct
download URL - not out of carelessness, but because the manifest had nowhere else to put
one. The effect was to make the license gate compare zenodo.org against zenodo.org, so
the preprint/published check silently never fired.

`record_url` gives that value a home and makes the comparison the right one: the
question is whether the license was read on the same host that served the files, not
whether the paper and its data live in the same place. A publisher article pointing at a
Zenodo record is the ordinary arrangement and should be silent.
"""

import pytest

from chem_mat_data.agent import manifest as manifest_module
from chem_mat_data.agent.manifest import Manifest
from chem_mat_data.agent.gates import gate_license


@pytest.fixture
def work_root(tmp_path, monkeypatch):
    """Redirect the pipeline working directory into a temporary folder."""
    monkeypatch.setattr(manifest_module, 'WORK_DIR', tmp_path / '.dataset_work')
    return tmp_path / '.dataset_work'


def build(source_url=None, record_url=None, evidence='https://zenodo.org/records/1'):
    """Build a manifest with a clean license recorded on ``evidence``."""
    manifest = Manifest.create('demo', source_url=source_url)
    manifest.record_url = record_url
    manifest.license = {
        'identifier': 'CC-BY-4.0',
        'evidence_url': evidence,
        'redistribution': 'allowed',
        'attribution': 'Cite the paper.',
        'upstream_database': None,
    }
    return manifest


def divergence_warnings(report) -> list:
    """Return the host-divergence warnings from a report."""
    return [i.message for i in report.warnings if 'license was established on' in i.message]


# -- the noise this removes --

def test_paper_and_data_on_different_hosts_is_silent(work_root):
    """
    The ordinary arrangement: article on a publisher, data on Zenodo.

    Comparing the publication against the license evidence made this warn on almost
    every real run, which is how a warning stops being read.
    """
    manifest = build(
        source_url='https://link.springer.com/article/10.1007/s11030-025-11196-5',
        record_url='https://zenodo.org/records/14392754',
        evidence='https://zenodo.org/records/14392754',
    )

    assert not divergence_warnings(gate_license(manifest))


# -- the signal it keeps --

def test_license_read_on_a_different_host_than_the_record_warns(work_root):
    """The case the check exists for: a preprint against its published version."""
    manifest = build(
        source_url='https://chemrxiv.org/engage/chemrxiv/article-details/abc',
        record_url='https://chemrxiv.org/engage/chemrxiv/article-details/abc',
        evidence='https://pubs.acs.org/doi/10.1021/example',
    )

    warnings = divergence_warnings(gate_license(manifest))
    assert warnings
    assert 'the data record' in warnings[0]


def test_falls_back_to_source_url_when_no_record_recorded(work_root):
    """Runs predating record_url must still get the check."""
    manifest = build(
        source_url='https://chemrxiv.org/engage/chemrxiv/article-details/abc',
        evidence='https://pubs.acs.org/doi/10.1021/example',
    )

    warnings = divergence_warnings(gate_license(manifest))
    assert warnings
    assert 'the run source' in warnings[0]


def test_no_urls_at_all_is_silent(work_root):
    assert not divergence_warnings(gate_license(build()))


# -- persistence --

def test_record_url_round_trips(work_root):
    manifest = Manifest.create('demo', source_url='https://example.org/article')
    manifest.record_url = 'https://zenodo.org/records/1'
    manifest.save()

    reloaded = Manifest.load('demo')

    assert reloaded.source_url == 'https://example.org/article'
    assert reloaded.record_url == 'https://zenodo.org/records/1'


def test_record_url_defaults_to_none(work_root):
    assert Manifest.create('demo').record_url is None


def test_a_manifest_without_record_url_still_loads(work_root):
    """Manifests written before the field existed must not break on resume."""
    manifest = Manifest.create('demo', source_url='https://example.org/article')
    manifest.save()

    path = manifest.path_for('demo')
    text = '\n'.join(
        line for line in path.read_text().splitlines() if 'record_url' not in line
    )
    path.write_text(text)

    assert Manifest.load('demo').record_url is None
