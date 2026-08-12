"""
Tests for the two license fields the gate matches against fixed vocabularies.

`redistribution` and `upstream_license` are compared, not read. Prose in either used to
fail for the wrong reason: a sentence saying the upstream is CC0 escalated exactly like
a sentence saying it forbids redistribution, because neither is on the permissive list.
An escalation that fires either way carries no information, so the gate now rejects
prose explicitly and says what shape it wanted.

The real elanos_bp run recorded a full paragraph in `redistribution`, which the early
return on an approved override had been hiding.
"""

import pytest

from chem_mat_data.agent import manifest as manifest_module
from chem_mat_data.agent.manifest import Manifest
from chem_mat_data.agent.gates import gate_license, looks_like_identifier


@pytest.fixture
def work_root(tmp_path, monkeypatch):
    """Redirect the pipeline working directory into a temporary folder."""
    monkeypatch.setattr(manifest_module, 'WORK_DIR', tmp_path / '.dataset_work')
    return tmp_path / '.dataset_work'


def build(**overrides) -> Manifest:
    """Build a manifest with an otherwise-clean license record."""
    record = {
        'identifier': 'CC-BY-4.0',
        'evidence_url': 'https://zenodo.org/records/1',
        'redistribution': 'allowed',
        'attribution': 'Cite the paper.',
        'upstream_database': None,
    }
    record.update(overrides)
    manifest = Manifest.create('demo')
    manifest.license = record
    return manifest


# -- looks_like_identifier --

@pytest.mark.parametrize(
    'value',
    ['CC-BY-4.0', 'MIT', 'CC0-1.0', 'proprietary', 'none', 'public domain', 'Apache-2.0'],
)
def test_identifiers_are_recognised(value):
    assert looks_like_identifier(value)


@pytest.mark.parametrize(
    'value',
    [
        'No open license. EPA asserts copyright worldwide over EPI Suite.',
        'academic and non-commercial use only; systematic redistribution prohibited',
        'allowed by the data record itself (Zenodo record 14392754 is CC-BY-4.0)',
        'The terms permit use but not redistribution of substantial portions.',
    ],
)
def test_prose_is_not_mistaken_for_an_identifier(value):
    assert not looks_like_identifier(value)


def test_empty_is_not_an_identifier():
    assert not looks_like_identifier(None)
    assert not looks_like_identifier('')


# -- redistribution --

@pytest.mark.parametrize('value', ['allowed', 'permitted', 'yes', 'ALLOWED', ' allowed '])
def test_allowed_vocabulary_passes(work_root, value):
    assert gate_license(build(redistribution=value)).ok


@pytest.mark.parametrize('value', ['forbidden', 'no', 'restricted', 'unclear'])
def test_denied_vocabulary_is_reported_as_a_verdict(work_root, value):
    """A stated refusal must read as a refusal, not as a formatting complaint."""
    report = gate_license(build(redistribution=value))

    assert not report.ok
    assert any('concluded redistribution is' in i.message for i in report.errors)


def test_prose_in_redistribution_is_named_as_prose(work_root):
    """The exact shape the real run produced."""
    report = gate_license(
        build(
            redistribution=(
                'allowed by the data record itself (Zenodo record 14392754 is '
                'CC-BY-4.0, open access); the upstream measured values, however, come '
                'from the US EPA EPI Suite'
            )
        )
    )

    assert not report.ok
    assert any('must be one of' in i.message for i in report.errors)


# -- upstream_license --

def test_prose_in_upstream_license_is_named_as_prose(work_root):
    report = gate_license(
        build(
            upstream_database='US EPA EPI Suite',
            upstream_terms_url='https://example.gov/terms',
            upstream_license=(
                'No open license. EPA asserts copyright worldwide and grants only '
                'personal use.'
            ),
        )
    )

    assert not report.ok
    assert any('must be a license identifier' in i.message for i in report.errors)


def test_proprietary_upstream_escalates_as_a_restriction(work_root):
    """`proprietary` is the identifier for stated-but-restrictive terms."""
    report = gate_license(
        build(
            upstream_database='US EPA EPI Suite',
            upstream_terms_url='https://example.gov/terms',
            upstream_license='proprietary',
        )
    )

    assert not report.ok
    assert any('not permissive' in i.message for i in report.errors)


def test_permissive_upstream_identifier_passes(work_root):
    """
    The case the old behaviour could not express.

    A genuinely open upstream, stated as an identifier, must pass - previously any
    description of it escalated regardless of what it said.
    """
    report = gate_license(
        build(
            upstream_database='PubChem',
            upstream_terms_url='https://example.org/terms',
            upstream_license='CC0-1.0',
        )
    )

    assert report.ok, [i.message for i in report.errors]


def test_the_documented_skill_examples_behave_as_documented(work_root):
    """Guards against the examples drifting out of step with the gate again."""
    clean = gate_license(build(upstream_database=None))
    assert clean.ok

    named = gate_license(
        build(
            upstream_database='US EPA EPI Suite (values from the SRC PHYSPROP database)',
            upstream_terms_url='https://www.epa.gov/terms',
            upstream_license='proprietary',
        )
    )
    assert not named.ok
    assert any('not permissive' in i.message for i in named.errors)
