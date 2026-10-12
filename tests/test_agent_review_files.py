"""
The reviewing stages must leave a report on disk, and it must say who wrote it.

Stages 1 and 6 exist to have someone with no stake in the outcome read the licence and
the finished dataset. In one real run that property was lost without a trace: all three
subagents were spawned with a ``name``, which routes an agent through a mailbox, so the
spawn call returned a receipt rather than a report and the completion arrived as an idle
notification carrying no text. The agents finished their work; none of it came back.
Both stages were then done by the agent that had written the script, and the run passed
every gate.

Nothing mechanical noticed. The only reason anyone found out is that the agent chose to
write a paragraph admitting it. These tests cover the checks that make the same failure
loud: the report has to exist, has to be substantial, and has to name its author.
"""

import pytest

from chem_mat_data.agent import manifest as manifest_module
from chem_mat_data.agent.manifest import Manifest, PASSED, STAGES
from chem_mat_data.agent.gates import check_review, REVIEW_MIN_WORDS
from chem_mat_data.agent.verification import VerificationReport


#: A review long enough to clear the word floor, so tests exercise the check they name
#: rather than tripping over the length one.
BODY = ' '.join(['checked the artifacts against the paper and found no divergence'] * 12)


@pytest.fixture
def work_root(tmp_path, monkeypatch):
    """Redirect the pipeline working directory into a temporary folder."""
    monkeypatch.setattr(manifest_module, 'WORK_DIR', tmp_path / '.dataset_work')
    return tmp_path / '.dataset_work'


def write_review(manifest: Manifest, stage: str, text: str) -> None:
    """Write a review file for the given stage, creating the folder."""
    path = manifest.review_path(stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def run_check(manifest: Manifest, stage: str = 'verification') -> VerificationReport:
    """Run the review check on its own and hand back the report."""
    report = VerificationReport(results_dir=manifest.reviews_dir)
    check_review(manifest, stage, 'final review', report)
    return report


def test_a_missing_review_is_an_error(work_root):
    """
    The regression this file exists for.

    A run whose reviewer never reported used to pass silently. It must now fail, and
    the message must name the cause, because the agent hitting this will not know that
    naming a subagent is what lost its report.
    """
    manifest = Manifest.create('demo')

    report = run_check(manifest)

    assert not report.ok
    assert 'reviews/verification.md' in report.errors[0].message.replace('\\', '/')
    assert 'name' in report.errors[0].message


def test_a_real_review_passes(work_root):
    """The ordinary case: a subagent wrote a report, and the gate is satisfied."""
    manifest = Manifest.create('demo')
    write_review(manifest, 'verification', f'reviewer: subagent\n\n{BODY}')

    report = run_check(manifest)

    assert report.ok
    assert report.warnings == []


def test_a_human_review_passes(work_root):
    """
    The documented escape.

    Requiring a subagent specifically would block a run the maintainer reviewed
    personally, which is a stronger guarantee rather than a weaker one.
    """
    manifest = Manifest.create('demo')
    write_review(manifest, 'verification', f'reviewer: human\n\n{BODY}')

    report = run_check(manifest)

    assert report.ok


def test_a_self_review_is_reported_but_not_fatal(work_root):
    """
    Self-review is allowed and recorded, never silent.

    Blocking it outright would push an agent whose subagent failed into either lying
    about the author or abandoning a finished run. A warning keeps the honest path
    open while ensuring `--strict` puts the decision in front of a human.
    """
    manifest = Manifest.create('demo')
    write_review(manifest, 'verification', f'reviewer: self\n\n{BODY}')

    report = run_check(manifest)

    assert report.ok, 'a declared self-review should not fail the check outright'
    assert len(report.warnings) == 1
    assert report.unacknowledged_warnings, '--strict has to stop on this'
    assert 'same agent wrote' in report.warnings[0].message


def test_an_unsigned_review_is_an_error(work_root):
    """
    A review with no author is the failure mode being guarded against.

    The bad outcome is not an empty file - it is a plausible one written by the agent
    under review. Without a declared author there is nothing to tell the two apart.
    """
    manifest = Manifest.create('demo')
    write_review(manifest, 'verification', BODY)

    report = run_check(manifest)

    assert not report.ok
    assert 'does not say who wrote it' in report.errors[0].message


def test_an_unrecognised_author_is_an_error(work_root):
    """A creative label must not read as a pass."""
    manifest = Manifest.create('demo')
    write_review(manifest, 'verification', f'reviewer: the-pipeline\n\n{BODY}')

    report = run_check(manifest)

    assert not report.ok


def test_a_token_review_is_an_error(work_root):
    """A file that exists but says nothing is the same as no review."""
    manifest = Manifest.create('demo')
    write_review(manifest, 'verification', 'reviewer: subagent\n\nLooks good.')

    report = run_check(manifest)

    assert not report.ok
    assert 'nearly empty' in report.errors[0].message
    assert len('Looks good.'.split()) < REVIEW_MIN_WORDS


def test_the_author_line_survives_markdown_decoration(work_root):
    """
    An agent writing markdown will bold the label sooner or later.

    Rejecting `**reviewer:** subagent` would teach the reader that the check is about
    formatting rather than about authorship.
    """
    manifest = Manifest.create('demo')
    write_review(manifest, 'verification', f'**reviewer:** `subagent`\n\n{BODY}')

    report = run_check(manifest)

    assert report.ok, [i.message for i in report.errors]


def test_both_stages_have_their_own_file(work_root):
    """
    The licence review and the final review are separate obligations.

    They are done at opposite ends of the run by different agents, and one standing in
    for the other would leave one of the two stages unreviewed.
    """
    manifest = Manifest.create('demo')
    write_review(manifest, 'license', f'reviewer: subagent\n\n{BODY}')

    assert run_check(manifest, 'license').ok
    assert not run_check(manifest, 'verification').ok
    assert manifest.review_path('license') != manifest.review_path('verification')


def test_the_review_path_rejects_a_bogus_stage(work_root):
    """A typo in a stage name should not silently create a file nothing checks."""
    manifest = Manifest.create('demo')

    with pytest.raises(Exception):
        manifest.review_path('verificaton')


def test_the_verification_gate_requires_both_reviews(work_root, tmp_path):
    """
    End to end through the real gate, not just the helper.

    A unit-tested check that was never wired into a gate would pass this file while
    changing nothing about what the pipeline enforces.
    """
    from chem_mat_data.agent.gates import gate_verification

    manifest = Manifest.create('demo', source_url='https://example.org/paper')
    for stage in STAGES:
        manifest.mark(stage, PASSED)

    messages = ' '.join(i.message for i in gate_verification(manifest).errors)

    assert 'license review' in messages
    assert 'verification review' in messages
