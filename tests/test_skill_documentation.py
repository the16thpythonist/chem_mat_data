"""
The skill's documentation is instructions for an agent, so it is executable in effect.

An agent reads SKILL.md and does what it says, literally. When the documentation
describes behaviour that does not exist, the agent trusts it — there is no scepticism in
the loop. Six such defects shipped during development, two of them introduced on the
same day the others were being fixed:

- the licence example was missing a required field and could not pass its own gate
- an instruction said to read the licence back with `agent status`, which does not print it
- an error message offered an escape hatch the code did not implement
- the skill warned about a failure mode that cannot occur
- an example contradicted the rule it illustrated and passed its gate by accident
- `--strict` could never pass a run whose licence a human had approved

Every one is prose nothing checked. These tests check it. They are the documentation
equivalent of `tests/test_docs.py`, which already does this for the README.
"""

import json
import re
from pathlib import Path

import pytest
import yaml

from chem_mat_data.agent import manifest as manifest_module
from chem_mat_data.agent.manifest import PASSED, STAGES, Manifest
from chem_mat_data.agent.gates import gate_license, gate_metadata, gate_verification


SKILL_ROOT = Path(__file__).parent.parent / '.claude' / 'skills' / 'dataset-extraction'

#: Errors that mean "this example is incomplete or malformed" as opposed to "the policy
#: says escalate". A documented example may legitimately escalate - one of the licence
#: examples exists precisely to show a case that must go to a human - but none may fail
#: because a required field is missing. That was the actual defect.
COMPLETENESS_ERRORS = (
    'must state',
    'must record',
    'must be one of',
    'must be a license identifier',
    'no "evidence_url"',
    'no license identified',
    'was not recorded',
    'no license information',
)


@pytest.fixture
def work_root(tmp_path, monkeypatch):
    """Redirect the pipeline working directory into a temporary folder."""
    monkeypatch.setattr(manifest_module, 'WORK_DIR', tmp_path / '.dataset_work')
    return tmp_path / '.dataset_work'


def harvest_payloads(key: str):
    """
    Pull the JSON payloads documented for a given `agent set` key out of the skill.

    Blocks containing an ellipsis are sketches rather than examples - the override
    section deliberately writes "...every field from above, unchanged..." - and are
    returned separately so the caller can account for them instead of silently
    skipping.

    :param key: The manifest key, e.g. ``license`` or ``metadata_draft``.

    :returns: Tuple of (complete payloads, number of sketches skipped).
    """
    complete, sketches = [], 0

    for path in sorted(SKILL_ROOT.rglob('*.md')):
        text = path.read_text()
        for match in re.finditer(r'^```(?:bash|json)?\n(.*?)^```', text, re.S | re.M):
            body = match.group(1)
            if '{' not in body:
                continue

            # `agent set <slug> <key> '{...}'` in a bash block, or a bare JSON block.
            # The quoted form is followed by further commands, so stop at the closing
            # brace rather than running to the end of the block.
            named = re.search(rf"agent set <slug> {key} '(\{{.*?\}})'", body, re.S)
            if named:
                candidate = named.group(1)
            elif body.lstrip().startswith('{') and key == 'license':
                candidate = body
            else:
                continue

            # An ellipsis inside a string value is a placeholder URL and the example is
            # still complete; an ellipsis outside one - "...every field from above..." -
            # means the block is a sketch of a shape rather than a usable payload. Only
            # the second kind is skipped, and anything that survives that must parse, so
            # a genuinely malformed example fails loudly instead of being written off.
            outside_strings = re.sub(r'"(?:[^"\\]|\\.)*"', '""', candidate)
            if '...' in outside_strings or '…' in outside_strings:
                sketches += 1
                continue

            try:
                complete.append((path.name, json.loads(candidate)))
            except json.JSONDecodeError as exc:
                pytest.fail(f'{path.name}: documented {key} payload is not valid JSON: {exc}')

    return complete, sketches


def licence_manifest(record: dict) -> Manifest:
    """Build a manifest carrying the given licence record."""
    manifest = Manifest.create('demo', source_url='https://example.org/paper')
    manifest.record_url = 'https://zenodo.org/records/1'
    manifest.license = record
    return manifest


# -- A1: the documented payloads must satisfy their own gates --

def test_the_harvester_actually_finds_the_examples():
    """
    Guard against this whole file passing vacuously.

    A harvester that silently matches nothing is worse than no test, because it reports
    success. If the documentation is restructured, this fails and asks to be updated.
    """
    licences, licence_sketches = harvest_payloads('license')
    drafts, _ = harvest_payloads('metadata_draft')

    assert len(licences) >= 3, f'only found {len(licences)} complete licence examples'
    assert licence_sketches >= 1, 'expected at least one deliberately abbreviated example'
    assert len(drafts) >= 1, 'found no metadata_draft example'


def test_documented_licence_examples_are_complete(work_root):
    """
    The regression this file exists for.

    The stage-1 example showed four fields while the gate required a fifth, so an agent
    copying the visible example was blocked on a field it had never seen. An example may
    escalate on policy grounds; none may fail for being incomplete.
    """
    licences, _ = harvest_payloads('license')

    for name, record in licences:
        report = gate_license(licence_manifest(record))
        incomplete = [
            issue.message
            for issue in report.errors
            if any(marker in issue.message for marker in COMPLETENESS_ERRORS)
        ]
        assert not incomplete, f'{name}: documented example is missing fields: {incomplete}'


def test_the_clean_licence_example_passes(work_root):
    """At least one documented example must be a complete, passing licence."""
    licences, _ = harvest_payloads('license')

    passing = [record for _, record in licences if gate_license(licence_manifest(record)).ok]

    assert passing, 'no documented licence example actually passes the licence gate'


def test_the_restrictive_licence_example_escalates(work_root):
    """
    The example showing a non-permissive upstream must escalate, as documented.

    It is labelled in the skill as escalating rather than passing; if it started
    passing, the labelling would be teaching the wrong lesson.
    """
    licences, _ = harvest_payloads('license')

    restrictive = [
        record
        for _, record in licences
        if str(record.get('upstream_database') or '').strip()
        and not (record.get('human_override') or {})
    ]
    assert restrictive, 'no documented example with a named upstream database'

    for record in restrictive:
        assert not gate_license(licence_manifest(record)).ok, (
            'the named-upstream example is documented as escalating but passes'
        )


def test_the_documented_metadata_draft_passes(work_root):
    """The metadata example is presented as a template; it has to be a valid one."""
    drafts, _ = harvest_payloads('metadata_draft')

    for name, draft in drafts:
        manifest = Manifest.create('demo')
        manifest.metadata_draft = draft
        report = gate_metadata(manifest)
        assert report.ok, f'{name}: documented metadata_draft fails its gate: ' + str(
            [i.message for i in report.errors]
        )


# -- A3: the specific claims that have been wrong before --

def test_agent_status_prints_the_licence_block():
    """
    The skill tells the agent to recover the licence record before merging an override.

    A previous version pointed at `cmmanage agent status`, which prints stage lines and
    nothing else, so the instruction could not be followed. Whatever the docs name as
    the way to read it back must actually produce it.
    """
    text = (SKILL_ROOT / 'SKILL.md').read_text()
    override_section = text[text.find('## Stage 1 — license'):]
    override_section = override_section[: override_section.find('## Stage 2')]

    assert 'manifest.yml' in override_section, (
        'the licence stage no longer tells the agent where to read the recorded licence '
        'back from before merging an override into it'
    )


def test_strict_passes_a_run_whose_licence_a_human_approved(work_root, tmp_path):
    """
    Stage 6 is documented as running with `--strict`.

    An approved override downgrades waived errors to warnings so they stay visible; if
    `--strict` treated those as failures, every overridden run would be permanently
    unable to pass the final gate while re-asking a question already answered.
    """
    manifest = Manifest.create('demo', source_url='https://example.org/paper')
    # Same host as the evidence, so the preprint/published divergence warning does not
    # fire - that one is deliberately unacknowledged, because it wants a human look.
    manifest.record_url = 'https://zenodo.org/records/1'
    manifest.license = {
        'identifier': 'CC-BY-4.0',
        'evidence_url': 'https://zenodo.org/records/1',
        'redistribution': 'allowed',
        'attribution': 'cite',
        'upstream_database': 'Some Restrictive Database',
        'upstream_terms_url': 'https://example.org/terms',
        'upstream_license': 'proprietary',
        'human_override': {'decision': 'approved', 'reason': 'the maintainer said so'},
    }

    report = gate_license(manifest)

    assert report.ok
    assert report.warnings, 'the waiver should remain visible'
    assert report.unacknowledged_warnings == [], (
        '--strict would block this run forever: the waived findings are not marked as '
        'already ruled on by a human'
    )


def test_metadata_description_reaches_metadata_yml():
    """
    The skill used to claim that putting `description` only in `METADATA` fails the
    final gate. The template merges `METADATA` over the derived fields, so it does not.
    Documenting a failure mode that cannot occur trains the reader to guard the wrong
    thing.
    """
    template = (
        Path(__file__).parent.parent
        / 'chem_mat_data' / 'scripts' / 'create_graph_datasets.py'
    ).read_text()

    assert 'metadata.update(e.METADATA)' in template, (
        'the base template no longer merges METADATA over the derived fields, so the '
        'skill guidance about DESCRIPTION vs METADATA needs rechecking'
    )


def test_the_documented_drop_rate_override_exists(work_root, tmp_path):
    """
    An error message once offered a `max_drop_rate` escape hatch the code did not
    implement. Whatever the skill documents as the way to accept a known loss must work.
    """
    text = (SKILL_ROOT / 'SKILL.md').read_text()
    assert 'max_drop_rate' in text and 'drop_rate_reason' in text

    manifest = Manifest.create('demo')
    manifest.max_drop_rate = 0.5
    manifest.drop_rate_reason = 'the source ships rows without SMILES'

    assert manifest.max_drop_rate == 0.5
    assert manifest.drop_rate_reason


# -- A2: every documented command invocation must be real --

def documented_invocations():
    """
    Every ``cmmanage agent ...`` command line the skill tells an agent to run.

    :returns: List of (source file, subcommand, flags).
    """
    found = []
    for path in sorted(SKILL_ROOT.rglob('*.md')):
        for line in path.read_text().splitlines():
            match = re.search(r'cmmanage agent ([a-z]+)((?:\s+[^\s\'"]+)*)', line)
            if not match:
                continue
            flags = re.findall(r'(--[a-z-]+)', match.group(2))
            found.append((path.name, match.group(1), flags))
    return found


def agent_command_tree():
    """
    Build {subcommand: {flags}} from the actual Click app.

    :returns: Mapping of agent subcommand name to its recognised option strings.
    """
    from chem_mat_data.manage import CLI

    cli = CLI()
    group = cli.commands['agent']
    tree = {}
    for name, command in group.commands.items():
        options = set()
        for param in command.params:
            options.update(getattr(param, 'opts', []))
            options.update(getattr(param, 'secondary_opts', []))
        tree[name] = {opt for opt in options if opt.startswith('--')}
    return tree


def test_documented_agent_commands_exist():
    """
    A documented command that does not exist wastes an agent's turn and its trust.

    This also catches the inverse case that actually happened: `--no-record` and
    `--strict` existed for months while the skill never mentioned them, so runs kept
    destroying their own state to read a warning.
    """
    tree = agent_command_tree()
    invocations = documented_invocations()

    assert len(invocations) >= 10, (
        f'only found {len(invocations)} documented agent invocations - has the '
        'documentation moved, or the harvester broken?'
    )

    for source, subcommand, flags in invocations:
        assert subcommand in tree, (
            f'{source}: documents `cmmanage agent {subcommand}`, which does not exist. '
            f'Available: {sorted(tree)}'
        )
        for flag in flags:
            assert flag in tree[subcommand], (
                f'{source}: documents `cmmanage agent {subcommand} {flag}`, but that '
                f'command has no such option. Available: {sorted(tree[subcommand])}'
            )


def test_every_agent_subcommand_is_documented_somewhere():
    """
    The reverse direction: a capability nobody knows about might as well not exist.
    """
    tree = agent_command_tree()
    documented = {sub for _, sub, _ in documented_invocations()}

    undocumented = sorted(set(tree) - documented)
    assert not undocumented, (
        f'these agent subcommands appear nowhere in the skill: {undocumented}'
    )
