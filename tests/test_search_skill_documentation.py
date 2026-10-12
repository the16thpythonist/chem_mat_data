"""
Tests for the ``/dataset-search`` skill's documentation.

Same reasoning as ``test_skill_documentation.py``: the skill is instructions an
agent follows literally, so prose describing a command that does not exist, or a
flag spelled wrong, is a defect with the authority of code and none of the
verification. An agent cannot tell a typo from an instruction.

What is pinned here is everything mechanically checkable — that every documented
`cmmanage discover` invocation resolves against the real Click tree in both
directions, that the documented vocabularies match the module's, and that the
counts the prose states about the live catalog are still true.
"""
import re
import sys
from pathlib import Path

import pytest

if sys.version_info < (3, 10):
    pytest.skip('manage.py requires Python 3.10+ (pycomex dependency)', allow_module_level=True)

SKILL_ROOT = Path(__file__).parent.parent / '.claude' / 'skills' / 'dataset-search'
REPO_ROOT = Path(__file__).parent.parent


def skill_files():
    """Every markdown file making up the skill."""
    return sorted(SKILL_ROOT.rglob('*.md'))


def skill_text():
    """The whole skill as one string, for claim-level assertions."""
    return '\n'.join(path.read_text(encoding='utf-8') for path in skill_files())


# == the skill exists and is well-formed ==

def test_the_skill_is_present_with_its_references():
    assert (SKILL_ROOT / 'SKILL.md').exists()
    names = {path.name for path in skill_files()}
    assert {'sweep.md', 'verification.md', 'bookkeeping.md'} <= names


def test_the_frontmatter_declares_a_name_and_description():
    text = (SKILL_ROOT / 'SKILL.md').read_text(encoding='utf-8')
    assert text.startswith('---\n')
    frontmatter = text.split('---', 2)[1]
    assert re.search(r'^name:\s*dataset-search\s*$', frontmatter, re.M)
    description = re.search(r'^description:\s*(.+)$', frontmatter, re.M)
    assert description and len(description.group(1)) > 80, (
        'the description is what decides whether the skill is selected at all'
    )


def test_every_referenced_file_exists():
    """A reference to a file that is not there sends an agent looking for nothing."""
    for path in skill_files():
        for match in re.finditer(r'`references/([a-z_-]+\.md)`', path.read_text(encoding='utf-8')):
            assert (SKILL_ROOT / 'references' / match.group(1)).exists(), (
                f'{path.name} points at references/{match.group(1)}, which does not exist'
            )


# == the documented commands resolve ==

def documented_invocations():
    """
    Every ``cmmanage discover ...`` command line the skill tells an agent to run.

    :returns: List of (source file, subcommand, flags).
    """
    found = []
    for path in skill_files():
        for line in path.read_text(encoding='utf-8').splitlines():
            match = re.search(r'cmmanage discover ([a-z-]+)((?:\s+[^\s\'"]+)*)', line)
            if not match:
                continue
            flags = re.findall(r'(--[a-z-]+)', match.group(2))
            found.append((path.name, match.group(1), flags))
    return found


def discover_command_tree():
    """
    Build {subcommand: {flags}} from the actual Click app.

    :returns: Mapping of discover subcommand name to its recognised option strings.
    """
    from chem_mat_data.manage import CLI

    group = CLI().commands['discover']
    tree = {}
    for name, command in group.commands.items():
        options = set()
        for param in command.params:
            options.update(getattr(param, 'opts', []))
            options.update(getattr(param, 'secondary_opts', []))
        tree[name] = {opt for opt in options if opt.startswith('--')}
    return tree


def test_the_harvester_actually_finds_the_invocations():
    """Guard against passing vacuously if the documentation moves."""
    invocations = documented_invocations()
    assert len(invocations) >= 8, (
        f'only found {len(invocations)} documented discover invocations - has the '
        'documentation moved, or the harvester broken?'
    )


def test_documented_discover_commands_exist():
    tree = discover_command_tree()
    for source, subcommand, flags in documented_invocations():
        assert subcommand in tree, (
            f'{source}: documents `cmmanage discover {subcommand}`, which does not '
            f'exist. Available: {sorted(tree)}'
        )
        for flag in flags:
            assert flag in tree[subcommand], (
                f'{source}: documents `cmmanage discover {subcommand} {flag}`, but '
                f'that command has no such option. Available: {sorted(tree[subcommand])}'
            )


def test_every_discover_subcommand_is_documented_somewhere():
    """A capability nobody knows about might as well not exist."""
    tree = discover_command_tree()
    documented = {sub for _, sub, _ in documented_invocations()}
    undocumented = sorted(set(tree) - documented)
    assert not undocumented, (
        f'these discover subcommands appear nowhere in the skill: {undocumented}'
    )


# == the documented vocabularies match the code ==

def test_the_documented_decision_vocabulary_is_the_real_one():
    """
    The skill tells the agent to pass exactly these three values. If the module's
    vocabulary changes and the prose does not, every verdict call starts failing.
    """
    from chem_mat_data.discovery.ledger import DECISIONS

    text = skill_text()
    for decision in DECISIONS:
        assert f'`{decision}`' in text, (
            f'the vocabulary includes {decision!r} but the skill never mentions it'
        )
    # And nothing invented: any word the skill presents as a --decision value must
    # actually be accepted.
    for match in re.finditer(r"--decision\s+'?([a-z]+)", text):
        assert match.group(1) in DECISIONS, (
            f'the skill documents `--decision {match.group(1)}`, which the ledger rejects'
        )


def test_the_documented_screen_statuses_are_the_real_ones():
    from chem_mat_data.discovery import ledger

    statuses = {
        ledger.STATUS_NEW,
        ledger.STATUS_IN_CATALOG,
        ledger.STATUS_REJECTED,
        ledger.STATUS_VERIFIED,
        ledger.STATUS_LISTED,
    }
    text = skill_text()
    for status in statuses:
        assert f'`{status}`' in text, (
            f'`discover check` can report {status!r} and the skill never says so'
        )


def test_the_documented_verdict_columns_match_the_ledger():
    from chem_mat_data.discovery.ledger import VERDICT_COLUMNS

    text = skill_text()
    header = '| ' + ' | '.join(VERDICT_COLUMNS) + ' |'
    assert header in text, (
        f'the ledger writes columns {VERDICT_COLUMNS} but the skill documents a '
        'different header'
    )


# == the documented defaults match the CLI ==

def test_the_documented_threshold_is_the_cli_default():
    from chem_mat_data.manage import CLI

    stream = CLI().commands['discover'].commands['stream']
    threshold = next(p for p in stream.params if '--threshold' in getattr(p, 'opts', []))
    assert f'{threshold.default}' in skill_text(), (
        f'the CLI default threshold is {threshold.default} and the skill does not say so'
    )


def test_the_skill_overrides_the_limit_and_says_it_is_an_override():
    """
    The skill deliberately sweeps 300 where the CLI defaults to 1000, because the
    verification cost scales with what the sweep flags. If it stops saying which is
    which, the next reader will assume they agree.
    """
    from chem_mat_data.manage import CLI

    stream = CLI().commands['discover'].commands['stream']
    limit = next(p for p in stream.params if '--limit' in getattr(p, 'opts', []))
    text = skill_text()
    assert '--limit 300' in text
    assert f'{limit.default}' in text, (
        f"the skill must name the CLI's own default ({limit.default}) when overriding it"
    )


# == the claims the prose makes about live state ==

def test_the_documented_catalog_blind_spot_is_still_accurate():
    """
    The skill tells the agent how many catalog datasets cannot be matched by URL,
    because that is what makes `new` mean 'no URL matched' rather than 'absent'.

    Only the *unindexed* count is pinned, deliberately. The catalog total moves
    whenever any dataset is added or removed for reasons that have nothing to do
    with this skill - `tadf` was withdrawn over its licence hours after this test
    was written, turning 46 into 45 and the suite red. The unindexed count moves
    only when a source-less dataset is added or removed, which is both rarer and
    actually about the blind spot the prose describes.
    """
    from chem_mat_data.discovery.ledger import load_catalog

    index = load_catalog()
    text = skill_text()
    assert f'{len(index.unindexed)} catalog datasets declare no' in text, (
        f'{len(index.unindexed)} catalog datasets declare no `sources`, but the '
        'skill documents a different number - update the prose'
    )
    # And the total must NOT be baked in anywhere, or this drifts again.
    assert f'of the {index.total} catalog' not in text
    assert f'of {index.total} catalog' not in text


def test_the_run_directory_is_gitignored():
    """The skill writes run state to a path that must not be committed."""
    ignored = (REPO_ROOT / '.gitignore').read_text(encoding='utf-8')
    assert '.dataset_search/' in ignored


def test_the_committed_ledgers_the_skill_writes_exist():
    assert (REPO_ROOT / 'docs' / 'search_verdicts.md').exists(), (
        'the skill appends verdicts here; the file must exist with its header'
    )
    assert (REPO_ROOT / 'MISSING_DATASETS.md').exists()


def test_the_skill_keeps_licensing_out_of_scope():
    """
    The one boundary that must not erode: this skill rules on whether data exists,
    never on whether it may be shipped.
    """
    text = skill_text()
    assert 'Do not evaluate the licence.' in text, (
        'the subagent brief must explicitly forbid a licence judgement'
    )
    assert 'dataset-extraction' in text, (
        'the skill must name where licensing is actually decided'
    )


def test_the_skill_warns_about_naming_subagents():
    """
    A named subagent's report never arrives. This was learned the expensive way in
    the extraction skill and applies identically here.
    """
    text = skill_text()
    assert re.search(r'without a `name`', text), (
        'the skill must tell the agent to spawn verification subagents unnamed'
    )


def test_the_skill_says_a_listed_paper_is_still_verified():
    """
    `listed` is the one status that does not skip. If the skill ever documents it
    alongside the other three matches, runs will start skipping leads - which is
    the exact mistake the status was created to avoid.
    """
    from chem_mat_data.discovery.ledger import SKIP_STATUSES, STATUS_LISTED

    assert STATUS_LISTED not in SKIP_STATUSES
    text = skill_text()
    assert 'verify it anyway' in text or 'still gets a subagent' in text or (
        '`new` and `listed` both get a subagent' in text
    ), 'the skill must say that a listed paper is still verified'


def test_the_skill_says_which_url_to_record_a_verdict_under():
    """
    Recording under the resolved publisher URL silently orphans the row. This is
    invisible at the time and only costs a wasted subagent one sweep later, so the
    instruction is the only thing preventing it.
    """
    text = skill_text()
    assert 'exactly as it appears in the CSV' in text or 'exactly as the CSV has it' in text


def test_the_skill_documents_the_stream_column_guard():
    """The refusal is only useful if the reader knows what to do about it."""
    text = skill_text()
    assert '--include-title' in text
    assert 'refuses' in text or 'refusal' in text


def test_the_subagent_brief_exists_in_exactly_one_place():
    """
    The brief was duplicated in SKILL.md and references/verification.md. They
    drifted, and the copy the skill pointed at as "the brief in full" was the
    stale one - so a run would have sent the unfixed brief. One copy, or this
    happens again.
    """
    marker = 'Determine whether this publication releases a dataset'
    copies = [path.name for path in skill_files()
              if marker in path.read_text(encoding='utf-8')]
    assert copies == ['SKILL.md'], (
        f'the brief appears in {copies}; it must live in SKILL.md alone'
    )


def test_the_run_placeholder_is_a_bare_directory_name():
    """
    Every use writes `.dataset_search/<run>/…` and supplies the prefix itself, so
    defining `run` as a path produced `.dataset_search/.dataset_search/…` - which
    fails a shell redirect and silently succeeds for the Write tool, scattering
    subagent reports into a directory nobody looks in.
    """
    text = (SKILL_ROOT / 'SKILL.md').read_text(encoding='utf-8')
    assert 'run=.dataset_search/' not in text, (
        '`run` must hold the bare directory name, not a path'
    )
    assert 'run=$(date +%F)' in text


def test_every_leads_section_shape_is_documented():
    """
    `MISSING_DATASETS.md` has five sections with different column counts. A row
    wider than its header silently loses its trailing cells when rendered, and the
    URL is last - so an over-wide row drops exactly what /dataset-extraction needs.
    """
    import re

    leads = (REPO_ROOT / 'MISSING_DATASETS.md').read_text(encoding='utf-8')
    headers = re.findall(r'^\|\s*#\s*\|.*\|$', leads, re.M)
    widths = [len(h.strip('|').split('|')) for h in headers]
    assert len(set(widths)) > 1, 'if the sections agree now, simplify the prose instead'

    text = skill_text()
    assert 'do not share a header' in text, (
        'the skill must say the sections have different column counts'
    )
    # Pin the actual widths, not just the claim that they differ: the earlier
    # version of this test passed with the documented counts wrong and with the
    # shape table deleted entirely.
    spelled = ', '.join(str(w) for w in widths[:-1]) + f' and {widths[-1]} columns'
    assert spelled in text, (
        f'the real header widths are {widths}; the prose must say "{spelled}"'
    )
    # And the row count the prose quotes must be the real one.
    rows = len(re.findall(r'^\|\s*\d+\s*\|', leads, re.M))
    assert f'{rows} numbered rows' in text, (
        f'{rows} numbered rows in MISSING_DATASETS.md; the prose says otherwise'
    )


def test_shell_snippets_do_not_rely_on_variables_from_another_command():
    """
    Shell state does not carry between Bash invocations, so a block that reads
    `$run` set by an earlier, separate block silently gets an empty string - and
    writes the run's artefacts to the wrong place while exiting 0.

    `$run` may only be *assigned* and used inside the setup block that creates the
    directory. Everywhere else the skill uses the `<run>` placeholder.
    """
    import re

    text = (SKILL_ROOT / 'SKILL.md').read_text(encoding='utf-8')
    blocks = re.findall(r'```bash\n(.*?)```', text, re.S)
    for block in blocks:
        if 'run=$(date' in block:
            continue  # the setup block, which both assigns and uses it
        assert '$run' not in block, (
            'a shell snippet reads `$run` without assigning it; use the `<run>` '
            f'placeholder instead:\n{block}'
        )
