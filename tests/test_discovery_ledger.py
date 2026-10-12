"""
Tests for :mod:`chem_mat_data.discovery.ledger` and the two CLI commands built
on it (``cmmanage discover check`` and ``cmmanage discover verdict``).

The ledger is what stops the ``/dataset-search`` skill spending a verification
subagent on a paper somebody already answered, so the cases that matter most
here are the *matching* ones: the same paper written three different ways must
resolve to the same verdict, and a hit on the human-decided rejection list must
win over everything else.
"""
import sys

import pytest

if sys.version_info < (3, 10):
    pytest.skip('manage.py requires Python 3.10+ (pycomex dependency)', allow_module_level=True)

from click.testing import CliRunner

from chem_mat_data.discovery import ledger as ledger_mod
from chem_mat_data.discovery.ledger import (
    STATUS_IN_CATALOG,
    STATUS_LISTED,
    STATUS_NEW,
    STATUS_REJECTED,
    STATUS_VERIFIED,
    CatalogIndex,
    Verdict,
    VerdictLedger,
    load_catalog,
    load_leads,
    load_rejected,
    lookup_rejected,
    normalize_doi,
    normalize_url,
    screen,
    screen_many,
)
from chem_mat_data.manage import cli


# == normalisation ==

@pytest.mark.parametrize('value,expected', [
    ('https://doi.org/10.1021/acs.jcim.0c01041', '10.1021/acs.jcim.0c01041'),
    ('http://doi.org/10.1021/ACS.JCIM.0C01041', '10.1021/acs.jcim.0c01041'),
    ('10.1021/acs.jcim.0c01041', '10.1021/acs.jcim.0c01041'),
    ('doi:10.1021/acs.jcim.0c01041', '10.1021/acs.jcim.0c01041'),
    ('[10.1038/s43588-024-00616-5](https://doi.org/10.1038/s43588-024-00616-5)',
     '10.1038/s43588-024-00616-5'),
    ('see 10.1021/foo.', '10.1021/foo'),
    ('https://zenodo.org/records/14392754', None),
    ('', None),
    (None, None),
])
def test_normalize_doi(value, expected):
    assert normalize_doi(value) == expected


@pytest.mark.parametrize('a,b', [
    ('https://zenodo.org/records/123', 'http://zenodo.org/records/123/'),
    ('https://www.nature.com/articles/x', 'nature.com/articles/X'),
])
def test_normalize_url_equates_equivalent_forms(a, b):
    assert normalize_url(a) == normalize_url(b)


# == the catalog ==

def _write_catalog(tmp_path, body):
    path = tmp_path / 'metadata.yml'
    path.write_text(body, encoding='utf-8')
    return path


def test_catalog_indexes_sources_and_records_unindexed(tmp_path):
    path = _write_catalog(tmp_path, """
datasets:
  tmqm:
    compounds: 100
    sources:
      - https://doi.org/10.1021/acs.jcim.0c01041
      - https://github.com/uiocompcat/tmQM
  ames:
    compounds: 6512
""")
    index = load_catalog(path)
    assert index.total == 2
    assert index.lookup('10.1021/acs.jcim.0c01041') == 'tmqm'
    # The GitHub source has no DOI, so it is matched on its normalised URL.
    assert index.lookup('http://github.com/uiocompcat/tmQM/') == 'tmqm'
    # A dataset declaring no sources cannot be matched, and must be reported.
    assert index.unindexed == ['ames']
    assert index.lookup('https://doi.org/10.9999/unknown') is None


def test_catalog_missing_file_yields_empty_index(tmp_path):
    """Screening is an optimisation - a broken catalog must degrade, not raise."""
    index = load_catalog(tmp_path / 'nope.yml')
    assert isinstance(index, CatalogIndex)
    assert index.total == 0
    assert index.lookup('10.1021/foo') is None


# == the rejection list ==

def test_rejected_collects_every_identifier_in_a_section(tmp_path):
    path = tmp_path / 'rejected_datasets.md'
    path.write_text("""
# Rejected

### tmQMg-L — ligand library

| **Publication** | [10.1038/s43588-024-00616-5](https://doi.org/10.1038/s43588-024-00616-5) |
| **Data record** | https://github.com/uiocompcat/tmQMg-L |

### polymer_tg — glass transition

| **Publication** | [10.1021/acs.jcim.1c01031](https://doi.org/10.1021/acs.jcim.1c01031) |
""", encoding='utf-8')
    entries = load_rejected(path)
    assert len(entries) == 2
    assert lookup_rejected('10.1038/s43588-024-00616-5', entries).startswith('tmQMg-L')
    # A shared data record is exactly the case worth stopping on, even though
    # the publication itself differs.
    assert lookup_rejected('https://github.com/uiocompcat/tmQMg-L', entries).startswith('tmQMg-L')
    assert lookup_rejected('10.1021/acs.jcim.1c01031', entries).startswith('polymer_tg')
    assert lookup_rejected('10.9999/other', entries) is None


# == the verdict ledger ==

def test_verdict_round_trips_through_disk(tmp_path):
    path = tmp_path / 'search_verdicts.md'
    written = VerdictLedger.load(path)
    written.append(Verdict(
        url='https://doi.org/10.1021/foo',
        decision='confirmed',
        name='foo_db',
        record_url='https://zenodo.org/records/999',
        why='Zenodo record | 12k SMILES + logP',
        date='2026-08-19',
    ))
    read_back = VerdictLedger.load(path)
    assert len(read_back.verdicts) == 1
    verdict = read_back.verdicts[0]
    assert verdict.decision == 'confirmed'
    assert verdict.name == 'foo_db'
    # A pipe inside a cell must survive rather than inventing a column.
    assert verdict.why == 'Zenodo record | 12k SMILES + logP'


def test_verdict_lookup_matches_paper_and_record(tmp_path):
    path = tmp_path / 'search_verdicts.md'
    ledger = VerdictLedger.load(path)
    ledger.append(Verdict(
        url='https://doi.org/10.1021/foo',
        decision='rejected',
        record_url='https://zenodo.org/records/999',
        why='method paper',
    ))
    reloaded = VerdictLedger.load(path)
    assert reloaded.lookup('10.1021/FOO') is not None
    assert reloaded.lookup('http://zenodo.org/records/999/') is not None
    assert reloaded.lookup('10.1021/bar') is None


def test_verdict_newlines_are_collapsed_into_one_row(tmp_path):
    path = tmp_path / 'search_verdicts.md'
    ledger = VerdictLedger.load(path)
    ledger.append(Verdict(url='https://doi.org/10.1021/foo', decision='unclear',
                          why='ACS bot check,\nno preprint found'))
    assert VerdictLedger.load(path).verdicts[0].why == 'ACS bot check, no preprint found'


@pytest.mark.parametrize('decision', ['maybe', '', 'CONFIRMED', 'yes'])
def test_verdict_rejects_vocabulary_outside_the_fixed_set(tmp_path, decision):
    ledger = VerdictLedger.load(tmp_path / 'search_verdicts.md')
    with pytest.raises(ValueError):
        ledger.append(Verdict(url='https://doi.org/10.1021/foo', decision=decision))


def test_verdict_requires_a_url(tmp_path):
    ledger = VerdictLedger.load(tmp_path / 'search_verdicts.md')
    with pytest.raises(ValueError):
        ledger.append(Verdict(url='   ', decision='confirmed'))


def test_verdict_ledger_tolerates_a_damaged_file(tmp_path):
    """Hand-edits and added prose must not break a sweep."""
    path = tmp_path / 'search_verdicts.md'
    path.write_text("""# Verdicts

Some prose a human added.

| url | decision | name | record_url | why | date |
|---|---|---|---|---|---|
| https://doi.org/10.1021/good | confirmed | good_db |  | fine | 2026-01-01 |
this line is not a row at all
| broken row with too few cells |
| https://doi.org/10.1021/bad | not-a-decision | | | | |
""", encoding='utf-8')
    ledger = VerdictLedger.load(path)
    assert len(ledger.verdicts) == 1
    assert ledger.verdicts[0].url == 'https://doi.org/10.1021/good'


# == screening ==

def test_screen_precedence_puts_human_rejection_first(tmp_path):
    """
    A paper that is in the catalog *and* rejected must report as rejected: the
    human decision is the one a reader needs to see.
    """
    catalog = load_catalog(_write_catalog(tmp_path, """
datasets:
  thing:
    sources: [https://doi.org/10.1021/both]
"""))
    rejected_path = tmp_path / 'rejected.md'
    rejected_path.write_text('### thing was refused\n\nhttps://doi.org/10.1021/both\n', encoding='utf-8')
    rejected = load_rejected(rejected_path)

    result = screen('10.1021/both', catalog=catalog, rejected=rejected,
                    ledger=VerdictLedger(path=tmp_path / 'v.md'))
    assert result.status == STATUS_REJECTED
    assert result.skip is True


def test_screen_reports_each_status_and_skip_flag(tmp_path):
    catalog = load_catalog(_write_catalog(tmp_path, """
datasets:
  known:
    sources: [https://doi.org/10.1021/known]
"""))
    ledger = VerdictLedger(path=tmp_path / 'v.md')
    ledger.append(Verdict(url='https://doi.org/10.1021/judged', decision='rejected',
                          why='method paper'))

    results = screen_many(
        ['10.1021/known', '10.1021/judged', '10.1021/fresh'],
        catalog=catalog, rejected=[], ledger=ledger,
    )
    assert [r.status for r in results] == [STATUS_IN_CATALOG, STATUS_VERIFIED, STATUS_NEW]
    assert [r.skip for r in results] == [True, True, False]
    assert results[0].detail == 'known'
    assert 'rejected' in results[1].detail


# == the CLI ==

@pytest.fixture
def isolated_ledgers(tmp_path, monkeypatch):
    """Point all three ledger paths at a scratch directory."""
    catalog = tmp_path / 'metadata.yml'
    catalog.write_text("""
datasets:
  known:
    sources: [https://doi.org/10.1021/known]
  nosources:
    compounds: 5
""", encoding='utf-8')
    rejected = tmp_path / 'rejected_datasets.md'
    rejected.write_text('### refused thing\n\nhttps://doi.org/10.1021/refused\n', encoding='utf-8')
    verdicts = tmp_path / 'search_verdicts.md'
    leads = tmp_path / 'MISSING_DATASETS.md'
    leads.write_text(
        '| # | Dataset / Paper | Data Type | Size | URL |\n'
        '|---|---|---|---|---|\n'
        '| 3 | Polyuniverse | Polymer SMILES | >1M | https://doi.org/10.1021/listed |\n',
        encoding='utf-8',
    )

    monkeypatch.setattr(ledger_mod, 'CATALOG_PATH', catalog)
    monkeypatch.setattr(ledger_mod, 'REJECTED_PATH', rejected)
    monkeypatch.setattr(ledger_mod, 'VERDICTS_PATH', verdicts)
    monkeypatch.setattr(ledger_mod, 'LEADS_PATH', leads)
    return tmp_path


def test_check_and_verdict_are_registered():
    result = CliRunner().invoke(cli, ['discover', '--help'])
    assert result.exit_code == 0, result.output
    assert 'check' in result.output
    assert 'verdict' in result.output


def test_check_json_reports_every_status(isolated_ledgers):
    import json

    result = CliRunner().invoke(cli, [
        'discover', 'check', '--json',
        'https://doi.org/10.1021/known',
        'https://doi.org/10.1021/refused',
        'https://doi.org/10.1021/fresh',
    ])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert [row['status'] for row in payload] == [
        STATUS_IN_CATALOG, STATUS_REJECTED, STATUS_NEW,
    ]
    assert [row['skip'] for row in payload] == [True, True, False]


def test_check_new_only_prints_just_the_work_left(isolated_ledgers):
    result = CliRunner().invoke(cli, [
        'discover', 'check', '--new-only',
        'https://doi.org/10.1021/known',
        'https://doi.org/10.1021/fresh',
    ])
    assert result.exit_code == 0, result.output
    assert result.output.strip() == 'https://doi.org/10.1021/fresh'


def test_check_from_csv_honours_the_flagged_column(isolated_ledgers, tmp_path):
    csv_path = tmp_path / 'stream.csv'
    csv_path.write_text(
        'openalex_id,url,score,flagged,publication_year\n'
        'W1,https://doi.org/10.1021/flagged,0.9,1,2026\n'
        'W2,https://doi.org/10.1021/unflagged,0.01,0,2026\n',
        encoding='utf-8',
    )
    flagged = CliRunner().invoke(cli, [
        'discover', 'check', '--from-csv', str(csv_path), '--new-only',
    ])
    assert flagged.output.strip() == 'https://doi.org/10.1021/flagged'

    every = CliRunner().invoke(cli, [
        'discover', 'check', '--from-csv', str(csv_path), '--all', '--new-only',
    ])
    assert every.output.strip().splitlines() == [
        'https://doi.org/10.1021/flagged',
        'https://doi.org/10.1021/unflagged',
    ]


def test_check_without_targets_fails_loudly(isolated_ledgers):
    result = CliRunner().invoke(cli, ['discover', 'check'])
    assert result.exit_code == 1
    assert 'nothing to check' in result.output


def test_check_warns_about_catalog_entries_it_cannot_match(isolated_ledgers):
    result = CliRunner().invoke(cli, ['discover', 'check', 'https://doi.org/10.1021/fresh'])
    assert result.exit_code == 0, result.output
    # `new` must not be allowed to read as `definitely absent`.
    assert 'nosources' in result.output


def test_verdict_appends_and_then_screens_as_verified(isolated_ledgers):
    runner = CliRunner()
    recorded = runner.invoke(cli, [
        'discover', 'verdict', 'https://doi.org/10.1021/fresh',
        '--decision', 'confirmed', '--name', 'fresh_db',
        '--why', 'Zenodo record | 8k SMILES',
    ])
    assert recorded.exit_code == 0, recorded.output

    rechecked = runner.invoke(cli, [
        'discover', 'check', '--json', 'https://doi.org/10.1021/fresh',
    ])
    import json
    row = json.loads(rechecked.output)[0]
    assert row['status'] == STATUS_VERIFIED
    assert 'fresh_db' in row['detail']
    assert row['skip'] is True


def test_verdict_refuses_a_duplicate_and_quotes_the_prior_reason(isolated_ledgers):
    runner = CliRunner()
    runner.invoke(cli, [
        'discover', 'verdict', 'https://doi.org/10.1021/fresh',
        '--decision', 'rejected', '--why', 'method paper, no deposit',
    ])
    again = runner.invoke(cli, [
        'discover', 'verdict', 'https://doi.org/10.1021/fresh',
        '--decision', 'confirmed', '--why', 'changed my mind',
    ])
    assert again.exit_code == 1
    assert 'already recorded as rejected' in again.output
    assert 'method paper, no deposit' in again.output

    forced = runner.invoke(cli, [
        'discover', 'verdict', 'https://doi.org/10.1021/fresh',
        '--decision', 'confirmed', '--why', 'changed my mind', '--force',
    ])
    assert forced.exit_code == 0, forced.output


def test_verdict_rejects_an_invalid_decision(isolated_ledgers):
    result = CliRunner().invoke(cli, [
        'discover', 'verdict', 'https://doi.org/10.1021/fresh',
        '--decision', 'maybe', '--why', 'x',
    ])
    assert result.exit_code != 0


# == regression: the newest verdict must win ==

def test_a_forced_correction_supersedes_the_earlier_verdict(tmp_path):
    """
    The ledger is append-only, so a correction is a second row. If the first row
    kept the lookup key, `verdict --force` would write to disk and change
    nothing - reporting success for a no-op.
    """
    path = tmp_path / 'search_verdicts.md'
    ledger = VerdictLedger.load(path)
    ledger.append(Verdict(url='https://doi.org/10.1021/x', decision='unclear',
                          why='could not reach the record'))
    VerdictLedger.load(path).append(
        Verdict(url='https://doi.org/10.1021/x', decision='confirmed',
                name='x_db', why='Zenodo record found on a second look')
    )

    reloaded = VerdictLedger.load(path)
    assert len(reloaded.verdicts) == 2, 'both rows must survive on disk'
    assert reloaded.lookup('https://doi.org/10.1021/x').decision == 'confirmed'
    assert screen('https://doi.org/10.1021/x', catalog=CatalogIndex(), rejected=[],
                  ledger=reloaded, leads={}).detail.startswith('confirmed')


def test_force_correction_through_the_cli_takes_effect(isolated_ledgers):
    import json

    runner = CliRunner()
    runner.invoke(cli, ['discover', 'verdict', 'https://doi.org/10.1021/fresh',
                        '--decision', 'unclear', '--why', 'paywalled'])
    forced = runner.invoke(cli, ['discover', 'verdict', 'https://doi.org/10.1021/fresh',
                                 '--decision', 'confirmed', '--why', 'found it', '--force'])
    assert forced.exit_code == 0, forced.output

    rechecked = runner.invoke(cli, ['discover', 'check', '--json',
                                    'https://doi.org/10.1021/fresh'])
    assert 'confirmed' in json.loads(rechecked.output)[0]['detail']


# == the standing leads list ==

def test_leads_are_reported_but_never_skipped(tmp_path):
    """
    A lead is not a decision: those rows were confirmed from the title alone and
    26% of that population was later measured as mislabelled. Screening must say
    so without letting the paper escape verification.
    """
    leads_path = tmp_path / 'MISSING_DATASETS.md'
    leads_path.write_text(
        '| # | Dataset / Paper | Data Type | Size | URL |\n'
        '|---|---|---|---|---|\n'
        '| 3 | Polyuniverse | Polymer SMILES | >1M | https://doi.org/10.1039/D4DD00196F |\n',
        encoding='utf-8',
    )
    leads = load_leads(leads_path)
    result = screen('https://doi.org/10.1039/D4DD00196F', catalog=CatalogIndex(),
                    rejected=[], ledger=VerdictLedger(path=tmp_path / 'v.md'), leads=leads)
    assert result.status == STATUS_LISTED
    assert result.skip is False, 'a lead must still be verified'
    assert result.detail == 'Polyuniverse'


def test_a_verdict_outranks_a_lead(tmp_path):
    """Once judged, the verdict is the answer - the lead is only a reminder."""
    leads_path = tmp_path / 'MISSING_DATASETS.md'
    leads_path.write_text(
        '| # | Dataset / Paper | Data | Size | URL |\n|---|---|---|---|---|\n'
        '| 3 | Polyuniverse | x | y | https://doi.org/10.1021/x |\n',
        encoding='utf-8',
    )
    ledger = VerdictLedger(path=tmp_path / 'v.md')
    ledger.append(Verdict(url='https://doi.org/10.1021/x', decision='rejected',
                          why='method paper'))
    result = screen('https://doi.org/10.1021/x', catalog=CatalogIndex(), rejected=[],
                    ledger=ledger, leads=load_leads(leads_path))
    assert result.status == STATUS_VERIFIED
    assert result.skip is True


def test_missing_leads_file_degrades_to_empty(tmp_path):
    assert load_leads(tmp_path / 'nope.md') == {}


def test_cli_reports_a_listed_paper_as_still_worth_verifying(isolated_ledgers):
    import json

    result = CliRunner().invoke(cli, ['discover', 'check', '--json',
                                      'https://doi.org/10.1021/listed'])
    row = json.loads(result.output)[0]
    assert row['status'] == STATUS_LISTED
    assert row['skip'] is False
    assert row['source'] == 'MISSING_DATASETS.md'

    # And --new-only must still emit it, since it is work that remains.
    listed = CliRunner().invoke(cli, ['discover', 'check', '--new-only',
                                      'https://doi.org/10.1021/listed'])
    assert listed.output.strip() == 'https://doi.org/10.1021/listed'


# == titles are carried through screening ==

def test_titles_from_the_csv_reach_the_json_and_the_table(isolated_ledgers, tmp_path):
    import json

    csv_path = tmp_path / 'stream.csv'
    csv_path.write_text(
        'openalex_id,url,score,flagged,publication_year,title\n'
        'W1,https://doi.org/10.1021/fresh,0.9,1,2026,A dataset of measured pKa values\n',
        encoding='utf-8',
    )
    as_json = CliRunner().invoke(cli, ['discover', 'check', '--from-csv', str(csv_path),
                                       '--json'])
    assert json.loads(as_json.output)[0]['title'] == 'A dataset of measured pKa values'

    table = CliRunner().invoke(cli, ['discover', 'check', '--from-csv', str(csv_path)])
    assert 'measured pKa' in table.output


def test_a_csv_without_titles_still_screens(isolated_ledgers, tmp_path):
    """The title column is optional; its absence must not change screening."""
    import json

    csv_path = tmp_path / 'stream.csv'
    csv_path.write_text(
        'openalex_id,url,score,flagged,publication_year\n'
        'W1,https://doi.org/10.1021/fresh,0.9,1,2026\n',
        encoding='utf-8',
    )
    result = CliRunner().invoke(cli, ['discover', 'check', '--from-csv', str(csv_path),
                                      '--json'])
    row = json.loads(result.output)[0]
    assert row['title'] == ''
    assert row['status'] == STATUS_NEW


# == --new-only and --json compose ==

def test_new_only_and_json_compose(isolated_ledgers, tmp_path):
    """
    `--new-only` selects rows, `--json` selects a format. They used to collide:
    `--new-only` returned first and `--json` was silently ignored, which trapped
    the one caller that wants the work list *with* its titles.
    """
    import json

    csv_path = tmp_path / 'stream.csv'
    csv_path.write_text(
        'openalex_id,url,score,flagged,publication_year,title\n'
        'W1,https://doi.org/10.1021/known,0.9,1,2026,Already in the catalog\n'
        'W2,https://doi.org/10.1021/fresh,0.8,1,2026,A genuinely new one\n',
        encoding='utf-8',
    )
    result = CliRunner().invoke(cli, ['discover', 'check', '--from-csv', str(csv_path),
                                      '--new-only', '--json'])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert [row['url'] for row in payload] == ['https://doi.org/10.1021/fresh']
    assert payload[0]['title'] == 'A genuinely new one'
    assert payload[0]['openalex_id'] == 'W2'


def test_json_without_new_only_still_lists_everything(isolated_ledgers, tmp_path):
    """The unfiltered form must keep reporting settled rows, marked skip."""
    import json

    csv_path = tmp_path / 'stream.csv'
    csv_path.write_text(
        'openalex_id,url,score,flagged,publication_year\n'
        'W1,https://doi.org/10.1021/known,0.9,1,2026\n'
        'W2,https://doi.org/10.1021/fresh,0.8,1,2026\n',
        encoding='utf-8',
    )
    payload = json.loads(CliRunner().invoke(
        cli, ['discover', 'check', '--from-csv', str(csv_path), '--json']).output)
    assert len(payload) == 2
    assert [row['skip'] for row in payload] == [True, False]


def test_new_only_alone_stays_pipeable(isolated_ledgers, tmp_path):
    csv_path = tmp_path / 'stream.csv'
    csv_path.write_text(
        'openalex_id,url,score,flagged,publication_year,title\n'
        'W2,https://doi.org/10.1021/fresh,0.8,1,2026,A title that must not appear\n',
        encoding='utf-8',
    )
    out = CliRunner().invoke(cli, ['discover', 'check', '--from-csv', str(csv_path),
                                   '--new-only']).output
    assert out.strip() == 'https://doi.org/10.1021/fresh'


def test_the_openalex_id_reaches_the_output(isolated_ledgers, tmp_path):
    """
    The brief tells the subagent to write `reviews/<openalex_id>.md`. A DOI cannot
    be a filename - it contains a slash - so the short id is the only usable key,
    and screening has to carry it.
    """
    import json

    csv_path = tmp_path / 'stream.csv'
    csv_path.write_text(
        'openalex_id,url,score,flagged,publication_year\n'
        'W7153478152,https://doi.org/10.5281/zenodo.19519213,0.3,1,2026\n',
        encoding='utf-8',
    )
    row = json.loads(CliRunner().invoke(
        cli, ['discover', 'check', '--from-csv', str(csv_path), '--json']).output)[0]
    assert row['openalex_id'] == 'W7153478152'
    assert '/' not in row['openalex_id'], 'must be usable as a filename'
