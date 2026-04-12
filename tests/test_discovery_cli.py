"""
Integration-ish tests for the ``cmmanage discover`` CLI group.

The goal is not to exercise real OpenAlex or real torch — those paths are
covered by the unit tests in ``test_discovery_openalex.py`` and
``test_discovery_classifier.py``. Here we just verify that the CLI wiring
is correct: the group is registered, ``--help`` works, arguments are parsed,
and each command routes into the underlying module with the expected calls.
"""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from chem_mat_data.manage import cli


def test_discovery_group_is_registered():
    runner = CliRunner()
    result = runner.invoke(cli, ['discover', '--help'])
    assert result.exit_code == 0, result.output
    # All five commands must show up in the group help.
    for name in ('fetch', 'score', 'stream', 'model-download', 'model-info'):
        assert name in result.output


def test_discovery_group_panel_title_stays_discovery():
    """
    The CLI verb is ``discover`` but the top-level ``cmmanage --help`` panel
    heading must still read ``Discovery`` (via the ``display_title`` override
    on the group).
    """
    runner = CliRunner()
    result = runner.invoke(cli, ['--help'])
    assert result.exit_code == 0, result.output
    assert 'Discovery' in result.output


def test_discovery_run_command_is_gone():
    """The ``run`` command should no longer exist under the discover group."""
    runner = CliRunner()
    result = runner.invoke(cli, ['discover', 'run', '--help'])
    assert result.exit_code != 0


def test_discovery_fetch_help():
    runner = CliRunner()
    result = runner.invoke(cli, ['discover', 'fetch', '--help'])
    assert result.exit_code == 0, result.output
    assert '--limit' in result.output
    assert '--year-min' in result.output


def test_discovery_score_help():
    runner = CliRunner()
    result = runner.invoke(cli, ['discover', 'score', '--help'])
    assert result.exit_code == 0, result.output
    assert '--threshold' in result.output


def test_discovery_fetch_writes_jsonl(tmp_path):
    """`fetch` should call into fetch_works and persist yielded records."""
    out_path = tmp_path / 'papers.jsonl'

    def fake_fetch_works(*, state, limit, year_lower_bound, **kwargs):
        records = [
            {
                'openalex_id': f'https://openalex.org/W{i}',
                'title': f'Paper {i}',
                'abstract': 'Released a new materials dataset.',
                'publication_year': 2020,
            }
            for i in range(3)
        ]
        for rec in records:
            state.mark_seen(rec['openalex_id'])
            yield rec

    with patch(
        'chem_mat_data.manage.discovery_openalex_fetch_works',
        side_effect=fake_fetch_works,
    ):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ['discover', 'fetch', '--limit', '3', '--out', str(out_path)],
        )

    assert result.exit_code == 0, result.output
    assert out_path.exists()
    lines = [json.loads(line) for line in out_path.read_text().splitlines() if line]
    assert len(lines) == 3
    assert lines[0]['title'] == 'Paper 0'


def test_discovery_score_writes_all_and_flagged(tmp_path):
    """`score` should read an input JSONL and write both output files."""
    inp = tmp_path / 'in.jsonl'
    records = [
        {'openalex_id': 'W1', 'title': 'Negative', 'abstract': 'abstract 1'},
        {'openalex_id': 'W2', 'title': 'Positive', 'abstract': 'abstract 2'},
        {'openalex_id': 'W3', 'title': 'Mid', 'abstract': 'abstract 3'},
    ]
    inp.write_text('\n'.join(json.dumps(r) for r in records) + '\n')

    out_all = tmp_path / 'all.jsonl'
    out_flagged = tmp_path / 'flagged.jsonl'

    fake_classifier = MagicMock()
    fake_classifier.predict_score.side_effect = [0.01, 0.9, 0.3]

    with (
        patch(
            'chem_mat_data.manage.discovery_is_ml_stack_available',
            return_value=True,
        ),
        patch(
            'chem_mat_data.manage.discovery_load_classifier',
            return_value=fake_classifier,
        ),
        patch(
            'chem_mat_data.manage.discovery_ensure_classifier_model',
            return_value=Path(tmp_path / 'model'),
        ),
    ):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                'discover',
                'score',
                '--in',
                str(inp),
                '--out-all',
                str(out_all),
                '--out-flagged',
                str(out_flagged),
                '--threshold',
                '0.2',
                '--model-url',
                'http://example/ignored.tar.gz',
            ],
        )

    assert result.exit_code == 0, result.output
    assert out_all.exists()
    assert out_flagged.exists()

    all_records = [json.loads(line) for line in out_all.read_text().splitlines() if line]
    flagged_records = [
        json.loads(line) for line in out_flagged.read_text().splitlines() if line
    ]
    assert len(all_records) == 3
    # Threshold 0.2 → W2 (0.9) and W3 (0.3) should be flagged, W1 (0.01) not.
    flagged_ids = {r['openalex_id'] for r in flagged_records}
    assert flagged_ids == {'W2', 'W3'}


def test_discovery_model_download_invokes_ensure(tmp_path):
    """`model-download` should call ensure_classifier_model once."""
    with patch(
        'chem_mat_data.manage.discovery_ensure_classifier_model',
        return_value=tmp_path / 'model',
    ) as mocked:
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                'discover',
                'model-download',
                '--model-url',
                'http://example/classifier.tar.gz',
            ],
        )

    assert result.exit_code == 0, result.output
    mocked.assert_called_once()


def test_discovery_model_info_runs_without_download():
    """`model-info` should not trigger a download."""
    with patch(
        'chem_mat_data.manage.discovery_ensure_classifier_model'
    ) as mocked:
        runner = CliRunner()
        result = runner.invoke(cli, ['discover', 'model-info'])

    assert result.exit_code == 0, result.output
    mocked.assert_not_called()
