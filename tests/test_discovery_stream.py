"""
Unit and CLI tests for the ``cmmanage discover stream`` command.

Covers:

- :func:`chem_mat_data.discovery.io.build_csv_row` (pure projection logic,
  DOI → URL fallback, flag semantics, ``include_title`` toggle).
- The wired-up ``stream`` command itself, with ``fetch_works`` and
  ``load_classifier`` monkey-patched so the tests stay fast and offline.
"""
import sys
import csv
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

if sys.version_info < (3, 10):
    pytest.skip('manage.py requires Python 3.10+ (pycomex dependency)', allow_module_level=True)

from click.testing import CliRunner

from chem_mat_data.discovery.io import (
    STREAM_CSV_BASE_COLUMNS,
    build_csv_row,
)
from chem_mat_data.manage import cli


# ---------------------------------------------------------------------------
# build_csv_row unit tests
# ---------------------------------------------------------------------------


class TestBuildCsvRow:
    def _record(self, **overrides):
        base = {
            'openalex_id': 'https://openalex.org/W4401002345',
            'doi': 'https://doi.org/10.1038/s41586-024-00000-0',
            'title': 'A new benchmark dataset for molecular property prediction',
            'abstract': 'Abstract text.',
            'publication_year': 2024,
        }
        base.update(overrides)
        return base

    def test_basic_projection_has_all_base_columns(self):
        row = build_csv_row(self._record(), score=0.91, threshold=0.05)
        assert set(row.keys()) == set(STREAM_CSV_BASE_COLUMNS)
        # Dict ordering should match STREAM_CSV_BASE_COLUMNS for stable headers.
        assert tuple(row.keys()) == STREAM_CSV_BASE_COLUMNS

    def test_short_id_extracted_from_full_openalex_url(self):
        row = build_csv_row(self._record(), score=0.5, threshold=0.05)
        assert row['openalex_id'] == 'W4401002345'

    def test_short_id_is_empty_string_when_missing(self):
        row = build_csv_row(
            self._record(openalex_id=None), score=0.5, threshold=0.05
        )
        assert row['openalex_id'] == ''

    def test_doi_url_when_doi_is_full_url(self):
        row = build_csv_row(self._record(), score=0.5, threshold=0.05)
        assert row['url'] == 'https://doi.org/10.1038/s41586-024-00000-0'

    def test_doi_url_when_doi_is_bare_identifier(self):
        row = build_csv_row(
            self._record(doi='10.1038/s41586-024-00000-0'),
            score=0.5,
            threshold=0.05,
        )
        assert row['url'] == 'https://doi.org/10.1038/s41586-024-00000-0'

    def test_falls_back_to_openalex_landing_when_doi_missing(self):
        row = build_csv_row(
            self._record(doi=None), score=0.5, threshold=0.05
        )
        assert row['url'] == 'https://openalex.org/W4401002345'

    def test_falls_back_to_reconstructed_landing_when_only_short_id_in_record(self):
        rec = self._record(openalex_id=None, doi=None)
        row = build_csv_row(rec, score=0.5, threshold=0.05)
        # With both fields missing, we accept an empty URL rather than inventing one.
        assert row['url'] == ''

    def test_score_is_formatted_to_four_decimals(self):
        row = build_csv_row(self._record(), score=0.87654321, threshold=0.05)
        assert row['score'] == '0.8765'

    def test_flagged_is_1_when_score_above_threshold(self):
        row = build_csv_row(self._record(), score=0.06, threshold=0.05)
        assert row['flagged'] == 1

    def test_flagged_is_0_when_score_equal_to_threshold(self):
        # Strict > semantics, matching the `score` command.
        row = build_csv_row(self._record(), score=0.05, threshold=0.05)
        assert row['flagged'] == 0

    def test_flagged_is_0_when_score_below_threshold(self):
        row = build_csv_row(self._record(), score=0.01, threshold=0.05)
        assert row['flagged'] == 0

    def test_include_title_adds_trailing_column(self):
        row = build_csv_row(
            self._record(), score=0.5, threshold=0.05, include_title=True
        )
        assert 'title' in row
        assert list(row.keys())[-1] == 'title'
        assert row['title'].startswith('A new benchmark')

    def test_include_title_false_omits_title_column(self):
        row = build_csv_row(
            self._record(), score=0.5, threshold=0.05, include_title=False
        )
        assert 'title' not in row


# ---------------------------------------------------------------------------
# CLI tests for `cmmanage discover stream`
# ---------------------------------------------------------------------------


def _fake_records(n: int):
    """Build ``n`` canned OpenAlex-shaped records for the fake fetch."""
    return [
        {
            'openalex_id': f'https://openalex.org/W{1000 + i}',
            'doi': f'https://doi.org/10.1234/paper-{i}' if i % 2 == 0 else None,
            'title': f'Paper {i}',
            'abstract': f'Abstract for paper {i}.',
            'publication_year': 2020 + (i % 5),
        }
        for i in range(n)
    ]


def _patched_stream_env(records, scores):
    """
    Return the stack of context managers needed to run the stream command
    offline with a canned fetch and a canned classifier.
    """
    def fake_fetch_works(*, state, limit, year_lower_bound, **kwargs):
        for rec in records[:limit]:
            state.mark_seen(rec['openalex_id'])
            yield rec

    fake_classifier = MagicMock()
    fake_classifier.predict_score.side_effect = list(scores)

    return [
        patch(
            'chem_mat_data.manage.discovery_openalex_fetch_works',
            side_effect=fake_fetch_works,
        ),
        patch(
            'chem_mat_data.manage.discovery_is_ml_stack_available',
            return_value=True,
        ),
        patch(
            'chem_mat_data.manage.discovery_ensure_classifier_model',
            return_value=Path('/tmp/fake-model'),
        ),
        patch(
            'chem_mat_data.manage.discovery_load_classifier',
            return_value=fake_classifier,
        ),
    ]


def _read_csv(path: Path):
    with path.open(encoding='utf-8') as f:
        return list(csv.DictReader(f))


def test_stream_help():
    runner = CliRunner()
    result = runner.invoke(cli, ['discover', 'stream', '--help'])
    assert result.exit_code == 0, result.output
    for option in (
        '--out',
        '--limit',
        '--threshold',
        '--flagged-only',
        '--include-title',
    ):
        assert option in result.output


def test_stream_group_registered():
    runner = CliRunner()
    result = runner.invoke(cli, ['discover', '--help'])
    assert result.exit_code == 0
    assert 'stream' in result.output


def test_stream_writes_header_and_all_rows(tmp_path):
    out = tmp_path / 'stream.csv'
    records = _fake_records(3)
    scores = [0.91, 0.02, 0.50]

    patches = _patched_stream_env(records, scores)
    for p in patches:
        p.start()
    try:
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                'discover',
                'stream',
                '--out',
                str(out),
                '--limit',
                '3',
                '--threshold',
                '0.05',
                '--model-url',
                'http://example/ignored.tar.gz',
            ],
        )
    finally:
        for p in patches:
            p.stop()

    assert result.exit_code == 0, result.output
    assert out.exists()
    rows = _read_csv(out)
    assert len(rows) == 3
    assert rows[0]['openalex_id'] == 'W1000'
    assert rows[0]['url'] == 'https://doi.org/10.1234/paper-0'
    assert rows[0]['flagged'] == '1'
    assert rows[1]['flagged'] == '0'
    assert rows[2]['flagged'] == '1'  # 0.50 > 0.05
    # No DOI on index 1 → OpenAlex landing fallback.
    assert rows[1]['url'] == 'https://openalex.org/W1001'
    # include_title defaults to False.
    assert 'title' not in rows[0]


def test_stream_flagged_only_filters_output(tmp_path):
    out = tmp_path / 'stream.csv'
    records = _fake_records(3)
    scores = [0.91, 0.02, 0.50]

    patches = _patched_stream_env(records, scores)
    for p in patches:
        p.start()
    try:
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                'discover',
                'stream',
                '--out',
                str(out),
                '--limit',
                '3',
                '--threshold',
                '0.05',
                '--flagged-only',
                '--model-url',
                'http://example/ignored.tar.gz',
            ],
        )
    finally:
        for p in patches:
            p.stop()

    assert result.exit_code == 0, result.output
    rows = _read_csv(out)
    ids = [r['openalex_id'] for r in rows]
    assert 'W1001' not in ids  # score 0.02 below threshold, filtered out
    assert set(ids) == {'W1000', 'W1002'}
    assert all(r['flagged'] == '1' for r in rows)


def test_stream_include_title_adds_column(tmp_path):
    out = tmp_path / 'stream.csv'
    records = _fake_records(2)
    scores = [0.9, 0.1]

    patches = _patched_stream_env(records, scores)
    for p in patches:
        p.start()
    try:
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                'discover',
                'stream',
                '--out',
                str(out),
                '--limit',
                '2',
                '--include-title',
                '--model-url',
                'http://example/ignored.tar.gz',
            ],
        )
    finally:
        for p in patches:
            p.stop()

    assert result.exit_code == 0, result.output
    rows = _read_csv(out)
    assert 'title' in rows[0]
    assert rows[0]['title'] == 'Paper 0'


def test_stream_resume_skips_already_seen_ids(tmp_path):
    out = tmp_path / 'stream.csv'
    # Pre-seed the seen-IDs sidecar: W1000 and W1001 were supposedly processed
    # in a previous run, so only W1002 should land in the CSV this time.
    (tmp_path / 'stream.csv.ids').write_text(
        'https://openalex.org/W1000\nhttps://openalex.org/W1001\n',
        encoding='utf-8',
    )
    # The canned fetch respects FetchState.seen_ids (it is the real
    # fetch_works implementation that does the skipping; our fake reproduces
    # that behaviour here).
    records = _fake_records(3)

    def fake_fetch_works(*, state, limit, year_lower_bound, **kwargs):
        for rec in records:
            if rec['openalex_id'] in state.seen_ids:
                continue
            state.mark_seen(rec['openalex_id'])
            yield rec

    fake_classifier = MagicMock()
    fake_classifier.predict_score.side_effect = [0.99]

    with (
        patch(
            'chem_mat_data.manage.discovery_openalex_fetch_works',
            side_effect=fake_fetch_works,
        ),
        patch(
            'chem_mat_data.manage.discovery_is_ml_stack_available',
            return_value=True,
        ),
        patch(
            'chem_mat_data.manage.discovery_ensure_classifier_model',
            return_value=Path('/tmp/fake-model'),
        ),
        patch(
            'chem_mat_data.manage.discovery_load_classifier',
            return_value=fake_classifier,
        ),
    ):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                'discover',
                'stream',
                '--out',
                str(out),
                '--limit',
                '10',
                '--model-url',
                'http://example/ignored.tar.gz',
            ],
        )

    assert result.exit_code == 0, result.output
    rows = _read_csv(out)
    assert len(rows) == 1
    assert rows[0]['openalex_id'] == 'W1002'
