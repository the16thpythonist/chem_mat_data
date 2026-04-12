"""
Unit tests for :mod:`chem_mat_data.discovery.io`.

These tests exercise the on-disk JSONL helpers and the :class:`FetchState`
bookkeeping used by the OpenAlex fetcher to resume paginated runs without
re-downloading records.
"""
import json
import os

import pytest

from chem_mat_data.discovery.io import (
    FetchState,
    iter_jsonl,
    write_jsonl_line,
)


class TestIterJsonl:
    """Tests for ``iter_jsonl`` which streams JSON objects from a ``.jsonl`` file."""

    def test_reads_all_records_in_order(self, tmp_path):
        path = tmp_path / 'papers.jsonl'
        records = [
            {'id': 'W1', 'title': 'First paper'},
            {'id': 'W2', 'title': 'Second paper'},
            {'id': 'W3', 'title': 'Third paper'},
        ]
        with path.open('w', encoding='utf-8') as f:
            for r in records:
                f.write(json.dumps(r) + '\n')

        result = list(iter_jsonl(path))
        assert result == records

    def test_skips_blank_lines(self, tmp_path):
        path = tmp_path / 'papers.jsonl'
        path.write_text('\n{"id": "W1"}\n\n{"id": "W2"}\n\n', encoding='utf-8')
        result = list(iter_jsonl(path))
        assert result == [{'id': 'W1'}, {'id': 'W2'}]

    def test_skips_malformed_lines_without_raising(self, tmp_path):
        path = tmp_path / 'papers.jsonl'
        path.write_text(
            '{"id": "W1"}\n'
            'not-valid-json\n'
            '{"id": "W2"}\n',
            encoding='utf-8',
        )
        result = list(iter_jsonl(path))
        assert result == [{'id': 'W1'}, {'id': 'W2'}]

    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            list(iter_jsonl(tmp_path / 'does_not_exist.jsonl'))


class TestWriteJsonlLine:
    """Tests for ``write_jsonl_line`` which appends a single JSON object."""

    def test_appends_line_with_trailing_newline(self, tmp_path):
        path = tmp_path / 'out.jsonl'
        with path.open('w', encoding='utf-8') as f:
            write_jsonl_line(f, {'id': 'W1', 'title': 'Foo'})
            write_jsonl_line(f, {'id': 'W2', 'title': 'Bar'})

        content = path.read_text(encoding='utf-8')
        assert content == '{"id": "W1", "title": "Foo"}\n{"id": "W2", "title": "Bar"}\n'

    def test_preserves_unicode_without_escapes(self, tmp_path):
        path = tmp_path / 'out.jsonl'
        with path.open('w', encoding='utf-8') as f:
            write_jsonl_line(f, {'title': 'Schrödinger'})

        content = path.read_text(encoding='utf-8')
        assert 'Schrödinger' in content
        assert '\\u' not in content


class TestFetchState:
    """
    Tests for :class:`FetchState` which persists the pagination cursor and the
    set of already-seen OpenAlex IDs next to a target JSONL file.
    """

    def test_initial_state_is_empty(self, tmp_path):
        state = FetchState(tmp_path / 'papers.jsonl')
        assert state.cursor == '*'
        assert state.seen_ids == set()

    def test_load_reads_existing_cursor_and_ids(self, tmp_path):
        jsonl_path = tmp_path / 'papers.jsonl'
        cursor_path = tmp_path / 'papers.jsonl.cursor'
        ids_path = tmp_path / 'papers.jsonl.ids'

        cursor_path.write_text('abc123==', encoding='utf-8')
        ids_path.write_text(
            'https://openalex.org/W1\n'
            'https://openalex.org/W2\n',
            encoding='utf-8',
        )

        state = FetchState(jsonl_path)
        state.load()
        assert state.cursor == 'abc123=='
        assert state.seen_ids == {
            'https://openalex.org/W1',
            'https://openalex.org/W2',
        }

    def test_load_ignores_blank_lines_in_ids_file(self, tmp_path):
        jsonl_path = tmp_path / 'papers.jsonl'
        ids_path = tmp_path / 'papers.jsonl.ids'
        ids_path.write_text('\nW1\n\n  \nW2\n', encoding='utf-8')

        state = FetchState(jsonl_path)
        state.load()
        assert state.seen_ids == {'W1', 'W2'}

    def test_mark_seen_persists_immediately(self, tmp_path):
        jsonl_path = tmp_path / 'papers.jsonl'
        state = FetchState(jsonl_path)
        state.open()

        try:
            state.mark_seen('W1')
            state.mark_seen('W2')
            assert state.seen_ids == {'W1', 'W2'}

            # The ids file should already contain both entries without an
            # explicit ``save`` — this is what protects us against crashes
            # mid-run.
            ids_path = tmp_path / 'papers.jsonl.ids'
            content = ids_path.read_text(encoding='utf-8')
            assert 'W1' in content
            assert 'W2' in content
        finally:
            state.close()

    def test_update_cursor_writes_file(self, tmp_path):
        jsonl_path = tmp_path / 'papers.jsonl'
        state = FetchState(jsonl_path)
        state.update_cursor('next-page-token')

        cursor_path = tmp_path / 'papers.jsonl.cursor'
        assert cursor_path.read_text(encoding='utf-8') == 'next-page-token'

    def test_update_cursor_with_none_is_a_noop(self, tmp_path):
        jsonl_path = tmp_path / 'papers.jsonl'
        state = FetchState(jsonl_path)
        state.update_cursor('first')
        state.update_cursor(None)

        cursor_path = tmp_path / 'papers.jsonl.cursor'
        assert cursor_path.read_text(encoding='utf-8') == 'first'

    def test_creates_parent_directory_on_write(self, tmp_path):
        jsonl_path = tmp_path / 'subdir' / 'papers.jsonl'
        state = FetchState(jsonl_path)
        state.update_cursor('tok')
        assert os.path.isdir(tmp_path / 'subdir')
