"""
Unit tests for :mod:`chem_mat_data.discovery.openalex`.

The tests stub out the ``requests.Session.get`` call used by the fetch helper
so that no network traffic is required. This keeps the test suite fast and
deterministic while still exercising the full response-parsing and
pagination-control logic.
"""
from unittest.mock import MagicMock, patch

import pytest

from chem_mat_data.discovery.openalex import (
    DEFAULT_CONCEPT_IDS,
    build_filter_string,
    fetch_works,
    reconstruct_abstract,
    to_record,
)
from chem_mat_data.discovery.io import FetchState


class TestReconstructAbstract:
    """Tests for the inverted-index abstract reconstruction."""

    def test_empty_input_returns_empty_string(self):
        assert reconstruct_abstract(None) == ''
        assert reconstruct_abstract({}) == ''

    def test_simple_reconstruction(self):
        inv = {
            'A': [0],
            'novel': [1],
            'polymer': [2],
            'dataset': [3],
        }
        assert reconstruct_abstract(inv) == 'A novel polymer dataset'

    def test_repeated_tokens_are_placed_at_all_positions(self):
        inv = {
            'the': [0, 3],
            'cat': [1],
            'sat': [2],
            'mat': [4],
        }
        assert reconstruct_abstract(inv) == 'the cat sat the mat'

    def test_gaps_are_dropped_rather_than_empty_slots(self):
        # Missing position 1 should collapse, not leave an empty token.
        inv = {'alpha': [0], 'gamma': [2]}
        assert reconstruct_abstract(inv) == 'alpha gamma'


class TestBuildFilterString:
    """Tests for the filter-string construction used in OpenAlex requests."""

    def test_default_filter_contains_year_bound_and_concepts(self):
        s = build_filter_string(year_lower_bound=2015)
        parts = s.split(',')
        assert 'has_abstract:true' in parts
        assert 'publication_year:>2014' in parts
        # Both canonical concept IDs should be present, pipe-separated.
        concept_entry = next(p for p in parts if p.startswith('concepts.id:'))
        for cid in DEFAULT_CONCEPT_IDS:
            assert cid in concept_entry

    def test_custom_concepts_override_defaults(self):
        s = build_filter_string(
            year_lower_bound=2010,
            concept_ids=['https://openalex.org/C12345'],
        )
        concept_entry = next(p for p in s.split(',') if p.startswith('concepts.id:'))
        assert concept_entry == 'concepts.id:https://openalex.org/C12345'


class TestToRecord:
    """Tests for the OpenAlex-work → minimal-record projection."""

    def test_maps_all_fields(self):
        obj = {
            'id': 'https://openalex.org/W1',
            'doi': 'https://doi.org/10.1234/foo',
            'title': 'A Paper',
            'abstract_inverted_index': {'Hello': [0], 'world': [1]},
            'publication_year': 2021,
        }
        rec = to_record(obj)
        assert rec == {
            'openalex_id': 'https://openalex.org/W1',
            'doi': 'https://doi.org/10.1234/foo',
            'title': 'A Paper',
            'abstract': 'Hello world',
            'publication_year': 2021,
        }

    def test_missing_abstract_is_empty_string(self):
        obj = {'id': 'W1', 'title': 'T', 'abstract_inverted_index': None}
        rec = to_record(obj)
        assert rec['abstract'] == ''

    def test_missing_doi_is_none(self):
        obj = {'id': 'W1', 'title': 'T', 'abstract_inverted_index': {'x': [0]}}
        rec = to_record(obj)
        assert rec['doi'] is None


def _fake_response(items, next_cursor='next-token-1'):
    """Return a MagicMock that mimics a successful ``requests.Response``."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        'results': items,
        'meta': {'next_cursor': next_cursor},
    }
    return resp


class TestFetchWorks:
    """Tests for the resumable ``fetch_works`` generator."""

    def test_yields_records_until_limit(self, tmp_path):
        items_page_1 = [
            {
                'id': f'https://openalex.org/W{i}',
                'title': f'Paper {i}',
                'abstract_inverted_index': {'abstract': [0], str(i): [1]},
                'publication_year': 2020,
            }
            for i in range(3)
        ]
        items_page_2 = [
            {
                'id': f'https://openalex.org/W{i}',
                'title': f'Paper {i}',
                'abstract_inverted_index': {'abstract': [0]},
                'publication_year': 2020,
            }
            for i in range(3, 6)
        ]

        def fake_get(url, params, timeout=30):
            # Return different pages based on the cursor.
            if params['cursor'] == '*':
                return _fake_response(items_page_1, next_cursor='cursor-2')
            return _fake_response(items_page_2, next_cursor='cursor-3')

        state = FetchState(tmp_path / 'papers.jsonl')
        with patch(
            'chem_mat_data.discovery.openalex._session_get', side_effect=fake_get
        ):
            with state:
                records = list(fetch_works(state=state, limit=5, year_lower_bound=2010))

        assert len(records) == 5
        assert all('openalex_id' in r for r in records)
        # Cursor should have been advanced and persisted.
        assert state.cursor in {'cursor-2', 'cursor-3'}
        assert (tmp_path / 'papers.jsonl.cursor').exists()

    def test_skips_already_seen_ids(self, tmp_path):
        items = [
            {
                'id': f'https://openalex.org/W{i}',
                'title': f'Paper {i}',
                'abstract_inverted_index': {'a': [0]},
                'publication_year': 2020,
            }
            for i in range(3)
        ]

        def fake_get(url, params, timeout=30):
            return _fake_response(items, next_cursor='end')

        state = FetchState(tmp_path / 'papers.jsonl')
        # Pre-seed the seen set: the fetcher must skip these.
        (tmp_path / 'papers.jsonl.ids').write_text(
            'https://openalex.org/W0\nhttps://openalex.org/W1\n',
            encoding='utf-8',
        )

        with patch(
            'chem_mat_data.discovery.openalex._session_get', side_effect=fake_get
        ):
            with state:
                records = list(fetch_works(state=state, limit=10, year_lower_bound=2010))

        # Only W2 should remain, plus the loop should terminate when a page
        # contains nothing new (all results are duplicates on the second call).
        yielded_ids = [r['openalex_id'] for r in records]
        assert 'https://openalex.org/W0' not in yielded_ids
        assert 'https://openalex.org/W1' not in yielded_ids
        assert 'https://openalex.org/W2' in yielded_ids

    def test_stops_on_http_error(self, tmp_path):
        error_resp = MagicMock()
        error_resp.status_code = 500
        error_resp.json.return_value = {}

        state = FetchState(tmp_path / 'papers.jsonl')
        with patch(
            'chem_mat_data.discovery.openalex._session_get', return_value=error_resp
        ):
            with state:
                records = list(fetch_works(state=state, limit=5, year_lower_bound=2010))

        assert records == []

    def test_skips_a_fully_seen_page_and_continues(self, tmp_path):
        """
        A single mid-traversal page where every item is already in ``seen_ids``
        must not abort the fetch. The generator should advance the cursor past
        it and pick up fresh records from the following page.
        """
        seen_page = [
            {
                'id': f'https://openalex.org/W{i}',
                'title': f'Seen {i}',
                'abstract_inverted_index': {'a': [0]},
                'publication_year': 2020,
            }
            for i in range(3)
        ]
        fresh_page = [
            {
                'id': f'https://openalex.org/W{i}',
                'title': f'Fresh {i}',
                'abstract_inverted_index': {'abstract': [0]},
                'publication_year': 2020,
            }
            for i in range(100, 103)
        ]

        responses = iter(
            [
                _fake_response(seen_page, next_cursor='cursor-2'),
                _fake_response(fresh_page, next_cursor='cursor-3'),
                _fake_response([], next_cursor='cursor-4'),  # end-of-index
            ]
        )

        def fake_get(url, params, timeout=30):
            return next(responses)

        state = FetchState(tmp_path / 'papers.jsonl')
        # Pre-seed every ID of the first page so that the fetcher finds the
        # entire page already seen and must skip forward.
        (tmp_path / 'papers.jsonl.ids').write_text(
            ''.join(f'https://openalex.org/W{i}\n' for i in range(3)),
            encoding='utf-8',
        )

        with patch(
            'chem_mat_data.discovery.openalex._session_get', side_effect=fake_get
        ):
            with state:
                records = list(
                    fetch_works(state=state, limit=5, year_lower_bound=2010)
                )

        # Fresh page should have been reached and fully consumed.
        assert [r['openalex_id'] for r in records] == [
            'https://openalex.org/W100',
            'https://openalex.org/W101',
            'https://openalex.org/W102',
        ]

    def test_stops_after_max_consecutive_empty_pages(self, tmp_path):
        """
        If every page keeps coming back fully seen, the fetcher must still
        eventually give up rather than looping indefinitely.
        """
        from chem_mat_data.discovery.openalex import MAX_EMPTY_PAGES

        seen_items = [
            {
                'id': f'https://openalex.org/W{i}',
                'title': f'Seen {i}',
                'abstract_inverted_index': {'a': [0]},
                'publication_year': 2020,
            }
            for i in range(3)
        ]
        # Every call returns another all-seen page with a fresh cursor token.
        call_count = {'n': 0}

        def fake_get(url, params, timeout=30):
            call_count['n'] += 1
            return _fake_response(
                seen_items, next_cursor=f'cursor-{call_count["n"]}'
            )

        state = FetchState(tmp_path / 'papers.jsonl')
        (tmp_path / 'papers.jsonl.ids').write_text(
            ''.join(f'https://openalex.org/W{i}\n' for i in range(3)),
            encoding='utf-8',
        )

        with patch(
            'chem_mat_data.discovery.openalex._session_get', side_effect=fake_get
        ):
            with state:
                records = list(
                    fetch_works(state=state, limit=5, year_lower_bound=2010)
                )

        assert records == []
        # Must have bailed after exactly MAX_EMPTY_PAGES attempts, not looped forever.
        assert call_count['n'] == MAX_EMPTY_PAGES

    def test_records_without_title_or_abstract_are_skipped(self, tmp_path):
        items = [
            {
                'id': 'https://openalex.org/W1',
                'title': None,
                'abstract_inverted_index': {'a': [0]},
                'publication_year': 2020,
            },
            {
                'id': 'https://openalex.org/W2',
                'title': 'Has title',
                'abstract_inverted_index': None,
                'publication_year': 2020,
            },
            {
                'id': 'https://openalex.org/W3',
                'title': 'OK',
                'abstract_inverted_index': {'a': [0]},
                'publication_year': 2020,
            },
        ]

        def fake_get(url, params, timeout=30):
            return _fake_response(items, next_cursor='end')

        state = FetchState(tmp_path / 'papers.jsonl')
        with patch(
            'chem_mat_data.discovery.openalex._session_get', side_effect=fake_get
        ):
            with state:
                records = list(fetch_works(state=state, limit=10, year_lower_bound=2010))

        assert len(records) == 1
        assert records[0]['openalex_id'] == 'https://openalex.org/W3'
