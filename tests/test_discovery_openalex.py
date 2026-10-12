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
    BLACKLISTED_DOI_PREFIXES,
    DEFAULT_CONCEPT_IDS,
    DEFAULT_TYPES,
    _is_blacklisted_doi,
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
        assert 'language:en' in parts
        assert 'publication_year:>2014' in parts
        # Both canonical concept IDs should be present, pipe-separated.
        concept_entry = next(p for p in parts if p.startswith('concepts.id:'))
        for cid in DEFAULT_CONCEPT_IDS:
            assert cid in concept_entry

    def test_default_types_exclude_book_chapters_and_reviews(self):
        assert 'book-chapter' not in DEFAULT_TYPES
        assert 'review-article' not in DEFAULT_TYPES
        assert 'journal-article' in DEFAULT_TYPES
        assert 'proceedings-article' in DEFAULT_TYPES
        assert 'report' in DEFAULT_TYPES

    def test_custom_concepts_override_defaults(self):
        s = build_filter_string(
            year_lower_bound=2010,
            concept_ids=['https://openalex.org/C12345'],
        )
        concept_entry = next(p for p in s.split(',') if p.startswith('concepts.id:'))
        assert concept_entry == 'concepts.id:https://openalex.org/C12345'


class TestIsBlacklistedDoi:
    """Tests for the DOI-prefix blacklist helper."""

    def test_none_doi_is_not_blacklisted(self):
        assert _is_blacklisted_doi(None) is False

    def test_empty_doi_is_not_blacklisted(self):
        assert _is_blacklisted_doi('') is False

    def test_blacklisted_bare_doi(self):
        assert _is_blacklisted_doi('10.55277/researchhub.xyz') is True

    def test_blacklisted_full_url_doi(self):
        assert _is_blacklisted_doi('https://doi.org/10.46943/xi.conedu.2025') is True

    def test_clean_doi_is_not_blacklisted(self):
        assert _is_blacklisted_doi('https://doi.org/10.1038/s41586-023-06735-9') is False
        assert _is_blacklisted_doi('10.1021/acs.jctc.1c00471') is False

    def test_custom_prefixes(self):
        assert _is_blacklisted_doi('10.9999/foo', prefixes=('10.9999/',)) is True
        assert _is_blacklisted_doi('10.1234/bar', prefixes=('10.9999/',)) is False


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

    def test_blacklisted_doi_records_are_skipped(self, tmp_path):
        """Records with a blacklisted DOI prefix are skipped but marked seen."""
        items = [
            {
                'id': 'https://openalex.org/W1',
                'doi': 'https://doi.org/10.55277/researchhub.xyz',
                'title': 'Spam entry',
                'abstract_inverted_index': {'spam': [0]},
                'publication_year': 2025,
            },
            {
                'id': 'https://openalex.org/W2',
                'doi': 'https://doi.org/10.1038/s41586-023-06735-9',
                'title': 'Good paper',
                'abstract_inverted_index': {'real': [0], 'data': [1]},
                'publication_year': 2023,
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
        assert records[0]['openalex_id'] == 'https://openalex.org/W2'
        # The blacklisted record should still be marked as seen.
        assert 'https://openalex.org/W1' in state.seen_ids


# == the cursor must not outrun the classifier ==
#
# `fetch_works` requests full pages regardless of the caller's remaining limit.
# It used to persist `next_cursor` even when it broke out mid-page on `limit`,
# which put the unexamined remainder of that page permanently behind the cursor:
# never classified, and never revisited, because nothing else ever goes back.

def _page(start, count, cursor):
    """Build one fake OpenAlex response page of `count` usable records."""
    return {
        'results': [
            {
                'id': f'https://openalex.org/W{start + i}',
                'doi': f'https://doi.org/10.1234/w{start + i}',
                'title': f'Paper {start + i}',
                'abstract_inverted_index': {'A': [0], 'dataset': [1]},
                'publication_year': 2026,
            }
            for i in range(count)
        ],
        'meta': {'next_cursor': cursor},
    }


def _run_fetch(monkeypatch, tmp_path, pages, limit):
    """Drive fetch_works over canned pages, returning (records, final cursor)."""
    from chem_mat_data.discovery import openalex
    from chem_mat_data.discovery.io import FetchState

    served = []

    class _Response:
        status_code = 200

        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    def fake_get(url, params, timeout=30):
        served.append(params.get('cursor'))
        index = len(served) - 1
        if index >= len(pages):
            return _Response({'results': [], 'meta': {'next_cursor': None}})
        return _Response(pages[index])

    monkeypatch.setattr(openalex, '_session_get', fake_get)

    state = FetchState(tmp_path / 'out.csv')
    state.load()
    with state:
        records = list(openalex.fetch_works(state, limit=limit, page_size=10))
    return records, state.cursor


def test_a_mid_page_stop_does_not_advance_the_cursor(monkeypatch, tmp_path):
    """The regression: 6 of a 10-record page consumed, 4 must not be skipped."""
    pages = [_page(1, 10, 'cursor2'), _page(11, 10, 'cursor3')]
    records, cursor = _run_fetch(monkeypatch, tmp_path, pages, limit=6)

    assert len(records) == 6
    assert cursor == '*', (
        'the cursor advanced past 4 records that were never classified'
    )


def test_the_run_resumes_at_the_first_unexamined_record(monkeypatch, tmp_path):
    """
    The other half of the fix: holding the cursor must not re-yield what was
    already seen. The second run re-fetches the page and `seen_ids` filters it.
    """
    from chem_mat_data.discovery import openalex
    from chem_mat_data.discovery.io import FetchState

    pages = [_page(1, 10, 'cursor2'), _page(11, 10, 'cursor3')]
    first, _ = _run_fetch(monkeypatch, tmp_path, pages, limit=6)
    assert [r['openalex_id'] for r in first] == [f'https://openalex.org/W{i}' for i in range(1, 7)]

    # Second run over the same state: the same first page is served again.
    served = []

    class _Response:
        status_code = 200

        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    # Serve by cursor, not by call count. The earlier version handed out pages in
    # call order and so could not tell a held cursor from a advanced one - it was
    # a seen_ids dedup test wearing a cursor test's name.
    by_cursor = {'*': pages[0], 'cursor2': pages[1]}

    def fake_get(url, params, timeout=30):
        cursor = params.get('cursor')
        served.append(cursor)
        return _Response(by_cursor.get(cursor, {'results': [], 'meta': {'next_cursor': None}}))

    monkeypatch.setattr(openalex, '_session_get', fake_get)
    state = FetchState(tmp_path / 'out.csv')
    state.load()
    with state:
        second = list(openalex.fetch_works(state, limit=4, page_size=10))

    assert served[0] == '*', (
        'the second run must re-request the page the first one stopped inside'
    )
    assert [r['openalex_id'] for r in second] == [
        f'https://openalex.org/W{i}' for i in range(7, 11)
    ], 'the resumed run must pick up exactly where the first one stopped'


def test_consuming_a_page_exactly_still_advances(monkeypatch, tmp_path):
    """Hitting the limit on the final record consumed the whole page."""
    pages = [_page(1, 10, 'cursor2'), _page(11, 10, 'cursor3')]
    _, cursor = _run_fetch(monkeypatch, tmp_path, pages, limit=10)
    assert cursor == 'cursor2', 'a fully consumed page must not be re-fetched'
