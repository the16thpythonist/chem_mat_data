"""
OpenAlex client for the dataset-discovery pipeline.

This module is a slimmed-down version of the upstream ``fetch_oa_in_order.py``
from the Dataset-Discovery-Pipeline repository, refactored into reusable
functions that the ``cmmanage discover fetch`` command can drive.

The public surface is intentionally narrow:

- :func:`reconstruct_abstract` — expand the OpenAlex inverted-index abstract
  representation back into plain text.
- :func:`build_filter_string` — compose the OpenAlex ``filter=`` parameter
  from a year lower bound plus optional concept-ID whitelist.
- :func:`to_record` — project a raw OpenAlex work into the minimal record
  shape used everywhere downstream (``openalex_id`` / ``title`` / ``abstract``
  / ``publication_year``).
- :func:`fetch_works` — a generator that drives cursor pagination, de-duplicates
  against a :class:`~chem_mat_data.discovery.io.FetchState`, and yields up to a
  caller-specified number of new records.

The fetch helper keeps all network access behind the private ``_session_get``
function so that tests can monkey-patch it without having to build a fake
``requests`` module.
"""
from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from chem_mat_data.discovery.io import FetchState

#: Base URL for the OpenAlex ``/works`` endpoint.
OPENALEX_WORKS_URL = 'https://api.openalex.org/works'

#: Default concept IDs used when the caller does not supply an override.
#: The upstream pipeline uses these two concepts to narrow the search to
#: chemistry (C185592680) and materials science (C192562407).
DEFAULT_CONCEPT_IDS: List[str] = [
    'https://openalex.org/C185592680',
    'https://openalex.org/C192562407',
]

#: Default ``type=`` filter values accepted by the pipeline. Book chapters and
#: review articles are excluded because they almost never release reusable
#: molecular or materials datasets — they dominated the false-positive list
#: when the discovery stream was validated against the LLM identify agent.
DEFAULT_TYPES: List[str] = [
    'journal-article',
    'proceedings-article',
    'report',
]

#: DOI prefixes known to yield only spam, placeholder entries, or publishers
#: that exclusively produce content outside chemistry/materials science (e.g.
#: education conferences, Turkish book chapter publishers, Brazilian e-book
#: series). Records whose DOI starts with any of these prefixes are silently
#: skipped during fetching so they never reach the classifier.
BLACKLISTED_DOI_PREFIXES: Tuple[str, ...] = (
    '10.55277/',    # ResearchHub — spam / placeholder entries
    '10.46943/',    # CONEDU education conference (all 404s)
    '10.37609/',    # Turkish book chapter publisher (akya)
    '10.58830/',    # Turkish book chapter publisher (ozgur)
    '10.56238/',    # Brazilian book chapter publisher (sevened)
    '10.37885/',    # Brazilian book chapter publisher
    '10.63330/',    # Brazilian book chapter publisher (aurumpub)
    '10.29327/',    # Brazilian e-book publisher
)

#: Fields to request via ``select=`` to keep response payloads small.
SELECT_FIELDS = 'id,doi,title,abstract_inverted_index,publication_year'

#: Maximum page size allowed by OpenAlex.
MAX_PAGE_SIZE = 200

#: Number of consecutive fully-already-seen pages the fetcher will skip over
#: before giving up. This protects against runaway loops when the tail of the
#: index fully overlaps with ``seen_ids``, while still letting the fetcher
#: cross a contiguous run of fully-seen pages to reach fresh records on the
#: other side. At the maximum page size of 200 this lets us skip up to
#: ``20 * 200 = 4000`` already-seen records before concluding that we have
#: actually reached the end of the useful tail.
MAX_EMPTY_PAGES = 20


def reconstruct_abstract(inv_idx: Optional[Dict[str, List[int]]]) -> str:
    """
    Rebuild a plain-text abstract from the OpenAlex inverted index.

    OpenAlex returns abstracts as a ``{token: [positions]}`` mapping. This
    function reverses that into a space-joined string, preserving token order
    and repetitions. Missing positions collapse rather than becoming empty
    tokens — the upstream implementation does the same.

    :param inv_idx: The ``abstract_inverted_index`` field of an OpenAlex work,
        or ``None`` if the work has no abstract.

    :returns: The reconstructed abstract, or an empty string if the input is
        empty.
    """
    if not inv_idx:
        return ''
    positions = [p for ps in inv_idx.values() for p in ps]
    if not positions:
        return ''
    words: List[str] = [''] * (max(positions) + 1)
    for token, ps in inv_idx.items():
        for p in ps:
            words[p] = token
    return ' '.join(w for w in words if w)


def build_filter_string(
    year_lower_bound: int,
    concept_ids: Optional[List[str]] = None,
    types: Optional[List[str]] = None,
) -> str:
    """
    Compose the OpenAlex ``filter=`` parameter.

    The filter restricts results to English-language works that have an
    abstract, belong to at least one of the requested concept categories, and
    match one of the allowed publication types. Non-English papers are excluded
    because the TinyLlama classifier produces unreliable (typically very high)
    scores on text it cannot understand.

    :param year_lower_bound: Only works published from this year onward are
        returned. Converted to a strict-greater-than filter on the wire.
    :param concept_ids: Optional concept-ID whitelist. Defaults to
        :data:`DEFAULT_CONCEPT_IDS` (chemistry + materials science).
    :param types: Optional publication-type whitelist. Defaults to
        :data:`DEFAULT_TYPES`.

    :returns: A comma-joined filter string ready for the request query.
    """
    if concept_ids is None:
        concept_ids = DEFAULT_CONCEPT_IDS
    if types is None:
        types = DEFAULT_TYPES

    filters = [
        'has_abstract:true',
        'language:en',
        'concepts.id:' + '|'.join(concept_ids),
        'type:' + '|'.join(types),
        f'publication_year:>{year_lower_bound - 1}',
    ]
    return ','.join(filters)


def to_record(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Project a raw OpenAlex work into the minimal record shape.

    :param obj: Raw work dict as returned by OpenAlex.

    :returns: Dict with keys ``openalex_id``, ``doi``, ``title``, ``abstract``,
        ``publication_year``. ``doi`` is returned verbatim (OpenAlex may give
        either a bare ``10.xxxx/...`` identifier or a full ``https://doi.org/...``
        URL) so downstream callers can normalize it as needed.
    """
    return {
        'openalex_id': obj.get('id'),
        'doi': obj.get('doi'),
        'title': obj.get('title'),
        'abstract': reconstruct_abstract(obj.get('abstract_inverted_index')),
        'publication_year': obj.get('publication_year'),
    }


def _is_blacklisted_doi(
    doi: Optional[str],
    prefixes: Tuple[str, ...] = BLACKLISTED_DOI_PREFIXES,
) -> bool:
    """
    Return ``True`` if ``doi`` starts with any of the blacklisted prefixes.

    The function normalises both full URL forms (``https://doi.org/10.xxx``)
    and bare identifiers (``10.xxx``) before checking.
    """
    if not doi:
        return False

    bare = doi
    for schema in ('https://doi.org/', 'http://doi.org/'):
        if bare.startswith(schema):
            bare = bare[len(schema):]
            break

    return bare.startswith(prefixes)


def _build_session(retries: int = 3) -> requests.Session:
    """Construct a ``requests.Session`` with exponential-backoff retries."""
    session = requests.Session()
    retry_config = Retry(
        total=retries - 1,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    session.mount('https://', HTTPAdapter(max_retries=retry_config))
    return session


def _session_get(
    url: str, params: Dict[str, Any], timeout: int = 30
) -> Optional[requests.Response]:
    """
    Perform a single GET against OpenAlex.

    Isolated from :func:`fetch_works` so that tests can monkey-patch the
    network boundary without touching ``requests`` internals.
    """
    session = _build_session()
    try:
        return session.get(url, params=params, timeout=timeout)
    except requests.RequestException:
        return None


def fetch_works(
    state: FetchState,
    limit: int,
    year_lower_bound: int = 2010,
    concept_ids: Optional[List[str]] = None,
    types: Optional[List[str]] = None,
    page_size: int = MAX_PAGE_SIZE,
) -> Iterator[Dict[str, Any]]:
    """
    Yield up to ``limit`` new OpenAlex records, resuming from persisted state.

    The generator drives cursor pagination against the OpenAlex ``/works``
    endpoint and yields each new record that:

    1. Has not been seen before (according to ``state.seen_ids``).
    2. Has both a non-empty title and a non-empty abstract.

    Every yielded record is also registered with the :class:`FetchState` via
    :meth:`FetchState.mark_seen` so that interrupted runs can be resumed. The
    cursor is persisted after every batch.

    Iteration stops when any of the following occurs:

    - ``limit`` new records have been yielded.
    - OpenAlex returns an empty result page.
    - OpenAlex returns a non-200 response.
    - OpenAlex stops providing a ``next_cursor`` value.
    - A full page is received but every record in it is already in
      ``seen_ids`` (prevents infinite duplicate loops when the tail of the
      index overlaps with previously seen data).

    :param state: Persistent fetch state. Must already be opened (or used as
        a context manager) by the caller so that ``mark_seen`` writes can be
        flushed immediately.
    :param limit: Maximum number of new records to yield in this call.
    :param year_lower_bound: Only works from this year onward are requested.
    :param concept_ids: Optional concept-ID whitelist override.
    :param types: Optional publication-type whitelist override.
    :param page_size: Requested ``per-page`` value, capped at 200.

    :yields: Minimal record dicts as produced by :func:`to_record`.
    """
    if limit <= 0:
        return

    filter_string = build_filter_string(
        year_lower_bound=year_lower_bound,
        concept_ids=concept_ids,
        types=types,
    )
    page_size = min(page_size, MAX_PAGE_SIZE)
    fetched = 0
    consecutive_empty_pages = 0

    while fetched < limit:
        params = {
            'filter': filter_string,
            'select': SELECT_FIELDS,
            # Always request full pages regardless of the user's remaining
            # limit. API cost is the same for 25 vs 200 records, and fetching
            # full pages lets us skip past a large ``seen_ids`` set much
            # faster when resuming an existing run. We stop yielding once we
            # hit ``limit`` regardless of how many records the page holds.
            'per-page': page_size,
            'cursor': state.cursor,
            'sort': 'publication_year:desc,publication_date:desc',
        }
        response = _session_get(OPENALEX_WORKS_URL, params=params)
        if response is None or response.status_code != 200:
            return

        payload = response.json() or {}
        items = payload.get('results') or []
        if not items:
            return

        next_cursor = (payload.get('meta') or {}).get('next_cursor')
        if not next_cursor:
            return

        new_in_page = 0
        for obj in items:
            oid = obj.get('id')
            if not oid or oid in state.seen_ids:
                continue

            record = to_record(obj)
            if not record['title'] or not record['abstract']:
                # Still mark as seen so the next run does not re-examine it.
                state.mark_seen(oid)
                continue

            if _is_blacklisted_doi(record.get('doi')):
                state.mark_seen(oid)
                continue

            state.mark_seen(oid)
            fetched += 1
            new_in_page += 1
            yield record
            if fetched >= limit:
                break

        state.update_cursor(next_cursor)

        # When a page contains no new records we skip over it rather than
        # aborting — a single fully-already-seen page mid-traversal is
        # legitimate (e.g. when resuming a run that previously fetched parts
        # of an adjacent page). Only after ``MAX_EMPTY_PAGES`` consecutive
        # empty pages do we conclude we are actually at the end of the useful
        # index and stop, which still protects against runaway loops.
        if new_in_page == 0:
            consecutive_empty_pages += 1
            if consecutive_empty_pages >= MAX_EMPTY_PAGES:
                return
        else:
            consecutive_empty_pages = 0
